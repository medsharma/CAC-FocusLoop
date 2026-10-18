// FocusLoop face-track tool (macOS). Runs the app's own `VideoFaceAnalyzer` (Vision face landmarks) over every
// frame of a recorded video and writes `<video>.facetrack.csv` next to it: numbers only, no images.
//
// Needed for the iOS Simulator, whose Vision face-landmark model is broken: with the sidecar present the
// Simulator gets real blinks, yawns, brows and gaze for that video. Build + run with `tools/facetrack.sh`.
//
//   tools/facetrack.sh path/to/demo_face.mov [--start 0] [--duration 9999] [--mirrored] [--out file.csv]

import AVFoundation
import Foundation

@main
struct FaceTrackTool {
    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())
        guard let path = args.first, !path.hasPrefix("-") else {
            print("usage: facetrack <video> [--start s] [--duration s] [--mirrored] [--out file.csv]")
            exit(2)
        }
        args.removeFirst()
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        let start = Double(value("--start") ?? "") ?? 0
        let duration = Double(value("--duration") ?? "") ?? 1e6
        let video = URL(fileURLWithPath: path)
        let outURL = value("--out").map { URL(fileURLWithPath: $0) } ?? FaceTrack.sidecarURL(for: video)
        do {
            let track = try await analyse(video: video, start: start, duration: duration, mirrored: args.contains("--mirrored"))
            try track.csv().write(to: outURL, atomically: true, encoding: .utf8)
            summarize(track)
            print("wrote \(outURL.path)")
        } catch {
            print("facetrack failed: \(error)")
            exit(1)
        }
    }

    static func analyse(video: URL, start: Double, duration: Double, mirrored: Bool) async throws -> FaceTrack {
        let asset = AVURLAsset(url: video)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw NSError(domain: "facetrack", code: 1, userInfo: [NSLocalizedDescriptionKey: "no video track"])
        }
        let (transform, fps) = try await track.load(.preferredTransform, .nominalFrameRate)
        let orientation = VideoFaceAnalyzer.orientation(for: transform)
        let reader = try AVAssetReader(asset: asset)
        reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 600),
                                       duration: CMTime(seconds: duration, preferredTimescale: 600))
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? NSError(domain: "facetrack", code: 2) }
        let analyzer = VideoFaceAnalyzer(mode: .landmarks)
        analyzer.mirrored = mirrored
        var frames: [FaceTrack.Frame] = []
        var lastMesh = -Double.infinity
        var n = 0
        while let sb = output.copyNextSampleBuffer() {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let time = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            let wantMesh = time - lastMesh >= 1.0 / 15
            let r = analyzer.analyze(pixelBuffer: pb, orientation: orientation, t: time, wantMesh: wantMesh)
            if wantMesh, !r.mesh.isEmpty { lastMesh = time }
            frames.append(FaceTrack.Frame(time: time, sample: r.sample, mesh: wantMesh ? r.mesh : []))
            n += 1
            if n % 600 == 0 { print(String(format: "  %.0f s analysed", time - start)) }
        }
        if reader.status == .failed { throw reader.error ?? NSError(domain: "facetrack", code: 3) }
        return FaceTrack(frames: frames, metadata: [
            "video": video.lastPathComponent, "frames": String(frames.count), "fps": String(format: "%.2f", fps),
            "analyzer": "VideoFaceAnalyzer", "mode": analyzer.mode.rawValue, "mirrored": mirrored ? "1" : "0",
        ])
    }

    /// Feature sanity printout: the same numbers the app will compute from this track.
    static func summarize(_ track: FaceTrack) {
        let extractor = AttentionFeatureExtractor(workZone: .default)
        let hr = POSHeartRateEstimator()
        var blink = PerceptionBlinkDetector()
        var blinks = 0, faces = 0, offZone = 0, frames10 = 0
        var bpm: [Float] = []
        var lastHR = -1.0
        var maxJaw: Float = 0, maxYawns: Float = 0, maxNods: Float = 0
        var prev: Double?
        var faceSeconds = 0.0
        for f in track.frames {
            let s = f.sample(t: f.time)
            if s.faceDetected {
                faces += 1
                if let p = prev { faceSeconds += f.time - p }
                if blink.update(closure: ((s.blendShapes["eyeBlinkLeft"] ?? 0) + (s.blendShapes["eyeBlinkRight"] ?? 0)) / 2, t: f.time) != nil { blinks += 1 }
                maxJaw = max(maxJaw, s.blendShapes["jawOpen"] ?? 0)
            }
            prev = f.time
            extractor.ingest(face: s)
            if let rgb = s.skinRGB {
                hr.ingest(rgb: rgb, t: f.time, headMotion: extractor.headMotion)
                if let h = hr.latest, h.t != lastHR { lastHR = h.t; bpm.append(h.bpm); extractor.ingest(heartRate: h) }
            }
            if let fr = extractor.frame(at: f.time) {
                frames10 += 1
                if fr.faceDetected > 0.5 && fr.inWorkZone < 0.5 { offZone += 1 }
                maxYawns = max(maxYawns, fr.yawnCount); maxNods = max(maxNods, fr.nodCount)
            }
        }
        let n = max(1, track.frames.count)
        print(String(format: "frames %d, face in %.1f%%, blinks %d (%.1f/min of face time), max jawOpen %.2f",
                     track.frames.count, 100 * Double(faces) / Double(n), blinks, Double(blinks) / max(1, faceSeconds) * 60, maxJaw))
        print(String(format: "yawns (max in 60 s) %.0f, nods (max in 60 s) %.0f, off-work-zone %.0f%% of face time",
                     maxYawns, maxNods, 100 * Double(offZone) / Double(max(1, frames10))))
        let sorted = bpm.sorted()
        if !sorted.isEmpty { print(String(format: "camera heart rate: median %.0f BPM over %d estimates", sorted[sorted.count / 2], sorted.count)) }
    }
}
