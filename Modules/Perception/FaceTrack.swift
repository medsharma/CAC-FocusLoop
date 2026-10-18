import CoreGraphics
import Foundation
import simd

/// A face track: the `VideoFaceAnalyzer` output for every frame of a recorded video, keyed by the frame's
/// presentation time, as a small CSV sidecar next to the video (`<name>.facetrack.csv`). Numbers only: no
/// images, no pixels.
///
/// Why it exists: Vision's face-landmark model returns garbage in the iOS Simulator runtime (it works on
/// devices and on macOS), so in the Simulator blinks, yawns, brows and gaze can't be measured live.
/// `tools/facetrack.sh` runs the very same analyzer on the Mac and writes this file; in the Simulator
/// `VideoFaceSource` then plays the video and, for each frame it shows, takes that frame's numbers from the
/// track. Everything downstream (feature extraction, rPPG, classifier, forecaster, nudges) still runs live.
struct FaceTrack: Equatable {
    /// Blendshape columns, in `perceptionBlendShapeNames` order.
    static let blendShapeNames: [String] = [
        "eyeBlinkLeft", "eyeBlinkRight",
        "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
        "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
        "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
        "jawOpen", "browDownLeft", "browDownRight", "browInnerUp", "mouthClose",
    ]
    static let fileSuffix = ".facetrack.csv"
    static let formatLine = "# FocusLoop face track v1: numbers only (Vision face analysis per video frame; no images)."

    struct Frame: Equatable {
        /// Presentation time of the video frame (seconds from the start of the video).
        var time: Double
        var faceDetected: Bool
        /// `blendShapeNames` order.
        var shapes: [Float] = []
        var lookAtPoint: SIMD3<Float> = .zero
        var headYaw: Float = 0, headPitch: Float = 0, headRoll: Float = 0
        var skinRGB: SIMD3<Float>?
        /// Landmark points, normalized to the upright frame (top-left origin); may be empty (thinned to <= 15 Hz).
        var mesh: [CGPoint] = []

        init(time: Double, sample s: FaceSample, mesh: [CGPoint]) {
            self.time = time
            faceDetected = s.faceDetected
            if s.faceDetected {
                shapes = FaceTrack.blendShapeNames.map { s.blendShapes[$0] ?? 0 }
                lookAtPoint = s.lookAtPoint
                headYaw = s.headYaw; headPitch = s.headPitch; headRoll = s.headRoll
                skinRGB = s.skinRGB
                self.mesh = mesh
            }
        }

        init(time: Double, faceDetected: Bool) { self.time = time; self.faceDetected = faceDetected }

        /// The frame as a `FaceSample` stamped `t` (the caller's clock).
        func sample(t: Double) -> FaceSample {
            guard faceDetected else { return FaceSample(t: t, faceDetected: false) }
            var b: [String: Float] = [:]
            for (i, n) in FaceTrack.blendShapeNames.enumerated() where i < shapes.count { b[n] = shapes[i] }
            return FaceSample(t: t, faceDetected: true, blendShapes: b, lookAtPoint: lookAtPoint,
                              headYaw: headYaw, headPitch: headPitch, headRoll: headRoll, skinRGB: skinRGB)
        }
    }

    /// Sorted by `time`.
    var frames: [Frame]
    /// Free-form `key=value` metadata from the header (video name, analyzer mode...).
    var metadata: [String: String] = [:]

    init(frames: [Frame], metadata: [String: String] = [:]) {
        self.frames = frames.sorted { $0.time < $1.time }
        self.metadata = metadata
    }

    /// The frame shown at `time` (the latest frame at or before it, within `tolerance`), or nil.
    func frame(at time: Double, tolerance: Double = 0.1) -> Frame? {
        guard !frames.isEmpty else { return nil }
        var lo = 0, hi = frames.count - 1
        if time < frames[0].time { return frames[0].time - time <= tolerance ? frames[0] : nil }
        while lo < hi {
            let m = (lo + hi + 1) / 2
            if frames[m].time <= time + 1e-6 { lo = m } else { hi = m - 1 }
        }
        return time - frames[lo].time <= tolerance ? frames[lo] : nil
    }

    // MARK: Files

    /// `<video without extension>.facetrack.csv`.
    static func sidecarURL(for video: URL) -> URL {
        video.deletingPathExtension().appendingPathExtension("facetrack").appendingPathExtension("csv")
    }

    /// The sidecar next to `video`, if one exists.
    static func locate(for video: URL) -> URL? {
        let u = sidecarURL(for: video)
        return FileManager.default.fileExists(atPath: u.path) ? u : nil
    }

    static func load(_ url: URL) throws -> FaceTrack {
        try parse(String(contentsOf: url, encoding: .utf8))
    }

    enum ParseError: Error, Equatable { case notAFaceTrack, badRow(Int) }

    static func parse(_ text: String) throws -> FaceTrack {
        var meta: [String: String] = [:]
        var frames: [Frame] = []
        var sawHeader = false
        let n = blendShapeNames.count
        var lineNo = 0
        for raw in text.split(whereSeparator: \.isNewline) {
            lineNo += 1
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if line.hasPrefix("#") {
                for part in line.dropFirst().split(separator: " ") where part.contains("=") {
                    let kv = part.split(separator: "=", maxSplits: 1)
                    if kv.count == 2 { meta[String(kv[0])] = String(kv[1]) }
                }
                continue
            }
            if line.hasPrefix("time,") { sawHeader = true; continue }
            guard sawHeader else { throw ParseError.notAFaceTrack }
            let c = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            guard c.count >= 2 + n + 9, let time = Double(c[0]) else { throw ParseError.badRow(lineNo) }
            guard c[1] == "1" else { frames.append(Frame(time: time, faceDetected: false)); continue }
            var f = Frame(time: time, faceDetected: true)
            f.shapes = (0..<n).map { Float(c[2 + $0]) ?? 0 }
            var k = 2 + n
            func next() -> Float? { defer { k += 1 }; return Float(c[k]) }
            f.lookAtPoint = SIMD3(next() ?? 0, next() ?? 0, next() ?? 1)
            f.headYaw = next() ?? 0; f.headPitch = next() ?? 0; f.headRoll = next() ?? 0
            if let r = next(), let g = next(), let b = next() { f.skinRGB = SIMD3(r, g, b) }
            if k < c.count {
                let v = c[k].split(separator: " ").compactMap { Double($0) }
                f.mesh = stride(from: 0, to: v.count - 1, by: 2).map { CGPoint(x: v[$0], y: v[$0 + 1]) }
            }
            frames.append(f)
        }
        guard sawHeader else { throw ParseError.notAFaceTrack }
        return FaceTrack(frames: frames, metadata: meta)
    }

    func csv() -> String {
        var out = Self.formatLine + "\n"
        let meta = metadata.keys.sorted().map { "\($0)=\(metadata[$0]!.replacingOccurrences(of: " ", with: "_"))" }
        if !meta.isEmpty { out += "# " + meta.joined(separator: " ") + "\n" }
        out += (["time", "face"] + Self.blendShapeNames
                + ["lookX", "lookY", "lookZ", "headYaw", "headPitch", "headRoll", "skinR", "skinG", "skinB", "mesh"]).joined(separator: ",") + "\n"
        let empties = Array(repeating: "", count: Self.blendShapeNames.count + 10).joined(separator: ",")
        for f in frames {
            guard f.faceDetected else { out += String(format: "%.4f,0,", f.time) + empties + "\n"; continue }
            var row = [String(format: "%.4f", f.time), "1"]
            row += f.shapes.map { String(format: "%.4f", $0) }
            row += [f.lookAtPoint.x, f.lookAtPoint.y, f.lookAtPoint.z, f.headYaw, f.headPitch, f.headRoll].map { String(format: "%.5f", $0) }
            if let s = f.skinRGB { row += [s.x, s.y, s.z].map { String(format: "%.7f", $0) } } else { row += ["", "", ""] }
            row.append(f.mesh.map { String(format: "%.4f %.4f", $0.x, $0.y) }.joined(separator: " "))
            out += row.joined(separator: ",") + "\n"
        }
        return out
    }
}
