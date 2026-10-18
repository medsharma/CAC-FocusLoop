import AVFoundation
import CoreVideo
import XCTest
@testable import FocusLoop

/// Development fixture: the bundled public-domain NASA lecture `black-holes-01.mp4` (a presenter talking to the
/// camera; REAL footage of a real person). Decoded offline with `AVAssetReader` (faster than real time); frames
/// are only read in memory, nothing is written anywhere.
///
/// Vision's face-landmark model is broken in the iOS Simulator runtime (it returns values that grow without
/// bound), so there the live analyser runs in head-pose-only mode, and blinks are checked on the face track that
/// the same analyser produced on the Mac (`Fixtures/black-holes-01.facetrack.csv`, made with
/// `tools/facetrack.sh Resources/Lectures/black-holes-01.mp4 --start 318 --duration 23 --out ...`).
/// On a device the live-landmark assertions run too.
final class VideoFaceSourceTests: XCTestCase {

    /// 16 s of the closing outdoor shot: one presenter facing the camera the whole time.
    static let segment = (start: 323.5, duration: 16.0)

    struct Run {
        var frames = 0
        var faceFrames = 0
        var skinFrames = 0
        var faceSeconds = 0.0
        var blinks = 0
        var samples: [FaceSample] = []
        var featureFrames: [FeatureFrame] = []
        var heartRates: [HeartRateSample] = []
        var mode: VideoFaceAnalyzer.Mode = .landmarks

        var faceFraction: Double { Double(faceFrames) / Double(max(1, frames)) }
        var blinkRate: Double { Double(blinks) / max(1, faceSeconds) * 60 }
        var medianBPM: Float? { VideoFaceSourceTests.median(heartRates.map(\.bpm)) }
        var medianClosure: Float? {
            VideoFaceSourceTests.median(samples.filter(\.faceDetected).map {
                (($0.blendShapes["eyeBlinkLeft"] ?? 0) + ($0.blendShapes["eyeBlinkRight"] ?? 0)) / 2
            })
        }

        /// The same chain as `RecordedVideoFrameSource`: extractor + POS on each sample.
        mutating func feed(_ s: FaceSample, extractor: AttentionFeatureExtractor, hr: POSHeartRateEstimator,
                           blink: inout PerceptionBlinkDetector, prevT: inout Double?, lastHRT: inout Double) {
            frames += 1
            samples.append(s)
            if s.faceDetected {
                faceFrames += 1
                if let p = prevT { faceSeconds += s.t - p }
                let closure = ((s.blendShapes["eyeBlinkLeft"] ?? 0) + (s.blendShapes["eyeBlinkRight"] ?? 0)) / 2
                if blink.update(closure: closure, t: s.t) != nil { blinks += 1 }
            }
            prevT = s.t
            extractor.ingest(face: s)
            if let rgb = s.skinRGB {
                skinFrames += 1
                hr.ingest(rgb: rgb, t: s.t, headMotion: extractor.headMotion)
                if let h = hr.latest, h.t != lastHRT { lastHRT = h.t; heartRates.append(h); extractor.ingest(heartRate: h) }
            }
            if let f = extractor.frame(at: s.t) { featureFrames.append(f) }
        }

        func log(_ label: String) {
            let bpm = medianBPM.map { String(format: "%.0f", $0) } ?? "nil"
            let all = heartRates.map { String(Int($0.bpm)) }.joined(separator: ",")
            print(String(format: "[VideoFaceSourceTests] %@ (%@): face %.1f%%, skin %.1f%%, blinks %d in %.1f s (%.1f/min), median closure %.2f, HR median %@ from %d estimates [%@]",
                         label, mode.rawValue, 100 * faceFraction, 100 * Double(skinFrames) / Double(max(1, frames)), blinks,
                         faceSeconds, blinkRate, medianClosure ?? -1, bpm, heartRates.count, all))
        }
    }

    private static func fixtureURL() throws -> URL {
        try XCTUnwrap(Bundle.main.url(forResource: "black-holes-01", withExtension: "mp4"), "lecture fixture not bundled")
    }

    private static func trackURL() throws -> URL {
        try XCTUnwrap(Bundle(for: VideoFaceSourceTests.self).url(forResource: "black-holes-01.facetrack", withExtension: "csv"),
                      "face-track fixture not bundled")
    }

    /// Decodes [start, start + duration) and runs the analyser + the `RecordedVideoFrameSource` chain.
    private func analyse(start: Double, duration: Double, mode: VideoFaceAnalyzer.Mode? = nil,
                         pixelFormat: OSType = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) async throws -> Run {
        let asset = AVURLAsset(url: try Self.fixtureURL())
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let transform = try await track.load(.preferredTransform)
        let orientation = VideoFaceAnalyzer.orientation(for: transform)
        let reader = try AVAssetReader(asset: asset)
        reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 600),
                                       duration: CMTime(seconds: duration, preferredTimescale: 600))
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        XCTAssertTrue(reader.startReading())

        let analyzer = VideoFaceAnalyzer(mode: mode)
        let extractor = AttentionFeatureExtractor(workZone: .default)
        let hr = POSHeartRateEstimator()
        var blink = PerceptionBlinkDetector()
        var run = Run()
        var t0: Double?, prevT: Double?
        var lastHRT = -1.0
        while let sb = output.copyNextSampleBuffer() {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            if t0 == nil { t0 = pts }
            let s = analyzer.analyze(pixelBuffer: pb, orientation: orientation, t: pts - (t0 ?? pts), wantMesh: run.frames == 0).sample
            run.feed(s, extractor: extractor, hr: hr, blink: &blink, prevT: &prevT, lastHRT: &lastHRT)
        }
        run.mode = analyzer.mode
        return run
    }

    static func median(_ a: [Float]) -> Float? {
        guard !a.isEmpty else { return nil }
        let s = a.sorted()
        return s[s.count / 2]
    }

    /// Face found, finite pose, skin colour present, a physiological POS heart rate (accuracy is not asserted on
    /// compressed lecture video): what every mode must deliver.
    private func assertCommon(_ run: Run, file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertGreaterThan(run.frames, 300, file: file, line: line)
        XCTAssertGreaterThan(run.faceFraction, 0.9, "face detected in > 90 % of frames", file: file, line: line)
        for s in run.samples where s.faceDetected {
            XCTAssertTrue(s.headYaw.isFinite && s.headPitch.isFinite && s.headRoll.isFinite, file: file, line: line)
            XCTAssertLessThan(abs(s.headYaw), 1.6, file: file, line: line)
            XCTAssertLessThan(abs(s.headPitch), 1.6, file: file, line: line)
        }
        XCTAssertGreaterThan(Double(run.skinFrames), 0.85 * Double(run.faceFrames), "skinRGB on face frames", file: file, line: line)
        if let rgb = run.samples.compactMap(\.skinRGB).first {
            XCTAssertGreaterThan(rgb.x, rgb.z, "skin: red above blue", file: file, line: line)
        }
        let hr = try XCTUnwrap(run.medianBPM, "no rPPG estimate", file: file, line: line)
        XCTAssertTrue(hr.isFinite, file: file, line: line)
        XCTAssertGreaterThanOrEqual(hr, 45, file: file, line: line)
        XCTAssertLessThanOrEqual(hr, 150, file: file, line: line)
        // 10 Hz frames come out with a face, and talking is never read as yawning.
        XCTAssertGreaterThan(run.featureFrames.filter { $0.faceDetected == 1 }.count, 140, file: file, line: line)
        XCTAssertEqual(run.featureFrames.map(\.yawnCount).max() ?? -1, 0, "talking is not yawning", file: file, line: line)
    }

    /// Blinks and eyes from landmarks: a plausible blink rate for a person talking to camera (resting range is
    /// ~8-25/min; a little either side allowed on 16 s), and eyes that read open most of the time.
    private func assertEyes(_ run: Run, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertGreaterThanOrEqual(run.blinkRate, 5, file: file, line: line)
        XCTAssertLessThanOrEqual(run.blinkRate, 35, file: file, line: line)
        XCTAssertLessThan(run.medianClosure ?? 1, 0.2, file: file, line: line)
    }

    // MARK: Fixture

    /// Live analysis on this platform: landmarks on a device, head pose only in the Simulator.
    func testLiveAnalysisOnFixture() async throws {
        let run = try await analyse(start: Self.segment.start, duration: Self.segment.duration)
        run.log("live")
        try assertCommon(run)
        if run.mode == .landmarks {
            assertEyes(run)
        } else {
            XCTAssertFalse(VideoFaceAnalyzer.landmarksAvailable)
            XCTAssertEqual(run.blinks, 0, "pose-only mode reports neutral eyes")
        }
    }

    /// Landmarks on this platform must either work or be detected as implausible and fall back to pose only.
    func testLandmarkModeWorksOrFallsBack() async throws {
        let run = try await analyse(start: Self.segment.start, duration: 6, mode: .landmarks)
        if VideoFaceAnalyzer.landmarksAvailable { XCTAssertEqual(run.mode, .landmarks) }
        XCTAssertGreaterThan(run.faceFraction, 0.9)
        for s in run.samples where s.faceDetected {
            for v in s.blendShapes.values { XCTAssertTrue(v.isFinite && v >= 0 && v <= 1) }
        }
    }

    /// The Mac-computed face track (same analyser, Vision landmarks) replayed as `VideoFaceSource` does in the
    /// Simulator: the frame shown at each video time, through the live extractor + POS chain.
    func testFaceTrackOnFixture() throws {
        let track = try FaceTrack.load(try Self.trackURL())
        XCTAssertEqual(track.metadata["mode"], "landmarks")
        let extractor = AttentionFeatureExtractor(workZone: .default)
        let hr = POSHeartRateEstimator()
        var blink = PerceptionBlinkDetector()
        var run = Run()
        var prevT: Double?
        var lastHRT = -1.0
        let end = Self.segment.start + Self.segment.duration
        for f in track.frames where f.time >= Self.segment.start && f.time < end {
            run.feed(f.sample(t: f.time - Self.segment.start), extractor: extractor, hr: hr, blink: &blink,
                     prevT: &prevT, lastHRT: &lastHRT)
        }
        run.log("face track")
        try assertCommon(run)
        assertEyes(run)
        XCTAssertTrue(track.frames.contains { !$0.mesh.isEmpty }, "landmark mesh for the overlay")
    }

    func testBGRAFramesGiveTheSameAnswer() async throws {
        let run = try await analyse(start: Self.segment.start, duration: 4, pixelFormat: kCVPixelFormatType_32BGRA)
        XCTAssertGreaterThan(run.faceFraction, 0.9)
        XCTAssertGreaterThan(Double(run.skinFrames), 0.85 * Double(run.faceFrames))
    }

    func testNoFaceDuringAnimation() async throws {
        // 0-20 s of the lecture is space animation: no presenter on screen.
        let run = try await analyse(start: 2, duration: 3)
        XCTAssertLessThan(run.faceFraction, 0.1)
        XCTAssertTrue(run.samples.filter { !$0.faceDetected }.allSatisfy { $0.skinRGB == nil && $0.blendShapes.isEmpty })
    }

    // MARK: Face track format

    func testFaceTrackRoundTripsAndLooksUpByTime() throws {
        var shapes: [String: Float] = [:]
        for (i, n) in FaceTrack.blendShapeNames.enumerated() { shapes[n] = Float(i) / 40 }
        let s = FaceSample(t: 0, faceDetected: true, blendShapes: shapes, lookAtPoint: [0.1, -0.2, 0.97],
                           headYaw: 0.3, headPitch: -0.25, headRoll: 0.05, skinRGB: [0.3123456, 0.2, 0.15])
        let track = FaceTrack(frames: [
            FaceTrack.Frame(time: 0.5, sample: s, mesh: [CGPoint(x: 0.25, y: 0.5)]),
            FaceTrack.Frame(time: 0.0, faceDetected: false),
        ], metadata: ["video": "clip.mov"])
        let parsed = try FaceTrack.parse(track.csv())
        XCTAssertEqual(parsed.frames.count, 2)
        XCTAssertEqual(parsed.metadata["video"], "clip.mov")
        XCTAssertFalse(parsed.frames[0].faceDetected)
        let f = try XCTUnwrap(parsed.frame(at: 0.52))
        XCTAssertEqual(f.time, 0.5, accuracy: 1e-6)
        let back = f.sample(t: 9)
        XCTAssertEqual(back.t, 9)
        XCTAssertEqual(back.headYaw, 0.3, accuracy: 1e-5)
        XCTAssertEqual(back.blendShapes["jawOpen"] ?? -1, shapes["jawOpen"] ?? 0, accuracy: 1e-4)
        XCTAssertEqual(back.skinRGB?.x ?? 0, 0.3123456, accuracy: 1e-6)
        XCTAssertEqual(f.mesh, [CGPoint(x: 0.25, y: 0.5)])
        XCTAssertNil(parsed.frame(at: 5), "no frame within tolerance")
        XCTAssertEqual(FaceTrack.blendShapeNames, perceptionBlendShapeNames)
        XCTAssertThrowsError(try FaceTrack.parse("hello,world\n1,2\n"))
        XCTAssertEqual(FaceTrack.sidecarURL(for: URL(fileURLWithPath: "/x/demo_face.mov")).lastPathComponent, "demo_face.facetrack.csv")
    }

    // MARK: Real-time source

    func testVideoFaceSourcePlaysAndEmitsSamples() async throws {
        let src = VideoFaceSource(url: try Self.fixtureURL())
        try src.start()
        let stream = src.samples
        let collector = Task { () -> (count: Int, monotonic: Bool) in
            var count = 0, monotonic = true, lastT = -Double.infinity
            for await s in stream {
                if s.t <= lastT { monotonic = false }
                lastT = s.t
                count += 1
                if count >= 30 { break }
            }
            return (count, monotonic)
        }
        try await Task.sleep(nanoseconds: 4_000_000_000)
        src.stop()
        collector.cancel()
        let result = await collector.value
        XCTAssertGreaterThanOrEqual(result.count, 30, "fewer than 30 analysed frames in 4 s")
        XCTAssertTrue(result.monotonic, "sample clock must increase")
        XCTAssertTrue(src.player.isMuted)
    }

    func testMissingFileThrows() {
        let src = VideoFaceSource(url: URL(fileURLWithPath: "/nonexistent/clip.mov"))
        XCTAssertFalse(src.isAvailable)
        XCTAssertThrowsError(try src.start())
    }

    // MARK: Geometry

    func testOrientationFromTransform() {
        XCTAssertEqual(VideoFaceAnalyzer.orientation(for: .identity), .up)
        // iPhone portrait recording: stored landscape, rotated 90 degrees clockwise for display.
        XCTAssertEqual(VideoFaceAnalyzer.orientation(for: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 1080, ty: 0)), .right)
        XCTAssertEqual(VideoFaceAnalyzer.orientation(for: CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: 1920)), .left)
        XCTAssertEqual(VideoFaceAnalyzer.orientation(for: CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: 1920, ty: 1080)), .down)
    }

    func testBufferPointMapsUprightCornersToBufferCorners() {
        // Stored 1920 x 1080; upright portrait (.right) is 1080 x 1920. The upright top-left is the stored
        // bottom-left, the upright top-right is the stored top-left.
        let w: Float = 1920, h: Float = 1080
        XCTAssertEqual(VideoFaceAnalyzer.bufferPoint([0, 0], orientation: .right, bufW: w, bufH: h), [0, h - 1])
        XCTAssertEqual(VideoFaceAnalyzer.bufferPoint([h - 1, 0], orientation: .right, bufW: w, bufH: h), [0, 0])
        XCTAssertEqual(VideoFaceAnalyzer.bufferPoint([0, w - 1], orientation: .right, bufW: w, bufH: h), [w - 1, h - 1])
        XCTAssertEqual(VideoFaceAnalyzer.bufferPoint([10, 20], orientation: .up, bufW: w, bufH: h), [10, 20])
        XCTAssertEqual(VideoFaceAnalyzer.bufferPoint([0, 0], orientation: .left, bufW: w, bufH: h), [w - 1, 0])
    }

    func testRollingStatPercentiles() {
        var s = RollingStat()
        for i in 0..<100 { s.add(Float(i), t: Double(i) * 0.1, window: 100) }
        XCTAssertEqual(s.percentile(0.5, minCount: 10) ?? -1, 50, accuracy: 1)
        XCTAssertEqual(s.percentile(0.9, minCount: 10) ?? -1, 89, accuracy: 1)
        XCTAssertNil(s.percentile(0.5, minCount: 200))
        // Old samples leave the window.
        var w = RollingStat()
        for i in 0..<100 { w.add(i < 50 ? 0 : 1, t: Double(i), window: 10) }
        XCTAssertEqual(w.percentile(0.1, minCount: 1), 1)
    }
}

/// Recorded-video bookkeeping: import/remove, launch argument, honest labelling.
@MainActor
final class RecordedVideoLibraryTests: XCTestCase {
    private var dir: URL!

    override func setUp() async throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("RecordedVideoLibraryTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: dir)
    }

    func testImportReplacesAndRemoves() throws {
        let a = dir.appendingPathComponent("clip A.MOV"), b = dir.appendingPathComponent("clip.mp4")
        try Data([1, 2, 3]).write(to: a)
        try Data([4, 5]).write(to: b)
        let store = dir.appendingPathComponent("support")
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)

        let first = try RecordedVideoLibrary.importVideo(from: a, into: store)
        XCTAssertEqual(first.lastPathComponent, "FocusLoop-demo-recording.mov")
        let second = try RecordedVideoLibrary.importVideo(from: b, into: store)
        XCTAssertEqual(second.lastPathComponent, "FocusLoop-demo-recording.mp4")
        XCTAssertEqual(RecordedVideoLibrary.importedURL(in: store), second)
        XCTAssertFalse(FileManager.default.fileExists(atPath: first.path), "an earlier import is replaced")

        // A face track sits next to the video and is found as its sidecar; it is never mistaken for the video.
        let csv = dir.appendingPathComponent("clip.facetrack.csv")
        try FaceTrack(frames: [FaceTrack.Frame(time: 0, faceDetected: false)]).csv().write(to: csv, atomically: true, encoding: .utf8)
        let imported = try RecordedVideoLibrary.importFaceTrack(from: csv, into: store)
        XCTAssertEqual(RecordedVideoLibrary.importedURL(in: store), second)
        XCTAssertEqual(FaceTrack.locate(for: second), imported)
        XCTAssertThrowsError(try RecordedVideoLibrary.importFaceTrack(from: a, into: store), "not a face track")

        try RecordedVideoLibrary.removeImported(in: store)
        XCTAssertNil(RecordedVideoLibrary.importedURL(in: store))
        XCTAssertFalse(FileManager.default.fileExists(atPath: imported.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: a.path), "the user's original file is untouched")
    }

    func testLaunchArgument() throws {
        let f = dir.appendingPathComponent("face.mov")
        try Data([0]).write(to: f)
        XCTAssertEqual(RecordedVideoLibrary.launchArgumentURL(["app", "-recordedVideo", f.path]), f)
        XCTAssertNil(RecordedVideoLibrary.launchArgumentURL(["app", "-recordedVideo", "/missing.mov"]))
        XCTAssertNil(RecordedVideoLibrary.launchArgumentURL(["app", "-recordedVideo"]))
    }

    func testRecordedSourceIsRealButLabelledRecorded() throws {
        let src = RecordedVideoFrameSource(url: dir.appendingPathComponent("x.mov"), workZone: .default)
        XCTAssertFalse(src.isSynthetic)
        XCTAssertEqual(RecordedVideoFrameSource.sourceKind, "recorded")
        XCTAssertTrue(SessionRecord(sourceKind: RecordedVideoFrameSource.sourceKind).isRecorded)
    }

    func testOldSessionRecordsStillDecode() throws {
        let old = SessionRecord(duration: 12)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(old)) as? [String: Any])
        json.removeValue(forKey: "sourceKind")
        let decoded = try JSONDecoder().decode(SessionRecord.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(decoded.sourceKind)
        XCTAssertFalse(decoded.isRecorded)
        XCTAssertEqual(decoded.duration, 12)
    }

    func testRecordedSourceKindPersists() throws {
        let store = try SwiftDataSessionStore(inMemory: true, defaults: UserDefaults(suiteName: "RecordedVideoLibraryTests")!,
                                              tempDirectory: dir)
        let rec = SessionRecord(duration: 30, sourceKind: "recorded")
        try store.save(rec)
        XCTAssertEqual(try store.session(id: rec.id)?.sourceKind, "recorded")
    }
}
