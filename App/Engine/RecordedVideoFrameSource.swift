import AVFoundation
import Foundation
import OSLog
import QuartzCore

/// A recorded video of a real person → 10 Hz `FeatureFrame`s, exactly like `LiveFeatureFrameSource` does with
/// ARKit: `VideoFaceSource` (Vision landmarks on the frames being played) → `AttentionFeatureExtractor`
/// (+ `POSHeartRateEstimator` on the skin ROI colour). Runs in the Simulator, where there is no TrueDepth camera.
///
/// Honesty: the signals are real (a real person's real footage, analysed frame by frame), so `isSynthetic` is
/// false; but the same clip replays many times, so these sessions are labelled RECORDED and never teach the
/// real learners (see `AppServices`). There is no Core Motion here (phone pickups come only from gaze / pose).
///
/// Work zone: Vision head pose is centred on the video's own resting pose (`VideoFaceAnalyzer`), so the saved
/// zone's size is kept but it is re-centred on 0. Only numbers leave the analysis queue.
final class RecordedVideoFrameSource: FeatureFrameSource, SuspendableFrameSource {
    let frames: AsyncStream<FeatureFrame>
    let heartRate: AsyncStream<HeartRateSample>
    var isSynthetic: Bool { false }
    /// `SessionRecord.sourceKind` for sessions driven by this source.
    static let sourceKind = "recorded"

    let url: URL
    /// The player whose frames are analysed (muted, looping). Show it with an `AVPlayerLayer`.
    var player: AVPlayer { face.player }
    /// Upright video width / height.
    var videoAspect: CGFloat { face.videoAspect }

    private let frameCont: AsyncStream<FeatureFrame>.Continuation
    private let hrCont: AsyncStream<HeartRateSample>.Continuation
    private let queue = DispatchQueue(label: "focusloop.perception.recorded", qos: .userInitiated)
    private let face: VideoFaceSource
    private let extractor: AttentionFeatureExtractor
    private let hr: POSHeartRateEstimator
    private var task: Task<Void, Never>?
    private var lastHRT: Double = -1
    private let meshLock = NSLock()
    private var meshPoints: [CGPoint] = []
    private var suspendedAt: Double?
    private var lastLogT: Double = 0
    private var faceFrames = 0, frameCount = 0
    private static let log = Logger(subsystem: "org.focusloop.app", category: "recorded-video")

    /// - Parameters:
    ///   - workZone: the saved work zone; its half-widths are kept, its centre is reset to the video's resting pose.
    ///   - mirrored: selfie-mirrored footage.
    init(url: URL, workZone: WorkZone, mirrored: Bool = false) {
        self.url = url
        var zone = workZone
        zone.centerYaw = 0
        zone.centerPitch = 0
        extractor = AttentionFeatureExtractor(workZone: zone)
        // No persisted resting baseline: a replayed clip must not move the user's real HR baseline.
        hr = POSHeartRateEstimator(restingBaseline: nil)
        face = VideoFaceSource(url: url)
        face.mirrored = mirrored
        (frames, frameCont) = AsyncStream.makeStream(of: FeatureFrame.self, bufferingPolicy: .bufferingNewest(64))
        (heartRate, hrCont) = AsyncStream.makeStream(of: HeartRateSample.self, bufferingPolicy: .bufferingNewest(8))
        face.onFrame = { [weak self] pts in
            guard let self else { return }
            self.meshLock.lock(); self.meshPoints = pts; self.meshLock.unlock()
        }
    }

    /// Latest landmark points (normalized to the upright video frame, top-left origin). Never stored.
    func currentMesh() -> [CGPoint] {
        meshLock.lock(); defer { meshLock.unlock() }
        return meshPoints
    }

    func start() {
        face.timeOrigin = CACurrentMediaTime()
        face.onFailure = { [weak self] _ in self?.finishStreams() }
        do { try face.start() } catch { finishStreams(); return }
        let stream = face.samples
        task = Task.detached { [weak self] in
            for await s in stream { self?.queue.sync { self?.handle(face: s) } }
        }
    }

    /// Runs on `queue`.
    private func handle(face s: FaceSample) {
        extractor.ingest(face: s)
        if let rgb = s.skinRGB {
            hr.ingest(rgb: rgb, t: s.t, headMotion: extractor.headMotion)
            if let h = hr.latest, h.t != lastHRT {
                lastHRT = h.t
                extractor.ingest(heartRate: h)
                hrCont.yield(h)
            }
        }
        if let f = extractor.frame(at: s.t) {
            frameCont.yield(f)
            frameCount += 1
            if f.faceDetected > 0.5 { faceFrames += 1 }
            if f.t - lastLogT >= 10 {
                // Numbers only, for checking a recording in the console (never stored).
                lastLogT = f.t
                Self.log.info("recorded video t=\(f.t, format: .fixed(precision: 0))s face=\(100 * Double(self.faceFrames) / Double(max(1, self.frameCount)), format: .fixed(precision: 0))% blinkRate=\(f.blinkRate, format: .fixed(precision: 1))/min perclos=\(f.perclos, format: .fixed(precision: 2)) yawns=\(f.yawnCount, format: .fixed(precision: 0)) nods=\(f.nodCount, format: .fixed(precision: 0)) inZone=\(f.inWorkZone, format: .fixed(precision: 0)) dwell=\(f.offZoneDwell, format: .fixed(precision: 1))s hr=\(f.heartRate, format: .fixed(precision: 0)) q=\(f.heartRateQuality, format: .fixed(precision: 2)) via \(self.face.analysisDescription, privacy: .public)")
            }
        }
    }

    /// Background: pause the video (and so the analysis). The session clock skips the paused time.
    func suspend() {
        guard suspendedAt == nil else { return }
        suspendedAt = CACurrentMediaTime()
        face.stop()
    }

    func resume() {
        guard let since = suspendedAt else { return }
        suspendedAt = nil
        face.timeOrigin += CACurrentMediaTime() - since
        queue.sync { hr.reset() }
        try? face.start()
    }

    func stop() {
        task?.cancel(); task = nil
        face.stop()
        finishStreams()
    }

    private func finishStreams() {
        frameCont.finish(); hrCont.finish()
    }
}
