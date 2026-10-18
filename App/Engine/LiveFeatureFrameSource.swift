import Foundation
#if !targetEnvironment(simulator)
import ARKit
#endif

/// Real sensors → 10 Hz `FeatureFrame`s: ARKit face tracking (+ rPPG skin ROI), Core Motion, feature extractor.
/// Everything runs on one serial queue; only numbers leave it. Not available in the Simulator (use Replay).
final class LiveFeatureFrameSource: FeatureFrameSource, SuspendableFrameSource {
    let frames: AsyncStream<FeatureFrame>
    let heartRate: AsyncStream<HeartRateSample>
    var isSynthetic: Bool { false }

    /// Latest face-mesh landmarks (normalized, top-left origin) for the Science view. Never stored.
    private(set) var meshPoints: [CGPoint] = []
    /// Called with the resting HR baseline when the source stops, so the caller can persist it.
    var onBaselineUpdate: ((Float) -> Void)?

    private let frameCont: AsyncStream<FeatureFrame>.Continuation
    private let hrCont: AsyncStream<HeartRateSample>.Continuation
    private let queue = DispatchQueue(label: "focusloop.perception", qos: .userInitiated)
    private let face = ARKitFaceSource()
    private let motion = CoreMotionSource()
    private let extractor: AttentionFeatureExtractor
    private let hr: POSHeartRateEstimator
    private var tasks: [Task<Void, Never>] = []
    private var lastHRT: Double = -1
    private let meshLock = NSLock()
    private var suspendedAt: Double?

    static var isSupported: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return ARFaceTrackingConfiguration.isSupported
        #endif
    }

    init(workZone: WorkZone, restingBaseline: Float?) {
        extractor = AttentionFeatureExtractor(workZone: workZone)
        hr = POSHeartRateEstimator(restingBaseline: restingBaseline)
        (frames, frameCont) = AsyncStream.makeStream(of: FeatureFrame.self, bufferingPolicy: .bufferingNewest(64))
        (heartRate, hrCont) = AsyncStream.makeStream(of: HeartRateSample.self, bufferingPolicy: .bufferingNewest(8))
        #if !targetEnvironment(simulator)
        let roi = SkinROIExtractor()
        face.skinSampler = { frame, anchor in roi.sample(frame: frame, anchor: anchor) }
        #endif
        face.onFrame = { [weak self] pts in
            guard let self else { return }
            self.meshLock.lock(); self.meshPoints = pts; self.meshLock.unlock()
        }
    }

    func currentMesh() -> [CGPoint] {
        meshLock.lock(); defer { meshLock.unlock() }
        return meshPoints
    }

    func start() {
        let origin = ProcessInfo.processInfo.systemUptime
        face.timeOrigin = origin
        motion.timeOrigin = origin
        do { try face.start() } catch { frameCont.finish(); hrCont.finish(); return }
        motion.start()
        let faceStream = face.samples, motionStream = motion.samples
        tasks.append(Task.detached { [weak self] in
            for await s in faceStream { self?.queue.sync { self?.handle(face: s) } }
        })
        tasks.append(Task.detached { [weak self] in
            for await m in motionStream { self?.queue.sync { self?.extractor.ingest(motion: m) } }
        })
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
        if let f = extractor.frame(at: s.t) { frameCont.yield(f) }
    }

    /// Background: pause the AR session and Core Motion without ending the streams.
    func suspend() {
        guard suspendedAt == nil else { return }
        suspendedAt = ProcessInfo.processInfo.systemUptime
        face.stop(); motion.stop()
    }

    /// Foreground: restart the sensors. The session clock skips the time spent in the background, and the
    /// heart-rate window restarts (a camera gap would corrupt the pulse estimate).
    func resume() {
        guard let since = suspendedAt else { return }
        suspendedAt = nil
        let gap = ProcessInfo.processInfo.systemUptime - since
        queue.sync { hr.reset() }
        face.timeOrigin += gap
        motion.timeOrigin += gap
        do { try face.start() } catch { return }
        motion.start()
    }

    func stop() {
        tasks.forEach { $0.cancel() }; tasks.removeAll()
        face.stop(); motion.stop()
        // Read the estimator on its own queue (a frame may still be in flight there).
        if let b = queue.sync(execute: { hr.restingBaseline }) { onBaselineUpdate?(b) }
        frameCont.finish(); hrCont.finish()
    }
}
