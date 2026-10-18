import Foundation

/// Turns camera-rate `FaceSample`s, `MotionSample`s and `HeartRateSample`s into 10 Hz `FeatureFrame`s.
///
/// Pure Swift and deterministic: every computation uses the timestamps passed in (no clock reads).
/// All timestamps (face, motion, heart rate and `frame(at:)`) must share one clock, normally
/// seconds since session start. Not thread-safe: call from a single actor/queue.
///
/// Semantics follow `ml/schema/feature_schema.json` and the synthetic generator
/// `ml/synth/make_replay.py` so live frames match what the models were trained on:
///   * blinks: hysteresis 0.5 / 0.3 on mean(eyeBlinkLeft, eyeBlinkRight), detected at camera rate;
///     a blink is attributed to its onset time; trailing 60 s window.
///   * blinkRate = blinks in window / min(elapsed, 60 s) x 60.
///   * perclos = fraction of 10 Hz frames (face tracked) with mean closure > 0.8 over trailing 60 s.
///   * yawn = jawOpen > 0.6 for >= 1.5 s; yawnActive = jawOpen > 0.6 right now.
///   * nod = pitch drop >= 0.25 rad below a slow baseline and recovery within 2 s.
///   * work zone: combined direction = gaze + 0.3 x head (see `WorkZoneMath`).
///   * browTension = trailing 2 s mean of mean(browDownLeft, browDownRight).
///   * headMotion = |d headYaw/dt| + |d headPitch/dt|, lightly smoothed (rad/s).
public final class AttentionFeatureExtractor: FeatureExtractor {
    // MARK: Tunables
    public static let windowSeconds: Double = 60
    public static let longBlinkSeconds: Double = 0.5
    public static let perclosClosure: Float = 0.8
    public static let browSmoothingSeconds: Double = 2
    /// A face sample older than this at frame time counts as "no face".
    public static let faceStaleSeconds: Double = 0.5
    /// A heart-rate sample older than this at frame time counts as unavailable.
    public static let heartRateStaleSeconds: Double = 5
    /// EMA time constants.
    public static let headMotionTau: Double = 0.1
    public static let motionMagnitudeTau: Double = 0.3

    public var workZone: WorkZone

    // MARK: State
    private var latestFace: FaceSample?
    private var latestMotion: MotionSample?
    private var latestHR: HeartRateSample?

    private var blinkDetector = PerceptionBlinkDetector()
    private var yawnDetector = PerceptionYawnDetector()
    private var nodDetector = PerceptionNodDetector()

    private var blinks: [PerceptionBlinkDetector.Blink] = []
    private var yawnTimes: [Double] = []
    private var nodTimes: [Double] = []
    /// (t, closed>0.8, faceTracked) per emitted frame.
    private var perclosFrames: [(t: Double, closed: Bool)] = []
    /// (t, browMean) per face sample, trailing 2 s.
    private var browSamples: [(t: Double, v: Float)] = []

    private var prevHead: (t: Double, yaw: Float, pitch: Float)?
    /// Smoothed head angular speed (rad/s), updated at camera rate. Also usable for rPPG gating.
    public private(set) var headMotion: Float = 0
    private var motionMagnitude: Float = 0
    private var lastMotionT: Double?

    private var firstFrameT: Double?
    private var nextTick: Double?
    private var lastFrameT: Double?
    private var offZoneDwell: Double = 0
    private var lastDeviation: Float = 0

    public init(workZone: WorkZone = .default) {
        self.workZone = workZone
    }

    // MARK: FeatureExtractor

    public func ingest(face: FaceSample) {
        let t = face.t
        guard face.faceDetected else {
            latestFace = face
            blinkDetector.cancel(); yawnDetector.cancel()
            prevHead = nil
            return
        }
        latestFace = face
        let b = face.blendShapes
        let closure = Self.meanClosure(b)
        if let blink = blinkDetector.update(closure: closure, t: t) { blinks.append(blink) }
        if let y = yawnDetector.update(jawOpen: b["jawOpen"] ?? 0, t: t) { yawnTimes.append(y) }
        if let n = nodDetector.update(pitch: face.headPitch, t: t) { nodTimes.append(n) }

        let brow = ((b["browDownLeft"] ?? 0) + (b["browDownRight"] ?? 0)) / 2
        browSamples.append((t, brow))
        Self.prune(&browSamples, before: t - Self.browSmoothingSeconds) { $0.t }

        if let p = prevHead, t > p.t {
            let dt = t - p.t
            let speed = (abs(face.headYaw - p.yaw) + abs(face.headPitch - p.pitch)) / Float(dt)
            let a = Float(1 - exp(-dt / Self.headMotionTau))
            headMotion += a * (speed - headMotion)
        }
        prevHead = (t, face.headYaw, face.headPitch)
    }

    public func ingest(motion: MotionSample) {
        if let lt = lastMotionT, motion.t > lt {
            let a = Float(1 - exp(-(motion.t - lt) / Self.motionMagnitudeTau))
            motionMagnitude += a * (motion.accelerationMagnitude - motionMagnitude)
        } else if lastMotionT == nil {
            motionMagnitude = motion.accelerationMagnitude
        }
        lastMotionT = motion.t
        latestMotion = motion
    }

    public func ingest(heartRate: HeartRateSample) {
        latestHR = heartRate
    }

    /// Returns a frame when a 10 Hz tick is due at `t`, else nil. Frames are stamped on a fixed
    /// 0.1 s grid starting at the first call; if calls fall behind, missed ticks are skipped
    /// (one frame per call) rather than back-filled.
    public func frame(at t: Double) -> FeatureFrame? {
        let period = 1 / FeatureFrame.rateHz
        let tick: Double
        if let n = nextTick {
            guard t >= n - 1e-6 else { return nil }
            let k = ((t - n) / period + 1e-6).rounded(.down)
            tick = n + k * period
        } else {
            tick = t
            firstFrameT = t
        }
        nextTick = tick + period
        let dt = lastFrameT.map { tick - $0 } ?? period
        lastFrameT = tick
        return makeFrame(t: tick, dt: dt)
    }

    public func reset() {
        latestFace = nil; latestMotion = nil; latestHR = nil
        blinkDetector = PerceptionBlinkDetector(); yawnDetector = PerceptionYawnDetector(); nodDetector.reset()
        blinks.removeAll(); yawnTimes.removeAll(); nodTimes.removeAll()
        perclosFrames.removeAll(); browSamples.removeAll()
        prevHead = nil; headMotion = 0; motionMagnitude = 0; lastMotionT = nil
        firstFrameT = nil; nextTick = nil; lastFrameT = nil; offZoneDwell = 0; lastDeviation = 0
    }

    // MARK: Frame assembly

    private func makeFrame(t: Double, dt: Double) -> FeatureFrame {
        var f = FeatureFrame(t: t)
        let w0 = t - Self.windowSeconds
        Self.prune(&blinks, before: w0) { $0.onset }
        Self.prune(&yawnTimes, before: w0) { $0 }
        Self.prune(&nodTimes, before: w0) { $0 }
        Self.prune(&perclosFrames, before: w0 + 1e-6) { $0.t }

        let face = latestFace.flatMap { $0.faceDetected && t - $0.t <= Self.faceStaleSeconds ? $0 : nil }
        if let face {
            let b = face.blendShapes
            f.faceDetected = 1
            f.eyeBlinkLeft = b["eyeBlinkLeft"] ?? 0
            f.eyeBlinkRight = b["eyeBlinkRight"] ?? 0
            f.eyeLookInLeft = b["eyeLookInLeft"] ?? 0
            f.eyeLookInRight = b["eyeLookInRight"] ?? 0
            f.eyeLookOutLeft = b["eyeLookOutLeft"] ?? 0
            f.eyeLookOutRight = b["eyeLookOutRight"] ?? 0
            f.eyeLookUpLeft = b["eyeLookUpLeft"] ?? 0
            f.eyeLookUpRight = b["eyeLookUpRight"] ?? 0
            f.eyeLookDownLeft = b["eyeLookDownLeft"] ?? 0
            f.eyeLookDownRight = b["eyeLookDownRight"] ?? 0
            f.eyeSquintLeft = b["eyeSquintLeft"] ?? 0
            f.eyeSquintRight = b["eyeSquintRight"] ?? 0
            f.eyeWideLeft = b["eyeWideLeft"] ?? 0
            f.eyeWideRight = b["eyeWideRight"] ?? 0
            f.jawOpen = b["jawOpen"] ?? 0
            f.browDownLeft = b["browDownLeft"] ?? 0
            f.browDownRight = b["browDownRight"] ?? 0
            f.browInnerUp = b["browInnerUp"] ?? 0
            f.mouthClose = b["mouthClose"] ?? 0

            let gaze = WorkZoneMath.gazeDirection(lookAtPoint: face.lookAtPoint, headYaw: face.headYaw, headPitch: face.headPitch)
            f.gazeYaw = gaze.yaw; f.gazePitch = gaze.pitch
            f.headYaw = face.headYaw; f.headPitch = face.headPitch; f.headRoll = face.headRoll

            let c = WorkZoneMath.combinedDirection(gazeYaw: gaze.yaw, gazePitch: gaze.pitch,
                                                   headYaw: face.headYaw, headPitch: face.headPitch)
            lastDeviation = WorkZoneMath.deviation(yaw: c.yaw, pitch: c.pitch, zone: workZone)
            let inside = lastDeviation == 0
            f.inWorkZone = inside ? 1 : 0
            offZoneDwell = inside ? 0 : offZoneDwell + dt
            f.gazeDeviation = lastDeviation

            perclosFrames.append((t, Self.meanClosure(b) > Self.perclosClosure))
            f.yawnActive = yawnDetector.isOpen ? 1 : 0
            f.headMotion = headMotion
            if !browSamples.isEmpty {
                f.browTension = browSamples.reduce(0) { $0 + $1.v } / Float(browSamples.count)
            }
        } else {
            // No face: treated as outside the work zone (the student has turned away or left);
            // deviation holds its last value, other face features use the schema missing value 0.
            f.inWorkZone = 0
            offZoneDwell += dt
            f.gazeDeviation = lastDeviation
        }
        f.offZoneDwell = Float(offZoneDwell)

        // Trailing-window event statistics.
        let elapsed = min(Self.windowSeconds, t - (firstFrameT ?? t) + 1 / FeatureFrame.rateHz)
        f.blinkRate = Float(Double(blinks.count) / elapsed * 60)
        if !blinks.isEmpty {
            f.blinkMeanDuration = Float(blinks.reduce(0) { $0 + $1.duration } / Double(blinks.count))
        }
        f.longBlinkCount = Float(blinks.filter { $0.duration > Self.longBlinkSeconds }.count)
        if !perclosFrames.isEmpty {
            f.perclos = Float(perclosFrames.filter(\.closed).count) / Float(perclosFrames.count)
        }
        f.yawnCount = Float(yawnTimes.count)
        f.nodCount = Float(nodTimes.count)

        // Motion.
        if let m = latestMotion {
            f.phonePickup = m.pickedUp ? 1 : 0
            f.motionMagnitude = motionMagnitude
        }

        // Heart rate pass-through.
        if let hr = latestHR, t - hr.t <= Self.heartRateStaleSeconds, hr.bpm > 0 {
            f.heartRate = hr.bpm
            f.heartRateQuality = hr.quality
            if let base = hr.baselineBPM, base > 0 { f.heartRateDelta = hr.bpm - base }
        }
        return f
    }

    // MARK: Helpers

    static func meanClosure(_ b: [String: Float]) -> Float {
        ((b["eyeBlinkLeft"] ?? 0) + (b["eyeBlinkRight"] ?? 0)) / 2
    }

    /// Removes leading elements (time-ordered) whose key is before `cutoff`.
    static func prune<T>(_ a: inout [T], before cutoff: Double, key: (T) -> Double) {
        var i = 0
        while i < a.count, key(a[i]) < cutoff { i += 1 }
        if i > 0 { a.removeFirst(i) }
    }
}
