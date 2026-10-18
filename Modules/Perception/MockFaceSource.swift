import Foundation
import simd

/// SYNTHETIC face samples for tests, previews and the Simulator. Never real data.
///
/// Scriptable either with a list of `Segment`s (behaviours played back to back, optionally looped)
/// or with an arbitrary generator closure `(t) -> FaceSample`. `render(...)` produces the same samples
/// offline and deterministically for tests; `start()` plays them in real time on a private queue.
public final class MockFaceSource: FaceSignalSource {
    /// Always true: this source is SYNTHETIC.
    public let isSynthetic = true
    public static let label = "SYNTHETIC"

    public enum Behavior: Equatable, Sendable {
        /// Looking at the work zone centre, relaxed, blinking every `blinkEvery` s (0.2 s blinks).
        case focused(blinkEvery: Double)
        /// Gaze and head turned away by `yaw` radians (+ = user's left).
        case lookAway(yaw: Float)
        /// Eyes closed (closure 0.95) for the whole segment.
        case eyesClosed
        /// Heavy lids (closure 0.25) with a `blinkDuration` s blink every `blinkEvery` s.
        case drowsy(blinkEvery: Double, blinkDuration: Double)
        /// Jaw opens to 0.85 (0.5 s ramps in and out) for the whole segment.
        case yawn
        /// Head pitch dips by `depth` rad (half-sine) over the whole segment.
        case nod(depth: Float)
        /// Brows lowered by `amount` (0...1).
        case browTension(amount: Float)
        /// Face not tracked.
        case noFace
    }

    public struct Segment: Equatable, Sendable {
        public var duration: Double
        public var behavior: Behavior
        public init(_ behavior: Behavior, duration: Double) { self.behavior = behavior; self.duration = duration }
    }

    public var isAvailable: Bool { true }
    public let samples: AsyncStream<FaceSample>
    public let rateHz: Double
    public let loop: Bool
    public let generator: (Double) -> FaceSample
    /// Script length in seconds (nil for open-ended generators).
    public let duration: Double?

    private let continuation: AsyncStream<FaceSample>.Continuation
    private let queue = DispatchQueue(label: "org.focusloop.perception.mockface")
    private var timer: DispatchSourceTimer?

    public init(rateHz: Double = 30, duration: Double? = nil, loop: Bool = true, generator: @escaping (Double) -> FaceSample) {
        self.rateHz = rateHz; self.duration = duration; self.loop = loop; self.generator = generator
        var c: AsyncStream<FaceSample>.Continuation!
        samples = AsyncStream(bufferingPolicy: .bufferingNewest(120)) { c = $0 }
        continuation = c
    }

    public convenience init(script: [Segment], rateHz: Double = 30, loop: Bool = true) {
        let total = script.reduce(0) { $0 + $1.duration }
        self.init(rateHz: rateHz, duration: total, loop: loop) { t in
            MockFaceSource.sample(script: script, at: loop && total > 0 ? t.truncatingRemainder(dividingBy: total) : t, outputT: t)
        }
    }

    /// A gentle default demo loop (SYNTHETIC): focus, a glance away, drowsiness, a yawn and a nod.
    public static func demo(rateHz: Double = 30) -> MockFaceSource {
        MockFaceSource(script: [
            Segment(.focused(blinkEvery: 4), duration: 20),
            Segment(.lookAway(yaw: 0.9), duration: 6),
            Segment(.focused(blinkEvery: 4), duration: 10),
            Segment(.drowsy(blinkEvery: 2.5, blinkDuration: 0.8), duration: 12),
            Segment(.yawn, duration: 3),
            Segment(.nod(depth: 0.35), duration: 1.6),
            Segment(.focused(blinkEvery: 4), duration: 10),
        ], rateHz: rateHz)
    }

    deinit {
        timer?.cancel()
        continuation.finish()
    }

    public func start() throws {
        guard timer == nil else { return }
        let start = ProcessInfo.processInfo.systemUptime
        let gen = generator, cont = continuation, dur = duration, loop = loop
        let tm = DispatchSource.makeTimerSource(queue: queue)
        tm.schedule(deadline: .now(), repeating: 1 / rateHz)
        tm.setEventHandler { [weak self] in
            let t = ProcessInfo.processInfo.systemUptime - start
            if let d = dur, !loop, t > d { self?.stop(); return }
            cont.yield(gen(t))
        }
        timer = tm
        tm.resume()
    }

    public func stop() {
        timer?.cancel()
        timer = nil
    }

    /// Deterministic offline rendering of samples at `rateHz` over [from, to).
    public func render(from: Double = 0, to: Double) -> [FaceSample] {
        let n = Int(((to - from) * rateHz).rounded(.up))
        return (0..<n).map { generator(from + Double($0) / rateHz) }
    }

    // MARK: Script evaluation

    static func sample(script: [Segment], at t: Double, outputT: Double) -> FaceSample {
        var start = 0.0
        for seg in script {
            if t < start + seg.duration { return sample(seg, local: t - start, outputT: outputT) }
            start += seg.duration
        }
        guard let last = script.last else { return FaceSample(t: outputT, faceDetected: false) }
        return sample(last, local: last.duration, outputT: outputT)
    }

    static func neutralShapes() -> [String: Float] {
        var s: [String: Float] = [:]
        for name in perceptionBlendShapeNames { s[name] = 0.02 }
        s["eyeSquintLeft"] = 0.1; s["eyeSquintRight"] = 0.1
        s["browDownLeft"] = 0.1; s["browDownRight"] = 0.1
        s["mouthClose"] = 0.2; s["jawOpen"] = 0.03
        return s
    }

    static func sample(_ seg: Segment, local: Double, outputT: Double) -> FaceSample {
        var s = neutralShapes()
        var look = SIMD3<Float>(0, 0, 0.5)
        var yaw: Float = 0, pitch: Float = 0
        func setClosure(_ c: Float) { s["eyeBlinkLeft"] = c; s["eyeBlinkRight"] = c }
        switch seg.behavior {
        case let .focused(every):
            if every > 0, local.truncatingRemainder(dividingBy: every) >= every - 0.2 { setClosure(0.95) }
        case let .lookAway(a):
            yaw = a * 0.5
            look = SIMD3<Float>(0.5 * tan(a * 0.5), 0, 0.5)
        case .eyesClosed:
            setClosure(0.95)
        case let .drowsy(every, dur):
            setClosure(0.25)
            if every > 0, local.truncatingRemainder(dividingBy: every) >= every - dur { setClosure(0.95) }
        case .yawn:
            let ramp = max(0, min(1, local / 0.5, (seg.duration - local) / 0.5))
            s["jawOpen"] = max(0.03, Float(0.85 * ramp))
        case let .nod(depth):
            pitch = -depth * Float(sin(Double.pi * min(1, local / max(seg.duration, 1e-6))))
        case let .browTension(amount):
            s["browDownLeft"] = amount; s["browDownRight"] = amount
        case .noFace:
            return FaceSample(t: outputT, faceDetected: false)
        }
        return FaceSample(t: outputT, faceDetected: true, blendShapes: s, lookAtPoint: look,
                          headYaw: yaw, headPitch: pitch, headRoll: 0, skinRGB: nil)
    }
}
