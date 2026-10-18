import Foundation
import simd
#if !targetEnvironment(simulator)
import CoreMotion
#endif

/// Core Motion phone pickup / tilt detection as a `MotionSource`.
///
/// Device: `CMMotionManager` device motion at 30 Hz on a dedicated operation queue, fed through
/// `PerceptionPickupDetector` (rest pose learned after 2 s still; pickup = tilt > 25 deg or an
/// acceleration spike, with hysteresis). Timestamps are `CMDeviceMotion.timestamp` (uptime clock)
/// minus `timeOrigin`, the same clock as `ARKitFaceSource`.
///
/// Simulator: emits still samples at 10 Hz that never report a pickup.
public final class CoreMotionSource: MotionSource {
    public let samples: AsyncStream<MotionSample>
    public static let updateRateHz: Double = 30
    /// Subtracted from the Core Motion timestamp to produce `MotionSample.t`.
    public var timeOrigin: Double = 0
    /// Detector tunables (copied in on `start()`).
    public var detector = PerceptionPickupDetector()

    private let continuation: AsyncStream<MotionSample>.Continuation

    #if !targetEnvironment(simulator)
    private let manager = CMMotionManager()
    private let opQueue: OperationQueue = {
        let q = OperationQueue()
        q.name = "org.focusloop.perception.motion"
        q.maxConcurrentOperationCount = 1
        q.qualityOfService = .userInitiated
        return q
    }()
    #else
    private var timer: DispatchSourceTimer?
    private let simQueue = DispatchQueue(label: "org.focusloop.perception.motion.sim")
    #endif

    public init() {
        var c: AsyncStream<MotionSample>.Continuation!
        samples = AsyncStream(bufferingPolicy: .bufferingNewest(60)) { c = $0 }
        continuation = c
    }

    deinit {
        stop()
        continuation.finish()
    }

    /// True when real device-motion hardware is used (false in the Simulator mock).
    public var isHardware: Bool {
        #if !targetEnvironment(simulator)
        return manager.isDeviceMotionAvailable
        #else
        return false
        #endif
    }

    public func start() {
        #if !targetEnvironment(simulator)
        guard manager.isDeviceMotionAvailable, !manager.isDeviceMotionActive else { return }
        var det = detector
        det.reset()
        let origin = timeOrigin
        let cont = continuation
        manager.deviceMotionUpdateInterval = 1 / Self.updateRateHz
        manager.startDeviceMotionUpdates(to: opQueue) { motion, _ in
            guard let m = motion else { return }
            let g = SIMD3<Float>(Float(m.gravity.x), Float(m.gravity.y), Float(m.gravity.z))
            let a = SIMD3<Float>(Float(m.userAcceleration.x), Float(m.userAcceleration.y), Float(m.userAcceleration.z))
            cont.yield(det.update(t: m.timestamp - origin, gravity: g, userAcceleration: a))
        }
        #else
        guard timer == nil else { return }
        let origin = timeOrigin
        let cont = continuation
        let tm = DispatchSource.makeTimerSource(queue: simQueue)
        tm.schedule(deadline: .now(), repeating: 0.1)
        tm.setEventHandler {
            let t = ProcessInfo.processInfo.systemUptime - origin
            cont.yield(MotionSample(t: t, pickedUp: false, tilt: 0, accelerationMagnitude: 0))
        }
        timer = tm
        tm.resume()
        #endif
    }

    public func stop() {
        #if !targetEnvironment(simulator)
        if manager.isDeviceMotionActive { manager.stopDeviceMotionUpdates() }
        #else
        timer?.cancel()
        timer = nil
        #endif
    }
}
