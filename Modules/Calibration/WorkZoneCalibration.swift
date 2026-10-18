import Foundation

// MARK: - Samples

/// One combined gaze direction sample used for work-zone calibration.
/// `yaw = gazeYaw + headYaw`, `pitch = gazePitch + headPitch` (radians), at session/media time `t`.
public struct CalibrationGazeReading: Equatable, Sendable {
    public var t: Double
    public var yaw: Float
    public var pitch: Float

    public init(t: Double, yaw: Float, pitch: Float) {
        self.t = t; self.yaw = yaw; self.pitch = pitch
    }

    /// Combined gaze+head direction from a live/replayed frame. nil when no face is tracked.
    public init?(frame: FeatureFrame) {
        guard frame.faceDetected >= 0.5 else { return nil }
        self.init(t: frame.t, yaw: frame.gazeYaw + frame.headYaw, pitch: frame.gazePitch + frame.headPitch)
    }

    /// Adapts a FeatureFrame stream (live or replay) into calibration readings (frames without a face are dropped).
    public static func stream(from frames: AsyncStream<FeatureFrame>) -> AsyncStream<CalibrationGazeReading> {
        AsyncStream { cont in
            let task = Task {
                for await f in frames {
                    if let r = CalibrationGazeReading(frame: f) { cont.yield(r) }
                }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }
}

// MARK: - Builder

/// Turns center + corner samples into a `WorkZone`. The orchestrator may swap in Perception's
/// `WorkZoneMath.calibrate(center:corners:isPhoneScreen:)` behind this protocol.
public protocol CalibrationWorkZoneBuilder {
    func build(center: [(yaw: Float, pitch: Float)], corners: [[(yaw: Float, pitch: Float)]],
               isPhoneScreen: Bool) -> WorkZone
}

/// Default builder: per-axis median of the center samples; each corner reduced to its per-axis median;
/// half-width = max |corner median - center| over the corners, plus a fixed margin.
/// Falls back to `WorkZone.default` values for any axis without data.
public struct CalibrationMedianWorkZoneBuilder: CalibrationWorkZoneBuilder {
    public var margin: Float

    public init(margin: Float = 0.05) { self.margin = margin }

    public func build(center: [(yaw: Float, pitch: Float)], corners: [[(yaw: Float, pitch: Float)]],
                      isPhoneScreen: Bool) -> WorkZone {
        let d = WorkZone.default
        let cy = CalibrationStats.median(center.map(\.yaw)) ?? d.centerYaw
        let cp = CalibrationStats.median(center.map(\.pitch)) ?? d.centerPitch
        let cornerMedians: [(yaw: Float, pitch: Float)] = corners.compactMap { s in
            guard let y = CalibrationStats.median(s.map(\.yaw)), let p = CalibrationStats.median(s.map(\.pitch)) else { return nil }
            return (y, p)
        }
        let hy = cornerMedians.map { abs($0.yaw - cy) }.max().map { $0 + margin } ?? d.halfWidthYaw
        let hp = cornerMedians.map { abs($0.pitch - cp) }.max().map { $0 + margin } ?? d.halfWidthPitch
        return WorkZone(centerYaw: cy, centerPitch: cp, halfWidthYaw: hy, halfWidthPitch: hp,
                        calibratedAt: Date(), isPhoneScreen: isPhoneScreen)
    }
}

// MARK: - Targets & run logic

public enum CalibrationTarget: Int, CaseIterable, Sendable, Identifiable {
    case center, topLeft, topRight, bottomRight, bottomLeft
    public var id: Int { rawValue }

    /// Position in a unit square, x/y in -1...1 (y down), for drawing.
    public var unitPosition: (x: Double, y: Double) {
        switch self {
        case .center: (0, 0)
        case .topLeft: (-1, -1)
        case .topRight: (1, -1)
        case .bottomRight: (1, 1)
        case .bottomLeft: (-1, 1)
        }
    }

    public var instruction: String {
        switch self {
        case .center: "Look at the center"
        case .topLeft: "Look at the top-left corner"
        case .topRight: "Look at the top-right corner"
        case .bottomRight: "Look at the bottom-right corner"
        case .bottomLeft: "Look at the bottom-left corner"
        }
    }
}

/// Pure, clock-driven state machine for one calibration pass (center + 4 corners).
/// Each target is held for `dwell` seconds; samples in the first `settle` seconds are discarded
/// (saccade + fixation settling). A target that ends with too few samples (face lost) is repeated.
public struct WorkZoneCalibrationRun: Sendable {
    public var dwell: Double
    public var settle: Double
    public var minSamplesPerTarget: Int

    public private(set) var targetIndex = 0
    public private(set) var targetStart: Double?
    public private(set) var lastTime: Double?
    public private(set) var samples: [[(yaw: Float, pitch: Float)]]

    public init(dwell: Double = 2.0, settle: Double = 0.4, minSamplesPerTarget: Int = 5) {
        self.dwell = dwell; self.settle = settle; self.minSamplesPerTarget = minSamplesPerTarget
        samples = Array(repeating: [], count: CalibrationTarget.allCases.count)
    }

    public var isFinished: Bool { targetIndex >= CalibrationTarget.allCases.count }
    public var currentTarget: CalibrationTarget? { isFinished ? nil : CalibrationTarget(rawValue: targetIndex) }

    /// Progress 0...1 through the current target's dwell.
    public var progress: Double {
        guard let s = targetStart, let t = lastTime, !isFinished else { return isFinished ? 1 : 0 }
        return min(1, max(0, (t - s) / dwell))
    }

    /// Feed one reading. Returns true when the run has just finished.
    @discardableResult
    public mutating func ingest(_ r: CalibrationGazeReading) -> Bool {
        guard !isFinished else { return false }
        lastTime = r.t
        guard let start = targetStart else { targetStart = r.t; return false }
        let elapsed = r.t - start
        if elapsed >= dwell {
            if samples[targetIndex].count >= minSamplesPerTarget {
                targetIndex += 1
            } else {
                samples[targetIndex].removeAll()   // repeat this target
            }
            targetStart = r.t
            return isFinished
        }
        if elapsed >= settle { samples[targetIndex].append((r.yaw, r.pitch)) }
        return false
    }

    public func makeWorkZone(builder: any CalibrationWorkZoneBuilder, isPhoneScreen: Bool) -> WorkZone {
        builder.build(center: samples[CalibrationTarget.center.rawValue],
                      corners: Array(samples.dropFirst()), isPhoneScreen: isPhoneScreen)
    }
}

// MARK: - Store

/// Persists the calibrated cones in UserDefaults (laptop and phone-screen zones kept separately).
public final class WorkZoneStore {
    public static let laptopKey = "focusloop.calibration.workZone.laptop"
    public static let phoneKey = "focusloop.calibration.workZone.phoneScreen"

    private let defaults: UserDefaults

    public init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    public func save(_ zone: WorkZone) {
        guard let data = try? JSONEncoder().encode(zone) else { return }
        defaults.set(data, forKey: zone.isPhoneScreen ? Self.phoneKey : Self.laptopKey)
    }

    public func load(isPhoneScreen: Bool) -> WorkZone? {
        guard let data = defaults.data(forKey: isPhoneScreen ? Self.phoneKey : Self.laptopKey) else { return nil }
        return try? JSONDecoder().decode(WorkZone.self, from: data)
    }

    /// Calibrated zone, else `WorkZone.default` (with the requested `isPhoneScreen`).
    public func current(isPhoneScreen: Bool) -> WorkZone {
        if let z = load(isPhoneScreen: isPhoneScreen) { return z }
        var d = WorkZone.default
        d.isPhoneScreen = isPhoneScreen
        return d
    }

    public func clear() {
        defaults.removeObject(forKey: Self.laptopKey)
        defaults.removeObject(forKey: Self.phoneKey)
    }
}

// MARK: - Simulation (SYNTHETIC)

/// SYNTHETIC gaze generator for the Simulator: nominal target directions plus small noise.
public struct CalibrationGazeSimulator: Sendable {
    public var isPhoneScreen: Bool
    private var rng: CalibrationSplitMix64

    public init(isPhoneScreen: Bool, seed: UInt64 = 7) {
        self.isPhoneScreen = isPhoneScreen
        rng = CalibrationSplitMix64(seed: seed)
    }

    /// Nominal (yaw, pitch) for a target: a laptop beside the phone sits off to one side and slightly
    /// below; the phone screen is straight ahead and narrower.
    public func nominal(_ target: CalibrationTarget) -> (yaw: Float, pitch: Float) {
        let (cy, cp, hy, hp): (Float, Float, Float, Float) = isPhoneScreen ? (0, -0.08, 0.12, 0.2) : (0.3, -0.1, 0.26, 0.16)
        let p = target.unitPosition
        return (cy + Float(p.x) * hy, cp - Float(p.y) * hp)
    }

    public mutating func reading(for target: CalibrationTarget, at t: Double) -> CalibrationGazeReading {
        let n = nominal(target)
        let ny = Float(Double.random(in: -0.015...0.015, using: &rng))
        let np = Float(Double.random(in: -0.015...0.015, using: &rng))
        return CalibrationGazeReading(t: t, yaw: n.yaw + ny, pitch: n.pitch + np)
    }
}
