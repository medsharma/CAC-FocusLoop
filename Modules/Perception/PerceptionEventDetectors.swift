import Foundation

// Small, pure, deterministic state machines used by `AttentionFeatureExtractor`.
// All take explicit timestamps so they are testable without hardware.

/// Blink detection on mean eye closure with hysteresis: onset when closure rises above
/// `onThreshold`, offset when it falls below `offThreshold`.
public struct PerceptionBlinkDetector: Sendable {
    public struct Blink: Equatable, Sendable {
        public var onset: Double
        public var offset: Double
        public var duration: Double { offset - onset }
    }

    public var onThreshold: Float = 0.5
    public var offThreshold: Float = 0.3
    public private(set) var onsetTime: Double?

    public init(onThreshold: Float = 0.5, offThreshold: Float = 0.3) {
        self.onThreshold = onThreshold; self.offThreshold = offThreshold
    }

    public var isClosed: Bool { onsetTime != nil }

    /// Feed one sample of mean eye closure (0...1). Returns a completed blink at its offset.
    public mutating func update(closure: Float, t: Double) -> Blink? {
        if let on = onsetTime {
            if closure < offThreshold {
                onsetTime = nil
                return Blink(onset: on, offset: t)
            }
        } else if closure > onThreshold {
            onsetTime = t
        }
        return nil
    }

    /// Abandon an in-progress blink (e.g. face lost).
    public mutating func cancel() { onsetTime = nil }
}

/// Yawn: `jawOpen` above `threshold` sustained for at least `minDuration`. One event per opening,
/// reported at the moment the duration criterion is first met.
public struct PerceptionYawnDetector: Sendable {
    public var threshold: Float = 0.6
    public var minDuration: Double = 1.5
    public private(set) var openSince: Double?
    private var counted = false

    public init(threshold: Float = 0.6, minDuration: Double = 1.5) {
        self.threshold = threshold; self.minDuration = minDuration
    }

    /// True while the jaw is above threshold (matches the synthetic generator's `yawnActive`).
    public var isOpen: Bool { openSince != nil }

    /// Returns the event time when a yawn is confirmed.
    public mutating func update(jawOpen: Float, t: Double) -> Double? {
        if jawOpen > threshold {
            if openSince == nil { openSince = t; counted = false }
            if !counted, let s = openSince, t - s >= minDuration - 1e-9 {
                counted = true
                return t
            }
        } else {
            openSince = nil; counted = false
        }
        return nil
    }

    public mutating func cancel() { openSince = nil; counted = false }
}

/// Head nod: pitch drops at least `dropThreshold` below a slow baseline, then recovers to within
/// `recoveryTolerance` of the baseline within `maxDuration` of the drop. Event reported at recovery.
/// The baseline is an exponential average of pitch (time constant `baselineTau`) frozen during a drop.
public struct PerceptionNodDetector: Sendable {
    public var dropThreshold: Float = 0.25
    public var recoveryTolerance: Float = 0.1
    public var maxDuration: Double = 2.0
    public var baselineTau: Double = 4.0

    public private(set) var baseline: Float?
    private var lastT: Double?
    /// Time the pitch first fell below baseline - recoveryTolerance (start of the dip).
    private var dipStart: Double?
    /// True once the dip went deeper than `dropThreshold`.
    private var deepEnough = false
    /// Set when a dip exceeded `maxDuration`: wait for recovery without counting.
    private var expired = false

    public init(dropThreshold: Float = 0.25, recoveryTolerance: Float = 0.1, maxDuration: Double = 2.0) {
        self.dropThreshold = dropThreshold; self.recoveryTolerance = recoveryTolerance; self.maxDuration = maxDuration
    }

    public mutating func update(pitch: Float, t: Double) -> Double? {
        defer { lastT = t }
        guard let b = baseline, let lt = lastT else {
            baseline = pitch
            return nil
        }
        let dt = max(0, t - lt)
        let below = pitch < b - recoveryTolerance
        if dipStart == nil {
            if below {
                dipStart = t; deepEnough = pitch <= b - dropThreshold; expired = false
            } else {
                let a = Float(1 - exp(-dt / baselineTau))
                baseline = b + a * (pitch - b)
            }
            return nil
        }
        // In a dip (baseline frozen).
        if pitch <= b - dropThreshold { deepEnough = true }
        if let s = dipStart, t - s > maxDuration { expired = true }
        if !below {
            let isNod = deepEnough && !expired
            dipStart = nil; deepEnough = false; expired = false
            return isNod ? t : nil
        }
        if expired {
            // Head stays down (e.g. looking at desk): let the baseline adapt so we re-arm.
            let a = Float(1 - exp(-dt / baselineTau))
            baseline = b + a * (pitch - b)
        }
        return nil
    }

    public mutating func reset() {
        baseline = nil; lastT = nil; dipStart = nil; deepEnough = false; expired = false
    }
}

/// Phone pickup / tilt detection from Core Motion values, factored out for testing.
///
/// * Rest pose: learned when the phone is still (user acceleration < `stillAcceleration` and the
///   gravity direction within `stillTilt` of its running mean) for `restLearnSeconds`.
/// * Pickup: tilt from rest > `pickupTilt`, or a user-acceleration spike > `spikeAcceleration`.
/// * Release (hysteresis): tilt < `releaseTilt` and acceleration < `releaseAcceleration` for
///   `releaseHoldSeconds`, or a new rest pose learned after `relearnWhilePickedUpSeconds` of stillness
///   (phone set down somewhere else).
public struct PerceptionPickupDetector: Sendable {
    public var pickupTilt: Float = 25 * .pi / 180
    public var releaseTilt: Float = 15 * .pi / 180
    public var spikeAcceleration: Float = 0.35
    public var releaseAcceleration: Float = 0.08
    public var releaseHoldSeconds: Double = 1.0
    public var stillAcceleration: Float = 0.03
    public var stillTilt: Float = 2 * .pi / 180
    public var restLearnSeconds: Double = 2.0
    /// Stillness needed to accept a new rest pose while picked up (phone set down in a new spot).
    /// Longer than `restLearnSeconds` so holding the phone steadily doesn't count as "set down".
    public var relearnWhilePickedUpSeconds: Double = 5.0

    public private(set) var restGravity: SIMD3<Float>?
    public private(set) var pickedUp = false
    private var stillSince: Double?
    private var stillSum = SIMD3<Float>(repeating: 0)
    private var stillCount = 0
    private var releaseSince: Double?

    public init() {}

    /// `gravity`: gravity vector in device frame (any length > 0). `userAcceleration` in g.
    public mutating func update(t: Double, gravity: SIMD3<Float>, userAcceleration: SIMD3<Float>) -> MotionSample {
        let g = Self.normalized(gravity)
        let accel = (userAcceleration * userAcceleration).sum().squareRoot()

        // --- rest-pose learning ---
        let mean = stillCount > 0 ? Self.normalized(stillSum) : g
        if accel < stillAcceleration && Self.angle(g, mean) < stillTilt {
            if stillSince == nil { stillSince = t; stillSum = .zero; stillCount = 0 }
            stillSum += g; stillCount += 1
            let needed = pickedUp ? relearnWhilePickedUpSeconds : restLearnSeconds
            if let s = stillSince, t - s >= needed - 1e-9 {
                restGravity = Self.normalized(stillSum)
                if pickedUp { pickedUp = false; releaseSince = nil }
            }
        } else {
            stillSince = nil; stillSum = .zero; stillCount = 0
        }

        guard let rest = restGravity else {
            return MotionSample(t: t, pickedUp: false, tilt: 0, accelerationMagnitude: accel)
        }
        let tilt = Self.angle(g, rest)

        // --- pickup with hysteresis ---
        if !pickedUp {
            if tilt > pickupTilt || accel > spikeAcceleration { pickedUp = true; releaseSince = nil }
        } else if tilt < releaseTilt && accel < releaseAcceleration {
            if releaseSince == nil { releaseSince = t }
            if let r = releaseSince, t - r >= releaseHoldSeconds - 1e-9 { pickedUp = false; releaseSince = nil }
        } else {
            releaseSince = nil
        }
        return MotionSample(t: t, pickedUp: pickedUp, tilt: tilt, accelerationMagnitude: accel)
    }

    public mutating func reset() { self = PerceptionPickupDetector() }

    static func normalized(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let l = (v * v).sum().squareRoot()
        return l > 1e-6 ? v / l : SIMD3<Float>(0, 0, -1)
    }

    static func angle(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        acos(min(1, max(-1, (a * b).sum())))
    }
}
