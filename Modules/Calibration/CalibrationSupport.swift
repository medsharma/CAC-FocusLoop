import Foundation

// Small helpers shared inside the Calibration module (prefixed to stay globally unique in the single app target).

/// Deterministic, seedable RNG (SplitMix64) so SART sequences and sampling jitter are reproducible in tests.
public struct CalibrationSplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    /// Seeded from the system RNG.
    public init() {
        var g = SystemRandomNumberGenerator()
        state = g.next()
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Descriptive statistics used by the SART labeler and the work-zone builder.
public enum CalibrationStats {
    public static func mean(_ x: [Double]) -> Double? {
        x.isEmpty ? nil : x.reduce(0, +) / Double(x.count)
    }

    /// Sample standard deviation (n - 1). nil for fewer than 2 values.
    public static func standardDeviation(_ x: [Double]) -> Double? {
        guard x.count >= 2, let m = mean(x) else { return nil }
        let ss = x.reduce(0) { $0 + ($1 - m) * ($1 - m) }
        return (ss / Double(x.count - 1)).squareRoot()
    }

    /// Coefficient of variation = sample SD / mean. nil for fewer than 2 values or a non-positive mean.
    public static func coefficientOfVariation(_ x: [Double]) -> Double? {
        guard let m = mean(x), m > 0, let sd = standardDeviation(x) else { return nil }
        return sd / m
    }

    /// Percentile with linear interpolation between closest ranks (numpy's default). `p` in 0...1.
    public static func percentile(_ x: [Double], _ p: Double) -> Double? {
        guard !x.isEmpty else { return nil }
        let s = x.sorted()
        let pos = min(max(p, 0), 1) * Double(s.count - 1)
        let lo = Int(pos.rounded(.down)), hi = Int(pos.rounded(.up))
        let frac = pos - Double(lo)
        return s[lo] + (s[hi] - s[lo]) * frac
    }

    public static func median(_ x: [Float]) -> Float? {
        guard !x.isEmpty else { return nil }
        let s = x.sorted()
        let n = s.count
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
    }
}

public enum CalibrationEnvironment {
    /// True when running in the iOS Simulator (no ARKit face tracking): simulated flows are offered.
    public static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }
}

/// Selects the frames whose session time lies in [start, end).
func calibrationFrames(_ frames: [FeatureFrame], from start: Double, to end: Double) -> [FeatureFrame] {
    frames.filter { $0.t >= start && $0.t < end }
}
