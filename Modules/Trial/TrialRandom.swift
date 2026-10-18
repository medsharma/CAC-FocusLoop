import Foundation

/// Deterministic, seedable RNG (SplitMix64) so trial decisions are reproducible in tests.
/// Production uses a time-derived seed; tests inject a fixed one.
public struct TrialSeededRNG: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    /// Seeded from the system RNG (non-reproducible), for production.
    public static func system() -> TrialSeededRNG {
        var g = SystemRandomNumberGenerator()
        return TrialSeededRNG(seed: g.next())
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Sampling helpers used by the trial engines (Gamma via Marsaglia & Tsang 2000, Beta via two Gammas).
enum TrialDistributions {
    /// Uniform in the open interval (0, 1).
    static func uniform<G: RandomNumberGenerator>(_ rng: inout G) -> Double {
        // 53 random bits, shifted off zero.
        (Double(rng.next() >> 11) + 0.5) / 9_007_199_254_740_992.0
    }

    /// Standard normal via Box-Muller.
    static func normal<G: RandomNumberGenerator>(_ rng: inout G) -> Double {
        let u1 = uniform(&rng), u2 = uniform(&rng)
        return (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2)
    }

    /// Gamma(shape, 1).
    static func gamma<G: RandomNumberGenerator>(shape: Double, _ rng: inout G) -> Double {
        if shape < 1 {
            // Boost: Gamma(a) = Gamma(a + 1) * U^(1/a)
            let g = gamma(shape: shape + 1, &rng)
            return g * pow(uniform(&rng), 1 / shape)
        }
        let d = shape - 1.0 / 3.0
        let c = 1.0 / (9 * d).squareRoot()
        while true {
            var x: Double, v: Double
            repeat {
                x = normal(&rng)
                v = 1 + c * x
            } while v <= 0
            v = v * v * v
            let u = uniform(&rng)
            if u < 1 - 0.0331 * x * x * x * x { return d * v }
            if log(u) < 0.5 * x * x + d * (1 - v + log(v)) { return d * v }
        }
    }

    /// Beta(alpha, beta).
    static func beta<G: RandomNumberGenerator>(alpha: Double, beta: Double, _ rng: inout G) -> Double {
        let x = gamma(shape: alpha, &rng)
        let y = gamma(shape: beta, &rng)
        let s = x + y
        return s > 0 ? x / s : 0.5
    }

    /// Draws an index from a discrete distribution (weights need not be normalised).
    static func categorical<G: RandomNumberGenerator>(_ weights: [Double], _ rng: inout G) -> Int {
        let total = weights.reduce(0, +)
        guard total > 0 else { return 0 }
        var u = uniform(&rng) * total
        for (i, w) in weights.enumerated() {
            u -= w
            if u < 0 { return i }
        }
        return weights.count - 1
    }
}
