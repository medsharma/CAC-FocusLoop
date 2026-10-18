import Foundation

/// SYNTHETIC skin-colour trace for rPPG tests. Every parameter mirrors
/// `ml/rppg_reference/synthetic_params.json` (the Python cross-check `test_params_match_swift`
/// greps the `static let` lines below, so keep the literal values in sync with the JSON).
///
/// Model (linear RGB, 0...1):
///   c_i(t) = I(t) * base_i * (1 + pulseRelAmplitude * w_i * p(t)) + noiseSigma * n_i
///   p(t)   = (sin(2 pi f t) + h * sin(4 pi f t + phi)) / (1 + h)
///   I(t)   = 1 + driftAmplitude * sin(2 pi driftFrequency t) + driftSlope * t
/// Pulsatile channel weights w = [0.33, 0.77, 0.53] are the normalised blood-volume-pulse signature
/// from de Haan & van Leest 2014; pulseRelAmplitude 0.4 % of DC is in the range reported for facial
/// rPPG. Timestamps t_k = k / fsNominal + jitter * (2u - 1).
/// RNG: splitmix64 seeded with seedBase + bpm; per sample: one jitter uniform, then R, G, B normals
/// (Box-Muller cosine branch, two uniforms each, u1 replaced by 1 - u1).
enum RPPGSynth {
    static let fsNominal = 30.0
    static let jitter = 0.003
    static let duration = 30.0
    static let baseRGB: [Double] = [0.42, 0.28, 0.22]
    static let pulseRelAmplitude = 0.004
    static let pulseChannelWeights: [Double] = [0.33, 0.77, 0.53]
    static let harmonicRatio = 0.35
    static let harmonicPhase = 0.8
    static let driftAmplitude = 0.06
    static let driftFrequency = 0.04
    static let driftSlope = 0.003
    static let noiseSigma = 0.0003
    static let seedBase: UInt64 = 1000
    static let bpms: [Double] = [54, 72, 96, 120]

    struct Sample { var t: Double; var rgb: SIMD3<Float> }

    /// splitmix64 (Steele, Lea & Flood 2014), bit-identical to the Python reference.
    struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func uniform() -> Double { Double(next() >> 11) * 0x1.0p-53 }
        mutating func normal() -> Double {
            let u1 = 1 - uniform(), u2 = uniform()
            return (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2)
        }
    }

    /// - Parameters:
    ///   - bpm: true pulse rate; pass `pulseAmplitude: 0` for a pure-noise trace.
    ///   - seed: defaults to seedBase + bpm.
    static func generate(bpm: Double, duration: Double = duration, pulseAmplitude: Double = pulseRelAmplitude,
                         noiseSigma: Double = noiseSigma, seed: UInt64? = nil) -> [Sample] {
        var rng = SplitMix64(state: seed ?? (seedBase + UInt64(bpm)))
        let f = bpm / 60
        let n = Int(duration * fsNominal)
        var out: [Sample] = []
        out.reserveCapacity(n)
        for k in 0..<n {
            let t = Double(k) / fsNominal + jitter * (2 * rng.uniform() - 1)
            let p = (sin(2 * Double.pi * f * t) + harmonicRatio * sin(4 * Double.pi * f * t + harmonicPhase))
                / (1 + harmonicRatio)
            let illum = 1 + driftAmplitude * sin(2 * Double.pi * driftFrequency * t) + driftSlope * t
            var c = [Double](repeating: 0, count: 3)
            for i in 0..<3 {
                c[i] = illum * baseRGB[i] * (1 + pulseAmplitude * pulseChannelWeights[i] * p)
                    + noiseSigma * rng.normal()
            }
            out.append(Sample(t: t, rgb: SIMD3<Float>(Float(c[0]), Float(c[1]), Float(c[2]))))
        }
        return out
    }
}
