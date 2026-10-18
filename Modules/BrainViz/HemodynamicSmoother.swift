import Foundation

/// Two-stage exponential lag, ported from the Axonal Labs website (`brain.js` stepBold).
/// Gives activations the slow, blood-flow-like rise and fall of a BOLD signal.
///
/// Note: `<lecture>.brain.json` is already shifted for hemodynamic lag offline, so here this
/// is purely visual easing between 1 Hz samples — it does not add another physiological delay.
public struct HemodynamicSmoother {
    public var tauFast: Float = 0.45
    public var tauSlow: Float = 0.75
    private var fast: [Float]
    private var slow: [Float]

    public init(count: Int) {
        fast = .init(repeating: 0, count: count)
        slow = fast
    }

    /// Advances by `dt` seconds toward `targets` and returns the smoothed values.
    public mutating func step(targets: [Float], dt: Float) -> [Float] {
        let a = min(1, dt / tauFast), b = min(1, dt / tauSlow)
        for i in fast.indices {
            let t = i < targets.count ? targets[i] : 0
            fast[i] += (t - fast[i]) * a
            slow[i] += (fast[i] - slow[i]) * b
        }
        return slow
    }
}
