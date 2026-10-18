import CoreGraphics
import Foundation

/// Pure formatting / mapping helpers used by the UI. No SwiftUI, so they are unit-testable.
public enum UIFormat {
    /// "4:07" or "1:02:03".
    public static func timer(_ seconds: Double) -> String {
        let s = max(0, Int(seconds.isFinite ? seconds : 0))
        let h = s / 3600, m = (s % 3600) / 60, sec = s % 60
        return h > 0 ? String(format: "%d:%02d:%02d", h, m, sec) : String(format: "%d:%02d", m, sec)
    }

    /// "Back in 2.8 s". Values are clamped to 0...99.9 and shown with one decimal.
    public static func backIn(_ seconds: Double) -> String {
        "Back in \(oneDecimal(seconds)) s"
    }

    public static func oneDecimal(_ v: Double) -> String {
        let c = min(99.9, max(0, v.isFinite ? v : 0))
        return String(format: "%.1f", c)
    }

    /// "72%" from 0...1.
    public static func percent(_ fraction: Double, decimals: Int = 0) -> String {
        let f = fraction.isFinite ? fraction : 0
        return String(format: "%.\(decimals)f%%", f * 100)
    }

    /// Signed percentage points: "+12 pts", "−4 pts", "0 pts".
    public static func signedPoints(_ fraction: Double) -> String {
        let v = Int((fraction * 100).rounded())
        if v > 0 { return "+\(v) pts" }
        if v < 0 { return "\u{2212}\(-v) pts" }
        return "0 pts"
    }

    public static func bpm(_ v: Float?) -> String {
        guard let v, v > 0, v.isFinite else { return "--" }
        return String(Int(v.rounded()))
    }

    /// Correlation "r = 0.62" (or "r = n/a").
    public static func correlation(_ r: Double?) -> String {
        guard let r, r.isFinite else { return "r = n/a" }
        return String(format: "r = %.2f", r).replacingOccurrences(of: "-", with: "\u{2212}")
    }

    /// Degrees from radians, "12°".
    public static func degrees(_ radians: Float) -> String {
        "\(Int((Double(radians) * 180 / .pi).rounded()))\u{00B0}"
    }

    /// Minutes, "42 min" or "1 h 05 min".
    public static func minutes(_ minutes: Double) -> String {
        let m = Int(minutes.rounded())
        return m >= 60 ? String(format: "%d h %02d min", m / 60, m % 60) : "\(m) min"
    }

    /// Hour of day 0-23 to "9 AM" style.
    public static func hour(_ h: Int) -> String {
        let hh = ((h % 24) + 24) % 24
        let suffix = hh < 12 ? "AM" : "PM"
        let v = hh % 12 == 0 ? 12 : hh % 12
        return "\(v) \(suffix)"
    }
}

/// How long the "Back in X.X s" banner stays visible after a refocus.
public enum RefocusBanner {
    public static let visibleSeconds: Double = 4

    /// Seconds to show, or nil when the banner should be hidden.
    public static func value(lastRefocusSeconds: Double?, lastRefocusAt: Double?, elapsed: Double) -> Double? {
        guard let s = lastRefocusSeconds, let at = lastRefocusAt else { return nil }
        let age = elapsed - at
        return (age >= 0 && age < visibleSeconds) ? s : nil
    }
}

/// Lapse-risk halo mapping around the focus orb.
public enum RiskHalo {
    /// Opacity 0.04...0.65, eased so low risk stays nearly invisible.
    public static func opacity(probability p: Float) -> Double {
        let x = Double(min(1, max(0, p.isFinite ? p : 0)))
        return 0.04 + 0.61 * x * x
    }
    /// Scale of the halo relative to the orb, 1.05...1.45.
    public static func scale(probability p: Float) -> Double {
        let x = Double(min(1, max(0, p.isFinite ? p : 0)))
        return 1.05 + 0.4 * x
    }
}

/// Per-state motion parameters for the focus orb (blended by state probabilities).
public struct OrbMotion: Equatable, Sendable {
    /// Breathing period, seconds.
    public var period: Double
    /// Scale amplitude of the breathing/pulse.
    public var amplitude: Double
    /// Random-ish positional jitter, points.
    public var jitter: Double
    /// Downward drift ("sinking"), points.
    public var sink: Double
    /// Brightness multiplier.
    public var glow: Double

    public init(period: Double, amplitude: Double, jitter: Double, sink: Double, glow: Double) {
        self.period = period; self.amplitude = amplitude; self.jitter = jitter; self.sink = sink; self.glow = glow
    }

    public static func of(_ s: AttentionState) -> OrbMotion {
        switch s {
        case .focused: OrbMotion(period: 6.0, amplitude: 0.045, jitter: 0, sink: 0, glow: 1.0)
        case .distracted: OrbMotion(period: 3.2, amplitude: 0.03, jitter: 9, sink: 0, glow: 0.95)
        case .drowsy: OrbMotion(period: 9.0, amplitude: 0.03, jitter: 0, sink: 22, glow: 0.62)
        case .stressed: OrbMotion(period: 0.85, amplitude: 0.06, jitter: 1.5, sink: 0, glow: 1.1)
        }
    }

    /// Probability-weighted blend (weights need not sum to 1; empty -> focused).
    public static func blend(probabilities: [Float]) -> OrbMotion {
        let total = probabilities.reduce(0, +)
        guard total > 0, probabilities.count == AttentionState.allCases.count else { return .of(.focused) }
        var r = OrbMotion(period: 0, amplitude: 0, jitter: 0, sink: 0, glow: 0)
        for (i, s) in AttentionState.allCases.enumerated() {
            let w = Double(probabilities[i] / total), m = OrbMotion.of(s)
            // Blend frequency (1/period) rather than period, so fast pulses dominate smoothly.
            r.period += w / m.period
            r.amplitude += w * m.amplitude
            r.jitter += w * m.jitter
            r.sink += w * m.sink
            r.glow += w * m.glow
        }
        r.period = 1 / max(r.period, 0.01)
        return r
    }
}

/// Contiguous state segments from a time-ordered list of (t, state) points.
public enum StateSegmenter {
    public static func segments(_ points: [(t: Double, state: AttentionState)], end: Double? = nil) -> [SessionReport.Segment] {
        guard let first = points.first else { return [] }
        var out: [SessionReport.Segment] = []
        var start = first.t, cur = first.state
        for p in points.dropFirst() where p.state != cur {
            out.append(.init(start: start, end: p.t, state: cur))
            start = p.t; cur = p.state
        }
        let last = end ?? (points.last!.t + 1)
        out.append(.init(start: start, end: max(last, start), state: cur))
        return out
    }

    /// Fraction of total segment time spent in each state (`AttentionState.allCases` order).
    public static func fractions(_ segs: [SessionReport.Segment]) -> [Double] {
        var acc = Array(repeating: 0.0, count: AttentionState.allCases.count)
        for s in segs { acc[s.state.index] += max(0, s.end - s.start) }
        let total = acc.reduce(0, +)
        return total > 0 ? acc.map { $0 / total } : acc
    }
}

public enum UIStats {
    /// Pearson correlation, nil if fewer than 3 points or zero variance.
    public static func pearson(_ x: [Double], _ y: [Double]) -> Double? {
        let n = min(x.count, y.count)
        guard n >= 3 else { return nil }
        let xs = x.prefix(n), ys = y.prefix(n)
        let mx = xs.reduce(0, +) / Double(n), my = ys.reduce(0, +) / Double(n)
        var sxy = 0.0, sxx = 0.0, syy = 0.0
        for (a, b) in zip(xs, ys) {
            sxy += (a - mx) * (b - my); sxx += (a - mx) * (a - mx); syy += (b - my) * (b - my)
        }
        guard sxx > 0, syy > 0 else { return nil }
        return sxy / (sxx * syy).squareRoot()
    }

    /// Min-max scale to 0...1 (constant input -> all 0.5).
    public static func minMax(_ v: [Double]) -> [Double] {
        guard let lo = v.min(), let hi = v.max() else { return [] }
        guard hi > lo else { return v.map { _ in 0.5 } }
        return v.map { ($0 - lo) / (hi - lo) }
    }

    public static func median(_ v: [Double]) -> Double? {
        guard !v.isEmpty else { return nil }
        let s = v.sorted(), n = s.count
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
    }

    /// Exponential moving average with smoothing factor alpha.
    public static func ema(_ v: [Double], alpha: Double) -> [Double] {
        var out: [Double] = []; out.reserveCapacity(v.count)
        var acc: Double?
        for x in v {
            acc = acc.map { $0 + alpha * (x - $0) } ?? x
            out.append(acc!)
        }
        return out
    }
}

/// Symmetric x-axis domain for the Personal Trial whisker chart, padded and never narrower than ±0.2.
public enum EffectAxis {
    public static func domain(_ effects: [ArmEffect]) -> ClosedRange<Double> {
        let m = effects.flatMap { [abs($0.lower), abs($0.upper), abs($0.effect)] }.max() ?? 0
        let r = max(0.2, min(1, (m * 1.15 * 10).rounded(.up) / 10))
        return -r...r
    }

    /// Plain-language verdict for an effect interval.
    /// Experimental arms (cool-light pulse) never get a "helps" verdict: they are under test only.
    public static func verdict(_ e: ArmEffect) -> String {
        if e.arm.isExperimental { return "Experimental, under test" }
        if e.n < 3 { return "Too few tries yet" }
        if e.lower > 0 { return "Likely helps you" }
        if e.upper < 0 { return "Likely not for you" }
        return "Still learning"
    }
}

/// Maps normalized (0...1, origin top-left) face-mesh points into a view rect, preserving aspect (fit).
public enum FaceMeshMapper {
    public static func map(_ points: [CGPoint], into rect: CGRect, mirror: Bool = true) -> [CGPoint] {
        let side = min(rect.width, rect.height)
        let ox = rect.midX - side / 2, oy = rect.midY - side / 2
        return points.map { p in
            let x = mirror ? 1 - p.x : p.x
            return CGPoint(x: ox + x * side, y: oy + p.y * side)
        }
    }
}

/// Gauge normalisation for science-view meters.
public enum GaugeScale {
    /// Blink rate: 0...40 per minute.
    public static func blinkRate(_ v: Float) -> Double { clamp01(Double(v) / 40) }
    /// PERCLOS: 0...0.4 maps to full scale (0.15+ is high).
    public static func perclos(_ v: Float) -> Double { clamp01(Double(v) / 0.4) }
    /// Gaze deviation: 0...0.6 rad.
    public static func gazeDeviation(_ v: Float) -> Double { clamp01(Double(v) / 0.6) }
    /// Head pose: -0.6...0.6 rad to 0...1 with 0.5 = straight ahead.
    public static func headAngle(_ v: Float) -> Double { clamp01(0.5 + Double(v) / 1.2) }
    public static func clamp01(_ x: Double) -> Double { x.isFinite ? min(1, max(0, x)) : 0 }
}
