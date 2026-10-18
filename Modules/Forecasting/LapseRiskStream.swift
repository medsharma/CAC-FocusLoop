import Foundation

/// Produces the forecast line that runs AHEAD of "now" for `LiveSessionModel.forecastLine`.
///
/// It is an extrapolation of the current forecaster output, not a second model:
///   logit p(now + k) = logit(base) + (logit p_now - logit(base)) * decay(k)        // mean reversion
///                    + slope * k * decay(k)                                         // damped recent trend
///                    + driftGain  * (drift(ph + k) - drift(ph))                     // content ahead (if any)
///                    + valleyGain * (valleyProx(ph + k) - valleyProx(ph))
/// with decay(k) = exp(-k / reversionTau). Content terms use the precomputed brain timeline
/// (predicted brain response to content, average brain), which is known ahead of the playhead.
public final class LapseRiskStream {
    public var horizon: Double
    public var step: Double
    /// Long-run lapse probability the line relaxes toward.
    public var baseRate: Float = 0.15
    public var reversionTau: Double = 30
    public var driftGain: Float = 0.45
    public var valleyGain: Float = 1.0
    /// Max |trend| in logit units per second.
    public var maxSlope: Float = 0.15

    public private(set) var history: [LapseRisk] = []
    public private(set) var line: [(t: Double, p: Float)] = []
    private let historyLength = 8

    public init(horizon: Double = FeatureFrame.forecastHorizonSeconds, step: Double = 1) {
        self.horizon = horizon
        self.step = step
    }

    public func reset() { history.removeAll(); line.removeAll() }

    /// Push the latest forecaster output and get the updated forecast line (first point = now).
    /// - Parameters:
    ///   - timeline: the lecture's content timeline, or nil in free study.
    ///   - playhead: lecture playhead in seconds that corresponds to `risk.t`.
    @discardableResult
    public func update(risk: LapseRisk, timeline: ContentTimeline? = nil, playhead: Double? = nil) -> [(t: Double, p: Float)] {
        history.append(risk)
        if history.count > historyLength { history.removeFirst(history.count - historyLength) }
        line = extrapolate(from: risk, slope: trendSlope(), timeline: timeline, playhead: playhead)
        return line
    }

    /// Recent slope of the risk in logit units per second (least squares over the history), clamped.
    public func trendSlope() -> Float {
        guard history.count >= 3 else { return 0 }
        let ts = history.map { $0.t }, ys = history.map { Self.logit($0.probability) }
        let mt = ts.reduce(0, +) / Double(ts.count), my = ys.reduce(0, +) / Float(ys.count)
        var num: Double = 0, den: Double = 0
        for (t, y) in zip(ts, ys) { num += (t - mt) * Double(y - my); den += (t - mt) * (t - mt) }
        guard den > 1e-9 else { return 0 }
        return max(-maxSlope, min(maxSlope, Float(num / den)))
    }

    /// Pure extrapolation (exposed for tests).
    public func extrapolate(from risk: LapseRisk, slope: Float, timeline: ContentTimeline?,
                            playhead: Double?) -> [(t: Double, p: Float)] {
        let l0 = Self.logit(risk.probability), lb = Self.logit(baseRate)
        let n = max(1, Int((horizon / step).rounded()))
        var c0: ContentFeatures?
        if let tl = timeline, let ph = playhead { c0 = tl.features(at: ph) }
        var out: [(t: Double, p: Float)] = [(risk.t, risk.probability)]
        for i in 1...n {
            let k = Double(i) * step
            let decay = Float(exp(-k / reversionTau))
            var z = lb + (l0 - lb) * decay + slope * Float(k) * decay
            if let tl = timeline, let ph = playhead, let c0 {
                let ck = tl.features(at: min(tl.duration, ph + k))
                z += driftGain * (ck.driftRisk - c0.driftRisk)
                z += valleyGain * (Self.valleyProximity(ck) - Self.valleyProximity(c0))
            }
            out.append((risk.t + k, Self.sigmoid(z)))
        }
        return out
    }

    public static func valleyProximity(_ c: ContentFeatures) -> Float {
        c.inValley ? 1 : max(0, 1 - min(60, max(0, c.secondsToValley)) / 15)
    }

    static func logit(_ p: Float) -> Float {
        let q = min(0.999, max(0.001, p))
        return log(q / (1 - q))
    }

    static func sigmoid(_ z: Float) -> Float { 1 / (1 + exp(-z)) }
}
