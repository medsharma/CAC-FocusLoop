import Foundation

/// Rule-based lapse forecaster that needs no model file. Same interface as the Core ML forecaster.
///
/// It scores short-term TRENDS over the 20 s window (rising blink rate, gaze dispersion, time off the
/// work zone, PERCLOS slope, heart rate above baseline) plus, when a lecture is playing, the content's
/// predicted drift risk and valley proximity (predicted brain response to content, average brain).
/// Every weight is non-negative, so the probability is monotonically non-decreasing in each signal.
/// A Python mirror (`ml/forecast/heuristic.py`) evaluates the same formula on the SYNTHETIC corpus.
public final class HeuristicLapseForecaster: LapseForecaster {

    /// Engineered signals the heuristic scores. All are "higher = more lapse-prone".
    public struct Signals: Equatable, Sendable {
        /// Blink rate, last 5 s minus first 5 s of the window (blinks/min).
        public var blinkRateTrend: Float = 0
        /// sqrt(var(gazeYaw) + var(gazePitch)) over the last 10 s (rad).
        public var gazeDispersion: Float = 0
        /// Latest seconds continuously outside the work zone.
        public var offZoneDwell: Float = 0
        /// PERCLOS, last 5 s minus first 5 s of the window.
        public var perclosSlope: Float = 0
        /// Mean heart rate above the personal baseline over the last 5 s, good-quality readings only (bpm).
        public var heartRateDelta: Float = 0
        /// Latest content drift risk (Default vs Dorsal Attention), 0 without content.
        public var driftRisk: Float = 0
        /// 1 inside a predicted valley, ramps 0 -> 1 over the 15 s before one starts.
        public var valleyProximity: Float = 0
        public init() {}
    }

    /// Non-negative weights on each normalised signal. Defaults were sanity-checked against the
    /// SYNTHETIC corpus only (see ml/reports/forecast_ablation_SYNTHETIC.md); refit on real data.
    public struct Weights: Codable, Equatable, Sendable {
        public var bias: Float = -2.2
        public var blinkRateTrend: Float = 0.35      // per 5 blinks/min
        public var gazeDispersion: Float = 0.9       // per 0.1 rad
        public var offZoneDwell: Float = 0.25        // per s (capped at 10 s)
        public var perclosSlope: Float = 0.5         // per 0.05
        public var heartRateDelta: Float = 0.3       // per 5 bpm
        public var driftRisk: Float = 0.45           // per unit drift (z difference)
        public var valleyProximity: Float = 1.0
        public init() {}
    }

    public var weights: Weights
    public var personalThreshold: PersonalThreshold
    /// DEMONSTRATION ONLY: Demo Director lowers the effective threshold.
    public var demoDirector = false

    public init(weights: Weights = Weights(), threshold: PersonalThreshold = PersonalThreshold(initial: 0.5)) {
        self.weights = weights
        self.personalThreshold = threshold
    }

    public var threshold: Float { personalThreshold.effective(demoDirector: demoDirector) }

    public func forecast(frames: [FeatureFrame], content: [ContentFeatures], t: Double) -> LapseRisk {
        let s = Self.signals(frames: frames, content: content)
        let p = probability(s, usesContent: !content.isEmpty)
        return LapseRisk(t: t, probability: p, threshold: threshold, usedContent: !content.isEmpty, source: "heuristic")
    }

    public func observe(prediction: LapseRisk, lapseOccurred: Bool) {
        personalThreshold.observe(probability: prediction.probability, lapseOccurred: lapseOccurred)
    }

    /// Logistic score of the signals. Monotone non-decreasing in every signal (weights >= 0).
    public func probability(_ s: Signals, usesContent: Bool) -> Float {
        let w = weights
        var z = w.bias
        z += max(0, w.blinkRateTrend) * max(0, s.blinkRateTrend) / 5
        z += max(0, w.gazeDispersion) * s.gazeDispersion / 0.1
        z += max(0, w.offZoneDwell) * min(10, max(0, s.offZoneDwell))
        z += max(0, w.perclosSlope) * max(0, s.perclosSlope) / 0.05
        z += max(0, w.heartRateDelta) * max(0, s.heartRateDelta) / 5
        if usesContent {
            z += max(0, w.driftRisk) * s.driftRisk
            z += max(0, w.valleyProximity) * min(1, max(0, s.valleyProximity))
        }
        return 1 / (1 + exp(-z))
    }

    /// Computes the engineered signals from a window (any length; 10 Hz frames, 1 Hz content).
    public static func signals(frames: [FeatureFrame], content: [ContentFeatures]) -> Signals {
        var s = Signals()
        let hz = Int(FeatureFrame.rateHz)
        if !frames.isEmpty {
            let five = min(frames.count, 5 * hz), ten = min(frames.count, 10 * hz)
            let head = frames.prefix(five), tail = frames.suffix(five), last10 = frames.suffix(ten)
            s.blinkRateTrend = mean(tail.map(\.blinkRate)) - mean(head.map(\.blinkRate))
            s.perclosSlope = mean(tail.map(\.perclos)) - mean(head.map(\.perclos))
            let tracked = last10.filter { $0.faceDetected > 0.5 }
            if tracked.count > 1 {
                s.gazeDispersion = (variance(tracked.map(\.gazeYaw)) + variance(tracked.map(\.gazePitch))).squareRoot()
            }
            s.offZoneDwell = frames.last?.offZoneDwell ?? 0
            let hr = tail.filter { $0.heartRateQuality > 0.3 && $0.heartRate > 0 }.map(\.heartRateDelta)
            s.heartRateDelta = hr.isEmpty ? 0 : mean(hr)
        }
        if let c = content.last {
            s.driftRisk = c.driftRisk
            s.valleyProximity = c.inValley ? 1 : max(0, 1 - min(60, max(0, c.secondsToValley)) / 15)
        }
        return s
    }

    private static func mean<C: Collection>(_ x: C) -> Float where C.Element == Float {
        x.isEmpty ? 0 : x.reduce(0, +) / Float(x.count)
    }

    private static func variance(_ x: [Float]) -> Float {
        guard x.count > 1 else { return 0 }
        let m = mean(x)
        return x.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Float(x.count)
    }
}
