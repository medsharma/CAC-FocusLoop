import Foundation

/// Per-user alert threshold for the lapse forecaster.
///
/// Primary rule: stochastic quantile tracking so that about `targetAlertRate` of predictions alert
/// (`value += step * (alerted - targetAlertRate)`), which converges to the (1 - rate) quantile of
/// this user's own risk distribution. Secondary rule (weak): outcome feedback from
/// `observe(probability:lapseOccurred:)`: false alarms nudge the threshold up, missed lapses nudge it down.
/// Persist with `serialized()` / `restore(from:)` (Codable JSON).
public struct PersonalThreshold: Codable, Equatable, Sendable {
    /// Fraction of 1 Hz predictions that should raise an alert (default 10%).
    public var targetAlertRate: Float
    /// Current threshold on P(lapse within 15 s).
    public private(set) var value: Float
    /// Minimum adaptation step (the step anneals from larger values down to this).
    public var learningRate: Float
    /// Strength of the outcome (hit / false alarm / miss) feedback relative to the rate rule.
    public var feedbackWeight: Float
    public var lowerBound: Float
    public var upperBound: Float

    public private(set) var observations: Int = 0
    public private(set) var alerts: Int = 0
    public private(set) var hits: Int = 0
    public private(set) var falseAlarms: Int = 0
    public private(set) var misses: Int = 0

    public init(initial: Float = 0.5, targetAlertRate: Float = 0.10, learningRate: Float = 0.005,
                feedbackWeight: Float = 0.2, lowerBound: Float = 0.05, upperBound: Float = 0.95) {
        self.targetAlertRate = targetAlertRate
        self.learningRate = learningRate
        self.feedbackWeight = feedbackWeight
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.value = min(upperBound, max(lowerBound, initial))
    }

    /// Update from one prediction and whether a lapse actually began within the horizon after it.
    public mutating func observe(probability p: Float, lapseOccurred: Bool) {
        let alerted = p >= value
        observations += 1
        if alerted { alerts += 1 }
        switch (alerted, lapseOccurred) {
        case (true, true): hits += 1
        case (true, false): falseAlarms += 1
        case (false, true): misses += 1
        case (false, false): break
        }
        // Anneal: big steps for the first few dozen observations, then `learningRate`.
        let step = max(learningRate, 1 / Float(observations + 20))
        var delta = (alerted ? 1 : 0) - targetAlertRate
        if alerted && !lapseOccurred { delta += feedbackWeight * targetAlertRate }
        if !alerted && lapseOccurred { delta -= feedbackWeight * targetAlertRate }
        value = min(upperBound, max(lowerBound, value + step * delta))
    }

    /// Observed alert rate so far.
    public var alertRate: Float { observations == 0 ? 0 : Float(alerts) / Float(observations) }
    /// Observed precision of alerts so far (nil before the first alert).
    public var precision: Float? { alerts == 0 ? nil : Float(hits) / Float(alerts) }

    // MARK: Demo Director

    /// DEMONSTRATION ONLY (Demo Director, analogous to `PacingConfig.demo`): multiplies the threshold
    /// so pre-emptive nudges are easy to trigger on camera. Never used for any reported result.
    public static let demoScale: Float = 0.6

    /// The threshold in force. `demoDirector` is DEMONSTRATION ONLY.
    public func effective(demoDirector: Bool) -> Float {
        demoDirector ? max(lowerBound, value * Self.demoScale) : value
    }

    // MARK: Persistence

    public func serialized() throws -> Data { try JSONEncoder().encode(self) }
    public static func restore(from data: Data) throws -> PersonalThreshold {
        try JSONDecoder().decode(PersonalThreshold.self, from: data)
    }
}
