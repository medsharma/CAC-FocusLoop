import Foundation

// FROZEN CONTRACT (Phase 0).

/// Global feature flags. Persisted in UserDefaults by the App layer.
public struct FeatureFlags: Codable, Equatable, Sendable {
    /// DEMONSTRATION ONLY: lowers thresholds and speeds up the bandit display so states are easy to
    /// trigger on camera. Never enabled by default; never used for any reported result.
    public var demoDirector = false
    /// Force the synthetic Replay stream even on a device with ARKit.
    public var forceReplay = false
    /// Stretch goal: Apple Watch wrist tap via WatchConnectivity.
    public var watchTap = false
    /// Show experience-sampling prompts during sessions.
    public var experienceSampling = true
    public init() {}
}

/// Tunable thresholds. `demo` variant is for Demo Director mode ONLY.
public struct PacingConfig: Codable, Equatable, Sendable {
    /// Seconds of sustained non-focused evidence before any reactive nudge.
    public var reactiveDebounce: Double = 3
    /// Minimum seconds between nudges.
    public var cooldown: Double = 45
    /// Window in which returning to focused counts as success.
    public var rewardWindow: Double = 20
    /// Penalty subtracted from reward when the user dismisses a nudge.
    public var dismissalPenalty: Double = 0.5
    /// Probability floor for the no-nudge control arm.
    public var controlProbability: Double = 0.1
    /// Minimum seconds between experience-sampling prompts.
    public var experienceSamplingInterval: Double = 8 * 60
    public init() {}

    /// DEMONSTRATION ONLY (Demo Director).
    public static var demo: PacingConfig {
        var c = PacingConfig()
        c.reactiveDebounce = 1.5
        c.cooldown = 12
        // Experience sampling keeps the spec's hard limit (at most one prompt every 8 minutes) even here.
        return c
    }
}
