import Foundation

/// Decides when to show the "Were you focused just now?" prompt and turns answers into labels.
///
/// Rules: at most one prompt per `PacingConfig.experienceSamplingInterval` (8 min by default); the next
/// prompt becomes eligible at `lastPrompt + interval + U(0, jitterFraction x interval)` (randomised so the
/// user can't anticipate it); never while a nudge is active; an unanswered prompt expires after
/// `promptTimeout` seconds. Not thread-safe: call from the session engine's actor.
public final class ExperienceSampler {
    public var pacing: PacingConfig
    public var isEnabled: Bool
    public var jitterFraction: Double
    public var promptTimeout: Double
    public var windowSeconds: Double = FeatureFrame.stateWindowSeconds

    public private(set) var lastPromptAt: Double?
    public private(set) var nextEligibleAt: Double
    /// Session time of the prompt currently on screen, if any.
    public private(set) var pendingPromptAt: Double?

    private var rng: CalibrationSplitMix64

    public init(pacing: PacingConfig = PacingConfig(), isEnabled: Bool = true, sessionStart: Double = 0,
                jitterFraction: Double = 0.5, promptTimeout: Double = 60, seed: UInt64? = nil) {
        self.pacing = pacing; self.isEnabled = isEnabled
        self.jitterFraction = max(0, jitterFraction); self.promptTimeout = promptTimeout
        rng = seed.map { CalibrationSplitMix64(seed: $0) } ?? CalibrationSplitMix64()
        nextEligibleAt = 0
        nextEligibleAt = scheduleAfter(sessionStart)
    }

    private func scheduleAfter(_ t: Double) -> Double {
        let interval = pacing.experienceSamplingInterval
        let jitter = jitterFraction > 0 ? Double.random(in: 0...(jitterFraction * interval), using: &rng) : 0
        return t + interval + jitter
    }

    /// Call periodically (e.g. 1 Hz). Returns true exactly when a prompt should be shown now;
    /// the sampler then records it as pending.
    public func shouldPrompt(at t: Double, nudgeActive: Bool) -> Bool {
        if let p = pendingPromptAt {
            guard t - p >= promptTimeout else { return false }
            pendingPromptAt = nil   // expired, unanswered
        }
        guard isEnabled, !nudgeActive, t >= nextEligibleAt else { return false }
        lastPromptAt = t
        pendingPromptAt = t
        nextEligibleAt = scheduleAfter(t)
        return true
    }

    /// Record the user's answer to the pending prompt. Returns a window covering the `windowSeconds`
    /// before the prompt: `.focused` (isLapse false) or `reason.state` (isLapse true; `.distracted` if no reason).
    public func answer(focused: Bool, reason: NotFocusedReason?, frames: [FeatureFrame], sessionID: UUID) -> LabeledWindow? {
        guard let p = pendingPromptAt else { return nil }
        pendingPromptAt = nil
        return Self.window(focused: focused, reason: reason, promptTime: p, frames: frames, sessionID: sessionID,
                           windowSeconds: windowSeconds)
    }

    /// User closed the prompt without answering.
    public func dismiss() { pendingPromptAt = nil }

    public static func window(focused: Bool, reason: NotFocusedReason?, promptTime: Double, frames: [FeatureFrame],
                              sessionID: UUID, windowSeconds: Double = FeatureFrame.stateWindowSeconds) -> LabeledWindow {
        let start = promptTime - windowSeconds
        let label: AttentionState = focused ? .focused : (reason?.state ?? .distracted)
        return LabeledWindow(sessionID: sessionID, start: start, end: promptTime, label: label, isLapse: !focused,
                             source: .experienceSampling, frames: calibrationFrames(frames, from: start, to: promptTime))
    }
}
