import Foundation

/// Turns the state stream after each decision into a `NudgeOutcome`. Handles `noNudge` (control)
/// decisions identically to real nudges, which is what makes the IPW contrast valid.
///
/// Reactive reward: 1 if the user is back to `focused`, sustained for >= `refocusSustain` s (2 s),
/// with that focused run starting within `rewardWindow` (20 s) of the decision; `refocusSeconds` is
/// the time from the decision to the start of that run. Otherwise 0, emitted once the window (plus the
/// sustain time) has passed.
///
/// Pre-emptive reward: 1 if no lapse occurred within the forecast horizon (15 s). A lapse is either
/// reported explicitly (`noteLapse`) or seen as a non-focused state sustained >= `refocusSustain` s.
/// The outcome is emitted at the horizon, or early (reward 0) as soon as a lapse is seen.
///
/// Dismissal: if the user dismissed the nudge, `reward = max(0, reward - dismissalPenalty)`.
///
/// Not thread-safe: drive it from the same actor as the engine.
public final class RewardTracker {
    public var pacing: PacingConfig
    public var refocusSustain: Double
    public var forecastHorizon: Double

    private struct Pending {
        var decision: NudgeDecision
        var dismissed = false
        /// Start of the current focused run (reactive) / non-focused run (pre-emptive) after the decision.
        var runStart: Double?
        var lapseAt: Double?
    }

    private var pending: [Pending] = []

    public init(pacing: PacingConfig = PacingConfig(), refocusSustain: Double = 2,
                forecastHorizon: Double = FeatureFrame.forecastHorizonSeconds) {
        self.pacing = pacing
        self.refocusSustain = refocusSustain
        self.forecastHorizon = forecastHorizon
    }

    public var pendingDecisions: [NudgeDecision] { pending.map(\.decision) }

    /// Start tracking a decision (any arm, including `noNudge`).
    public func track(_ decision: NudgeDecision) {
        pending.append(Pending(decision: decision))
    }

    /// User dismissed the nudge from this decision (defaults to the most recent one).
    public func noteDismissal(decisionID: UUID? = nil) {
        guard let i = decisionID.flatMap({ id in pending.firstIndex { $0.decision.id == id } })
                ?? (pending.isEmpty ? nil : pending.count - 1) else { return }
        guard pending[i].decision.arm != .noNudge else { return }
        pending[i].dismissed = true
    }

    /// An externally detected lapse (e.g. the session's lapse detector) at time `t`.
    public func noteLapse(t: Double) {
        for i in pending.indices where pending[i].decision.policy == .preemptive {
            let d = pending[i].decision
            if t >= d.t, t <= d.t + forecastHorizon, pending[i].lapseAt == nil { pending[i].lapseAt = t }
        }
    }

    /// Feed the latest classified state. Returns any outcomes that became final at `t`.
    @discardableResult
    public func observe(state: AttentionState, t: Double) -> [NudgeOutcome] {
        var done: [NudgeOutcome] = []
        var keep: [Pending] = []
        for var p in pending {
            if let o = step(&p, state: state, t: t) { done.append(o) } else { keep.append(p) }
        }
        pending = keep
        return done
    }

    /// Finalises everything still pending at `t` (e.g. at session end) with what is known so far.
    public func flush(t: Double) -> [NudgeOutcome] {
        let out = pending.map { p -> NudgeOutcome in
            switch p.decision.policy {
            case .reactive:
                if let s = p.runStart, t - s >= refocusSustain, s - p.decision.t <= pacing.rewardWindow {
                    return outcome(p, t: t, success: true, refocusSeconds: s - p.decision.t, lapseAvoided: nil)
                }
                return outcome(p, t: t, success: false, refocusSeconds: nil, lapseAvoided: nil)
            case .preemptive:
                let avoided = p.lapseAt == nil
                return outcome(p, t: t, success: avoided, refocusSeconds: nil, lapseAvoided: avoided)
            }
        }
        pending.removeAll()
        return out
    }

    public func reset() { pending.removeAll() }

    // MARK: Internals

    private func step(_ p: inout Pending, state: AttentionState, t: Double) -> NudgeOutcome? {
        let d = p.decision
        guard t >= d.t else { return nil }
        switch d.policy {
        case .reactive:
            if state == .focused {
                if p.runStart == nil { p.runStart = t }
            } else {
                p.runStart = nil
            }
            if let s = p.runStart, s - d.t <= pacing.rewardWindow, t - s >= refocusSustain {
                return outcome(p, t: t, success: true, refocusSeconds: s - d.t, lapseAvoided: nil)
            }
            if t - d.t > pacing.rewardWindow, p.runStart == nil || (p.runStart! - d.t) > pacing.rewardWindow {
                return outcome(p, t: t, success: false, refocusSeconds: nil, lapseAvoided: nil)
            }
            return nil
        case .preemptive:
            if t <= d.t + forecastHorizon {
                if state == .focused {
                    p.runStart = nil
                } else if p.runStart == nil {
                    p.runStart = t
                }
                if let s = p.runStart, t - s >= refocusSustain, p.lapseAt == nil { p.lapseAt = s }
            }
            if p.lapseAt != nil {
                return outcome(p, t: t, success: false, refocusSeconds: nil, lapseAvoided: false)
            }
            if t >= d.t + forecastHorizon {
                return outcome(p, t: t, success: true, refocusSeconds: nil, lapseAvoided: true)
            }
            return nil
        }
    }

    private func outcome(_ p: Pending, t: Double, success: Bool, refocusSeconds: Double?,
                         lapseAvoided: Bool?) -> NudgeOutcome {
        var reward = success ? 1.0 : 0.0
        if p.dismissed { reward = max(0, reward - pacing.dismissalPenalty) }
        return NudgeOutcome(decisionID: p.decision.id, t: t, reward: reward, refocused: success && p.decision.policy == .reactive,
                            refocusSeconds: refocusSeconds, dismissed: p.dismissed, lapseAvoided: lapseAvoided)
    }
}
