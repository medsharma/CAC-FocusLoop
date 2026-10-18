import Foundation

/// DEMONSTRATION ONLY (Demo Director). Wraps the real trial policy and, while `enabled`, replaces Thompson
/// sampling with a fixed demo script so every beat of a demo shows the nudge the script promises:
///
/// * pre-emptive: `recallQuestion` in Study Mode when a recall question is unlocked and not yet asked this
///   session (and the trial's own eligibility rule allows it), else `preemptiveGlow`. The first pre-emptive
///   nudge waits for a full forecast window (20 s of frames), so a demo opens on a calm, focused start.
/// * distracted: `putMeDown` when the phone is picked up, else `ambientGlow`
/// * drowsy: `stretchBreak`
/// * stressed: `pacedBreathing`
///
/// Decisions are logged honestly as deterministic: `randomizationProbability = 1`, `armProbabilities = [arm: 1]`.
/// They are NOT randomised, so they carry no causal information, and they are never handed to the base
/// policy: `decide` does not touch the bandit (no trial record, no pacer change) and `record(outcome:)` is
/// dropped. Demos therefore never contaminate the real or the SYNTHETIC Personal Trial. (The session pipeline
/// separately keeps demo sessions away from the personal head, the label store and the forecaster threshold.)
///
/// Pacing still follows the trial engine's rules (cooldown, reactive debounce, see `TrialPacer`) with
/// `PacingConfig.demo`, but on a pacer owned by this wrapper, so demo pacing never leaks into the trial
/// engine's persisted pacing. When `enabled` is false every call is forwarded to the base policy unchanged.
///
/// Thread-safe: the pipeline actor decides and records while the main actor flips `enabled` and reads
/// posteriors.
final class DemoScriptPolicy: InterventionPolicy, TrialPacingControl, @unchecked Sendable {
    let base: InterventionPolicy
    let basePacing: TrialPacingControl?

    private let lock = NSLock()
    private var _enabled = false
    private var pacer = TrialPacer(config: .demo)
    private var recallUnlockTimes: [Double] = []
    private var askedRecall: Set<Double> = []
    private var playhead: Double = 0

    /// Seconds before the first pre-emptive nudge: the forecaster's full input window.
    static let preemptiveWarmup = FeatureFrame.forecastWindowSeconds

    init(base: InterventionPolicy, pacing: TrialPacingControl?) {
        self.base = base
        self.basePacing = pacing
    }

    /// DEMONSTRATION ONLY. True while a Demo Director session runs.
    var enabled: Bool { lock.withLock { _enabled } }

    /// Called by the session engine at the start of every session.
    func beginSession(enabled: Bool, lecture: LectureInfo?) {
        lock.withLock {
            _enabled = enabled
            recallUnlockTimes = lecture?.recallQuestions.map(\.after).sorted() ?? []
            askedRecall = []
            playhead = 0
            pacer = TrialPacer(config: .demo)
        }
    }

    /// Lecture playhead (Study Mode), so the script knows whether a recall question is unlocked.
    func notePlayhead(_ t: Double) { lock.withLock { playhead = t } }

    /// The fixed demo arm for a decision, or nil when the script has no nudge for that state.
    /// `recallUnlocked`: an unlocked recall question that hasn't been asked yet this session.
    static func scriptedArm(policy: NudgePolicyKind, context: NudgeContext, recallUnlocked: Bool) -> NudgeArm? {
        switch policy {
        case .preemptive:
            let recallEligible = TrialEligibility.arms(policy: .preemptive, context: context).contains(.recallQuestion)
            return recallEligible && recallUnlocked ? .recallQuestion : .preemptiveGlow
        case .reactive:
            switch context.state {
            case .focused: return nil
            case .distracted: return context.phonePickedUp ? .putMeDown : .ambientGlow
            case .drowsy: return .stretchBreak
            case .stressed: return .pacedBreathing
            }
        }
    }

    // MARK: InterventionPolicy

    func decide(policy: NudgePolicyKind, context: NudgeContext) -> NudgeDecision? {
        guard enabled else { return base.decide(policy: policy, context: context) }
        return lock.withLock { () -> NudgeDecision? in
            if policy == .reactive { pacer.noteState(context.state, t: context.t) }
            if policy == .preemptive && context.t < Self.preemptiveWarmup { return nil }
            guard pacer.allows(policy: policy, t: context.t) else { return nil }
            // `RecallQuestionNudge` asks the latest unlocked question; ask each one at most once.
            let latest = recallUnlockTimes.filter { $0 <= playhead }.max()
            let available = latest.map { !askedRecall.contains($0) } ?? false
            guard let arm = Self.scriptedArm(policy: policy, context: context, recallUnlocked: available) else { return nil }
            if arm == .recallQuestion, let latest { askedRecall.insert(latest) }
            pacer.noteDecision(at: context.t)
            return NudgeDecision(t: context.t, policy: policy, context: context, arm: arm,
                                 randomizationProbability: 1, armProbabilities: [arm.rawValue: 1])
        }
    }

    /// Demo outcomes are dropped: a fixed (non-randomised) demo nudge must not teach any bandit.
    func record(outcome: NudgeOutcome, for decision: NudgeDecision) {
        if Self.isScripted(decision) { return }
        base.record(outcome: outcome, for: decision)
    }

    /// Unchanged posteriors of the base policy: a demo session does not pretend to learn.
    func posteriors() -> [ArmPosterior] { base.posteriors() }
    func effectEstimates() -> [ArmEffect] { base.effectEstimates() }
    func serialized() throws -> Data { try base.serialized() }
    func restore(from data: Data) throws { try base.restore(from: data) }

    /// A decision made by the demo script (deterministic, a real nudge). Thompson decisions always keep a
    /// no-nudge control floor, and sensing-only decisions are always `noNudge`, so neither matches.
    static func isScripted(_ d: NudgeDecision) -> Bool {
        d.randomizationProbability >= 1 && d.arm != .noNudge && d.armProbabilities.count == 1
    }

    // MARK: TrialPacingControl

    func noteState(_ state: AttentionState, t: Double) {
        if enabled { lock.withLock { pacer.noteState(state, t: t) } } else { basePacing?.noteState(state, t: t) }
    }

    func setPacing(_ config: PacingConfig) {
        // DEMONSTRATION ONLY pacing stays on this wrapper's pacer; the trial engine's pacing is untouched.
        if enabled { lock.withLock { pacer.config = config } } else { basePacing?.setPacing(config) }
    }

    func resetSession() {
        if enabled { lock.withLock { pacer.resetSession() } } else { basePacing?.resetSession() }
    }
}
