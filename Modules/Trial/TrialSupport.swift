import Foundation

// Shared building blocks for the trial engines: configuration, priors file, eligibility rules,
// pacing (cooldown + reactive debounce) and the control-floor randomisation mix.

/// Engine-level knobs that are not part of the frozen `PacingConfig`.
public struct TrialEngineConfig: Codable, Equatable, Sendable {
    /// Posterior draws used to compute the Thompson-sampling selection probabilities.
    public var monteCarloDraws: Int = 2_000
    /// Minimum selection probability of every eligible arm (MRT-style probability clipping),
    /// so inverse-probability weights stay bounded. Applied on top of the control floor.
    public var minArmProbability: Double = 0.02
    /// Sustained `focused` seconds that count as "back to focused" (reward tracker).
    public var refocusSustain: Double = 2
    /// Forecast horizon for pre-emptive rewards ("lapse avoided within the horizon").
    public var forecastHorizon: Double = FeatureFrame.forecastHorizonSeconds
    /// Bootstrap replicates for the IPW effect intervals.
    public var bootstrapReplicates: Int = 500
    /// Interval level for effect estimates.
    public var intervalLevel: Double = 0.9
    public init() {}
}

/// `nudge_priors.json` as written by `ml/brain/score_nudges.py`:
/// `{"mock": true, "priors": [{"arm": "...", "alpha": 2, "beta": 3}, ...]}`.
/// When `mock` is true the priors are MOCK and must be labelled as such wherever shown.
public struct NudgePriorsFile: Codable, Equatable, Sendable {
    public var mock: Bool
    public var priors: [NudgePrior]

    public init(mock: Bool, priors: [NudgePrior]) { self.mock = mock; self.priors = priors }

    /// Loads `nudge_priors.json` from the bundle root, or nil when missing/undecodable
    /// (the engine then falls back to Beta(1, 1) for every arm).
    public static func load(bundle: Bundle = .main) -> NudgePriorsFile? {
        guard let url = bundle.url(forResource: "nudge_priors", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? decode(data)
    }

    /// Decodes the file, skipping entries whose arm name is unknown to this build.
    public static func decode(_ data: Data) throws -> NudgePriorsFile {
        struct Loose: Decodable {
            struct Entry: Decodable { var arm: String; var alpha: Double; var beta: Double }
            var mock: Bool?
            var priors: [Entry]
        }
        let loose = try JSONDecoder().decode(Loose.self, from: data)
        let priors = loose.priors.compactMap { e -> NudgePrior? in
            guard let arm = NudgeArm(rawValue: e.arm), e.alpha > 0, e.beta > 0 else { return nil }
            return NudgePrior(arm: arm, alpha: e.alpha, beta: e.beta)
        }
        return NudgePriorsFile(mock: loose.mock ?? true, priors: priors)
    }
}

/// Which arms may be randomised among for a decision.
public enum TrialEligibility {
    /// Eligible arms (always including `noNudge`), in `NudgeArm.allCases` order.
    /// Rules: `putMeDown` only when the phone is picked up; `recapCard` and `recallQuestion` only in
    /// Study Mode; `recallQuestion` only for pre-emptive decisions with an upcoming predicted valley.
    public static func arms(policy: NudgePolicyKind, context: NudgeContext) -> [NudgeArm] {
        let base: [NudgeArm]
        switch policy {
        case .reactive: base = NudgeArm.reactiveArms(for: context.state)
        case .preemptive: base = NudgeArm.preemptiveArms
        }
        let filtered = base.filter { arm in
            switch arm {
            case .putMeDown: return context.phonePickedUp
            case .recapCard: return context.studyMode
            case .recallQuestion: return policy == .preemptive && context.studyMode && context.upcomingValleyInSeconds != nil
            default: return true
            }
        }
        // A focused user gets no reactive nudge at all (not even a logged control decision).
        if policy == .reactive && filtered.allSatisfy({ $0 == .noNudge }) { return [] }
        var out = filtered
        if !out.contains(.noNudge) { out.append(.noNudge) }
        return NudgeArm.allCases.filter { out.contains($0) }
    }
}

/// Pacing: minimum gap between decisions and sustained evidence before a reactive nudge.
/// Every decision (including a `noNudge` control decision) starts the cooldown, so treatment and
/// control are subject to identical pacing and their reward windows do not overlap.
public struct TrialPacer: Codable, Equatable, Sendable {
    public var config: PacingConfig
    /// Start time of the current run of non-focused evidence (nil while focused).
    public private(set) var nonFocusedSince: Double?
    public private(set) var lastStateT: Double?
    public private(set) var lastDecisionT: Double?

    public init(config: PacingConfig) { self.config = config }

    /// Feed the latest classified state (call at the classifier rate, ~1 Hz).
    public mutating func noteState(_ state: AttentionState, t: Double) {
        if let last = lastStateT, t < last { resetSession() }   // time went backwards: new session
        lastStateT = t
        if state == .focused {
            nonFocusedSince = nil
        } else if nonFocusedSince == nil {
            nonFocusedSince = t
        }
    }

    /// Seconds of sustained non-focused evidence at `t` (0 if focused).
    public func sustainedNonFocused(at t: Double) -> Double {
        guard let s = nonFocusedSince else { return 0 }
        return max(0, t - s)
    }

    public func inCooldown(at t: Double) -> Bool {
        guard let last = lastDecisionT else { return false }
        return t - last < config.cooldown
    }

    /// Whether a decision of this kind is allowed at `t`.
    public func allows(policy: NudgePolicyKind, t: Double) -> Bool {
        if inCooldown(at: t) { return false }
        if policy == .reactive {
            guard nonFocusedSince != nil else { return false }
            return sustainedNonFocused(at: t) >= config.reactiveDebounce
        }
        return true
    }

    public mutating func noteDecision(at t: Double) { lastDecisionT = t }

    public mutating func resetSession() {
        nonFocusedSince = nil
        lastStateT = nil
        lastDecisionT = nil
    }
}

/// Mixes Thompson-sampling probabilities with the per-arm floor and the control floor:
/// `pi(a) = (1 - c - K*eps) * piTS(a) + eps + c * 1[a == noNudge]`, so `pi(noNudge) >= c`
/// and every eligible arm has at least `eps`. (If `c + K*eps > 1`, `eps` is shrunk to fit.)
enum TrialRandomization {
    static func mix(thompson: [Double], arms: [NudgeArm], controlProbability: Double,
                    minArmProbability: Double) -> [Double] {
        let k = Double(arms.count)
        let c = min(max(controlProbability, 0), 1)
        let eps = max(0, min(minArmProbability, (1 - c) / max(k, 1)))
        let free = max(0, 1 - c - k * eps)
        var p = zip(arms, thompson).map { arm, ts in free * ts + eps + (arm == .noNudge ? c : 0) }
        let s = p.reduce(0, +)
        if s > 0 { p = p.map { $0 / s } }
        return p
    }
}

/// Pacing hooks both trial engines expose beyond `InterventionPolicy` (the session engine should
/// call `noteState` with every classified state, ~1 Hz, and `resetSession` when a session starts).
public protocol TrialPacingControl: AnyObject {
    func noteState(_ state: AttentionState, t: Double)
    func setPacing(_ config: PacingConfig)
    func resetSession()
}

extension ThompsonTrialEngine: TrialPacingControl {}
extension LinearThompsonEngine: TrialPacingControl {}
