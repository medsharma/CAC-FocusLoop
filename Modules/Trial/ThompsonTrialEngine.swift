import Foundation

/// One logged decision and (once known) its outcome.
public struct TrialRecord: Codable, Equatable, Sendable {
    public var decision: NudgeDecision
    public var outcome: NudgeOutcome?
    public init(decision: NudgeDecision, outcome: NudgeOutcome? = nil) { self.decision = decision; self.outcome = outcome }
}

/// The n-of-1 micro-randomised trial policy (default engine).
///
/// Thompson sampling with Beta-Bernoulli posteriors per (state, arm) for reactive decisions and per arm
/// for pre-emptive ones, initialised from `nudge_priors.json` (default Beta(1, 1)). Each decision:
///   1. pacing (cooldown; reactive also needs >= `reactiveDebounce` s of sustained non-focused evidence),
///   2. eligibility (`TrialEligibility`),
///   3. Thompson probabilities `piTS(a) = P(a has the highest posterior draw)`, estimated by Monte Carlo
///      over `monteCarloDraws` joint posterior draws,
///   4. mixed with the control floor and a small per-arm floor (`TrialRandomization.mix`),
///   5. the arm is drawn from exactly that mixed distribution, so the logged `randomizationProbability`
///      is the true probability of the realised choice (not an approximation of a different mechanism).
///
/// The logged probabilities feed `IPWEffectEstimator` for causal effects vs the no-nudge control,
/// following the micro-randomised trial design of Klasnja et al. (2015) and the just-in-time adaptive
/// intervention framework of Nahum-Shani et al. (2018).
///
/// Thread-safe: every public entry point takes an internal lock, because the session pipeline (an
/// actor) decides and records while the main actor reads posteriors/effects and serialises the state.
public final class ThompsonTrialEngine: InterventionPolicy, @unchecked Sendable {
    public var pacing: PacingConfig { locked { _pacing } }
    private var _pacing: PacingConfig
    public let config: TrialEngineConfig
    /// True when the priors came from a MOCK `nudge_priors.json`.
    public let priorsAreMock: Bool
    public weak var logger: TrialLogger?

    private var rng: TrialSeededRNG
    private var priorByArm: [NudgeArm: NudgePrior]
    private var table: [String: ArmPosterior] = [:]
    private var pacer: TrialPacer
    private var _records: [TrialRecord] = []
    private let lock = NSRecursiveLock()
    public var records: [TrialRecord] { locked { _records } }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock(); defer { lock.unlock() }
        return try body()
    }

    /// - Parameters:
    ///   - priors: per-arm Beta priors (arms not listed get Beta(1, 1)); the same prior seeds every state.
    ///   - seed: RNG seed; nil = non-reproducible system seed.
    public init(priors: [NudgePrior] = [], priorsAreMock: Bool = false, pacing: PacingConfig = PacingConfig(),
                config: TrialEngineConfig = TrialEngineConfig(), seed: UInt64? = nil, logger: TrialLogger? = nil) {
        self._pacing = pacing
        self.config = config
        self.priorsAreMock = priorsAreMock
        self.logger = logger
        self.rng = seed.map(TrialSeededRNG.init(seed:)) ?? .system()
        self.priorByArm = Dictionary(priors.map { ($0.arm, $0) }, uniquingKeysWith: { _, b in b })
        self.pacer = TrialPacer(config: pacing)
        buildTable()
    }

    /// Convenience: priors from the bundled `nudge_priors.json` (Beta(1, 1) if absent).
    public convenience init(bundle: Bundle, pacing: PacingConfig = PacingConfig(), seed: UInt64? = nil,
                            logger: TrialLogger? = nil) {
        let file = NudgePriorsFile.load(bundle: bundle)
        self.init(priors: file?.priors ?? [], priorsAreMock: file?.mock ?? false, pacing: pacing, seed: seed, logger: logger)
    }

    // MARK: Pacing

    /// Feed the classified state (~1 Hz). `decide` also notes the context state itself.
    public func noteState(_ state: AttentionState, t: Double) { locked { pacer.noteState(state, t: t) } }

    /// Switch pacing (e.g. Demo Director, DEMONSTRATION ONLY). Keeps the pacing timers.
    public func setPacing(_ config: PacingConfig) { locked { _pacing = config; pacer.config = config } }

    /// Call at the start of each session (session-relative clocks restart at 0).
    public func resetSession() { locked { pacer.resetSession() } }

    // MARK: InterventionPolicy

    public func decide(policy: NudgePolicyKind, context: NudgeContext) -> NudgeDecision? {
        let decision: NudgeDecision? = locked { decideLocked(policy: policy, context: context) }
        if let decision { logger?.log(decision: decision) }
        return decision
    }

    private func decideLocked(policy: NudgePolicyKind, context: NudgeContext) -> NudgeDecision? {
        if policy == .reactive { pacer.noteState(context.state, t: context.t) }
        guard pacer.allows(policy: policy, t: context.t) else { return nil }
        let arms = TrialEligibility.arms(policy: policy, context: context)
        guard !arms.isEmpty else { return nil }
        let probs = selectionProbabilities(policy: policy, state: context.state, arms: arms)
        let i = TrialDistributions.categorical(probs, &rng)
        var dict: [String: Double] = [:]
        for (a, p) in zip(arms, probs) { dict[a.rawValue] = p }
        let decision = NudgeDecision(t: context.t, policy: policy, context: context, arm: arms[i],
                                     randomizationProbability: probs[i], armProbabilities: dict)
        pacer.noteDecision(at: context.t)
        _records.append(TrialRecord(decision: decision))
        return decision
    }

    public func record(outcome: NudgeOutcome, for decision: NudgeDecision) {
        locked { recordLocked(outcome: outcome, for: decision) }
        logger?.log(outcome: outcome)
    }

    private func recordLocked(outcome: NudgeOutcome, for decision: NudgeDecision) {
        let key = Self.key(policy: decision.policy, state: decision.context.state, arm: decision.arm)
        if var post = table[key] {
            let r = min(max(outcome.reward, 0), 1)
            post.alpha += r
            post.beta += 1 - r
            post.pulls += 1
            table[key] = post
        }
        if let i = _records.lastIndex(where: { $0.decision.id == decision.id }) {
            _records[i].outcome = outcome
        } else {
            _records.append(TrialRecord(decision: decision, outcome: outcome))
        }
    }

    public func posteriors() -> [ArmPosterior] {
        locked { Self.allKeys().compactMap { table[$0.key] } }
    }

    /// The bootstrap runs on a snapshot outside the lock, so the pipeline is never blocked by it.
    public func effectEstimates() -> [ArmEffect] {
        let snapshot = records
        return IPWEffectEstimator(level: config.intervalLevel, replicates: config.bootstrapReplicates)
            .estimates(records: snapshot)
    }

    public func serialized() throws -> Data {
        let snap = locked { Snapshot(version: 1, posteriors: posteriors(), records: _records, pacing: _pacing) }
        return try JSONEncoder().encode(snap)
    }

    public func restore(from data: Data) throws {
        let snap = try JSONDecoder().decode(Snapshot.self, from: data)
        locked {
            buildTable()
            for p in snap.posteriors {
                let key = Self.key(policy: p.policy, state: p.state, arm: p.arm)
                if table[key] != nil { table[key] = p }
            }
            _records = snap.records
            _pacing = snap.pacing
            pacer = TrialPacer(config: snap.pacing)
        }
    }

    // MARK: Inspection

    /// Thompson probabilities (before floors) and the mixed selection probabilities for display/tests.
    public func selectionProbabilities(policy: NudgePolicyKind, context: NudgeContext) -> [NudgeArm: Double] {
        let arms = TrialEligibility.arms(policy: policy, context: context)
        let p = locked { selectionProbabilities(policy: policy, state: context.state, arms: arms) }
        return Dictionary(uniqueKeysWithValues: zip(arms, p))
    }

    /// Posterior probability that each arm is the best among `arms` (Monte Carlo).
    public func probabilityBest(policy: NudgePolicyKind, state: AttentionState?, arms: [NudgeArm],
                                draws: Int? = nil) -> [NudgeArm: Double] {
        let p = locked { thompsonProbabilities(policy: policy, state: state, arms: arms, draws: draws ?? config.monteCarloDraws) }
        return Dictionary(uniqueKeysWithValues: zip(arms, p))
    }

    public func posterior(policy: NudgePolicyKind, state: AttentionState?, arm: NudgeArm) -> ArmPosterior? {
        locked { table[Self.key(policy: policy, state: state, arm: arm)] }
    }

    // MARK: Internals

    private func selectionProbabilities(policy: NudgePolicyKind, state: AttentionState, arms: [NudgeArm]) -> [Double] {
        let ts = thompsonProbabilities(policy: policy, state: policy == .reactive ? state : nil, arms: arms,
                                       draws: config.monteCarloDraws)
        return TrialRandomization.mix(thompson: ts, arms: arms, controlProbability: _pacing.controlProbability,
                                      minArmProbability: config.minArmProbability)
    }

    private func thompsonProbabilities(policy: NudgePolicyKind, state: AttentionState?, arms: [NudgeArm],
                                       draws: Int) -> [Double] {
        guard arms.count > 1 else { return arms.map { _ in 1 } }
        let params = arms.map { arm -> (Double, Double) in
            let p = table[Self.key(policy: policy, state: policy == .reactive ? state : nil, arm: arm)]
            return (p?.alpha ?? 1, p?.beta ?? 1)
        }
        var wins = [Double](repeating: 0, count: arms.count)
        let n = max(1, draws)
        for _ in 0..<n {
            var best = -1.0, bestI = 0
            for (i, (a, b)) in params.enumerated() {
                let x = TrialDistributions.beta(alpha: a, beta: b, &rng)
                if x > best { best = x; bestI = i }
            }
            wins[bestI] += 1
        }
        return wins.map { $0 / Double(n) }
    }

    private func buildTable() {
        table = [:]
        for k in Self.allKeys() {
            let prior = priorByArm[k.arm]
            table[k.key] = ArmPosterior(state: k.state, policy: k.policy, arm: k.arm,
                                        alpha: prior?.alpha ?? 1, beta: prior?.beta ?? 1, pulls: 0)
        }
    }

    static func key(policy: NudgePolicyKind, state: AttentionState?, arm: NudgeArm) -> String {
        "\(policy.rawValue).\(policy == .reactive ? (state?.rawValue ?? "pre") : "pre").\(arm.rawValue)"
    }

    private static func allKeys() -> [(key: String, policy: NudgePolicyKind, state: AttentionState?, arm: NudgeArm)] {
        var out: [(String, NudgePolicyKind, AttentionState?, NudgeArm)] = []
        for s in AttentionState.allCases {
            for a in NudgeArm.reactiveArms(for: s) {
                out.append((key(policy: .reactive, state: s, arm: a), .reactive, s, a))
            }
        }
        for a in NudgeArm.preemptiveArms {
            out.append((key(policy: .preemptive, state: nil, arm: a), .preemptive, nil, a))
        }
        return out.map { (key: $0.0, policy: $0.1, state: $0.2, arm: $0.3) }
    }

    private struct Snapshot: Codable {
        var version: Int
        var posteriors: [ArmPosterior]
        var records: [TrialRecord]
        var pacing: PacingConfig
    }
}
