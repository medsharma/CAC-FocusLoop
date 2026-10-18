import Foundation

/// Optional contextual upgrade: linear Thompson sampling (Agrawal & Goyal 2013) over `NudgeContext`
/// features. OFF by default; select it with `TrialPolicyFactory.make(kind: .linearThompson, ...)`.
/// The Beta-Bernoulli `ThompsonTrialEngine` remains the default engine.
///
/// Per arm, a Bayesian ridge regression of reward on the features
///   [1, state one-hot (4), lapse risk, predicted engagement, session minute/60 (capped),
///    sin/cos(hour of day), minutes since last nudge/30 (capped), pre-emptive flag]
/// with posterior `theta ~ N(A^-1 b, v^2 A^-1)`, `A = lambda I + sum x x^T`, `b = sum r x`.
/// Pacing, eligibility, the control floor, Monte Carlo selection probabilities and IPW effects are
/// identical to the Beta-Bernoulli engine, so the micro-randomised trial analysis is unchanged.
public final class LinearThompsonEngine: InterventionPolicy {
    public static let featureCount = 12

    public private(set) var pacing: PacingConfig
    public let config: TrialEngineConfig
    /// Posterior scale `v` (exploration strength).
    public let explorationScale: Double
    public weak var logger: TrialLogger?

    private var rng: TrialSeededRNG
    private var pacer: TrialPacer
    private var models: [NudgeArm: ArmModel] = [:]
    /// Empirical success/failure counts per (policy, state, arm), for the posterior display only.
    private var display: [String: ArmPosterior] = [:]
    public private(set) var records: [TrialRecord] = []

    struct ArmModel: Codable, Equatable {
        var a: [Double]   // d x d row-major
        var b: [Double]
        init(lambda: Double, d: Int) {
            a = [Double](repeating: 0, count: d * d)
            for i in 0..<d { a[i * d + i] = lambda }
            b = [Double](repeating: 0, count: d)
        }
    }

    public init(priors: [NudgePrior] = [], pacing: PacingConfig = PacingConfig(), config: TrialEngineConfig = TrialEngineConfig(),
                explorationScale: Double = 0.5, ridge: Double = 1, seed: UInt64? = nil, logger: TrialLogger? = nil) {
        self.pacing = pacing
        self.config = config
        self.explorationScale = explorationScale
        self.logger = logger
        self.rng = seed.map(TrialSeededRNG.init(seed:)) ?? .system()
        self.pacer = TrialPacer(config: pacing)
        let priorByArm = Dictionary(priors.map { ($0.arm, $0) }, uniquingKeysWith: { _, b in b })
        let d = Self.featureCount
        for arm in NudgeArm.allCases {
            var m = ArmModel(lambda: ridge, d: d)
            // Prior mean reward enters through the intercept: b0 = lambda * prior mean.
            if let p = priorByArm[arm] { m.b[0] = ridge * p.alpha / (p.alpha + p.beta) } else { m.b[0] = ridge * 0.5 }
            models[arm] = m
        }
        buildDisplay(priors: priorByArm)
    }

    // MARK: Pacing

    public func noteState(_ state: AttentionState, t: Double) { pacer.noteState(state, t: t) }
    public func setPacing(_ config: PacingConfig) { pacing = config; pacer.config = config }
    public func resetSession() { pacer.resetSession() }

    // MARK: InterventionPolicy

    public func decide(policy: NudgePolicyKind, context: NudgeContext) -> NudgeDecision? {
        if policy == .reactive { pacer.noteState(context.state, t: context.t) }
        guard pacer.allows(policy: policy, t: context.t) else { return nil }
        let arms = TrialEligibility.arms(policy: policy, context: context)
        guard !arms.isEmpty else { return nil }
        let x = Self.features(context, policy: policy)
        let ts = thompsonProbabilities(arms: arms, x: x, draws: config.monteCarloDraws)
        let probs = TrialRandomization.mix(thompson: ts, arms: arms, controlProbability: pacing.controlProbability,
                                           minArmProbability: config.minArmProbability)
        let i = TrialDistributions.categorical(probs, &rng)
        var dict: [String: Double] = [:]
        for (a, p) in zip(arms, probs) { dict[a.rawValue] = p }
        let decision = NudgeDecision(t: context.t, policy: policy, context: context, arm: arms[i],
                                     randomizationProbability: probs[i], armProbabilities: dict)
        pacer.noteDecision(at: context.t)
        records.append(TrialRecord(decision: decision))
        logger?.log(decision: decision)
        return decision
    }

    public func record(outcome: NudgeOutcome, for decision: NudgeDecision) {
        let r = min(max(outcome.reward, 0), 1)
        let x = Self.features(decision.context, policy: decision.policy)
        let d = Self.featureCount
        if var m = models[decision.arm] {
            for i in 0..<d {
                m.b[i] += r * x[i]
                for j in 0..<d { m.a[i * d + j] += x[i] * x[j] }
            }
            models[decision.arm] = m
        }
        let key = ThompsonTrialEngine.key(policy: decision.policy, state: decision.context.state, arm: decision.arm)
        if var p = display[key] { p.alpha += r; p.beta += 1 - r; p.pulls += 1; display[key] = p }
        if let i = records.lastIndex(where: { $0.decision.id == decision.id }) {
            records[i].outcome = outcome
        } else {
            records.append(TrialRecord(decision: decision, outcome: outcome))
        }
        logger?.log(outcome: outcome)
    }

    /// Beta summaries of observed rewards per (state, arm) seeded by the priors (display only; the
    /// policy itself uses the linear model).
    public func posteriors() -> [ArmPosterior] {
        display.values.sorted { $0.id < $1.id }
    }

    public func effectEstimates() -> [ArmEffect] {
        IPWEffectEstimator(level: config.intervalLevel, replicates: config.bootstrapReplicates).estimates(records: records)
    }

    public func serialized() throws -> Data {
        let snap = Snapshot(version: 1, models: models.map { ArmEntry(arm: $0.key, model: $0.value) },
                            display: Array(display.values), records: records, pacing: pacing)
        return try JSONEncoder().encode(snap)
    }

    public func restore(from data: Data) throws {
        let snap = try JSONDecoder().decode(Snapshot.self, from: data)
        for e in snap.models where e.model.a.count == Self.featureCount * Self.featureCount { models[e.arm] = e.model }
        for p in snap.display { display[p.id] = p }
        records = snap.records
        pacing = snap.pacing
        pacer = TrialPacer(config: snap.pacing)
    }

    /// Selection probabilities for a context (display/tests).
    public func selectionProbabilities(policy: NudgePolicyKind, context: NudgeContext) -> [NudgeArm: Double] {
        let arms = TrialEligibility.arms(policy: policy, context: context)
        let ts = thompsonProbabilities(arms: arms, x: Self.features(context, policy: policy), draws: config.monteCarloDraws)
        let p = TrialRandomization.mix(thompson: ts, arms: arms, controlProbability: pacing.controlProbability,
                                       minArmProbability: config.minArmProbability)
        return Dictionary(uniqueKeysWithValues: zip(arms, p))
    }

    // MARK: Features

    static func features(_ c: NudgeContext, policy: NudgePolicyKind) -> [Double] {
        var x = [Double](repeating: 0, count: featureCount)
        x[0] = 1
        x[1 + c.state.index] = 1
        x[5] = Double(min(max(c.lapseRisk, 0), 1))
        x[6] = Double(min(max(c.predictedEngagement, 0), 1))
        x[7] = Double(min(max(c.sessionMinute, 0), 60)) / 60
        let h = Double(c.hourOfDay) / 24 * 2 * Double.pi
        x[8] = sin(h)
        x[9] = cos(h)
        x[10] = Double(min(max(c.minutesSinceLastNudge, 0), 30)) / 30
        x[11] = policy == .preemptive ? 1 : 0
        return x
    }

    // MARK: Internals

    private func thompsonProbabilities(arms: [NudgeArm], x: [Double], draws: Int) -> [Double] {
        guard arms.count > 1 else { return arms.map { _ in 1 } }
        let d = Self.featureCount
        // Per arm: Cholesky A = L L^T, mean mu = A^-1 b. A draw is theta = mu + v * L^-T z, and the
        // score is theta . x = mu . x + v * (L^-1 x) . z, so only w = L^-1 x is needed per arm.
        var means: [Double] = []
        var ws: [[Double]] = []
        for arm in arms {
            guard let m = models[arm], let l = Self.cholesky(m.a, d) else {
                means.append(0.5); ws.append([Double](repeating: 0, count: d)); continue
            }
            let mu = Self.solveCholesky(l, m.b, d)
            means.append(zip(mu, x).map(*).reduce(0, +))
            ws.append(Self.forwardSubstitute(l, x, d))
        }
        // theta.x ~ N(mean, v^2 |w|^2): sample the scalar score directly.
        let sds = ws.map { explorationScale * $0.map { $0 * $0 }.reduce(0, +).squareRoot() }
        var wins = [Double](repeating: 0, count: arms.count)
        let n = max(1, draws)
        for _ in 0..<n {
            var best = -Double.infinity, bestI = 0
            for i in arms.indices {
                let s = means[i] + sds[i] * TrialDistributions.normal(&rng)
                if s > best { best = s; bestI = i }
            }
            wins[bestI] += 1
        }
        return wins.map { $0 / Double(n) }
    }

    static func cholesky(_ a: [Double], _ d: Int) -> [Double]? {
        var l = [Double](repeating: 0, count: d * d)
        for i in 0..<d {
            for j in 0...i {
                var s = a[i * d + j]
                for k in 0..<j { s -= l[i * d + k] * l[j * d + k] }
                if i == j {
                    guard s > 1e-12 else { return nil }
                    l[i * d + i] = s.squareRoot()
                } else {
                    l[i * d + j] = s / l[j * d + j]
                }
            }
        }
        return l
    }

    static func forwardSubstitute(_ l: [Double], _ b: [Double], _ d: Int) -> [Double] {
        var y = [Double](repeating: 0, count: d)
        for i in 0..<d {
            var s = b[i]
            for k in 0..<i { s -= l[i * d + k] * y[k] }
            y[i] = s / l[i * d + i]
        }
        return y
    }

    static func solveCholesky(_ l: [Double], _ b: [Double], _ d: Int) -> [Double] {
        let y = forwardSubstitute(l, b, d)
        var x = [Double](repeating: 0, count: d)
        for i in stride(from: d - 1, through: 0, by: -1) {
            var s = y[i]
            for k in (i + 1)..<d { s -= l[k * d + i] * x[k] }
            x[i] = s / l[i * d + i]
        }
        return x
    }

    private func buildDisplay(priors: [NudgeArm: NudgePrior]) {
        for s in AttentionState.allCases {
            for a in NudgeArm.reactiveArms(for: s) {
                let p = ArmPosterior(state: s, policy: .reactive, arm: a, alpha: priors[a]?.alpha ?? 1,
                                     beta: priors[a]?.beta ?? 1, pulls: 0)
                display[p.id] = p
            }
        }
        for a in NudgeArm.preemptiveArms {
            let p = ArmPosterior(state: nil, policy: .preemptive, arm: a, alpha: priors[a]?.alpha ?? 1,
                                 beta: priors[a]?.beta ?? 1, pulls: 0)
            display[p.id] = p
        }
    }

    private struct ArmEntry: Codable { var arm: NudgeArm; var model: ArmModel }
    private struct Snapshot: Codable {
        var version: Int
        var models: [ArmEntry]
        var display: [ArmPosterior]
        var records: [TrialRecord]
        var pacing: PacingConfig
    }
}

/// Which trial engine to build. `betaBernoulli` is the default; `linearThompson` is the optional
/// contextual upgrade, kept behind this flag.
public enum TrialPolicyKind: String, Codable, Sendable {
    case betaBernoulli, linearThompson
}

public enum TrialPolicyFactory {
    /// Builds the configured engine with priors from the bundled `nudge_priors.json` (Beta(1, 1) if absent).
    public static func make(kind: TrialPolicyKind = .betaBernoulli, bundle: Bundle = .main,
                            pacing: PacingConfig = PacingConfig(), seed: UInt64? = nil,
                            logger: TrialLogger? = nil) -> InterventionPolicy & TrialPacingControl {
        let file = NudgePriorsFile.load(bundle: bundle)
        switch kind {
        case .betaBernoulli:
            return ThompsonTrialEngine(priors: file?.priors ?? [], priorsAreMock: file?.mock ?? false,
                                       pacing: pacing, seed: seed, logger: logger)
        case .linearThompson:
            return LinearThompsonEngine(priors: file?.priors ?? [], pacing: pacing, seed: seed, logger: logger)
        }
    }
}
