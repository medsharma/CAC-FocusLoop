import XCTest
@testable import FocusLoop

/// Helpers shared by the trial tests.
enum TrialTestSupport {
    static func context(t: Double, state: AttentionState = .distracted, studyMode: Bool = true,
                        pickedUp: Bool = true, valley: Float? = 8, lapseRisk: Float = 0.6) -> NudgeContext {
        NudgeContext(t: t, state: state, lapseRisk: lapseRisk, predictedEngagement: 0.4, sessionMinute: Float(t / 60),
                     hourOfDay: 16, minutesSinceLastNudge: 2, studyMode: studyMode, phonePickedUp: pickedUp,
                     upcomingValleyInSeconds: valley)
    }

    /// Pacing with no debounce/cooldown, for simulations that only exercise the bandit.
    static var unpaced: PacingConfig {
        var p = PacingConfig()
        p.reactiveDebounce = 0
        p.cooldown = 0
        return p
    }
}

final class TrialEngineTests: XCTestCase {
    typealias S = TrialTestSupport

    // MARK: Randomisation probabilities

    func testProbabilitiesSumToOneAndLoggedProbabilityMatches() {
        let engine = ThompsonTrialEngine(pacing: S.unpaced, seed: 1)
        let rng = TrialSeededRNG(seed: 2)
        var r = rng
        for i in 0..<60 {
            let ctx = S.context(t: Double(i) * 10)
            guard let d = engine.decide(policy: .reactive, context: ctx) else { return XCTFail("no decision") }
            let eligible = TrialEligibility.arms(policy: .reactive, context: ctx)
            XCTAssertEqual(Set(d.armProbabilities.keys), Set(eligible.map(\.rawValue)))
            XCTAssertEqual(d.armProbabilities.values.reduce(0, +), 1, accuracy: 1e-9)
            XCTAssertEqual(d.randomizationProbability, d.armProbabilities[d.arm.rawValue] ?? -1, accuracy: 1e-12)
            XCTAssertTrue(d.armProbabilities.values.allSatisfy { $0 > 0 })
            let reward: Double = TrialDistributions.uniform(&r) < (d.arm == .haptic ? 0.8 : 0.3) ? 1 : 0
            engine.record(outcome: NudgeOutcome(decisionID: d.id, t: d.t + 5, reward: reward, refocused: reward > 0,
                                                refocusSeconds: nil, dismissed: false, lapseAvoided: nil), for: d)
        }
    }

    func testEmpiricalChoiceFrequencyMatchesLoggedProbability() {
        // Fixed posterior: the realised arm frequencies must match the logged probabilities.
        let priors = [NudgePrior(arm: .stretchBreak, alpha: 8, beta: 4), NudgePrior(arm: .lightPulse, alpha: 5, beta: 5),
                      NudgePrior(arm: .noNudge, alpha: 3, beta: 6)]
        let engine = ThompsonTrialEngine(priors: priors, pacing: S.unpaced, seed: 7)
        var counts: [NudgeArm: Int] = [:]
        var probs: [NudgeArm: Double] = [:]
        let n = 3_000
        for i in 0..<n {
            let d = engine.decide(policy: .reactive, context: S.context(t: Double(i), state: .drowsy))!
            counts[d.arm, default: 0] += 1
            for (k, v) in d.armProbabilities { probs[NudgeArm(rawValue: k)!, default: 0] += v / Double(n) }
        }
        for arm in [NudgeArm.stretchBreak, .lightPulse, .noNudge] {
            XCTAssertEqual(Double(counts[arm, default: 0]) / Double(n), probs[arm]!, accuracy: 0.03, "\(arm)")
        }
    }

    func testControlFloorHoldsEvenWhenControlLooksBad() {
        let priors = [NudgePrior(arm: .haptic, alpha: 500, beta: 5), NudgePrior(arm: .noNudge, alpha: 1, beta: 500)]
        let engine = ThompsonTrialEngine(priors: priors, pacing: S.unpaced, seed: 3)
        let ctx = S.context(t: 0)
        let p = engine.selectionProbabilities(policy: .reactive, context: ctx)
        XCTAssertGreaterThanOrEqual(p[.noNudge]!, PacingConfig().controlProbability - 1e-12)
        XCTAssertGreaterThanOrEqual(p.values.min()!, TrialEngineConfig().minArmProbability - 1e-12)
        XCTAssertEqual(p.values.reduce(0, +), 1, accuracy: 1e-9)

        var pacing = S.unpaced
        pacing.controlProbability = 0.25
        let e2 = ThompsonTrialEngine(priors: priors, pacing: pacing, seed: 3)
        var controls = 0
        for i in 0..<2_000 where e2.decide(policy: .reactive, context: S.context(t: Double(i)))?.arm == .noNudge { controls += 1 }
        XCTAssertGreaterThan(Double(controls) / 2_000, 0.22)
        // Pre-emptive too.
        let pp = e2.selectionProbabilities(policy: .preemptive, context: S.context(t: 0))
        XCTAssertGreaterThanOrEqual(pp[.noNudge]!, 0.25 - 1e-12)
    }

    // MARK: Eligibility

    func testEligibilityRules() {
        func arms(_ p: NudgePolicyKind, _ c: NudgeContext) -> Set<NudgeArm> { Set(TrialEligibility.arms(policy: p, context: c)) }
        // Distracted, no pickup, not study mode: no putMeDown, no recap.
        XCTAssertEqual(arms(.reactive, S.context(t: 0, studyMode: false, pickedUp: false)), [.ambientGlow, .haptic, .chime, .noNudge])
        XCTAssertEqual(arms(.reactive, S.context(t: 0, studyMode: false, pickedUp: true)), [.ambientGlow, .haptic, .chime, .putMeDown, .noNudge])
        XCTAssertEqual(arms(.reactive, S.context(t: 0, studyMode: true, pickedUp: false)), [.ambientGlow, .haptic, .chime, .recapCard, .noNudge])
        XCTAssertEqual(arms(.reactive, S.context(t: 0, state: .drowsy)), [.stretchBreak, .lightPulse, .noNudge])
        XCTAssertEqual(arms(.reactive, S.context(t: 0, state: .stressed)), [.pacedBreathing, .noNudge])
        XCTAssertTrue(arms(.reactive, S.context(t: 0, state: .focused)).isEmpty)
        // Recall question: pre-emptive + study mode + upcoming valley only.
        XCTAssertEqual(arms(.preemptive, S.context(t: 0, studyMode: true, valley: 10)), [.preemptiveGlow, .recallQuestion, .noNudge])
        XCTAssertEqual(arms(.preemptive, S.context(t: 0, studyMode: true, valley: nil)), [.preemptiveGlow, .noNudge])
        XCTAssertEqual(arms(.preemptive, S.context(t: 0, studyMode: false, valley: 10)), [.preemptiveGlow, .noNudge])
        // Every set includes the control.
        for s in AttentionState.allCases where s != .focused {
            XCTAssertTrue(arms(.reactive, S.context(t: 0, state: s)).contains(.noNudge))
        }
    }

    func testDecisionsOnlyUseEligibleArms() {
        let engine = ThompsonTrialEngine(pacing: S.unpaced, seed: 11)
        for i in 0..<300 {
            let ctx = S.context(t: Double(i), studyMode: i % 2 == 0, pickedUp: i % 3 == 0, valley: i % 5 == 0 ? 6 : nil)
            let policy: NudgePolicyKind = i % 4 == 0 ? .preemptive : .reactive
            let d = engine.decide(policy: policy, context: ctx)!
            XCTAssertTrue(TrialEligibility.arms(policy: policy, context: ctx).contains(d.arm))
            if d.arm == .putMeDown { XCTAssertTrue(ctx.phonePickedUp) }
            if d.arm == .recapCard { XCTAssertTrue(ctx.studyMode) }
            if d.arm == .recallQuestion { XCTAssertTrue(policy == .preemptive && ctx.studyMode && ctx.upcomingValleyInSeconds != nil) }
        }
    }

    // MARK: Pacing

    func testReactiveDebounceRequiresSustainedEvidence() {
        let engine = ThompsonTrialEngine(seed: 1)   // default pacing: debounce 3 s, cooldown 45 s
        XCTAssertNil(engine.decide(policy: .reactive, context: S.context(t: 10)))      // run starts at 10
        XCTAssertNil(engine.decide(policy: .reactive, context: S.context(t: 12)))      // 2 s < 3 s
        engine.noteState(.focused, t: 12.5)                                           // evidence broken
        XCTAssertNil(engine.decide(policy: .reactive, context: S.context(t: 13)))      // run restarts at 13
        XCTAssertNil(engine.decide(policy: .reactive, context: S.context(t: 15.5)))    // 2.5 s
        XCTAssertNotNil(engine.decide(policy: .reactive, context: S.context(t: 16)))   // 3 s sustained
        // noteState alone also builds evidence.
        let e2 = ThompsonTrialEngine(seed: 2)
        e2.noteState(.drowsy, t: 0)
        e2.noteState(.drowsy, t: 1)
        XCTAssertNotNil(e2.decide(policy: .reactive, context: S.context(t: 3, state: .drowsy)))
        // Focused never gets a reactive nudge.
        let e3 = ThompsonTrialEngine(seed: 3)
        XCTAssertNil(e3.decide(policy: .reactive, context: S.context(t: 100, state: .focused)))
    }

    func testCooldownBetweenNudgesIncludingControl() {
        var pacing = PacingConfig()
        pacing.controlProbability = 1   // every decision is the control: cooldown must still apply
        let engine = ThompsonTrialEngine(pacing: pacing, seed: 5)
        engine.noteState(.distracted, t: 0)
        let d = engine.decide(policy: .reactive, context: S.context(t: 3))
        XCTAssertEqual(d?.arm, .noNudge)
        XCTAssertNil(engine.decide(policy: .reactive, context: S.context(t: 20)))
        XCTAssertNil(engine.decide(policy: .preemptive, context: S.context(t: 47.9)))
        XCTAssertNotNil(engine.decide(policy: .preemptive, context: S.context(t: 48)))
        // Pre-emptive has no debounce but does respect the cooldown it just started.
        XCTAssertNil(engine.decide(policy: .reactive, context: S.context(t: 60)))
        XCTAssertNotNil(engine.decide(policy: .reactive, context: S.context(t: 93)))
    }

    func testDemoPacingAndSessionReset() {
        let engine = ThompsonTrialEngine(seed: 9)
        engine.setPacing(.demo)
        engine.noteState(.distracted, t: 0)
        XCTAssertNotNil(engine.decide(policy: .reactive, context: S.context(t: 1.5)))
        XCTAssertNil(engine.decide(policy: .reactive, context: S.context(t: 10)))
        XCTAssertNotNil(engine.decide(policy: .reactive, context: S.context(t: 13.5)))
        // New session: clock restarts at 0, pacing resets.
        engine.resetSession()
        engine.noteState(.distracted, t: 0)
        XCTAssertNotNil(engine.decide(policy: .reactive, context: S.context(t: 2)))
    }

    // MARK: Posterior update & priors

    func testPriorsInitialisePosteriorsAndUpdates() throws {
        let json = #"{"mock": true, "priors": [{"arm": "haptic", "alpha": 3, "beta": 2}, {"arm": "bogusArm", "alpha": 1, "beta": 1}]}"#
        let file = try NudgePriorsFile.decode(Data(json.utf8))
        XCTAssertTrue(file.mock)
        XCTAssertEqual(file.priors, [NudgePrior(arm: .haptic, alpha: 3, beta: 2)])
        let engine = ThompsonTrialEngine(priors: file.priors, priorsAreMock: file.mock, pacing: S.unpaced, seed: 1)
        XCTAssertTrue(engine.priorsAreMock)
        let post = engine.posterior(policy: .reactive, state: .distracted, arm: .haptic)!
        XCTAssertEqual(post.alpha, 3); XCTAssertEqual(post.beta, 2)
        XCTAssertEqual(engine.posterior(policy: .reactive, state: .distracted, arm: .chime)?.alpha, 1)
        // Every (state, arm) and pre-emptive arm is present.
        let expected = AttentionState.allCases.map { NudgeArm.reactiveArms(for: $0).count }.reduce(0, +) + NudgeArm.preemptiveArms.count
        XCTAssertEqual(engine.posteriors().count, expected)

        // Fractional reward (after dismissal penalty) updates alpha/beta proportionally.
        let d = NudgeDecision(t: 0, policy: .reactive, context: S.context(t: 0), arm: .haptic,
                              randomizationProbability: 0.5, armProbabilities: ["haptic": 0.5, "noNudge": 0.5])
        engine.record(outcome: NudgeOutcome(decisionID: d.id, t: 5, reward: 0.5, refocused: true, refocusSeconds: 4,
                                            dismissed: true, lapseAvoided: nil), for: d)
        let p2 = engine.posterior(policy: .reactive, state: .distracted, arm: .haptic)!
        XCTAssertEqual(p2.alpha, 3.5); XCTAssertEqual(p2.beta, 2.5); XCTAssertEqual(p2.pulls, 1)
        // The drowsy table is separate from distracted; pre-emptive table is separate from reactive.
        XCTAssertEqual(engine.posterior(policy: .preemptive, state: nil, arm: .noNudge)?.pulls, 0)
    }

    func testLoggerReceivesDecisionsAndOutcomes() {
        let logger = InMemoryTrialLogger()
        let engine = ThompsonTrialEngine(pacing: S.unpaced, seed: 4, logger: logger)
        let d = engine.decide(policy: .reactive, context: S.context(t: 1))!
        let o = NudgeOutcome(decisionID: d.id, t: 10, reward: 1, refocused: true, refocusSeconds: 6, dismissed: false, lapseAvoided: nil)
        engine.record(outcome: o, for: d)
        XCTAssertEqual(logger.decisions, [d])
        XCTAssertEqual(logger.outcomes, [o])
        XCTAssertEqual(logger.records.first?.outcome, o)
        logger.removeAll()
        XCTAssertTrue(logger.decisions.isEmpty)
    }

    // MARK: Persistence

    func testPersistenceRoundTrip() throws {
        let engine = ThompsonTrialEngine(priors: [NudgePrior(arm: .chime, alpha: 2, beta: 5)], pacing: S.unpaced, seed: 21)
        var r = TrialSeededRNG(seed: 99)
        for i in 0..<120 {
            let policy: NudgePolicyKind = i % 3 == 0 ? .preemptive : .reactive
            let state: AttentionState = [.distracted, .drowsy, .stressed][i % 3]
            let d = engine.decide(policy: policy, context: S.context(t: Double(i) * 30, state: state))!
            let reward: Double = TrialDistributions.uniform(&r) < 0.5 ? 1 : 0
            engine.record(outcome: NudgeOutcome(decisionID: d.id, t: d.t + 10, reward: reward, refocused: reward > 0,
                                                refocusSeconds: reward > 0 ? 5 : nil, dismissed: false,
                                                lapseAvoided: policy == .preemptive ? reward > 0 : nil), for: d)
        }
        let data = try engine.serialized()
        let restored = ThompsonTrialEngine(seed: 1)
        try restored.restore(from: data)
        XCTAssertEqual(restored.posteriors(), engine.posteriors())
        XCTAssertEqual(restored.records, engine.records)
        XCTAssertEqual(restored.pacing, engine.pacing)
        XCTAssertEqual(restored.effectEstimates(), engine.effectEstimates())
        XCTAssertFalse(engine.effectEstimates().isEmpty)
        XCTAssertThrowsError(try restored.restore(from: Data("nope".utf8)))
    }

    // MARK: Linear Thompson (optional engine, behind a flag)

    func testLinearEngineLearnsBestArmAndRoundTrips() throws {
        let truth: [NudgeArm: Double] = [.stretchBreak: 0.75, .lightPulse: 0.35, .noNudge: 0.3]
        let engine = LinearThompsonEngine(pacing: S.unpaced, seed: 12)
        var r = TrialSeededRNG(seed: 13)
        var last: [NudgeArm] = []
        for i in 0..<400 {
            let d = engine.decide(policy: .reactive, context: S.context(t: Double(i), state: .drowsy))!
            XCTAssertEqual(d.armProbabilities.values.reduce(0, +), 1, accuracy: 1e-9)
            XCTAssertGreaterThanOrEqual(d.armProbabilities["noNudge"]!, 0.1 - 1e-12)
            let reward: Double = TrialDistributions.uniform(&r) < truth[d.arm]! ? 1 : 0
            engine.record(outcome: NudgeOutcome(decisionID: d.id, t: d.t, reward: reward, refocused: reward > 0,
                                                refocusSeconds: nil, dismissed: false, lapseAvoided: nil), for: d)
            if i >= 300 { last.append(d.arm) }
        }
        let share = Double(last.filter { $0 == .stretchBreak }.count) / Double(last.count)
        XCTAssertGreaterThan(share, 0.6)
        let restored = LinearThompsonEngine(seed: 1)
        try restored.restore(from: try engine.serialized())
        XCTAssertEqual(restored.records, engine.records)
        XCTAssertEqual(restored.posteriors(), engine.posteriors())
        let p1 = engine.selectionProbabilities(policy: .reactive, context: S.context(t: 0, state: .drowsy))
        XCTAssertGreaterThan(p1[.stretchBreak]!, 0.6)
        // Factory flag.
        XCTAssertTrue(TrialPolicyFactory.make(kind: .linearThompson, seed: 1) is LinearThompsonEngine)
        XCTAssertTrue(TrialPolicyFactory.make(seed: 1) is ThompsonTrialEngine)
    }

    func testCholeskySolve() {
        // A = [[4, 2], [2, 3]], b = [2, 1] -> x = [0.5, 0]
        let l = LinearThompsonEngine.cholesky([4, 2, 2, 3], 2)!
        let x = LinearThompsonEngine.solveCholesky(l, [2, 1], 2)
        XCTAssertEqual(x[0], 0.5, accuracy: 1e-12)
        XCTAssertEqual(x[1], 0, accuracy: 1e-12)
    }

    // MARK: Distributions

    func testBetaSamplerMoments() {
        var rng = TrialSeededRNG(seed: 42)
        for (a, b) in [(2.0, 5.0), (0.5, 0.5), (30.0, 10.0)] {
            let xs = (0..<20_000).map { _ in TrialDistributions.beta(alpha: a, beta: b, &rng) }
            let mean = xs.reduce(0, +) / Double(xs.count)
            XCTAssertEqual(mean, a / (a + b), accuracy: 0.01)
            XCTAssertTrue(xs.allSatisfy { $0 >= 0 && $0 <= 1 })
        }
    }
}
