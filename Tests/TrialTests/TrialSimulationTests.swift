import XCTest
@testable import FocusLoop

/// SIMULATED user: every number here comes from a synthetic Bernoulli user, not real data.
final class TrialSimulationTests: XCTestCase {
    /// True P(refocus) per arm for the SIMULATED distracted user (haptic is truly best).
    static let truth: [NudgeArm: Double] = [
        .haptic: 0.75, .ambientGlow: 0.35, .chime: 0.45, .putMeDown: 0.5, .recapCard: 0.4, .noNudge: 0.3,
    ]
    static let best: NudgeArm = .haptic
    static let seeds = 20
    static let trialsPerSeed = 600

    struct SeedResult {
        var lastHundredBestShare: Double
        var trialsToIdentify: Int?
        var effects: [ArmEffect]
    }

    static func run(seed: UInt64, trials: Int = trialsPerSeed) -> SeedResult {
        var config = TrialEngineConfig()
        // Fewer draws than the 2,000 default keeps this 12,000-decision Debug-build test fast;
        // the logged probabilities stay exact for the mechanism actually used (see ThompsonTrialEngine).
        config.monteCarloDraws = 500
        let engine = ThompsonTrialEngine(pacing: TrialTestSupport.unpaced, config: config, seed: seed)
        var userRNG = TrialSeededRNG(seed: seed &* 7_919 &+ 17)
        let eligible = NudgeArm.reactiveArms(for: .distracted)
        var chosen: [NudgeArm] = []
        var identified: Int?
        for i in 1...trials {
            let ctx = TrialTestSupport.context(t: Double(i) * 60, state: .distracted, studyMode: true, pickedUp: true)
            let d = engine.decide(policy: .reactive, context: ctx)!
            chosen.append(d.arm)
            let success = TrialDistributions.uniform(&userRNG) < truth[d.arm]!
            engine.record(outcome: NudgeOutcome(decisionID: d.id, t: d.t + 10, reward: success ? 1 : 0, refocused: success,
                                                refocusSeconds: success ? 8 : nil, dismissed: false, lapseAvoided: nil), for: d)
            if identified == nil, i % 5 == 0 {
                let pBest = engine.probabilityBest(policy: .reactive, state: .distracted, arms: eligible, draws: 500)
                if pBest[best]! > 0.9 { identified = i }
            }
        }
        let last = chosen.suffix(100)
        return SeedResult(lastHundredBestShare: Double(last.filter { $0 == best }.count) / Double(last.count),
                          trialsToIdentify: identified,
                          effects: engine.effectEstimates().filter { $0.policy == .reactive })
    }

    func testSimulatedUserConvergenceAndIPWCoverage() {
        var shares: [Double] = []
        var identify: [Int] = []
        var covered: [NudgeArm: Int] = [:]
        var counted: [NudgeArm: Int] = [:]
        var widths: [NudgeArm: [Double]] = [:]
        for s in 0..<Self.seeds {
            let r = Self.run(seed: UInt64(1_000 + s))
            shares.append(r.lastHundredBestShare)
            if let t = r.trialsToIdentify { identify.append(t) }
            for e in r.effects {
                let trueEffect = Self.truth[e.arm]! - Self.truth[.noNudge]!
                counted[e.arm, default: 0] += 1
                if e.lower <= trueEffect && trueEffect <= e.upper { covered[e.arm, default: 0] += 1 }
                widths[e.arm, default: []].append(e.upper - e.lower)
            }
        }
        // Convergence: >= 70 % of the last 100 pulls go to the truly best arm. The ceiling is ~0.80
        // (10 % control floor + 2 % per-arm floor), so require it on average and in >= 18/20 seeds.
        let mean = shares.reduce(0, +) / Double(shares.count)
        XCTAssertGreaterThanOrEqual(mean, 0.7, "SIMULATED last-100 best-arm shares: \(shares)")
        XCTAssertGreaterThanOrEqual(shares.filter { $0 >= 0.7 }.count, Self.seeds - 2, "SIMULATED shares: \(shares)")
        // Identification: P(best) > 0.9 is reached in (almost) every seed.
        XCTAssertGreaterThanOrEqual(identify.count, Self.seeds - 1)

        // IPW recovery: true effects inside the 90 % intervals in most seeds.
        let totalCovered = covered.values.reduce(0, +), total = counted.values.reduce(0, +)
        let overall = Double(totalCovered) / Double(max(total, 1))
        XCTAssertGreaterThanOrEqual(overall, 0.75)
        for arm in Self.truth.keys where arm != .noNudge {
            let c = Double(covered[arm, default: 0]) / Double(max(counted[arm, default: 0], 1))
            XCTAssertGreaterThan(c, 0.5, "coverage for \(arm)")
            XCTAssertGreaterThanOrEqual(counted[arm, default: 0], Self.seeds - 1, "estimates for \(arm)")
        }

        let sortedID = identify.sorted()
        let median = sortedID.isEmpty ? -1 : sortedID[sortedID.count / 2]
        var report = "\n=== SIMULATED trial (synthetic Bernoulli user, \(Self.seeds) seeds x \(Self.trialsPerSeed) decisions) ===\n"
        report += "True P(refocus): " + Self.truth.sorted { $0.key.rawValue < $1.key.rawValue }
            .map { "\($0.key.rawValue)=\($0.value)" }.joined(separator: ", ") + "\n"
        report += String(format: "Best-arm share of last 100 pulls: mean %.2f, min %.2f, median %.2f, max %.2f; seeds >= 0.70: %d/%d\n",
                         mean, shares.min()!, shares.sorted()[shares.count / 2], shares.max()!,
                         shares.filter { $0 >= 0.7 }.count, shares.count)
        report += "Trials until posterior P(best) > 0.9: median \(median), min \(sortedID.first ?? -1), max \(sortedID.last ?? -1) (identified in \(identify.count)/\(Self.seeds) seeds)\n"
        report += String(format: "IPW 90%% bootstrap interval coverage overall: %.2f (%d/%d)\n", overall, totalCovered, total)
        for arm in NudgeArm.reactiveArms(for: .distracted) where arm != .noNudge {
            let w = widths[arm, default: []].sorted()
            report += String(format: "  %@: true effect %+.2f, coverage %d/%d, median interval width %.2f\n",
                             arm.rawValue, Self.truth[arm]! - Self.truth[.noNudge]!, covered[arm, default: 0],
                             counted[arm, default: 0], w.isEmpty ? .nan : w[w.count / 2])
        }
        print(report)
    }

    func testIPWEstimatorOnUniformRandomisationIsUnbiased() {
        // Fixed, known, non-uniform randomisation probabilities: Hajek difference recovers the truth.
        var rng = TrialSeededRNG(seed: 5)
        let arms: [NudgeArm] = [.stretchBreak, .lightPulse, .noNudge]
        let probs = [0.6, 0.15, 0.25]
        let truth: [NudgeArm: Double] = [.stretchBreak: 0.7, .lightPulse: 0.4, .noNudge: 0.3]
        var records: [TrialRecord] = []
        for i in 0..<4_000 {
            let k = TrialDistributions.categorical(probs, &rng)
            let arm = arms[k]
            let d = NudgeDecision(t: Double(i), policy: .reactive, context: TrialTestSupport.context(t: Double(i), state: .drowsy),
                                  arm: arm, randomizationProbability: probs[k],
                                  armProbabilities: Dictionary(uniqueKeysWithValues: zip(arms.map(\.rawValue), probs)))
            let r: Double = TrialDistributions.uniform(&rng) < truth[arm]! ? 1 : 0
            records.append(TrialRecord(decision: d, outcome: NudgeOutcome(decisionID: d.id, t: d.t, reward: r, refocused: r > 0,
                                                                         refocusSeconds: nil, dismissed: false, lapseAvoided: nil)))
        }
        let effects = IPWEffectEstimator(level: 0.9, replicates: 400).estimates(records: records)
        XCTAssertEqual(effects.count, 2)
        for e in effects {
            let t = truth[e.arm]! - truth[.noNudge]!
            XCTAssertEqual(e.effect, t, accuracy: 0.06)
            XCTAssertLessThanOrEqual(e.lower, e.effect)
            XCTAssertGreaterThanOrEqual(e.upper, e.effect)
            XCTAssertEqual(e.level, 0.9)
            XCTAssertLessThan(e.upper - e.lower, 0.15)
        }
        // Missing control -> no estimate.
        let noControl = records.filter { $0.decision.arm != .noNudge }
        XCTAssertTrue(IPWEffectEstimator().estimates(records: noControl).isEmpty)
        // Pending (no outcome) records are ignored.
        let pending = records.map { TrialRecord(decision: $0.decision) }
        XCTAssertTrue(IPWEffectEstimator().estimates(records: pending).isEmpty)
    }
}
