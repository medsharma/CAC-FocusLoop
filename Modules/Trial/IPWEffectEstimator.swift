import Foundation

/// Per-user causal effect of each arm vs the no-nudge control, from the micro-randomised trial log.
///
/// Design: each decision point randomises among the eligible arms with a logged probability
/// (micro-randomised trial, Klasnja et al. 2015, "Microrandomized trials: An experimental design for
/// developing just-in-time adaptive interventions", Health Psychology 34(S):1220-1228; JITAI framework,
/// Nahum-Shani et al. 2018, "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health",
/// Annals of Behavioral Medicine 52(6):446-462).
///
/// Estimator: for arm `a` and policy `k`, restrict to decisions of policy `k` where both `a` and
/// `noNudge` were eligible (both appear in `armProbabilities`). On that set compute the Hajek
/// (self-normalised inverse-probability-weighted) mean reward of each arm,
///   mu_a = sum_i 1[A_i = a] R_i / p_i(a)  /  sum_i 1[A_i = a] / p_i(a),
/// and report `effect = mu_a - mu_noNudge`. The Hajek form is used rather than Horvitz-Thompson because
/// it is bounded in [-1, 1] and has much lower variance when probabilities drift as the bandit learns.
///
/// Interval: nonparametric bootstrap **percentile** interval (default 90 %), resampling decisions with
/// replacement and recomputing the Hajek difference; replicates lacking either arm are discarded.
/// The bootstrap RNG is seeded deterministically from the data size so the UI does not jitter.
///
/// Arms with no observed outcome (or with no control outcome in the eligible set) are omitted.
public struct IPWEffectEstimator: Sendable {
    public var level: Double
    public var replicates: Int

    public init(level: Double = 0.9, replicates: Int = 500) {
        self.level = level
        self.replicates = replicates
    }

    private struct Obs {
        var isArm: Bool
        var reward: Double
        var weight: Double
    }

    public func estimates(records: [TrialRecord]) -> [ArmEffect] {
        let completed = records.compactMap { r -> (NudgeDecision, NudgeOutcome)? in
            guard let o = r.outcome else { return nil }
            return (r.decision, o)
        }
        var out: [ArmEffect] = []
        for policy in NudgePolicyKind.allCases {
            let ofPolicy = completed.filter { $0.0.policy == policy }
            for arm in NudgeArm.allCases where arm != .noNudge {
                if let e = estimate(arm: arm, policy: policy, data: ofPolicy) { out.append(e) }
            }
        }
        return out
    }

    func estimate(arm: NudgeArm, policy: NudgePolicyKind, data: [(NudgeDecision, NudgeOutcome)]) -> ArmEffect? {
        // Observations on decisions where both arms were eligible and one of them was chosen.
        var obs: [Obs] = []
        for (d, o) in data {
            guard let pa = d.armProbabilities[arm.rawValue], let p0 = d.armProbabilities[NudgeArm.noNudge.rawValue]
            else { continue }
            if d.arm == arm, pa > 0 {
                obs.append(Obs(isArm: true, reward: o.reward, weight: 1 / pa))
            } else if d.arm == .noNudge, p0 > 0 {
                obs.append(Obs(isArm: false, reward: o.reward, weight: 1 / p0))
            } else {
                // Decision where a third arm was chosen: contributes zero to both Hajek sums, but
                // still belongs to the population the bootstrap resamples.
                obs.append(Obs(isArm: false, reward: 0, weight: 0))
            }
        }
        let nArm = obs.filter { $0.isArm }.count
        let nCtl = obs.filter { !$0.isArm && $0.weight > 0 }.count
        guard nArm > 0, nCtl > 0, let point = Self.hajekDifference(obs) else { return nil }

        var rng = TrialSeededRNG(seed: 0xC0FFEE ^ UInt64(obs.count) ^ (UInt64(nArm) << 20) ^ UInt64(NudgeArm.allCases.firstIndex(of: arm) ?? 0))
        var reps: [Double] = []
        reps.reserveCapacity(replicates)
        var sample = [Obs](repeating: obs[0], count: obs.count)
        for _ in 0..<max(0, replicates) {
            for j in 0..<obs.count {
                sample[j] = obs[Int(rng.next() % UInt64(obs.count))]
            }
            if let v = Self.hajekDifference(sample) { reps.append(v) }
        }
        let lower: Double, upper: Double
        if reps.count >= 20 {
            reps.sort()
            let tail = (1 - level) / 2
            lower = Self.quantile(reps, tail)
            upper = Self.quantile(reps, 1 - tail)
        } else {
            lower = -1; upper = 1
        }
        return ArmEffect(arm: arm, policy: policy, effect: point, lower: min(lower, point), upper: max(upper, point),
                         level: level, n: nArm)
    }

    private static func hajekDifference(_ obs: [Obs]) -> Double? {
        var swA = 0.0, swrA = 0.0, sw0 = 0.0, swr0 = 0.0
        for o in obs where o.weight > 0 {
            if o.isArm { swA += o.weight; swrA += o.weight * o.reward } else { sw0 += o.weight; swr0 += o.weight * o.reward }
        }
        guard swA > 0, sw0 > 0 else { return nil }
        return swrA / swA - swr0 / sw0
    }

    private static func quantile(_ sorted: [Double], _ q: Double) -> Double {
        let pos = q * Double(sorted.count - 1)
        let lo = Int(pos.rounded(.down)), hi = min(lo + 1, sorted.count - 1)
        let f = pos - Double(lo)
        return sorted[lo] * (1 - f) + sorted[hi] * f
    }
}
