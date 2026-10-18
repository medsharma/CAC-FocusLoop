import Foundation

/// Loads the Beta priors written offline by `ml/brain/score_nudges.py` (`nudge_priors.json`, bundle root).
///
/// File format: `{"mock": true, "note": "...", "priors": [{"arm": "<NudgeArm rawValue>", "alpha": a, "beta": b}, ...]}`
/// (a bare `[NudgePrior]` array is accepted too). The result always has exactly one prior per
/// `NudgeArm.allCases`, in that order: arms missing from the file (or unknown/invalid entries) fall back
/// to a neutral Beta(1, 1). If the file is missing or unreadable, every arm gets Beta(1, 1).
///
/// While the brain model is MOCK, `isMock` is true and the priors carry no evidence about real nudges.
public enum NudgePriorsLoader {
    public struct Result: Equatable, Sendable {
        public var priors: [NudgePrior]
        /// True when the file says it was produced by the MOCK brain model.
        public var isMock: Bool
        /// False when the file was missing/unreadable and every arm got Beta(1, 1).
        public var loadedFromFile: Bool

        public func prior(for arm: NudgeArm) -> NudgePrior {
            priors.first { $0.arm == arm } ?? NudgePrior(arm: arm, alpha: 1, beta: 1)
        }
    }

    public static let resourceName = "nudge_priors"

    /// Beta(1, 1) for every arm.
    public static var fallback: [NudgePrior] { NudgeArm.allCases.map { NudgePrior(arm: $0, alpha: 1, beta: 1) } }

    public static func load(bundle: Bundle = .main, name: String = resourceName) -> Result {
        guard let url = bundle.url(forResource: name, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let result = try? decode(data) else {
            return Result(priors: fallback, isMock: false, loadedFromFile: false)
        }
        return result
    }

    /// Convenience: just the priors (with fallback).
    public static func priors(bundle: Bundle = .main) -> [NudgePrior] { load(bundle: bundle).priors }

    public static func decode(_ data: Data) throws -> Result {
        let dec = JSONDecoder()
        let entries: [Entry]
        let mock: Bool
        if let file = try? dec.decode(File.self, from: data) {
            entries = file.priors
            mock = file.mock ?? false
        } else {
            entries = try dec.decode([Entry].self, from: data)
            mock = false
        }
        var byArm: [NudgeArm: NudgePrior] = [:]
        for e in entries {
            guard let arm = NudgeArm(rawValue: e.arm), e.alpha > 0, e.beta > 0,
                  e.alpha.isFinite, e.beta.isFinite, byArm[arm] == nil else { continue }
            byArm[arm] = NudgePrior(arm: arm, alpha: e.alpha, beta: e.beta)
        }
        let priors = NudgeArm.allCases.map { byArm[$0] ?? NudgePrior(arm: $0, alpha: 1, beta: 1) }
        return Result(priors: priors, isMock: mock, loadedFromFile: true)
    }

    private struct Entry: Decodable {
        let arm: String
        let alpha: Double
        let beta: Double
    }

    private struct File: Decodable {
        let mock: Bool?
        let priors: [Entry]
    }
}
