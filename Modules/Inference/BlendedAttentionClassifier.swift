import Foundation

/// The classifier the session engine should use: base (Core ML model, or the heuristic when the
/// model is missing) + rule-based safety nets + the user's personal head.
///
/// Per call:
/// 1. Heuristic estimate (always computed; cheap).
/// 2. Missing face: if the face was tracked in fewer than half of the last 3 s, the heuristic's
///    missing-face estimate is used as-is (distracted ~0.68, ~0.86 with a phone pickup; see
///    `HeuristicAttentionClassifier`). The model never saw face-less windows and the personal
///    head's embedding is meaningless then. Source `.heuristic`.
/// 3. Base model estimate. If its max probability < `confidenceThreshold` (0.45), it is averaged
///    with the heuristic (`.average`, source `.blended`) or replaced by it (`.heuristic`).
/// 4. Personal head: if it returns probabilities for the base embedding,
///    p = (1 - w) * p_base + w * p_personal with w = min(cap, n / (n + k)), n = labelCount,
///    k = 30, cap = 0.7. Source `.blended`.
///
/// Source: `.model` = pure model output, `.heuristic` = pure heuristic, `.blended` = any mix.
///
/// The returned `embedding` is the base classifier's embedding (model embedding when a model is
/// used, else the heuristic's engineered one). Train the personal head with exactly that
/// embedding: `head.update(embedding: estimate.embedding, label: ...)`.
public final class BlendedAttentionClassifier: AttentionClassifier {

    public enum LowConfidencePolicy: String, Codable, Sendable {
        /// Average base and heuristic probabilities.
        case average
        /// Use the heuristic probabilities alone.
        case heuristic
    }

    public struct Config: Codable, Equatable, Sendable {
        /// k in w = n / (n + k).
        public var personalK: Float = 30
        public var maxPersonalWeight: Float = 0.7
        /// Base max-probability below which the heuristic is brought in.
        public var confidenceThreshold: Float = 0.45
        public var lowConfidencePolicy: LowConfidencePolicy = .average
        /// Override with the heuristic's missing-face estimate when the face is gone.
        public var missingFaceOverride = true
        public init() {}
    }

    /// What happened on the last `classify` call (for the science view / debugging).
    public struct Diagnostics: Equatable, Sendable {
        public var baseIsModel: Bool
        public var baseConfidence: Float
        public var usedHeuristicFallback: Bool
        public var faceMissing: Bool
        public var personalWeight: Float
        public var heuristicEvidence: HeuristicAttentionClassifier.Evidence?
    }

    /// Model classifier, or nil to run on the heuristic alone.
    public let base: AttentionClassifier?
    public let heuristic: HeuristicAttentionClassifier
    public let personalHead: PersonalHead?
    public var config: Config
    public private(set) var lastDiagnostics: Diagnostics?

    public init(base: AttentionClassifier?, heuristic: HeuristicAttentionClassifier = HeuristicAttentionClassifier(),
                personalHead: PersonalHead?, config: Config = Config()) {
        // A heuristic passed as base is the same as no model.
        self.base = (base is HeuristicAttentionClassifier) ? nil : base
        self.heuristic = (base as? HeuristicAttentionClassifier) ?? heuristic
        self.personalHead = personalHead
        self.config = config
    }

    /// Uses the bundled Core ML model when present, otherwise the heuristic alone.
    public static func makeDefault(personalHead: PersonalHead?, bundle: Bundle = .main,
                                   config: Config = Config()) -> BlendedAttentionClassifier {
        BlendedAttentionClassifier(base: CoreMLAttentionClassifier(bundle: bundle), personalHead: personalHead,
                                   config: config)
    }

    /// True when a Core ML (or other non-heuristic) base classifier is in use.
    public var usesModel: Bool { base != nil }

    /// w = min(cap, n / (n + k)); 0 for n <= 0.
    public static func personalWeight(labelCount n: Int, k: Float = 30, cap: Float = 0.7) -> Float {
        guard n > 0 else { return 0 }
        let w = Float(n) / (Float(n) + max(k, 0))
        return min(max(cap, 0), w)
    }

    public func classify(window: [FeatureFrame]) -> StateEstimate {
        let t = window.last?.t ?? 0
        // 1. Heuristic (evidence is nil only for an empty window).
        let analysis = window.isEmpty ? nil : heuristic.analyze(window: window)
        let evidence = analysis?.0
        let heurEmbedding = analysis?.1 ?? Array(repeating: 0, count: HeuristicAttentionClassifier.embeddingDimension)
        let k = AttentionState.allCases.count
        let heurProbs = evidence.map { heuristic.probabilities(for: $0) } ?? Array(repeating: 1 / Float(k), count: k)

        var diag = Diagnostics(baseIsModel: base != nil, baseConfidence: 0, usedHeuristicFallback: false,
                               faceMissing: evidence?.faceMissing ?? false, personalWeight: 0,
                               heuristicEvidence: evidence)

        // 2. Missing face -> heuristic missing-face estimate.
        if config.missingFaceOverride, evidence?.faceMissing == true {
            let embedding = base == nil ? heurEmbedding : []
            diag.baseConfidence = heurProbs.max() ?? 0
            lastDiagnostics = diag
            return StateEstimate(t: t, probabilities: heurProbs, source: .heuristic, embedding: embedding)
        }

        // 3. Base estimate with confidence fallback.
        var probs: [Float]
        var source: EstimateSource
        let embedding: [Float]
        if let base {
            let b = base.classify(window: window)
            let bp = InferenceMath.normalizedProbabilities(b.probabilities)
            diag.baseConfidence = bp.max() ?? 0
            embedding = b.embedding
            if diag.baseConfidence < config.confidenceThreshold {
                diag.usedHeuristicFallback = true
                switch config.lowConfidencePolicy {
                case .average:
                    probs = zip(bp, heurProbs).map { ($0 + $1) / 2 }
                    source = .blended
                case .heuristic:
                    probs = heurProbs
                    source = .heuristic
                }
            } else {
                probs = bp
                source = b.source
            }
        } else {
            probs = heurProbs
            source = .heuristic
            embedding = heurEmbedding
            diag.baseConfidence = heurProbs.max() ?? 0
        }

        // 4. Personal head.
        if let head = personalHead, !embedding.isEmpty, let personal = head.predict(embedding: embedding),
           personal.count == probs.count {
            let w = Self.personalWeight(labelCount: head.labelCount, k: config.personalK, cap: config.maxPersonalWeight)
            if w > 0 {
                probs = zip(probs, personal).map { (1 - w) * $0 + w * $1 }
                source = .blended
                diag.personalWeight = w
            }
        }

        lastDiagnostics = diag
        return StateEstimate(t: t, probabilities: InferenceMath.normalizedProbabilities(probs), source: source,
                             embedding: embedding)
    }
}
