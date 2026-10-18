import CoreML
import Foundation

/// Runs the trained temporal network (`AttentionStateModel.mlmodelc`, compiled by Xcode from
/// `Resources/Models/AttentionStateModel.mlpackage`) over a 10 s window.
///
/// Uses `MLModel` + `MLMultiArray` by feature name (no generated class), so the app compiles and
/// runs without the model file; the failable initialisers return nil when it is absent or its I/O
/// does not match.
///
/// I/O (by name, configurable via `IO`):
/// * input  `window`: Float32 (1, 100, 42) of RAW `FeatureFrame.vector` values, oldest first
///   (normalisation is baked into the model). Short windows are front-padded by repeating the
///   first frame.
/// * output `probabilities`: (1, 4) in `AttentionState.allCases` order (renormalised; softmaxed
///   if it looks like logits).
/// * output `embedding`: (1, E), optional.
///
/// `classify` runs synchronously on the caller's queue; the session engine calls it at 1 Hz.
/// If a prediction fails it returns a uniform distribution with an empty embedding (confidence
/// 0.25), which `BlendedAttentionClassifier` treats as low confidence and covers with the heuristic.
public final class CoreMLAttentionClassifier: AttentionClassifier {

    public struct IO: Equatable, Sendable {
        public var inputName = "window"
        public var probabilitiesName = "probabilities"
        public var embeddingName = "embedding"
        public var timesteps = Int((FeatureFrame.stateWindowSeconds * FeatureFrame.rateHz).rounded())
        public var featureCount = FeatureFrame.featureNames.count
        public init() {}
    }

    public static let defaultModelName = "AttentionStateModel"

    public let model: MLModel
    public let io: IO
    private let inputDataType: MLMultiArrayDataType
    /// Last prediction error (nil after a successful prediction).
    public private(set) var lastError: Error?

    /// Loads `<modelName>.mlmodelc` from `bundle`; nil if it is not bundled or does not load.
    public convenience init?(bundle: Bundle = .main, modelName: String = CoreMLAttentionClassifier.defaultModelName,
                             io: IO = IO(), computeUnits: MLComputeUnits = .all) {
        guard let url = bundle.url(forResource: modelName, withExtension: "mlmodelc") else { return nil }
        self.init(compiledModelURL: url, io: io, computeUnits: computeUnits)
    }

    /// Loads a compiled model directory; nil if it does not load or its I/O does not match `io`.
    public convenience init?(compiledModelURL url: URL, io: IO = IO(), computeUnits: MLComputeUnits = .all) {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        guard let model = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }
        self.init(model: model, io: io)
    }

    /// Wraps an already-loaded model; nil if its I/O does not match `io`.
    public init?(model: MLModel, io: IO = IO()) {
        let desc = model.modelDescription
        guard let input = desc.inputDescriptionsByName[io.inputName], input.type == .multiArray,
              let probs = desc.outputDescriptionsByName[io.probabilitiesName], probs.type == .multiArray
        else { return nil }
        if let constraint = input.multiArrayConstraint {
            let shape = constraint.shape.map(\.intValue)
            // Expect (1, T, F) or (T, F); a mismatching fixed shape cannot be fed.
            if !shape.isEmpty, shape.suffix(2) != [io.timesteps, io.featureCount] { return nil }
            switch constraint.dataType {
            case .float32, .double, .float16: inputDataType = constraint.dataType
            default: inputDataType = .float32
            }
        } else {
            inputDataType = .float32
        }
        self.model = model
        self.io = io
    }

    public func classify(window: [FeatureFrame]) -> StateEstimate {
        let t = window.last?.t ?? 0
        let k = AttentionState.allCases.count
        let fallback = StateEstimate(t: t, probabilities: Array(repeating: 1 / Float(k), count: k), source: .model)
        guard !window.isEmpty else { return fallback }
        do {
            let input = try makeInput(window: window)
            let provider = try MLDictionaryFeatureProvider(dictionary: [io.inputName: MLFeatureValue(multiArray: input)])
            let out = try model.prediction(from: provider)
            guard let probArray = out.featureValue(for: io.probabilitiesName)?.multiArrayValue else {
                throw InferenceCoreMLError.missingOutput(io.probabilitiesName)
            }
            let probs = InferenceMath.normalizedProbabilities(Self.floats(probArray))
            let embedding = out.featureValue(for: io.embeddingName)?.multiArrayValue.map(Self.floats) ?? []
            lastError = nil
            return StateEstimate(t: t, probabilities: probs, source: .model, embedding: embedding)
        } catch {
            lastError = error
            return fallback
        }
    }

    /// Builds the (1, T, F) input array from the window.
    func makeInput(window: [FeatureFrame]) throws -> MLMultiArray {
        let rows = Self.paddedVectors(window: window, timesteps: io.timesteps)
        let f = io.featureCount
        let array = try MLMultiArray(shape: [1, NSNumber(value: io.timesteps), NSNumber(value: f)], dataType: inputDataType)
        if inputDataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: io.timesteps * f)
            for (i, row) in rows.enumerated() {
                for j in 0..<f { ptr[i * f + j] = j < row.count ? row[j] : 0 }
            }
        } else {
            for (i, row) in rows.enumerated() {
                for j in 0..<f { array[i * f + j] = NSNumber(value: j < row.count ? row[j] : 0) }
            }
        }
        return array
    }

    /// Last `timesteps` frame vectors, oldest first; shorter windows are front-padded by
    /// repeating the first frame.
    static func paddedVectors(window: [FeatureFrame], timesteps: Int) -> [[Float]] {
        guard let first = window.first else { return [] }
        let tail = window.suffix(timesteps).map(\.vector)
        let pad = timesteps - tail.count
        return pad > 0 ? Array(repeating: first.vector, count: pad) + tail : tail
    }

    static func floats(_ a: MLMultiArray) -> [Float] {
        (0..<a.count).map { a[$0].floatValue }
    }
}

enum InferenceCoreMLError: Error {
    case missingOutput(String)
}
