import CoreML
import Foundation

/// Core ML lapse forecaster: P(lapse within 15 s) from the last 20 s of `FeatureFrame`s, plus the last
/// 20 s of `ContentFeatures` when a lecture is playing.
///
/// Bundled models (Xcode compiles `Resources/Models/*.mlpackage` to `.mlmodelc` at the bundle root):
///   * `LapseForecaster`          inputs "features" (1,200,42) + "content" (1,20,11), raw values
///   * `LapseForecasterNoContent` input  "features" (1,200,42), raw values
///   output "lapse_probability" (1,1). Normalisation is baked into the models.
/// Uses the content model when content is supplied and that model exists, otherwise the features-only
/// model. If prediction fails it falls back to `HeuristicLapseForecaster` (source "heuristic").
/// The initialiser returns nil when neither model is in the bundle; use the heuristic then.
public final class CoreMLLapseForecaster: LapseForecaster, @unchecked Sendable {
    public static let contentModelName = "LapseForecaster"
    public static let featuresOnlyModelName = "LapseForecasterNoContent"
    public static let featuresInput = "features"
    public static let contentInput = "content"
    public static let outputName = "lapse_probability"

    private let contentModel: MLModel?
    private let featuresModel: MLModel?
    private let fallback: HeuristicLapseForecaster

    /// Guarded by `lock`: the pipeline actor observes outcomes while the main actor saves the threshold
    /// and toggles Demo Director.
    public var personalThreshold: PersonalThreshold { lock.withLock { _personalThreshold } }
    private var _personalThreshold: PersonalThreshold
    private let lock = NSLock()
    /// Sliding window for callers that push frames one at a time (`ingest` + `forecast(t:)`).
    public private(set) var buffer = ForecastWindowBuffer()
    /// DEMONSTRATION ONLY: Demo Director lowers the effective threshold.
    public var demoDirector: Bool {
        get { lock.withLock { _demoDirector } }
        set { lock.withLock { _demoDirector = newValue } }
    }
    private var _demoDirector = false
    /// Metadata written by ml/forecast/export_coreml.py (nil if the JSON is not bundled).
    public let metadata: Metadata?

    public struct Metadata: Codable, Sendable {
        public var synthetic: Bool?
        public var alert_threshold: Float?
        public var alert_rate: Float?
    }

    public var hasContentModel: Bool { contentModel != nil }
    public var hasFeaturesOnlyModel: Bool { featuresModel != nil }

    /// - Parameters:
    ///   - bundle: where the compiled `.mlmodelc` folders live (default `.main`).
    ///   - threshold: a restored personal threshold; default starts at the exported model's
    ///     validation threshold for a 10% alert rate (or 0.5).
    public init?(bundle: Bundle = .main, threshold: PersonalThreshold? = nil,
                 configuration: MLModelConfiguration = MLModelConfiguration()) {
        func load(_ name: String) -> MLModel? {
            guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else { return nil }
            return try? MLModel(contentsOf: url, configuration: configuration)
        }
        contentModel = load(Self.contentModelName)
        featuresModel = load(Self.featuresOnlyModelName)
        guard contentModel != nil || featuresModel != nil else { return nil }
        if let url = bundle.url(forResource: Self.contentModelName, withExtension: "json"),
           let data = try? Data(contentsOf: url) {
            metadata = try? JSONDecoder().decode(Metadata.self, from: data)
        } else {
            metadata = nil
        }
        let initial = threshold ?? PersonalThreshold(initial: metadata?.alert_threshold ?? 0.5,
                                                     targetAlertRate: metadata?.alert_rate ?? 0.10)
        _personalThreshold = initial
        fallback = HeuristicLapseForecaster(threshold: initial)
    }

    public var threshold: Float { lock.withLock { _personalThreshold.effective(demoDirector: _demoDirector) } }

    // MARK: Buffer convenience

    public func ingest(frame: FeatureFrame) { buffer.append(frame) }
    public func ingest(content: ContentFeatures) { buffer.append(content: content) }
    public func clearContent() { buffer.clearContent() }
    public func resetBuffer() { buffer.reset() }

    /// Forecast from the internal sliding window.
    public func forecast(t: Double) -> LapseRisk {
        forecast(frames: buffer.frames, content: buffer.content, t: t)
    }

    // MARK: LapseForecaster

    public func forecast(frames: [FeatureFrame], content: [ContentFeatures], t: Double) -> LapseRisk {
        let useContent = !content.isEmpty && contentModel != nil
        guard !frames.isEmpty, let model = useContent ? contentModel : (featuresModel ?? contentModel) else {
            return heuristic(frames: frames, content: content, t: t)
        }
        do {
            var inputs: [String: Any] = [
                Self.featuresInput: try Self.array(ForecastWindowBuffer.frameMatrix(frames),
                                                   rows: ForecastWindowBuffer.frameCount,
                                                   cols: ForecastWindowBuffer.featureCount)
            ]
            let modelNeedsContent = model === contentModel
            if modelNeedsContent {
                // Content model used without content only if the features-only model is missing.
                let c = ForecastWindowBuffer.contentMatrix(content.isEmpty ? [.none] : content) ?? []
                inputs[Self.contentInput] = try Self.array(c, rows: ForecastWindowBuffer.contentCount,
                                                           cols: ForecastWindowBuffer.contentFeatureCount)
            }
            let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
            guard let arr = out.featureValue(for: Self.outputName)?.multiArrayValue, arr.count > 0 else {
                return heuristic(frames: frames, content: content, t: t)
            }
            let p = min(1, max(0, arr[0].floatValue))
            guard p.isFinite else { return heuristic(frames: frames, content: content, t: t) }
            return LapseRisk(t: t, probability: p, threshold: threshold,
                             usedContent: modelNeedsContent && !content.isEmpty, source: "model")
        } catch {
            return heuristic(frames: frames, content: content, t: t)
        }
    }

    public func observe(prediction: LapseRisk, lapseOccurred: Bool) {
        lock.withLock { _personalThreshold.observe(probability: prediction.probability, lapseOccurred: lapseOccurred) }
    }

    /// Replace the threshold (e.g. restored from `SessionStore.modelWeights(key:)`).
    public func restoreThreshold(_ t: PersonalThreshold) { lock.withLock { _personalThreshold = t } }

    // MARK: Private

    private func heuristic(frames: [FeatureFrame], content: [ContentFeatures], t: Double) -> LapseRisk {
        fallback.personalThreshold = personalThreshold
        fallback.demoDirector = demoDirector
        return fallback.forecast(frames: frames, content: content, t: t)
    }

    private static func array(_ values: [Float], rows: Int, cols: Int) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, NSNumber(value: rows), NSNumber(value: cols)], dataType: .float32)
        let n = min(values.count, rows * cols)
        let ptr = a.dataPointer.bindMemory(to: Float.self, capacity: rows * cols)
        values.withUnsafeBufferPointer { src in
            for i in 0..<n { ptr[i] = src[i] }
        }
        if n < rows * cols { for i in n..<(rows * cols) { ptr[i] = 0 } }
        return a
    }
}
