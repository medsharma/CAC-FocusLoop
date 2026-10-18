import Foundation

/// Sliding window for the lapse forecaster: the last 20 s of 10 Hz `FeatureFrame`s (200) and the last
/// 20 s of 1 Hz `ContentFeatures` (20). Also turns arbitrary-length inputs into the exact fixed-size,
/// row-major matrices the Core ML model expects (same padding rule as `ml/forecast/data.py`:
/// too short -> repeat the OLDEST available entry at the front; too long -> keep the newest).
public struct ForecastWindowBuffer: Sendable {
    public static let frameCount = Int(FeatureFrame.forecastWindowSeconds * FeatureFrame.rateHz)   // 200
    public static let contentCount = Int(FeatureFrame.forecastWindowSeconds)                       // 20
    public static let featureCount = FeatureFrame.featureNames.count                               // 42
    public static let contentFeatureCount = ContentFeatures.featureNames.count                     // 11

    public private(set) var frames: [FeatureFrame] = []
    public private(set) var content: [ContentFeatures] = []

    public init() {}

    /// True once a full 20 s of frames has been seen.
    public var isWarm: Bool { frames.count >= Self.frameCount }

    public mutating func append(_ frame: FeatureFrame) {
        frames.append(frame)
        if frames.count > Self.frameCount { frames.removeFirst(frames.count - Self.frameCount) }
    }

    /// Append the content features for one playhead second (call at 1 Hz while a lecture plays).
    public mutating func append(content c: ContentFeatures) {
        content.append(c)
        if content.count > Self.contentCount { content.removeFirst(content.count - Self.contentCount) }
    }

    /// Drop content (e.g. the lecture stopped), so the features-only model is used.
    public mutating func clearContent() { content.removeAll() }

    public mutating func reset() { frames.removeAll(); content.removeAll() }

    /// Exactly `frameCount` frames, oldest first. Empty input -> empty output.
    public static func fixedFrames(_ frames: [FeatureFrame]) -> [FeatureFrame] {
        fixed(frames, count: frameCount)
    }

    /// Exactly `contentCount` entries, oldest first. Empty input -> empty output (no content).
    public static func fixedContent(_ content: [ContentFeatures]) -> [ContentFeatures] {
        fixed(content, count: contentCount)
    }

    /// Row-major [200 x 42] raw feature values (zeros if `frames` is empty).
    public static func frameMatrix(_ frames: [FeatureFrame]) -> [Float] {
        let f = fixedFrames(frames)
        guard !f.isEmpty else { return Array(repeating: 0, count: frameCount * featureCount) }
        return f.flatMap(\.vector)
    }

    /// Row-major [20 x 11] raw content values, or nil when there is no content.
    public static func contentMatrix(_ content: [ContentFeatures]) -> [Float]? {
        let c = fixedContent(content)
        guard !c.isEmpty else { return nil }
        return c.flatMap(\.vector)
    }

    private static func fixed<T>(_ xs: [T], count: Int) -> [T] {
        guard let first = xs.first else { return [] }
        if xs.count >= count { return Array(xs.suffix(count)) }
        return Array(repeating: first, count: count - xs.count) + xs
    }
}
