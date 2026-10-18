import Foundation

/// Thin `ContentTimeline` wrapper around the frozen `BrainTimeline` (Phase 0 contract).
/// Predicted brain response to this content (average brain), never the student's own brain.
public final class BrainContentTimeline: ContentTimeline {
    public let timeline: BrainTimeline

    public init(_ timeline: BrainTimeline) { self.timeline = timeline }

    /// Loads `<name>.brain.json` from the bundle root.
    public convenience init(named name: String, bundle: Bundle = .main) throws {
        self.init(try BrainTimeline.load(name, bundle: bundle))
    }

    public var lectureID: String { timeline.lectureID }
    public var isMock: Bool { timeline.mock ?? false }
    public var duration: Double { timeline.duration }

    public func activations(at t: Double) -> [Float] { timeline.activations(at: t) }

    public func networkValues(at t: Double) -> [Float] {
        BrainNetwork.allCases.map { Self.sample(timeline.networks[$0.key] ?? [], at: t * timeline.hz) }
    }

    public func engagement(at t: Double) -> Float { timeline.engagement(at: t) }
    public func driftRisk(at t: Double) -> Float { timeline.driftRisk(at: t) }

    public func valleys() -> [(start: Double, end: Double)] {
        (timeline.valleys ?? []).map { ($0.start, $0.end) }
    }

    public func upcomingValley(after t: Double, within horizon: Double = 30) -> (start: Double, end: Double)? {
        timeline.upcomingValley(after: t, within: horizon).map { ($0.start, $0.end) }
    }

    public func features(at t: Double) -> ContentFeatures {
        let inside = valleys().contains { t >= $0.start && t < $0.end }
        let next = valleys().first { $0.start >= t }.map { Float($0.start - t) } ?? 60
        return ContentFeatures(networks: networkValues(at: t), engagement: engagement(at: t),
                               driftRisk: driftRisk(at: t), secondsToValley: inside ? 0 : min(60, next),
                               inValley: inside)
    }

    private static func sample(_ s: [Float], at x: Double) -> Float {
        guard !s.isEmpty else { return 0 }
        let c = max(0, min(Double(s.count - 1), x))
        let i = Int(c), j = min(i + 1, s.count - 1), f = Float(c - Double(i))
        return s[i] * (1 - f) + s[j] * f
    }
}
