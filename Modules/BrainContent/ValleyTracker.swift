import Foundation

/// Follows the lecture playhead through the predicted low-engagement segments ("valleys") of a
/// `ContentTimeline` and emits enter / leave / approaching events.
///
/// Predicted brain response to this content (average brain), never the student's own brain.
///
/// Typical use (Study Mode, 1 Hz or faster):
/// ```swift
/// let tracker = ValleyTracker(timeline: content, approachHorizon: 10)
/// for event in tracker.update(playhead: player.currentTime) {
///     switch event {
///     case .approaching(let v, let s): pulse Study Mode UI / consider a pre-emptive nudge (s seconds out)
///     case .entered(let v):            log SessionEvent.valleyEntered
///     case .left:                      break
///     }
/// }
/// ```
/// Not thread-safe: call from one actor (the session engine).
public final class ValleyTracker {
    public struct Valley: Equatable, Sendable {
        public let start: Double
        public let end: Double
        public init(start: Double, end: Double) { self.start = start; self.end = end }
        public var duration: Double { end - start }
        /// Half-open: `start <= t < end`.
        public func contains(_ t: Double) -> Bool { t >= start && t < end }
    }

    public enum Event: Equatable, Sendable {
        /// The next valley starts within `approachHorizon` seconds (fired once per valley per pass).
        case approaching(Valley, secondsUntil: Double)
        case entered(Valley)
        case left(Valley)
    }

    /// Sorted by start, non-overlapping.
    public let valleys: [Valley]
    /// "Valley approaching within N s" horizon for `update(playhead:)`.
    public var approachHorizon: Double

    private var currentIndex: Int?
    private var announced: Set<Int> = []
    private var lastPlayhead: Double?

    public init(valleys: [(start: Double, end: Double)], approachHorizon: Double = 10) {
        self.valleys = valleys.filter { $0.end > $0.start }
            .map { Valley(start: $0.start, end: $0.end) }
            .sorted { $0.start < $1.start }
        self.approachHorizon = approachHorizon
    }

    public convenience init(timeline: ContentTimeline, approachHorizon: Double = 10) {
        self.init(valleys: timeline.valleys(), approachHorizon: approachHorizon)
    }

    // MARK: Queries (stateless)

    public func valleyIndex(at t: Double) -> Int? { valleys.firstIndex { $0.contains(t) } }
    public func valley(at t: Double) -> Valley? { valleyIndex(at: t).map { valleys[$0] } }
    public func isInValley(at t: Double) -> Bool { valleyIndex(at: t) != nil }

    /// First valley starting strictly after `t`.
    public func nextValley(after t: Double) -> Valley? { valleys.first { $0.start > t } }

    /// Seconds until the next valley starts; 0 inside a valley; nil if none remain.
    public func timeToNextValley(at t: Double) -> Double? {
        if isInValley(at: t) { return 0 }
        return nextValley(after: t).map { $0.start - t }
    }

    /// True when a valley starts within `horizon` seconds (false while already inside one).
    public func isValleyApproaching(at t: Double, within horizon: Double) -> Bool {
        guard !isInValley(at: t), let next = nextValley(after: t) else { return false }
        return next.start - t <= horizon
    }

    // MARK: Event stream (stateful)

    /// Advance to `t` (seeking in either direction is fine). Returns events in order: left, entered, approaching.
    @discardableResult
    public func update(playhead t: Double) -> [Event] {
        if let last = lastPlayhead, t < last {
            // Seek backwards: valleys ahead of the playhead may be announced again.
            announced = announced.filter { valleys[$0].start <= t }
        }
        lastPlayhead = t

        var events: [Event] = []
        let inside = valleyIndex(at: t)
        if inside != currentIndex {
            if let old = currentIndex { events.append(.left(valleys[old])) }
            if let new = inside {
                events.append(.entered(valleys[new]))
                announced.insert(new)
            }
            currentIndex = inside
        }
        if let i = valleys.firstIndex(where: { $0.start > t }), !announced.contains(i) {
            let until = valleys[i].start - t
            if until <= approachHorizon {
                events.append(.approaching(valleys[i], secondsUntil: until))
                announced.insert(i)
            }
        }
        return events
    }

    public func reset() {
        currentIndex = nil
        announced = []
        lastPlayhead = nil
    }
}
