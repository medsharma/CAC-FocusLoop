import Foundation
import SwiftData

/// Lightweight per-session row for History / Attention Profile lists (no blob decoding).
public struct SessionHistorySummary: Equatable, Sendable, Identifiable {
    public var id: UUID
    public var startedAt: Date
    public var duration: Double
    public var focusedSeconds: Double
    public var lectureID: String?
    public var contentIsMock: Bool
    public var isReplay: Bool
    public var focusedFraction: Double { duration > 0 ? min(1, focusedSeconds / duration) : 0 }
}

/// Focused minutes on one calendar day (start of day in the given calendar).
public struct DailyFocus: Equatable, Sendable, Identifiable {
    public var day: Date
    public var focusedMinutes: Double
    public var sessionCount: Int
    /// True if any of that day's sessions ran on the SYNTHETIC replay stream.
    public var includesSynthetic: Bool
    public var id: Date { day }
}

/// History queries over `StoredSession` summary columns.
@MainActor
public struct SessionHistoryQueries {
    public let context: ModelContext

    public init(context: ModelContext) { self.context = context }
    public init(store: SwiftDataSessionStore) { self.context = store.context }

    /// Most recent `n` sessions, newest first.
    public func lastSessions(_ n: Int) throws -> [SessionHistorySummary] {
        var d = FetchDescriptor<StoredSession>(sortBy: [SortDescriptor(\.startedAt, order: .reverse)])
        d.fetchLimit = max(0, n)
        return try context.fetch(d).map(Self.summary)
    }

    /// Full records of the most recent `n` sessions (decodes blobs), newest first.
    public func lastRecords(_ n: Int) throws -> [SessionRecord] {
        var d = FetchDescriptor<StoredSession>(sortBy: [SortDescriptor(\.startedAt, order: .reverse)])
        d.fetchLimit = max(0, n)
        return try context.fetch(d).map(SwiftDataSessionStore.record(from:))
    }

    /// Sessions started in the calendar week containing `now`, newest first.
    public func sessionsThisWeek(now: Date = Date(), calendar: Calendar = .current) throws -> [SessionHistorySummary] {
        guard let week = calendar.dateInterval(of: .weekOfYear, for: now) else { return [] }
        return try sessions(from: week.start, to: week.end)
    }

    /// Sessions with `start <= startedAt < end`, newest first.
    public func sessions(from start: Date, to end: Date) throws -> [SessionHistorySummary] {
        let d = FetchDescriptor<StoredSession>(predicate: #Predicate { $0.startedAt >= start && $0.startedAt < end },
                                               sortBy: [SortDescriptor(\.startedAt, order: .reverse)])
        return try context.fetch(d).map(Self.summary)
    }

    /// Focused minutes per day for the last `days` days ending today (oldest first, zero-filled).
    public func focusedMinutesPerDay(days: Int = 7, now: Date = Date(),
                                     calendar: Calendar = .current) throws -> [DailyFocus] {
        guard days > 0 else { return [] }
        let today = calendar.startOfDay(for: now)
        guard let first = calendar.date(byAdding: .day, value: -(days - 1), to: today),
              let end = calendar.date(byAdding: .day, value: 1, to: today) else { return [] }
        var buckets: [Date: DailyFocus] = [:]
        for i in 0..<days {
            if let day = calendar.date(byAdding: .day, value: i, to: first) {
                buckets[day] = DailyFocus(day: day, focusedMinutes: 0, sessionCount: 0, includesSynthetic: false)
            }
        }
        for s in try sessions(from: first, to: end) {
            let day = calendar.startOfDay(for: s.startedAt)
            buckets[day]?.focusedMinutes += s.focusedSeconds / 60
            buckets[day]?.sessionCount += 1
            if s.isReplay { buckets[day]?.includesSynthetic = true }
        }
        return buckets.values.sorted { $0.day < $1.day }
    }

    static func summary(_ s: StoredSession) -> SessionHistorySummary {
        SessionHistorySummary(id: s.id, startedAt: s.startedAt, duration: s.duration, focusedSeconds: s.focusedSeconds,
                              lectureID: s.lectureID, contentIsMock: s.contentIsMock, isReplay: s.isReplay)
    }
}
