import Foundation

/// Chart-ready data for the weekly "Attention Profile" screen (Swift Charts). Numbers only.
/// If `contentIsMock` / `isReplay` is true, the UI must show MOCK / SYNTHETIC badges.
public struct AttentionProfileData: Equatable, Sendable {

    public struct DayPoint: Equatable, Sendable, Identifiable {
        /// Start of the calendar day.
        public var date: Date
        public var focusedMinutes: Double
        public var totalMinutes: Double
        public var sessionCount: Int
        public var id: Date { date }
    }

    public struct HourPoint: Equatable, Sendable, Identifiable {
        /// Session start hour, 0-23.
        public var hour: Int
        public var focusedFraction: Double
        public var minutes: Double
        public var sessionCount: Int
        /// True when this hour has enough data to rank (>= 2 sessions or >= 10 min).
        public var isEligible: Bool
        public var id: Int { hour }
    }

    /// One value per session (median refocus seconds, resting HR...), in start order.
    public struct SessionPoint: Equatable, Sendable, Identifiable {
        public var sessionID: UUID
        public var date: Date
        public var value: Double
        public var id: UUID { sessionID }
    }

    /// Caller-supplied history of trial effect estimates (e.g. one snapshot saved per session).
    public struct ArmEffectSnapshot: Codable, Equatable, Sendable {
        public var date: Date
        public var effects: [ArmEffect]
        public init(date: Date, effects: [ArmEffect]) { self.date = date; self.effects = effects }
    }

    /// One arm's effect estimate at one point in time ("intervals that narrow across sessions").
    public struct ArmEffectPoint: Equatable, Sendable, Identifiable {
        public var date: Date
        public var arm: NudgeArm
        public var policy: NudgePolicyKind
        public var effect: Double
        public var lower: Double
        public var upper: Double
        public var level: Double
        public var n: Int
        public var id: String { "\(policy.rawValue).\(arm.rawValue).\(date.timeIntervalSinceReferenceDate)" }
    }

    public var periodStart: Date
    public var periodEnd: Date
    /// Aggregate over the sessions in the period (input to the coach).
    public var summary: AggregateSummary
    public var dailyFocusedMinutes: [DayPoint]
    public var hourlyFocus: [HourPoint]
    public var refocusTrend: [SessionPoint]
    public var heartRateTrend: [SessionPoint]
    public var armEffectHistory: [ArmEffectPoint]

    public var contentIsMock: Bool { summary.contentIsMock }
    public var isReplay: Bool { summary.isReplay }
}

extension AttentionProfileBuilder {

    /// Weekly (by default) profile for charts. Sessions outside `[now - days, now]` are ignored.
    /// - Parameters:
    ///   - armEffectHistory: saved effect snapshots, oldest first. When empty, `currentArmEffects`
    ///     (if any) becomes a single snapshot at `now`.
    public static func profile(sessions: [SessionRecord],
                               currentArmEffects: [ArmEffect] = [],
                               armEffectHistory: [AttentionProfileData.ArmEffectSnapshot] = [],
                               now: Date = Date(), days: Int = 7,
                               calendar: Calendar = .current, config: Config = Config()) -> AttentionProfileData {
        let today = calendar.startOfDay(for: now)
        let start = calendar.date(byAdding: .day, value: -(max(1, days) - 1), to: today) ?? today
        let inPeriod = sessions.filter { $0.startedAt >= start && $0.startedAt <= now }
            .sorted { $0.startedAt < $1.startedAt }

        // Daily focused minutes (every day in the window, zeros included).
        var daily: [AttentionProfileData.DayPoint] = []
        for d in 0..<max(1, days) {
            guard let day = calendar.date(byAdding: .day, value: d, to: start) else { continue }
            let ss = inPeriod.filter { calendar.isDate($0.startedAt, inSameDayAs: day) }
            let focusedTicks = ss.reduce(0) { $0 + $1.ticks.filter { $0.state == .focused }.count }
            daily.append(.init(date: day, focusedMinutes: Double(focusedTicks) / 60,
                               totalMinutes: ss.reduce(0) { $0 + sessionSeconds($1) } / 60,
                               sessionCount: ss.count))
        }

        let hourly = hourStats(inPeriod, calendar: calendar).map {
            AttentionProfileData.HourPoint(
                hour: $0.hour, focusedFraction: $0.focusedFraction, minutes: $0.seconds / 60,
                sessionCount: $0.sessionCount,
                isEligible: $0.sessionCount >= config.bestHoursMinSessions || $0.seconds >= config.bestHoursMinSeconds)
        }

        let refocus = inPeriod.compactMap { s -> AttentionProfileData.SessionPoint? in
            guard let m = median(refocusTimes(in: s)) else { return nil }
            return .init(sessionID: s.id, date: s.startedAt, value: m)
        }
        let hr = inPeriod.compactMap { s -> AttentionProfileData.SessionPoint? in
            guard let r = restingHeartRate(s.ticks, config: config) else { return nil }
            return .init(sessionID: s.id, date: s.startedAt, value: r)
        }

        var snapshots = armEffectHistory.sorted { $0.date < $1.date }
        if snapshots.isEmpty, !currentArmEffects.isEmpty {
            snapshots = [.init(date: now, effects: currentArmEffects)]
        }
        let effectPoints = snapshots.flatMap { snap in
            snap.effects.filter { $0.arm != .noNudge }.map {
                AttentionProfileData.ArmEffectPoint(date: snap.date, arm: $0.arm, policy: $0.policy, effect: $0.effect,
                                                    lower: $0.lower, upper: $0.upper, level: $0.level, n: $0.n)
            }
        }
        let latestEffects = currentArmEffects.isEmpty ? (snapshots.last?.effects ?? []) : currentArmEffects

        return AttentionProfileData(
            periodStart: start, periodEnd: now,
            summary: summary(sessions: inPeriod, armEffects: latestEffects, calendar: calendar, config: config),
            dailyFocusedMinutes: daily, hourlyFocus: hourly, refocusTrend: refocus, heartRateTrend: hr,
            armEffectHistory: effectPoints)
    }
}
