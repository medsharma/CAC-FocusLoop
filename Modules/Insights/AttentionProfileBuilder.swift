import Foundation

/// Pure functions that turn persisted `SessionRecord`s into the aggregated numbers the AI coach
/// (and the History / Attention Profile screens) consume. Numbers only; nothing here sees raw
/// sensor data, video or user text.
///
/// Conventions:
/// * `SessionTick`s are 1 Hz, `t` = seconds since the session started.
/// * A "lapse" is any non-`focused` state. A *sustained* lapse lasts at least
///   `Config.sustainedLapseSeconds`.
/// * `contentIsMock` / `isReplay` propagate: true if ANY contributing session is MOCK / SYNTHETIC.
public enum AttentionProfileBuilder {

    public struct Config: Equatable, Sendable {
        /// Minimum length of a non-focused run that counts as the session's drop point.
        public var sustainedLapseSeconds: Double = 5
        /// Heart-rate ticks below this rPPG quality are ignored (ticks with no quality value are kept).
        public var minHeartRateQuality: Float = 0.5
        /// Plausible BPM range; anything outside is treated as a bad reading.
        public var heartRateRange: ClosedRange<Float> = 40...180
        /// A start hour is eligible for "best hours" with at least this many sessions...
        public var bestHoursMinSessions = 2
        /// ...or at least this many seconds of ticks.
        public var bestHoursMinSeconds: Double = 600
        public var maxBestHours = 3
        /// Minimum Study Mode ticks before a content-drift correlation is reported.
        public var minCorrelationPoints = 10
        public init() {}
    }

    // MARK: - Aggregate summary

    /// Aggregated summary over many sessions (weekly profile, History).
    public static func summary(sessions: [SessionRecord], armEffects: [ArmEffect] = [],
                               calendar: Calendar = .current, config: Config = Config()) -> AggregateSummary {
        let fractions = stateFractions(sessions)
        return AggregateSummary(
            sessionCount: sessions.count,
            totalMinutes: sessions.reduce(0) { $0 + sessionSeconds($1) } / 60,
            focusedFraction: fractions[AttentionState.focused.index],
            stateFractions: fractions,
            typicalDropMinute: typicalDropMinute(sessions, config: config),
            medianRefocusSeconds: median(sessions.flatMap(refocusTimes(in:))),
            bestHours: bestHours(sessions, calendar: calendar, config: config),
            restingHeartRate: restingHeartRate(sessions.flatMap(\.ticks), config: config),
            heartRateTrendPerSession: heartRateTrendPerSession(sessions, config: config),
            contentDriftCorrelation: contentDriftCorrelation(sessions, config: config),
            contentIsMock: sessions.contains { $0.contentIsMock },
            isReplay: sessions.contains { $0.isReplay },
            armEffects: armEffects)
    }

    /// Summary of one session, for the session report. (`heartRateTrendPerSession` is nil: it needs 2+ sessions.)
    public static func summary(for session: SessionRecord, armEffects: [ArmEffect] = [],
                               calendar: Calendar = .current, config: Config = Config()) -> AggregateSummary {
        summary(sessions: [session], armEffects: armEffects, calendar: calendar, config: config)
    }

    // MARK: - Components (public for tests and for the UI/Persistence layers)

    /// Seconds of data in a session: `duration` if set, else the tick span at 1 Hz.
    public static func sessionSeconds(_ s: SessionRecord) -> Double {
        if s.duration > 0 { return s.duration }
        return Double(s.ticks.count)
    }

    /// Fraction of ticks per state, `AttentionState.allCases` order. All zeros when there are no ticks.
    public static func stateFractions(_ sessions: [SessionRecord]) -> [Double] {
        var counts = [Double](repeating: 0, count: AttentionState.allCases.count)
        var total = 0.0
        for s in sessions {
            for tick in s.ticks {
                counts[tick.state.index] += 1
                total += 1
            }
        }
        guard total > 0 else { return counts }
        return counts.map { $0 / total }
    }

    /// Session time (s) at which the earliest sustained lapse (non-focused run >= `minSeconds`) begins.
    /// Falls back to the earliest `.lapseDetected` event when the session has no ticks.
    public static func firstSustainedLapseTime(in session: SessionRecord, minSeconds: Double) -> Double? {
        let ticks = session.ticks.sorted { $0.t < $1.t }
        if ticks.isEmpty {
            return session.events.compactMap { e -> Double? in
                if case let .lapseDetected(t, _) = e { return t }
                return nil
            }.min()
        }
        var i = 0
        while i < ticks.count {
            guard ticks[i].state != .focused else { i += 1; continue }
            let start = ticks[i].t
            var j = i
            while j + 1 < ticks.count, ticks[j + 1].state != .focused { j += 1 }
            let end = j + 1 < ticks.count ? ticks[j + 1].t : ticks[j].t + 1
            if end - start >= minSeconds { return max(0, start) }
            i = j + 1
        }
        return nil
    }

    /// The session minute in which the first sustained lapse most often begins.
    /// Mode of the per-session minute bins when one repeats; otherwise their median.
    public static func typicalDropMinute(_ sessions: [SessionRecord], config: Config = Config()) -> Double? {
        let minutes = sessions.compactMap { firstSustainedLapseTime(in: $0, minSeconds: config.sustainedLapseSeconds) }
            .map { floor($0 / 60) }
        guard !minutes.isEmpty else { return nil }
        var counts: [Double: Int] = [:]
        for m in minutes { counts[m, default: 0] += 1 }
        let maxCount = counts.values.max() ?? 0
        if maxCount >= 2 {
            return counts.filter { $0.value == maxCount }.keys.min()
        }
        return median(minutes)
    }

    /// Refocus durations (s) from `.refocused` events, in session order.
    public static func refocusTimes(in session: SessionRecord) -> [Double] {
        session.events.compactMap { e -> (Double, Double)? in
            if case let .refocused(t, after) = e, after.isFinite, after >= 0 { return (t, after) }
            return nil
        }.sorted { $0.0 < $1.0 }.map(\.1)
    }

    public struct HourStat: Equatable, Sendable {
        public var hour: Int
        public var sessionCount: Int
        public var seconds: Double
        public var focusedFraction: Double
    }

    /// Focused fraction grouped by the session's start hour (0-23), hours with ticks only, ascending.
    public static func hourStats(_ sessions: [SessionRecord], calendar: Calendar = .current) -> [HourStat] {
        var bins: [Int: (sessions: Int, ticks: Double, focused: Double)] = [:]
        for s in sessions where !s.ticks.isEmpty {
            let h = calendar.component(.hour, from: s.startedAt)
            var b = bins[h] ?? (0, 0, 0)
            b.sessions += 1
            b.ticks += Double(s.ticks.count)
            b.focused += Double(s.ticks.filter { $0.state == .focused }.count)
            bins[h] = b
        }
        return bins.keys.sorted().map { h in
            let b = bins[h]!
            return HourStat(hour: h, sessionCount: b.sessions, seconds: b.ticks, focusedFraction: b.focused / b.ticks)
        }
    }

    /// Start hours with enough data (>= 2 sessions or >= 10 min), best focused fraction first.
    public static func bestHours(_ sessions: [SessionRecord], calendar: Calendar = .current,
                                 config: Config = Config()) -> [Int] {
        hourStats(sessions, calendar: calendar)
            .filter { $0.sessionCount >= config.bestHoursMinSessions || $0.seconds >= config.bestHoursMinSeconds }
            .sorted { $0.focusedFraction != $1.focusedFraction ? $0.focusedFraction > $1.focusedFraction : $0.hour < $1.hour }
            .prefix(config.maxBestHours)
            .map(\.hour)
    }

    /// Median heart rate over good-quality ticks in the `focused` state.
    public static func restingHeartRate(_ ticks: [SessionTick], config: Config = Config()) -> Double? {
        median(ticks.compactMap { tick -> Double? in
            guard tick.state == .focused, let bpm = tick.heartRate, bpm.isFinite,
                  config.heartRateRange.contains(bpm) else { return nil }
            if let q = tick.heartRateQuality, q < config.minHeartRateQuality { return nil }
            return Double(bpm)
        })
    }

    /// Least-squares slope (BPM per session) of per-session resting HR, sessions in start order.
    /// Needs 2+ sessions with heart-rate data.
    public static func heartRateTrendPerSession(_ sessions: [SessionRecord], config: Config = Config()) -> Double? {
        let ys = sessions.sorted { $0.startedAt < $1.startedAt }
            .compactMap { restingHeartRate($0.ticks, config: config) }
        guard ys.count >= 2 else { return nil }
        return slope(xs: ys.indices.map(Double.init), ys: ys)
    }

    /// Pearson r between the lecture's predicted drift risk and actual 1 - P(focused), over Study Mode ticks.
    public static func contentDriftCorrelation(_ sessions: [SessionRecord], config: Config = Config()) -> Double? {
        var xs: [Double] = []
        var ys: [Double] = []
        for s in sessions {
            for tick in s.ticks {
                guard let drift = tick.predictedDriftRisk, drift.isFinite else { continue }
                xs.append(Double(drift))
                ys.append(1 - pFocused(tick))
            }
        }
        guard xs.count >= config.minCorrelationPoints else { return nil }
        return pearson(xs, ys)
    }

    /// P(focused) for a tick (falls back to the hard state if probabilities are missing).
    public static func pFocused(_ tick: SessionTick) -> Double {
        let i = AttentionState.focused.index
        if i < tick.probabilities.count, tick.probabilities[i].isFinite {
            return Double(min(1, max(0, tick.probabilities[i])))
        }
        return tick.state == .focused ? 1 : 0
    }

    // MARK: - Statistics helpers

    public static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let s = values.sorted()
        let m = s.count / 2
        return s.count % 2 == 1 ? s[m] : (s[m - 1] + s[m]) / 2
    }

    /// Pearson correlation; nil if fewer than 2 points or either series is constant.
    public static func pearson(_ xs: [Double], _ ys: [Double]) -> Double? {
        let n = min(xs.count, ys.count)
        guard n >= 2 else { return nil }
        let mx = xs.prefix(n).reduce(0, +) / Double(n)
        let my = ys.prefix(n).reduce(0, +) / Double(n)
        var sxy = 0.0, sxx = 0.0, syy = 0.0
        for k in 0..<n {
            let dx = xs[k] - mx, dy = ys[k] - my
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy
        }
        guard sxx > 1e-12, syy > 1e-12 else { return nil }
        return max(-1, min(1, sxy / (sxx * syy).squareRoot()))
    }

    /// Ordinary least-squares slope; nil if fewer than 2 points or x is constant.
    public static func slope(xs: [Double], ys: [Double]) -> Double? {
        let n = min(xs.count, ys.count)
        guard n >= 2 else { return nil }
        let mx = xs.prefix(n).reduce(0, +) / Double(n)
        let my = ys.prefix(n).reduce(0, +) / Double(n)
        var sxy = 0.0, sxx = 0.0
        for k in 0..<n {
            sxy += (xs[k] - mx) * (ys[k] - my)
            sxx += (xs[k] - mx) * (xs[k] - mx)
        }
        guard sxx > 1e-12 else { return nil }
        return sxy / sxx
    }
}
