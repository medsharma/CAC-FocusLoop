import Foundation
@testable import FocusLoop

/// Hand-built fixtures for the Insights tests (no files, no randomness).
enum InsightsFixtures {
    static var utc: Calendar {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c
    }

    static func date(_ y: Int, _ m: Int, _ d: Int, _ h: Int, _ min: Int = 0) -> Date {
        utc.date(from: DateComponents(year: y, month: m, day: d, hour: h, minute: min))!
    }

    static func probs(_ s: AttentionState, pFocused: Float? = nil) -> [Float] {
        if let p = pFocused { return [p, (1 - p) / 3, (1 - p) / 3, (1 - p) / 3] }
        var v: [Float] = [0.05, 0.05, 0.05, 0.05]
        v[s.index] = 0.85
        return v
    }

    /// 1 Hz ticks from a run-length list of states.
    static func ticks(_ runs: [(AttentionState, Int)], heartRate: Float? = nil, quality: Float? = 0.9) -> [SessionTick] {
        var out: [SessionTick] = []
        var t = 0.0
        for (state, n) in runs {
            for _ in 0..<n {
                out.append(SessionTick(t: t, state: state, probabilities: probs(state), lapseRisk: 0.1,
                                       heartRate: heartRate, heartRateQuality: heartRate == nil ? nil : quality,
                                       playhead: nil, predictedEngagement: nil, predictedDriftRisk: nil))
                t += 1
            }
        }
        return out
    }

    static func session(_ runs: [(AttentionState, Int)], start: Date = date(2026, 10, 20, 10),
                        heartRate: Float? = nil, events: [SessionEvent] = [],
                        mock: Bool = false, replay: Bool = false) -> SessionRecord {
        let ts = ticks(runs, heartRate: heartRate)
        return SessionRecord(startedAt: start, duration: Double(ts.count), contentIsMock: mock, isReplay: replay,
                             ticks: ts, events: events)
    }

    static func summary(sessions: Int = 4, minutes: Double = 120, focused: Double = 0.7, drop: Double? = 12,
                        refocus: Double? = 8.4, bestHours: [Int] = [16, 9], hr: Double? = 68, hrTrend: Double? = -0.8,
                        corr: Double? = 0.42, mock: Bool = false, replay: Bool = false,
                        effects: [ArmEffect] = []) -> AggregateSummary {
        let rest = (1 - focused) / 3
        return AggregateSummary(sessionCount: sessions, totalMinutes: minutes, focusedFraction: focused,
                                stateFractions: [focused, rest, rest, rest], typicalDropMinute: drop,
                                medianRefocusSeconds: refocus, bestHours: bestHours, restingHeartRate: hr,
                                heartRateTrendPerSession: hrTrend, contentDriftCorrelation: corr,
                                contentIsMock: mock, isReplay: replay, armEffects: effects)
    }

    static let crossingZero: [ArmEffect] = [
        ArmEffect(arm: .haptic, policy: .reactive, effect: 0.12, lower: -0.05, upper: 0.29, level: 0.9, n: 14),
        ArmEffect(arm: .chime, policy: .reactive, effect: -0.02, lower: -0.2, upper: 0.16, level: 0.9, n: 11),
        ArmEffect(arm: .noNudge, policy: .reactive, effect: 0, lower: 0, upper: 0, level: 0.9, n: 6),
    ]

    static let hapticHelps: [ArmEffect] = [
        ArmEffect(arm: .haptic, policy: .reactive, effect: 0.18, lower: 0.05, upper: 0.31, level: 0.9, n: 24),
        ArmEffect(arm: .chime, policy: .reactive, effect: 0.04, lower: -0.1, upper: 0.18, level: 0.9, n: 19),
    ]

    /// Sentences in an insight (split on ". ", "! ", "? " and the final terminator).
    static func sentenceCount(_ text: String) -> Int {
        var body = text
        if let r = body.range(of: #"\s*\((based on MOCK|SYNTHETIC)[^)]*\)$"#, options: .regularExpression) {
            body.removeSubrange(r)
        }
        // A terminator only ends a sentence when followed by whitespace or end of text.
        var count = 0
        var idx = body.startIndex
        while idx < body.endIndex {
            let c = body[idx]
            let next = body.index(after: idx)
            if ".!?".contains(c), next == body.endIndex || body[next] == " " { count += 1 }
            idx = next
        }
        return max(1, count)
    }
}
