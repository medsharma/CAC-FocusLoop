import XCTest
@testable import FocusLoop

final class AttentionProfileBuilderTests: XCTestCase {
    typealias F = InsightsFixtures
    let cal = InsightsFixtures.utc

    func testStateFractionsAndTotals() {
        let s = F.session([(.focused, 30), (.distracted, 15), (.drowsy, 10), (.stressed, 5)])
        let sum = AttentionProfileBuilder.summary(for: s, calendar: cal)
        XCTAssertEqual(sum.sessionCount, 1)
        XCTAssertEqual(sum.totalMinutes, 1, accuracy: 1e-9)
        XCTAssertEqual(sum.focusedFraction, 0.5, accuracy: 1e-9)
        XCTAssertEqual(sum.stateFractions.count, 4)
        XCTAssertEqual(sum.stateFractions[1], 0.25, accuracy: 1e-9)
        XCTAssertEqual(sum.stateFractions[2], 10.0 / 60, accuracy: 1e-9)
        XCTAssertEqual(sum.stateFractions[3], 5.0 / 60, accuracy: 1e-9)
        XCTAssertEqual(sum.stateFractions.reduce(0, +), 1, accuracy: 1e-9)
        XCTAssertNil(sum.heartRateTrendPerSession, "trend needs 2+ sessions")
    }

    func testEmptyInputIsSafe() {
        let sum = AttentionProfileBuilder.summary(sessions: [], calendar: cal)
        XCTAssertEqual(sum.sessionCount, 0)
        XCTAssertEqual(sum.focusedFraction, 0)
        XCTAssertNil(sum.typicalDropMinute)
        XCTAssertNil(sum.medianRefocusSeconds)
        XCTAssertTrue(sum.bestHours.isEmpty)
        XCTAssertFalse(sum.contentIsMock)
        XCTAssertFalse(sum.isReplay)
    }

    func testFirstSustainedLapseIgnoresShortBlips() {
        // 2 s blip at t=30, sustained 10 s lapse (distracted then drowsy) from t=125.
        let s = F.session([(.focused, 30), (.distracted, 2), (.focused, 93), (.distracted, 6), (.drowsy, 4), (.focused, 20)])
        XCTAssertEqual(AttentionProfileBuilder.firstSustainedLapseTime(in: s, minSeconds: 5), 125)
        XCTAssertEqual(AttentionProfileBuilder.typicalDropMinute([s]), 2)
    }

    func testLapseAtEndOfSessionCounts() {
        let s = F.session([(.focused, 70), (.drowsy, 5)])
        XCTAssertEqual(AttentionProfileBuilder.firstSustainedLapseTime(in: s, minSeconds: 5), 70)
    }

    func testDropMinuteFallsBackToLapseEvents() {
        let s = SessionRecord(duration: 600, events: [.lapseDetected(t: 400, state: .distracted),
                                                      .lapseDetected(t: 250, state: .drowsy)])
        XCTAssertEqual(AttentionProfileBuilder.typicalDropMinute([s]), 4)
    }

    func testTypicalDropMinuteIsModeThenMedian() {
        func lapseAt(_ sec: Int) -> SessionRecord { F.session([(.focused, sec), (.distracted, 10), (.focused, 5)]) }
        XCTAssertEqual(AttentionProfileBuilder.typicalDropMinute([lapseAt(300), lapseAt(330), lapseAt(540)]), 5)
        // No repeated minute: median of 2, 6, 10.
        XCTAssertEqual(AttentionProfileBuilder.typicalDropMinute([lapseAt(120), lapseAt(360), lapseAt(600)]), 6)
        XCTAssertNil(AttentionProfileBuilder.typicalDropMinute([F.session([(.focused, 100)])]))
    }

    func testMedianRefocus() {
        let a = F.session([(.focused, 10)], events: [.refocused(t: 50, afterSeconds: 10), .refocused(t: 20, afterSeconds: 4),
                                                     .stateChanged(t: 3, from: .focused, to: .distracted)])
        let b = F.session([(.focused, 10)], events: [.refocused(t: 9, afterSeconds: 6), .refocused(t: 70, afterSeconds: 8)])
        XCTAssertEqual(AttentionProfileBuilder.refocusTimes(in: a), [4, 10])
        XCTAssertEqual(AttentionProfileBuilder.summary(sessions: [a, b], calendar: cal).medianRefocusSeconds, 7)
        XCTAssertEqual(AttentionProfileBuilder.summary(sessions: [b], calendar: cal).medianRefocusSeconds, 7)
        XCTAssertEqual(AttentionProfileBuilder.summary(sessions: [a, b, F.session([(.focused, 5)],
            events: [.refocused(t: 1, afterSeconds: 30)])], calendar: cal).medianRefocusSeconds, 8)
    }

    func testBestHoursRespectsDataThreshold() {
        let nine1 = F.session([(.focused, 240), (.distracted, 60)], start: F.date(2026, 10, 20, 9))     // 80%, 5 min
        let nine2 = F.session([(.focused, 240), (.distracted, 60)], start: F.date(2026, 10, 21, 9, 30)) // 80%, 5 min
        let three = F.session([(.focused, 648), (.drowsy, 72)], start: F.date(2026, 10, 20, 15))        // 90%, 12 min
        let eight = F.session([(.focused, 300)], start: F.date(2026, 10, 20, 20))                       // 100% but 5 min, 1 session
        let eleven = F.session([(.focused, 300), (.distracted, 400)], start: F.date(2026, 10, 22, 11))   // 43%, 11.7 min
        let best = AttentionProfileBuilder.bestHours([nine1, nine2, three, eight, eleven], calendar: cal)
        XCTAssertEqual(best, [15, 9, 11])
        let stats = AttentionProfileBuilder.hourStats([nine1, nine2], calendar: cal)
        XCTAssertEqual(stats.count, 1)
        XCTAssertEqual(stats[0].sessionCount, 2)
        XCTAssertEqual(stats[0].focusedFraction, 0.8, accuracy: 1e-9)
    }

    func testRestingHeartRateFiltersQualityStateAndRange() {
        func tick(_ s: AttentionState, _ hr: Float?, _ q: Float?) -> SessionTick {
            SessionTick(t: 0, state: s, probabilities: F.probs(s), lapseRisk: 0, heartRate: hr, heartRateQuality: q,
                        playhead: nil, predictedEngagement: nil, predictedDriftRisk: nil)
        }
        let ticks = [tick(.focused, 60, 0.9), tick(.focused, 62, 0.8), tick(.focused, 64, nil),
                     tick(.focused, 120, 0.1),       // low quality
                     tick(.distracted, 90, 0.9),     // not focused
                     tick(.focused, 250, 0.9),       // implausible
                     tick(.focused, nil, 0.9)]
        XCTAssertEqual(AttentionProfileBuilder.restingHeartRate(ticks), 62)
        XCTAssertNil(AttentionProfileBuilder.restingHeartRate([tick(.stressed, 80, 0.9)]))
    }

    func testHeartRateSlopePerSession() {
        // Out of order on purpose: slope uses start order.
        let s1 = F.session([(.focused, 20)], start: F.date(2026, 10, 20, 9), heartRate: 70)
        let s2 = F.session([(.focused, 20)], start: F.date(2026, 10, 21, 9), heartRate: 68)
        let s3 = F.session([(.focused, 20)], start: F.date(2026, 10, 22, 9), heartRate: 66)
        let noHR = F.session([(.focused, 20)], start: F.date(2026, 10, 23, 9))
        let sum = AttentionProfileBuilder.summary(sessions: [s3, s1, noHR, s2], calendar: cal)
        XCTAssertEqual(sum.heartRateTrendPerSession!, -2, accuracy: 1e-9)
        XCTAssertEqual(sum.restingHeartRate, 68)
        XCTAssertNil(AttentionProfileBuilder.heartRateTrendPerSession([s1]))
    }

    func testContentDriftCorrelation() {
        func study(_ pairs: [(drift: Float, pFocused: Float)]) -> SessionRecord {
            let ticks = pairs.enumerated().map { i, p in
                SessionTick(t: Double(i), state: p.pFocused >= 0.5 ? .focused : .distracted,
                            probabilities: F.probs(.focused, pFocused: p.pFocused), lapseRisk: 0, heartRate: nil,
                            heartRateQuality: nil, playhead: Double(i), predictedEngagement: 1 - p.drift,
                            predictedDriftRisk: p.drift)
            }
            return SessionRecord(duration: Double(ticks.count), lectureID: "sample_lecture", contentIsMock: true, ticks: ticks)
        }
        // Perfect alignment: 1 - P(focused) == drift.
        let xs: [Float] = (0..<20).map { Float($0 % 7) / 7 }
        let aligned = study(xs.map { ($0, 1 - $0) })
        XCTAssertEqual(AttentionProfileBuilder.contentDriftCorrelation([aligned])!, 1, accuracy: 1e-6)
        let opposite = study(xs.map { ($0, $0) })
        XCTAssertEqual(AttentionProfileBuilder.contentDriftCorrelation([opposite])!, -1, accuracy: 1e-6)
        // Hand-computed Pearson for a small mixed set, pooled with a free-study session (ignored).
        let mixed = study([(0.1, 0.9), (0.2, 0.7), (0.3, 0.8), (0.4, 0.5), (0.5, 0.6),
                           (0.6, 0.3), (0.7, 0.4), (0.8, 0.2), (0.9, 0.3), (1.0, 0.1)])
        let free = F.session([(.focused, 30), (.distracted, 30)])
        let yNot: [Double] = [0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.6, 0.8, 0.7, 0.9]
        let xD: [Double] = (1...10).map { Double($0) / 10 }
        let expected = AttentionProfileBuilder.pearson(xD, yNot)!
        XCTAssertEqual(expected, 0.9388, accuracy: 1e-3)  // 0.68 / sqrt(0.825 * 0.636)
        XCTAssertEqual(AttentionProfileBuilder.contentDriftCorrelation([mixed, free])!, expected, accuracy: 1e-5)
        // Not enough Study Mode ticks -> nil.
        XCTAssertNil(AttentionProfileBuilder.contentDriftCorrelation([free]))
        XCTAssertNil(AttentionProfileBuilder.contentDriftCorrelation([study([(0.1, 0.9), (0.9, 0.1)])]))
    }

    func testMockAndReplayPropagation() {
        let plain = F.session([(.focused, 10)])
        let mock = F.session([(.focused, 10)], mock: true)
        let replay = F.session([(.focused, 10)], replay: true)
        let a = AttentionProfileBuilder.summary(sessions: [plain, mock], calendar: cal)
        XCTAssertTrue(a.contentIsMock); XCTAssertFalse(a.isReplay)
        let b = AttentionProfileBuilder.summary(sessions: [replay, plain], calendar: cal)
        XCTAssertFalse(b.contentIsMock); XCTAssertTrue(b.isReplay)
        let c = AttentionProfileBuilder.summary(sessions: [plain], calendar: cal)
        XCTAssertFalse(c.contentIsMock); XCTAssertFalse(c.isReplay)
    }

    func testArmEffectsPassThrough() {
        let sum = AttentionProfileBuilder.summary(sessions: [F.session([(.focused, 10)])], armEffects: F.hapticHelps, calendar: cal)
        XCTAssertEqual(sum.armEffects, F.hapticHelps)
    }

    func testWeeklyProfileData() {
        let now = F.date(2026, 10, 25, 18)
        let old = F.session([(.focused, 600)], start: F.date(2026, 10, 10, 9))            // outside window
        let mon = F.session([(.focused, 120), (.distracted, 60)], start: F.date(2026, 10, 20, 9), heartRate: 70,
                            events: [.refocused(t: 150, afterSeconds: 6)])
        let tue = F.session([(.focused, 180)], start: F.date(2026, 10, 21, 9), heartRate: 66,
                            events: [.refocused(t: 50, afterSeconds: 4), .refocused(t: 90, afterSeconds: 8)], replay: true)
        let snapshots = [
            AttentionProfileData.ArmEffectSnapshot(date: F.date(2026, 10, 20, 10), effects: F.crossingZero),
            AttentionProfileData.ArmEffectSnapshot(date: F.date(2026, 10, 21, 10), effects: F.hapticHelps),
        ]
        let p = AttentionProfileBuilder.profile(sessions: [tue, old, mon], armEffectHistory: snapshots,
                                                now: now, days: 7, calendar: cal)
        XCTAssertEqual(p.dailyFocusedMinutes.count, 7)
        XCTAssertEqual(p.dailyFocusedMinutes.first?.date, F.date(2026, 10, 19, 0))
        let monPoint = p.dailyFocusedMinutes.first { $0.date == F.date(2026, 10, 20, 0) }!
        XCTAssertEqual(monPoint.focusedMinutes, 2, accuracy: 1e-9)
        XCTAssertEqual(monPoint.totalMinutes, 3, accuracy: 1e-9)
        XCTAssertEqual(p.dailyFocusedMinutes.filter { $0.sessionCount == 0 }.count, 5)
        XCTAssertEqual(p.summary.sessionCount, 2)
        XCTAssertTrue(p.isReplay)
        XCTAssertEqual(p.hourlyFocus.map(\.hour), [9])
        XCTAssertTrue(p.hourlyFocus[0].isEligible)
        XCTAssertEqual(p.hourlyFocus[0].focusedFraction, 300.0 / 360, accuracy: 1e-9)
        XCTAssertEqual(p.refocusTrend.map(\.value), [6, 6])
        XCTAssertEqual(p.heartRateTrend.map(\.value), [70, 66])
        XCTAssertEqual(p.summary.heartRateTrendPerSession!, -4, accuracy: 1e-9)
        XCTAssertEqual(p.armEffectHistory.count, 4, "noNudge excluded")
        XCTAssertFalse(p.armEffectHistory.contains { $0.arm == .noNudge })
        XCTAssertEqual(p.summary.armEffects, F.hapticHelps, "latest snapshot feeds the summary")
    }
}
