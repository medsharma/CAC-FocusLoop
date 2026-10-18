import XCTest
@testable import FocusLoop

final class PersistenceReportTests: XCTestCase {
    typealias F = PersistenceFixtures

    func testSegmentsMergeRunsAndAbsorbShortBlips() {
        // 0-9 focused, 10 distracted (1 s blip), 11-19 focused, 20-24 drowsy, 25 focused (last, 1 s).
        var ticks: [SessionTick] = []
        for i in 0..<26 {
            let s: AttentionState = i == 10 ? .distracted : (20..<25).contains(i) ? .drowsy : .focused
            ticks.append(F.tick(Double(i), s))
        }
        let rec = SessionRecord(duration: 26, ticks: ticks.shuffled())
        let r = ReportBuilder.build(record: rec, timeline: nil, lectureTitle: nil, effects: [])
        XCTAssertEqual(r.segments, [
            .init(start: 0, end: 20, state: .focused),
            .init(start: 20, end: 26, state: .drowsy),
        ])
        XCTAssertEqual(r.focusedFraction, 20.0 / 26, accuracy: 1e-9)   // raw ticks, not smoothed segments
    }

    func testLeadingShortRunJoinsNext() {
        let ticks = [F.tick(0, .stressed)] + (1..<6).map { F.tick(Double($0), .focused) }
        let segs = ReportBuilder.build(record: SessionRecord(duration: 6, ticks: ticks), timeline: nil,
                                       lectureTitle: nil, effects: []).segments
        XCTAssertEqual(segs, [.init(start: 0, end: 6, state: .focused)])
    }

    func testRefocusHeartRateNudgesAndPassThrough() {
        var rec = F.record()
        rec.events.append(.refocused(t: 5, afterSeconds: 4))
        rec.events.append(.refocused(t: 50, afterSeconds: 6))
        rec.events.append(.nudge(F.decision(t: 45, policy: .reactive, arm: .noNudge)))
        rec.ticks[3].heartRateQuality = 0.2       // filtered out
        rec.ticks[4].heartRate = nil              // no reading
        let effects = [ArmEffect(arm: .haptic, policy: .reactive, effect: 0.2, lower: -0.1, upper: 0.5, level: 0.9, n: 7)]
        let r = ReportBuilder.build(record: rec, timeline: nil, lectureTitle: "Photosynthesis", effects: effects)

        XCTAssertEqual(r.refocusTimes, [4, 10, 6])         // session order by t: 5, 30, 50
        XCTAssertEqual(r.medianRefocusSeconds, 6)
        XCTAssertEqual(r.heartRate.count, 58)
        XCTAssertFalse(r.heartRate.contains { $0.t == 3 || $0.t == 4 })
        // 20th percentile of the 70...74 cycle values.
        XCTAssertEqual(try XCTUnwrap(r.restingHeartRate), 70, accuracy: 1)
        XCTAssertEqual(r.nudgeCount, 2)                    // noNudge control is not a shown nudge
        XCTAssertEqual(r.preemptiveCount, 1)
        XCTAssertEqual(r.effects, effects)
        XCTAssertEqual(r.insights, rec.insights)
        XCTAssertEqual(r.lectureTitle, "Photosynthesis")
        XCTAssertTrue(r.contentIsMock)
        XCTAssertTrue(r.isReplay)

        let withBaseline = ReportBuilder.build(record: rec, timeline: nil, lectureTitle: nil, effects: [],
                                               restingHeartRate: 63)
        XCTAssertEqual(withBaseline.restingHeartRate, 63)
    }

    func testPearsonKnownAnswers() {
        let x = (1...20).map(Double.init)
        XCTAssertEqual(try XCTUnwrap(ReportBuilder.pearson(x, x.map { 2 * $0 + 3 })), 1, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(ReportBuilder.pearson(x, x.map { -0.5 * $0 })), -1, accuracy: 1e-12)
        // Textbook pair (x, x^2 symmetric around 0) -> r = 0.
        let sym = (-10...10).map(Double.init)
        XCTAssertEqual(try XCTUnwrap(ReportBuilder.pearson(sym, sym.map { $0 * $0 })), 0, accuracy: 1e-12)
        // Hand-computed: r = 0.8 for this set, repeated 4x (n = 20).
        let a: [Double] = [1, 2, 3, 4, 5], b: [Double] = [2, 1, 4, 3, 5]
        let r = ReportBuilder.pearson(Array([a, a, a, a].joined()), Array([b, b, b, b].joined()))
        XCTAssertEqual(try XCTUnwrap(r), 0.8, accuracy: 1e-12)
        XCTAssertNil(ReportBuilder.pearson(Array(x.prefix(19)), Array(x.prefix(19))))   // < 20 points
        XCTAssertNil(ReportBuilder.pearson(x, Array(repeating: 1, count: 20)))           // zero variance
    }

    /// Lecture where predicted drift rises linearly and the student's P(focused) falls linearly.
    private func linearLectureRecord(n: Int = 60) -> SessionRecord {
        let ticks = (0..<n).map { i -> SessionTick in
            let p = Float(1 - Double(i) / Double(n))
            return F.tick(Double(i), p > 0.5 ? .focused : .distracted, pFocused: p, playhead: 100 + Double(i))
        }
        return SessionRecord(duration: Double(n), lectureID: "test-lecture", contentIsMock: true, ticks: ticks)
    }

    func testDriftOverlayCorrelationAndScaling() throws {
        let rising = PersistenceTestTimeline { Float(($0 - 100) / 60) }
        let r = ReportBuilder.build(record: linearLectureRecord(), timeline: rising, lectureTitle: nil, effects: [])
        XCTAssertEqual(r.predictedDrift.count, 60)
        XCTAssertEqual(r.actualDrift.count, 60)
        XCTAssertEqual(r.predictedDrift.first?.t, 100)      // playhead axis
        XCTAssertEqual(r.predictedDrift.map(\.value).min(), 0)
        XCTAssertEqual(r.predictedDrift.map(\.value).max(), 1)
        XCTAssertEqual(r.actualDrift.map(\.value).min(), 0)
        XCTAssertEqual(try XCTUnwrap(r.actualDrift.map(\.value).max()), 1, accuracy: 1e-12)
        // Linear vs (smoothed) linear: only the truncated edge windows differ.
        XCTAssertGreaterThan(try XCTUnwrap(r.driftCorrelation), 0.99)

        let falling = PersistenceTestTimeline { Float(1 - ($0 - 100) / 60) }
        let r2 = ReportBuilder.build(record: linearLectureRecord(), timeline: falling, lectureTitle: nil, effects: [])
        XCTAssertLessThan(try XCTUnwrap(r2.driftCorrelation), -0.99)

        let short = ReportBuilder.build(record: linearLectureRecord(n: 15), timeline: rising, lectureTitle: nil, effects: [])
        XCTAssertNil(short.driftCorrelation)
        XCTAssertEqual(short.predictedDrift.count, 15)

        // No lecture -> no overlay.
        let free = ReportBuilder.build(record: SessionRecord(duration: 30, ticks: (0..<30).map { F.tick(Double($0), .focused) }),
                                       timeline: nil, lectureTitle: nil, effects: [])
        XCTAssertTrue(free.predictedDrift.isEmpty)
        XCTAssertNil(free.driftCorrelation)
        XCTAssertNil(free.biggestDriftPlayhead)
        XCTAssertNil(free.biggestDriftActivations)
    }

    func testFallsBackToTickPredictedDriftWithoutTimeline() throws {
        var rec = linearLectureRecord()
        for i in rec.ticks.indices { rec.ticks[i].predictedDriftRisk = Float(i) / 60 }
        let r = ReportBuilder.build(record: rec, timeline: nil, lectureTitle: nil, effects: [])
        XCTAssertEqual(r.predictedDrift.count, 60)
        XCTAssertGreaterThan(try XCTUnwrap(r.driftCorrelation), 0.99)
        XCTAssertNotNil(r.biggestDriftPlayhead)
        XCTAssertNil(r.biggestDriftActivations)
    }

    func testBiggestDriftMoment() throws {
        // Focused (0.9) everywhere except a deep 5 s dip centred on t = 30 (playhead 230).
        let ticks = (0..<60).map { i -> SessionTick in
            let dip = (28...32).contains(i)
            return F.tick(Double(i), dip ? .distracted : .focused, pFocused: dip ? 0 : 0.9, playhead: 200 + Double(i))
        }
        let tl = PersistenceTestTimeline { _ in 0.5 }
        let r = ReportBuilder.build(record: SessionRecord(duration: 60, lectureID: "x", ticks: ticks), timeline: tl,
                                    lectureTitle: nil, effects: [])
        XCTAssertEqual(r.biggestDriftPlayhead, 230)
        XCTAssertEqual(r.biggestDriftActivations, tl.activations(at: 230))
        XCTAssertNil(r.driftCorrelation)                    // constant prediction: zero variance
        XCTAssertEqual(r.actualDrift.first { $0.t == 230 }?.value, 1)
    }

    func testSmoothingIsCentredFiveSecondMean() {
        let ticks = (0..<10).map { F.tick(Double($0), .focused, pFocused: $0 == 5 ? 0 : 1) }
        let s = ReportBuilder.smoothedNotFocused(ticks)
        XCTAssertEqual(s[5], 0.2, accuracy: 1e-6)   // 1 of 5 samples in [2.5, 7.5]
        XCTAssertEqual(s[3], 0.2, accuracy: 1e-6)
        XCTAssertEqual(s[2], 0, accuracy: 1e-6)
        XCTAssertEqual(s[8], 0, accuracy: 1e-6)
    }

    func testRealReplayTimelineBuilds() throws {
        // MOCK lecture timeline from the bundle, end to end.
        let tl = try BrainContentTimeline(named: "sample_lecture")
        let ticks = (0..<120).map { F.tick(Double($0), .focused, pFocused: Float(0.5 + 0.4 * sin(Double($0) / 9)), playhead: Double($0)) }
        let r = ReportBuilder.build(record: SessionRecord(duration: 120, lectureID: tl.lectureID, contentIsMock: tl.isMock,
                                                          ticks: ticks),
                                    timeline: tl, lectureTitle: "Sample (MOCK)", effects: [])
        XCTAssertEqual(r.predictedDrift.count, 120)
        XCTAssertEqual(r.biggestDriftActivations?.count, 7)
        XCTAssertTrue(r.contentIsMock)
    }
}
