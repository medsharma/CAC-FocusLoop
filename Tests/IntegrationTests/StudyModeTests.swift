import XCTest
@testable import FocusLoop

final class StudyModeTests: XCTestCase {
    private func lecture() -> LectureInfo {
        LectureInfo(id: "t", title: "T", subject: "S", source: "src", sourceURL: "", license: "CC0-1.0",
                    attribution: "a", videoFile: nil, durationSeconds: 100,
                    recaps: [.init(start: 10, end: 20, text: "A"),
                             .init(start: 30, end: 50, text: "B"),
                             .init(start: 60, end: 70, text: "C")],
                    recallQuestions: [.init(after: 40, question: "Q2", choices: ["x", "y"], correctIndex: 1),
                                      .init(after: 15, question: "Q1", choices: ["x", "y"], correctIndex: 0),
                                      .init(after: 80, question: "Q3", choices: ["x", "y"], correctIndex: 0)])
    }

    // MARK: StudyContentProvider

    func testRecapContainingOrMostRecent() {
        let p = StudyContentProvider(lecture: lecture())
        XCTAssertNil(p.currentRecap(at: 5), "nothing before the first recap starts")
        XCTAssertEqual(p.currentRecap(at: 10)?.text, "A")
        XCTAssertEqual(p.currentRecap(at: 25)?.text, "A", "gap -> most recent before playhead")
        XCTAssertEqual(p.currentRecap(at: 30)?.text, "B")
        XCTAssertEqual(p.currentRecap(at: 49.9)?.text, "B")
        XCTAssertEqual(p.currentRecap(at: 99)?.text, "C")
    }

    func testRecallEligibilityAndTracking() {
        var p = StudyContentProvider(lecture: lecture())
        XCTAssertNil(p.nextRecallQuestion(at: 10))
        XCTAssertEqual(p.nextRecallQuestion(at: 15)?.question, "Q1", "after <= playhead is eligible")
        XCTAssertEqual(p.nextRecallQuestion(at: 45)?.question, "Q1", "earliest unasked first")
        // Peeking does not consume.
        XCTAssertEqual(p.nextRecallQuestion(at: 45)?.question, "Q1")

        let card = p.recallCard(at: 45)
        XCTAssertEqual(card?.kind, .recall)
        XCTAssertEqual(card?.body, "Q1")
        XCTAssertEqual(card?.correctIndex, 0)
        XCTAssertEqual(card?.choices, ["x", "y"])
        XCTAssertEqual(p.nextRecallQuestion(at: 45)?.question, "Q2")

        p.markAsked(lecture().recallQuestions[0])   // Q2
        XCTAssertNil(p.nextRecallQuestion(at: 45), "all eligible asked")
        XCTAssertEqual(p.remainingRecallCount, 1)
        XCTAssertEqual(p.nextRecallQuestion(at: 100)?.question, "Q3")
        XCTAssertNotNil(p.recallCard(at: 100))
        XCTAssertNil(p.recallCard(at: 100), "never repeats")

        p.markAsked(index: 99)  // out of range: ignored
        XCTAssertEqual(p.askedRecallIndices, [0, 1, 2])
        p.reset()
        XCTAssertEqual(p.remainingRecallCount, 3)
    }

    func testRecapCard() {
        let p = StudyContentProvider(lecture: lecture())
        XCTAssertNil(p.recapCard(at: 0))
        let c = p.recapCard(at: 35)
        XCTAssertEqual(c?.kind, .recap)
        XCTAssertEqual(c?.body, "B")
    }

    func testSampleLectureFallbackAndMockTimeline() throws {
        // The fixture stays loadable by id but is never listed in the UI.
        XCTAssertTrue(StudyLectureLibrary.allLectures().contains { $0.id == "sample_lecture" })
        XCTAssertFalse(StudyLectureLibrary.lectures().contains { $0.id == "sample_lecture" })
        XCTAssertNotNil(StudyLectureLibrary.lecture(id: "sample_lecture"))
        let sample = StudyLectureLibrary.sampleLecture
        XCTAssertNil(sample.videoFile)
        XCTAssertEqual(sample.durationSeconds, 180)
        XCTAssertNil(StudyLectureLibrary.videoURL(for: sample))
        let tl = try XCTUnwrap(StudyLectureLibrary.timeline(for: sample))
        XCTAssertTrue(tl.isMock)
        // Recaps tile the lecture; recall questions are answerable.
        XCTAssertEqual(sample.recaps.first?.start, 0)
        XCTAssertEqual(sample.recaps.last?.end, 180)
        for q in sample.recallQuestions { XCTAssertTrue(q.choices.indices.contains(q.correctIndex)) }
    }

    // MARK: Scrubber geometry

    func testTimeXRoundTripAndClamping() {
        XCTAssertEqual(StudyScrubberGeometry.x(for: 90, duration: 180, width: 300), 150, accuracy: 1e-9)
        XCTAssertEqual(StudyScrubberGeometry.x(for: -5, duration: 180, width: 300), 0)
        XCTAssertEqual(StudyScrubberGeometry.x(for: 500, duration: 180, width: 300), 300)
        XCTAssertEqual(StudyScrubberGeometry.time(forX: 75, duration: 180, width: 300), 45, accuracy: 1e-9)
        XCTAssertEqual(StudyScrubberGeometry.time(forX: -20, duration: 180, width: 300), 0)
        XCTAssertEqual(StudyScrubberGeometry.time(forX: 999, duration: 180, width: 300), 180)
        XCTAssertEqual(StudyScrubberGeometry.x(for: 10, duration: 0, width: 300), 0, "degenerate duration")
        for t in stride(from: 0.0, through: 180, by: 7.5) {
            let x = StudyScrubberGeometry.x(for: t, duration: 180, width: 333)
            XCTAssertEqual(StudyScrubberGeometry.time(forX: x, duration: 180, width: 333), t, accuracy: 1e-9)
        }
    }

    func testValleySegments() {
        let valleys: [(start: Double, end: Double)] = [(12, 35), (69, 90), (170, 200), (50, 50), (-10, 5)]
        let segs = StudyScrubberGeometry.valleySegments(valleys, duration: 180, width: 360)
        XCTAssertEqual(segs.count, 4, "zero-length valley dropped")
        XCTAssertEqual(segs[0].lowerBound, 24, accuracy: 1e-9)
        XCTAssertEqual(segs[0].upperBound, 70, accuracy: 1e-9)
        XCTAssertEqual(segs[1].lowerBound, 138, accuracy: 1e-9)
        XCTAssertEqual(segs[1].upperBound, 180, accuracy: 1e-9)
        XCTAssertEqual(segs[2].upperBound, 360, accuracy: 1e-9, "clamped to track end")
        XCTAssertEqual(segs[3].lowerBound, 0, accuracy: 1e-9, "clamped to track start")
        XCTAssertTrue(StudyScrubberGeometry.valleySegments(valleys, duration: 0, width: 360).isEmpty)

        XCTAssertEqual(StudyScrubberGeometry.valleyIndex(containing: 12, in: valleys), 0)
        XCTAssertNil(StudyScrubberGeometry.valleyIndex(containing: 35, in: valleys), "end is exclusive")
        XCTAssertEqual(StudyScrubberGeometry.valleyIndex(containing: 80, in: valleys), 1)
    }

    func testValleySegmentsMatchSampleTimeline() throws {
        let tl = try XCTUnwrap(StudyLectureLibrary.timeline(for: StudyLectureLibrary.sampleLecture))
        let segs = StudyScrubberGeometry.valleySegments(tl.valleys(), duration: 180, width: 180)
        XCTAssertEqual(segs.count, tl.valleys().count)
        for (s, v) in zip(segs, tl.valleys()) {
            XCTAssertEqual(Double(s.lowerBound), v.start, accuracy: 1e-9)
            XCTAssertEqual(Double(s.upperBound), v.end, accuracy: 1e-9)
        }
    }

    func testSparklineAndSamples() {
        let s = StudyScrubberGeometry.samples(count: 5, duration: 100) { Float($0 / 100) }
        XCTAssertEqual(s, [0, 0.25, 0.5, 0.75, 1])
        let pts = StudyScrubberGeometry.sparklinePoints([0, 1, 2, -1], width: 300, height: 30)
        XCTAssertEqual(pts.map(\.x), [0, 100, 200, 300])
        XCTAssertEqual(pts.map(\.y), [30, 0, 0, 30], "values clamp to 0...1, 1 = top")
        XCTAssertEqual(StudyScrubberGeometry.timeLabel(125.7), "2:05")
        XCTAssertEqual(StudyScrubberGeometry.timeLabel(-3), "0:00")
    }

    // MARK: Playback (timer path, no video)

    @MainActor
    func testTimerPlaybackAdvancesAndReportsPlayhead() {
        let c = StudyPlaybackController(lecture: StudyLectureLibrary.sampleLecture)
        XCTAssertFalse(c.hasVideo)
        var ticks: [Double] = []
        c.onTick = { ticks.append($0) }
        c.seek(to: 170)
        XCTAssertEqual(c.playhead, 170)
        c.play()
        let exp = expectation(description: "advances")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { exp.fulfill() }
        wait(for: [exp], timeout: 2)
        XCTAssertGreaterThan(c.playhead, 170.2)
        XCTAssertGreaterThan(ticks.count, 2)
        c.seek(to: 999)
        XCTAssertEqual(c.playhead, 180, "seek clamps to duration")
        c.tearDown()
        XCTAssertFalse(c.isPlaying)
    }
}
