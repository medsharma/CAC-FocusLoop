import XCTest
@testable import FocusLoop

final class HeuristicAttentionClassifierTests: XCTestCase {
    let clf = HeuristicAttentionClassifier()

    private func assertValid(_ e: StateEstimate, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(e.probabilities.count, 4, file: file, line: line)
        XCTAssertEqual(e.probabilities.reduce(0, +), 1, accuracy: 1e-4, file: file, line: line)
        XCTAssertEqual(e.embedding.count, HeuristicAttentionClassifier.embeddingDimension, file: file, line: line)
        XCTAssertTrue(e.embedding.allSatisfy(\.isFinite), file: file, line: line)
        XCTAssertEqual(e.source, .heuristic, file: file, line: line)
    }

    func testFocused() {
        let e = clf.classify(window: InferenceTestWindows.focused())
        assertValid(e)
        XCTAssertEqual(e.state, .focused)
        XCTAssertGreaterThan(e.confidence, 0.6)
        XCTAssertEqual(e.t, 9.9, accuracy: 1e-9)
    }

    func testDistractedGaze() {
        let e = clf.classify(window: InferenceTestWindows.distractedGaze())
        assertValid(e)
        XCTAssertEqual(e.state, .distracted)
        XCTAssertGreaterThan(e.confidence, 0.8)
    }

    func testPhonePickupIsDistracted() {
        let e = clf.classify(window: InferenceTestWindows.phonePickup())
        XCTAssertEqual(e.state, .distracted)
    }

    func testDrowsyHeavyLidsAndLongBlink() {
        let e = clf.classify(window: InferenceTestWindows.drowsy())
        assertValid(e)
        XCTAssertEqual(e.state, .drowsy)
        XCTAssertGreaterThan(e.confidence, 0.8)
    }

    func testYawnAndNodAreDrowsy() {
        let e = clf.classify(window: InferenceTestWindows.yawnAndNod())
        XCTAssertEqual(e.state, .drowsy)
    }

    func testStressed() {
        let e = clf.classify(window: InferenceTestWindows.stressed())
        assertValid(e)
        XCTAssertEqual(e.state, .stressed)
        XCTAssertGreaterThan(e.confidence, 0.8)
    }

    func testHeartRateIgnoredWhenQualityLow() {
        // HR up but rPPG quality poor, no brow tension: not stressed.
        let w = InferenceTestWindows.make { f, _ in f.heartRate = 90; f.heartRateDelta = 22; f.heartRateQuality = 0.1 }
        let (ev, _) = clf.analyze(window: w)
        XCTAssertFalse(ev.heartRateUsable)
        XCTAssertEqual(ev.stressed, 0, accuracy: 0.05)
        XCTAssertEqual(clf.classify(window: w).state, .focused)
        // Same HR with good quality pushes stress evidence up.
        let good = InferenceTestWindows.make { f, _ in f.heartRate = 90; f.heartRateDelta = 22 }
        XCTAssertGreaterThan(clf.analyze(window: good).0.stressed, 0.5)
    }

    func testTrailingCountersAloneDoNotTriggerDrowsy() {
        // perclos / longBlinkCount are trailing-60 s counters that linger after an episode.
        let w = InferenceTestWindows.make { f, _ in f.perclos = 0.3; f.longBlinkCount = 4; f.yawnCount = 2 }
        XCTAssertEqual(clf.classify(window: w).state, .focused)
    }

    func testMissingFaceIsModeratelyDistracted() {
        let e = clf.classify(window: InferenceTestWindows.faceMissing())
        let (ev, _) = clf.analyze(window: InferenceTestWindows.faceMissing())
        XCTAssertTrue(ev.faceMissing)
        XCTAssertEqual(e.state, .distracted)
        XCTAssertGreaterThan(e.confidence, 0.55)
        XCTAssertLessThan(e.confidence, 0.8)
        XCTAssertEqual(e.probability(of: .drowsy), e.probability(of: .stressed), accuracy: 1e-6)

        let p = clf.classify(window: InferenceTestWindows.faceMissing(pickup: true))
        XCTAssertEqual(p.state, .distracted)
        XCTAssertGreaterThan(p.confidence, e.confidence)
    }

    func testShortAndEmptyWindows() {
        let empty = clf.classify(window: [])
        XCTAssertEqual(empty.probabilities, [0.25, 0.25, 0.25, 0.25])
        XCTAssertEqual(empty.embedding.count, HeuristicAttentionClassifier.embeddingDimension)
        let short = clf.classify(window: Array(InferenceTestWindows.focused().prefix(3)))
        assertValid(short)
    }

    func testRunCount() {
        let xs: [Float] = [0, 0.9, 0.9, 0.9, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        XCTAssertEqual(HeuristicAttentionClassifier.runCount(xs, above: 0.8, minLength: 6), 2)
        XCTAssertEqual(HeuristicAttentionClassifier.runCount(xs, above: 0.8, minLength: 3), 3)
    }

    /// SYNTHETIC: the scripted replay stream, classified at 1 Hz over trailing 10 s windows,
    /// compared with the generator's `label` column at the window end.
    func testSyntheticReplayAccuracy() throws {
        let replay = try ReplayStream.load()
        XCTAssertTrue(replay.isSynthetic)
        var confusion = Array(repeating: Array(repeating: 0, count: 4), count: 4)
        var correct = 0, total = 0
        for end in stride(from: 10, through: replay.frames.count, by: 10) {
            guard let label = replay.labels[end - 1] else { continue }
            let window = Array(replay.frames[max(0, end - 100)..<end])
            let pred = clf.classify(window: window).state
            confusion[label.index][pred.index] += 1
            total += 1
            if pred == label { correct += 1 }
        }
        let accuracy = Double(correct) / Double(total)
        var lines = ["SYNTHETIC heuristic replay: accuracy \(String(format: "%.3f", accuracy)) (\(correct)/\(total) windows)"]
        for s in AttentionState.allCases {
            let row = confusion[s.index]
            let n = row.reduce(0, +)
            let recall = n > 0 ? Double(row[s.index]) / Double(n) : .nan
            lines.append("SYNTHETIC recall \(s.rawValue): \(String(format: "%.3f", recall)) n=\(n) row=\(row)")
            switch s {
            case .focused: XCTAssertGreaterThan(recall, 0.9)
            default: XCTAssertGreaterThan(recall, 0.8, "\(s)")
            }
        }
        print(lines.joined(separator: "\n"))
        XCTAssertGreaterThan(accuracy, 0.9)
    }
}
