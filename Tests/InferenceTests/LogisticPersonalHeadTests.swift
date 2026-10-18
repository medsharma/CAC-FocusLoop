import XCTest
@testable import FocusLoop

final class LogisticPersonalHeadTests: XCTestCase {

    /// Deterministic toy data: 4 well-separated Gaussian-ish clusters in 6-D.
    private func toy(_ n: Int, seed: UInt64 = 7) -> [([Float], AttentionState)] {
        var rng = InferenceSplitMix64(seed: seed)
        let centres: [[Float]] = [
            [2, 0, 0, 0, 5, 1], [0, 2, 0, 0, 5, 1], [0, 0, 2, 0, 5, 1], [0, 0, 0, 2, 5, 1],
        ]
        return (0..<n).map { i in
            let s = AttentionState.allCases[i % 4]
            let x = centres[s.index].map { $0 + Float.random(in: -0.4...0.4, using: &rng) }
            return (x, s)
        }
    }

    func testNilBeforeMinimumLabels() {
        let head = LogisticPersonalHead()
        XCTAssertNil(head.predict(embedding: [1, 0, 0, 0, 5, 1]))
        // 7 labels, 4 classes: still nil.
        for (x, s) in toy(7) { head.update(embedding: x, label: s) }
        XCTAssertEqual(head.labelCount, 7)
        XCTAssertNil(head.predict(embedding: [2, 0, 0, 0, 5, 1]))
        // 8th label: ready.
        let (x, s) = toy(8)[7]
        head.update(embedding: x, label: s)
        XCTAssertNotNil(head.predict(embedding: [2, 0, 0, 0, 5, 1]))
    }

    func testNilWithSingleClass() {
        let head = LogisticPersonalHead()
        for (x, _) in toy(12) { head.update(embedding: x, label: .focused) }
        XCTAssertEqual(head.labelCount, 12)
        XCTAssertNil(head.predict(embedding: [2, 0, 0, 0, 5, 1]))
        head.update(embedding: [0, 0, 2, 0, 5, 1], label: .drowsy)
        XCTAssertNotNil(head.predict(embedding: [2, 0, 0, 0, 5, 1]))
    }

    func testLearnsSeparableToyProblem() throws {
        let head = LogisticPersonalHead()
        for (x, s) in toy(80) { head.update(embedding: x, label: s) }
        let test = toy(40, seed: 99)
        var correct = 0
        for (x, s) in test {
            let p = try XCTUnwrap(head.predict(embedding: x))
            XCTAssertEqual(p.reduce(0, +), 1, accuracy: 1e-4)
            if p.indices.max(by: { p[$0] < p[$1] }) == s.index { correct += 1 }
        }
        XCTAssertGreaterThanOrEqual(Double(correct) / Double(test.count), 0.95)
    }

    func testLearnsAfterFewLabels() throws {
        // Two classes, 10 labels: should already separate them.
        let head = LogisticPersonalHead()
        for i in 0..<10 {
            head.update(embedding: i % 2 == 0 ? [1, 0, 0.1 * Float(i)] : [0, 1, 0.1 * Float(i)],
                        label: i % 2 == 0 ? .focused : .stressed)
        }
        let p = try XCTUnwrap(head.predict(embedding: [1, 0, 0.5]))
        XCTAssertEqual(p.indices.max(by: { p[$0] < p[$1] }), AttentionState.focused.index)
        let q = try XCTUnwrap(head.predict(embedding: [0, 1, 0.5]))
        XCTAssertEqual(q.indices.max(by: { q[$0] < q[$1] }), AttentionState.stressed.index)
    }

    func testPersistenceRoundTrip() throws {
        let head = LogisticPersonalHead()
        for (x, s) in toy(40) { head.update(embedding: x, label: s) }
        let data = try head.serialized()
        let restored = LogisticPersonalHead()
        try restored.restore(from: data)
        XCTAssertEqual(restored.labelCount, head.labelCount)
        XCTAssertEqual(restored.dimension, 6)
        XCTAssertEqual(restored.storedClassCounts, head.storedClassCounts)
        for (x, _) in toy(10, seed: 5) {
            let a = try XCTUnwrap(head.predict(embedding: x))
            let b = try XCTUnwrap(restored.predict(embedding: x))
            for (u, v) in zip(a, b) { XCTAssertEqual(u, v, accuracy: 1e-6) }
        }
        // Corrupt data throws and leaves the head untouched.
        XCTAssertThrowsError(try restored.restore(from: Data("{}".utf8)))
        XCTAssertEqual(restored.labelCount, 40)
    }

    func testReplayCapAndReset() {
        var cfg = LogisticPersonalHead.Config()
        cfg.replayCapacity = 20
        cfg.stepsPerUpdate = 2
        let head = LogisticPersonalHead(config: cfg)
        for (x, s) in toy(50) { head.update(embedding: x, label: s) }
        XCTAssertEqual(head.labelCount, 50)
        XCTAssertEqual(head.storedClassCounts.reduce(0, +), 20)
        head.reset()
        XCTAssertEqual(head.labelCount, 0)
        XCTAssertNil(head.dimension)
    }

    func testDimensionChangeResetsAndEmptyIgnored() {
        let head = LogisticPersonalHead()
        for (x, s) in toy(12) { head.update(embedding: x, label: s) }
        head.update(embedding: [], label: .focused)
        XCTAssertEqual(head.labelCount, 12)
        XCTAssertNil(head.predict(embedding: [1, 2, 3]))
        head.update(embedding: [1, 2, 3], label: .focused)
        XCTAssertEqual(head.dimension, 3)
        XCTAssertEqual(head.labelCount, 1)
    }

    func testLearnsOnHeuristicEmbeddingsOfSyntheticReplay() throws {
        // SYNTHETIC: train on every other 1 Hz window's heuristic embedding, test on the rest.
        let replay = try ReplayStream.load()
        let clf = HeuristicAttentionClassifier()
        let head = LogisticPersonalHead()
        var test: [([Float], AttentionState)] = []
        for (k, end) in stride(from: 20, through: replay.frames.count, by: 10).enumerated() {
            guard let label = replay.labels[end - 1] else { continue }
            let emb = clf.classify(window: Array(replay.frames[(end - 100 > 0 ? end - 100 : 0)..<end])).embedding
            if k % 2 == 0 { head.update(embedding: emb, label: label) } else { test.append((emb, label)) }
        }
        var correct = 0
        for (x, s) in test {
            let p = try XCTUnwrap(head.predict(embedding: x))
            if p.indices.max(by: { p[$0] < p[$1] }) == s.index { correct += 1 }
        }
        let acc = Double(correct) / Double(test.count)
        print("SYNTHETIC personal head on heuristic embeddings, held-out accuracy \(String(format: "%.3f", acc)) n=\(test.count)")
        XCTAssertGreaterThan(acc, 0.85)
    }
}
