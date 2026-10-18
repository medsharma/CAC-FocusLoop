import XCTest
@testable import FocusLoop

final class BlendedAttentionClassifierTests: XCTestCase {

    func testPersonalWeightMath() {
        XCTAssertEqual(BlendedAttentionClassifier.personalWeight(labelCount: 0), 0)
        XCTAssertEqual(BlendedAttentionClassifier.personalWeight(labelCount: 10), 10 / 40, accuracy: 1e-6)
        XCTAssertEqual(BlendedAttentionClassifier.personalWeight(labelCount: 30), 0.5, accuracy: 1e-6)
        XCTAssertEqual(BlendedAttentionClassifier.personalWeight(labelCount: 60), 60 / 90, accuracy: 1e-6)
        XCTAssertEqual(BlendedAttentionClassifier.personalWeight(labelCount: 70), 0.7, accuracy: 1e-6)
        XCTAssertEqual(BlendedAttentionClassifier.personalWeight(labelCount: 10_000), 0.7, accuracy: 1e-6)
        XCTAssertEqual(BlendedAttentionClassifier.personalWeight(labelCount: 10, k: 10, cap: 1), 0.5, accuracy: 1e-6)
    }

    func testConfidentModelPassesThrough() {
        let model = InferenceStubClassifier(probabilities: [0.1, 0.7, 0.1, 0.1])
        let b = BlendedAttentionClassifier(base: model, personalHead: nil)
        let e = b.classify(window: InferenceTestWindows.focused())
        XCTAssertEqual(e.source, .model)
        XCTAssertEqual(e.state, .distracted)
        XCTAssertEqual(e.embedding, [1, 2, 3])
        XCTAssertEqual(b.lastDiagnostics?.usedHeuristicFallback, false)
    }

    func testLowConfidenceAveragesWithHeuristic() {
        let model = InferenceStubClassifier(probabilities: [0.3, 0.3, 0.2, 0.2])
        let b = BlendedAttentionClassifier(base: model, personalHead: nil)
        let window = InferenceTestWindows.stressed()
        let h = HeuristicAttentionClassifier().classify(window: window)
        let e = b.classify(window: window)
        XCTAssertEqual(e.source, .blended)
        XCTAssertEqual(e.state, .stressed)
        XCTAssertTrue(b.lastDiagnostics?.usedHeuristicFallback ?? false)
        for i in 0..<4 {
            XCTAssertEqual(e.probabilities[i], (model.probabilities[i] + h.probabilities[i]) / 2, accuracy: 1e-5)
        }
        XCTAssertEqual(e.embedding, [1, 2, 3], "embedding stays in the model's space")
    }

    func testLowConfidenceHeuristicPolicyAndThreshold() {
        let model = InferenceStubClassifier(probabilities: [0.4, 0.3, 0.2, 0.1])
        var cfg = BlendedAttentionClassifier.Config()
        cfg.lowConfidencePolicy = .heuristic
        let b = BlendedAttentionClassifier(base: model, personalHead: nil, config: cfg)
        let e = b.classify(window: InferenceTestWindows.drowsy())
        XCTAssertEqual(e.source, .heuristic)
        XCTAssertEqual(e.state, .drowsy)

        cfg.confidenceThreshold = 0.35
        let b2 = BlendedAttentionClassifier(base: model, personalHead: nil, config: cfg)
        XCTAssertEqual(b2.classify(window: InferenceTestWindows.drowsy()).source, .model)
    }

    func testPersonalHeadBlending() {
        let model = InferenceStubClassifier(probabilities: [0.8, 0.1, 0.05, 0.05])
        let head = InferenceStubHead(labelCount: 30, output: [0, 0, 1, 0])
        let b = BlendedAttentionClassifier(base: model, personalHead: head)
        let e = b.classify(window: InferenceTestWindows.focused())
        XCTAssertEqual(e.source, .blended)
        XCTAssertEqual(b.lastDiagnostics?.personalWeight ?? 0, 0.5, accuracy: 1e-6)
        XCTAssertEqual(e.probabilities[0], 0.4, accuracy: 1e-5)
        XCTAssertEqual(e.probabilities[2], 0.525, accuracy: 1e-5)
        XCTAssertEqual(e.state, .drowsy)

        // Head not ready -> untouched.
        head.output = nil
        let f = b.classify(window: InferenceTestWindows.focused())
        XCTAssertEqual(f.source, .model)
        XCTAssertEqual(f.probabilities, model.probabilities)
    }

    func testHeuristicOnlyMode() {
        let b = BlendedAttentionClassifier(base: nil, personalHead: LogisticPersonalHead())
        XCTAssertFalse(b.usesModel)
        let e = b.classify(window: InferenceTestWindows.distractedGaze())
        XCTAssertEqual(e.source, .heuristic)
        XCTAssertEqual(e.state, .distracted)
        XCTAssertEqual(e.embedding.count, HeuristicAttentionClassifier.embeddingDimension)
        // Passing the heuristic as base is equivalent to no model.
        XCTAssertFalse(BlendedAttentionClassifier(base: HeuristicAttentionClassifier(), personalHead: nil).usesModel)
    }

    func testMissingFaceOverridesModel() {
        let model = InferenceStubClassifier(probabilities: [0.9, 0.05, 0.03, 0.02])
        let head = InferenceStubHead(labelCount: 100, output: [1, 0, 0, 0])
        let b = BlendedAttentionClassifier(base: model, personalHead: head)
        let e = b.classify(window: InferenceTestWindows.faceMissing())
        XCTAssertEqual(e.state, .distracted)
        XCTAssertEqual(e.source, .heuristic)
        XCTAssertTrue(e.embedding.isEmpty, "no model-space embedding without a face")
        XCTAssertEqual(model.calls, 0)
        XCTAssertEqual(b.lastDiagnostics?.faceMissing, true)
    }

    func testEndToEndHeuristicWithLearnedHeadOnSyntheticReplay() throws {
        // SYNTHETIC: label every 3rd second via the blended output's embedding, then check the
        // blended classifier still tracks the script.
        let replay = try ReplayStream.load()
        let head = LogisticPersonalHead()
        let b = BlendedAttentionClassifier(base: nil, personalHead: head)
        var correct = 0, total = 0
        for (k, end) in stride(from: 10, through: replay.frames.count, by: 10).enumerated() {
            guard let label = replay.labels[end - 1] else { continue }
            let e = b.classify(window: Array(replay.frames[max(0, end - 100)..<end]))
            if e.state == label { correct += 1 }
            total += 1
            if k % 3 == 0 { head.update(embedding: e.embedding, label: label) }
        }
        let acc = Double(correct) / Double(total)
        print("SYNTHETIC blended (heuristic + online personal head) replay accuracy \(String(format: "%.3f", acc))")
        XCTAssertGreaterThan(acc, 0.9)
        XCTAssertGreaterThanOrEqual(head.labelCount, 8)
    }

    func testRingBuffer() {
        var buf = InferenceWindowBuffer<Int>(capacity: 3)
        XCTAssertTrue(buf.isEmpty)
        XCTAssertNil(buf.last)
        buf.append(1); buf.append(2)
        XCTAssertEqual(buf.elements, [1, 2])
        XCTAssertEqual(buf.last, 2)
        buf.append(contentsOf: [3, 4, 5])
        XCTAssertTrue(buf.isFull)
        XCTAssertEqual(buf.elements, [3, 4, 5])
        XCTAssertEqual(buf.first, 3)
        XCTAssertEqual(buf.last, 5)
        buf.append(6)
        XCTAssertEqual(buf.elements, [4, 5, 6])
        buf.removeAll()
        XCTAssertEqual(buf.count, 0)
        buf.append(7)
        XCTAssertEqual(buf.elements, [7])

        var frames = InferenceWindowBuffer.stateWindow()
        XCTAssertEqual(frames.capacity, 100)
        XCTAssertEqual(InferenceWindowBuffer.forecastWindow().capacity, 200)
        for i in 0..<250 { frames.append(FeatureFrame(t: Double(i) / 10)) }
        XCTAssertEqual(frames.elements.count, 100)
        XCTAssertEqual(frames.elements.first?.t ?? 0, 15.0, accuracy: 1e-9)
        XCTAssertEqual(frames.elements.last?.t ?? 0, 24.9, accuracy: 1e-9)
    }
}
