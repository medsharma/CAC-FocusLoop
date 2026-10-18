import XCTest
@testable import FocusLoop

final class CoreMLAttentionClassifierTests: XCTestCase {

    func testInitReturnsNilWithoutModel() {
        // The test bundle never contains the model.
        XCTAssertNil(CoreMLAttentionClassifier(bundle: Bundle(for: Self.self)))
        XCTAssertNil(CoreMLAttentionClassifier(bundle: .main, modelName: "DefinitelyNotAModel"))
        let bogus = FileManager.default.temporaryDirectory.appendingPathComponent("missing.mlmodelc")
        XCTAssertNil(CoreMLAttentionClassifier(compiledModelURL: bogus))
    }

    func testMakeDefaultFallsBackToHeuristicWithoutModel() {
        let b = BlendedAttentionClassifier.makeDefault(personalHead: nil, bundle: Bundle(for: Self.self))
        XCTAssertFalse(b.usesModel)
        XCTAssertEqual(b.classify(window: InferenceTestWindows.focused()).source, .heuristic)
    }

    func testPaddingRepeatsFirstFrame() {
        let w = InferenceTestWindows.make(30) { f, i in f.heartRate = Float(i) }
        let rows = CoreMLAttentionClassifier.paddedVectors(window: w, timesteps: 100)
        XCTAssertEqual(rows.count, 100)
        XCTAssertTrue(rows.allSatisfy { $0.count == FeatureFrame.featureNames.count })
        let hr = FeatureFrame.featureNames.firstIndex(of: "heartRate")!
        XCTAssertEqual(rows[0][hr], 0)
        XCTAssertEqual(rows[69][hr], 0)
        XCTAssertEqual(rows[70][hr], 0)
        XCTAssertEqual(rows[71][hr], 1)
        XCTAssertEqual(rows[99][hr], 29)

        let long = InferenceTestWindows.make(150) { f, i in f.heartRate = Float(i) }
        let r2 = CoreMLAttentionClassifier.paddedVectors(window: long, timesteps: 100)
        XCTAssertEqual(r2.count, 100)
        XCTAssertEqual(r2[0][hr], 50)
        XCTAssertEqual(r2[99][hr], 149)
        XCTAssertTrue(CoreMLAttentionClassifier.paddedVectors(window: [], timesteps: 100).isEmpty)
    }

    func testProbabilityNormalisation() {
        XCTAssertEqual(InferenceMath.normalizedProbabilities([2, 2, 4, 0]), [0.25, 0.25, 0.5, 0])
        let logits = InferenceMath.normalizedProbabilities([-1, 0, 1, 2])
        XCTAssertEqual(logits.reduce(0, +), 1, accuracy: 1e-5)
        XCTAssertEqual(logits.indices.max(by: { logits[$0] < logits[$1] }), 3)
        XCTAssertEqual(InferenceMath.normalizedProbabilities([.nan, 0, 0, 0]), [0.25, 0.25, 0.25, 0.25])
        XCTAssertEqual(InferenceMath.normalizedProbabilities([0.5]), [0.25, 0.25, 0.25, 0.25])
    }
}
