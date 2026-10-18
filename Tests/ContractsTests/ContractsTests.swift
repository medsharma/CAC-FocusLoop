import XCTest
@testable import FocusLoop

final class ContractsTests: XCTestCase {
    func testFeatureVectorRoundTrip() {
        var f = FeatureFrame(t: 1.5)
        f.perclos = 0.3; f.heartRate = 72; f.gazeDeviation = 0.2
        let g = FeatureFrame(t: 1.5, vector: f.vector)
        XCTAssertEqual(f, g)
        XCTAssertEqual(f.vector.count, FeatureFrame.featureNames.count)
    }

    func testReplayLoadsAndIsSynthetic() throws {
        let r = try ReplayStream.load()
        XCTAssertTrue(r.isSynthetic)
        XCTAssertEqual(r.frames.count, 1800)
        XCTAssertTrue(r.labels.contains(.drowsy))
    }

    func testContentTimelineWrapper() throws {
        let c = try BrainContentTimeline(named: "sample_lecture")
        XCTAssertTrue(c.isMock)
        XCTAssertEqual(c.networkValues(at: 3).count, BrainNetwork.allCases.count)
        XCTAssertNotNil(c.upcomingValley(after: 60, within: 30))
        XCTAssertTrue(c.features(at: 20).inValley)
        XCTAssertEqual(c.features(at: 0).vector.count, ContentFeatures.featureNames.count)
    }

    func testStateEstimateArgmax() {
        let e = StateEstimate(t: 0, probabilities: [0.1, 0.2, 0.6, 0.1], source: .heuristic)
        XCTAssertEqual(e.state, .drowsy)
    }
}
