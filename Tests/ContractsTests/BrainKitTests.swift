import XCTest
@testable import FocusLoop

final class BrainKitTests: XCTestCase {
    func testMeshLoadsFromAppBundle() throws {
        let mesh = try BrainMesh.load()
        XCTAssertEqual(mesh.vertexCount, BrainMesh.expectedVertexCount)
        XCTAssertEqual(mesh.indices.count, 40_960 * 3)
    }

    func testSampleTimelineDecodes() throws {
        let tl = try BrainTimeline.load("sample_lecture")
        XCTAssertEqual(tl.mock, true)
        XCTAssertEqual(Set(tl.networks.keys), Set(BrainNetwork.allCases.map(\.key)))
        XCTAssertEqual(tl.activations(at: 10).count, 7)
    }
}
