import PDFKit
import SwiftUI
import XCTest
@testable import FocusLoop

@MainActor
final class PersistenceExportTests: XCTestCase {
    private var dir: URL!

    override func setUp() async throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("persist-export-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func rows(_ n: Int) -> some View {
        VStack(spacing: 0) {
            ForEach(0..<n, id: \.self) { i in
                Text("Row \(i)").frame(maxWidth: .infinity, minHeight: 40).background(i.isMultiple(of: 2) ? Color.blue : Color.gray)
            }
        }
    }

    private func report(replay: Bool, mock: Bool) -> SessionReport {
        var rec = PersistenceFixtures.record()
        rec.isReplay = replay
        rec.contentIsMock = mock
        return ReportBuilder.build(record: rec, timeline: nil, lectureTitle: nil, effects: [])
    }

    func testShortViewExportsSinglePagePDF() throws {
        let url = try ReportExporter.exportPDF(rows(3), report: report(replay: false, mock: false), directory: dir)
        let doc = try XCTUnwrap(PDFDocument(url: url))
        XCTAssertEqual(doc.pageCount, 1)
        let page = try XCTUnwrap(doc.page(at: 0))
        XCTAssertEqual(page.bounds(for: .mediaBox).size, ReportExporter.pageSize)
        XCTAssertTrue(url.lastPathComponent.hasPrefix("FocusLoop-Report-"))
        XCTAssertEqual(url.pathExtension, "pdf")
        XCTAssertEqual(url.deletingLastPathComponent().standardizedFileURL, dir.standardizedFileURL)
    }

    func testTallViewPaginatesAndAcceptsAnyView() throws {
        // 60 rows x 40 pt = 2400 pt; 720 pt of content per Letter page -> 4 pages.
        let url = try ReportExporter.exportPDF(AnyView(rows(60)), report: report(replay: true, mock: true), directory: dir)
        let doc = try XCTUnwrap(PDFDocument(url: url))
        XCTAssertEqual(doc.pageCount, 4)
        XCTAssertTrue(url.lastPathComponent.contains("-SYNTHETIC"))
        XCTAssertTrue(url.lastPathComponent.contains("-MOCK"))
        XCTAssertEqual(doc.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String,
                       "FocusLoop session report (SYNTHETIC data) (MOCK brain timeline)")
        // Text from the first and last rows lands in the document.
        XCTAssertTrue(doc.page(at: 0)?.string?.contains("Row 0") ?? false)
        XCTAssertTrue(doc.page(at: 3)?.string?.contains("Row 59") ?? false)
    }

    func testPNGExport() throws {
        let url = try ReportExporter.exportPNG(rows(5), report: nil, width: 300, scale: 2, directory: dir)
        let image = try XCTUnwrap(UIImage(contentsOfFile: url.path))
        XCTAssertEqual(image.size.width * image.scale, 600, accuracy: 1)
        XCTAssertEqual(image.size.height * image.scale, 400, accuracy: 1)
        let header = try Data(contentsOf: url).prefix(8)
        XCTAssertEqual(Array(header), [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])

        ReportExporter.removeExports(directory: dir)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    func testCortexSnapshotIsNonBlank() async throws {
        let acts: [Float] = [0.9, 0.2, 0.8, 0.3, 0.1, 0.7, 0.4]
        let start = Date()
        let snap = await CortexSnapshotter.snapshot(activations: acts, size: CGSize(width: 240, height: 180))
        let image = try XCTUnwrap(snap)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertEqual(image.size.width, 240, accuracy: 0.5)
        let coverage = CortexSnapshotter.coverage(of: image)
        XCTAssertGreaterThan(coverage, 0.05, "cortex should cover part of the frame")
        XCTAssertLessThan(coverage, 0.98, "background should stay transparent")
        XCTAssertLessThan(elapsed, 10)
        // Optional visual check: only when the test runner is given an output path
        // (TEST_RUNNER_FL_SNAPSHOT_OUT=...). App code never writes the snapshot to disk.
        if let out = ProcessInfo.processInfo.environment["FL_SNAPSHOT_OUT"], let png = image.pngData() {
            try png.write(to: URL(fileURLWithPath: out))
            print("CortexSnapshotter coverage=\(coverage) elapsed=\(elapsed)s -> \(out)")
        }
    }
}
