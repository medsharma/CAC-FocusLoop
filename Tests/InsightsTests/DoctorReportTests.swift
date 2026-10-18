import PDFKit
import XCTest
@testable import FocusLoop

@MainActor
final class DoctorReportTests: XCTestCase {

    // MARK: Fixtures

    /// A real (live-camera) session: randomized nudges, HR readings, a check-in answer.
    nonisolated static func realSession(daysAgo: Double, focusedUntil: Int = 40, hr: Float = 72) -> SessionRecord {
        var r = PersistenceFixtures.record(startedAt: Date(timeIntervalSince1970: 1_800_000_000 - daysAgo * 86_400))
        r.isReplay = false
        r.contentIsMock = false
        r.lectureID = nil
        r.insights = []
        r.ticks = (0..<120).map { i in
            let state: AttentionState = i < focusedUntil || i >= focusedUntil + 15 ? .focused : .distracted
            return PersistenceFixtures.tick(Double(i), state, hr: hr + Float(i % 3), hrQ: 0.9)
        }
        r.duration = 120
        return r
    }

    nonisolated static func replaySession() -> SessionRecord { PersistenceFixtures.record() }   // isReplay = true

    nonisolated static func recordedSession() -> SessionRecord {
        var r = realSession(daysAgo: 4); r.sourceKind = "recorded"; return r
    }

    nonisolated static func demoDirectorSession() -> SessionRecord {
        var r = realSession(daysAgo: 5); r.sourceKind = DoctorReportBuilder.demoDirectorSourceKind; return r
    }

    /// An older Demo Director record (no source kind) recognised by its scripted, deterministic nudge.
    nonisolated static func scriptedLegacySession() -> SessionRecord {
        var r = realSession(daysAgo: 6)
        let d = PersistenceFixtures.decision(t: 30, policy: .reactive, arm: .ambientGlow)
        let scripted = NudgeDecision(t: d.t, date: d.date, policy: d.policy, context: d.context, arm: .ambientGlow,
                                     randomizationProbability: 1, armProbabilities: [NudgeArm.ambientGlow.rawValue: 1])
        r.events = [.nudge(scripted)]
        return r
    }

    static let effects = [
        ArmEffect(arm: .ambientGlow, policy: .reactive, effect: 0.18, lower: 0.04, upper: 0.31, level: 0.9, n: 14),
        ArmEffect(arm: .haptic, policy: .reactive, effect: 0.05, lower: -0.10, upper: 0.20, level: 0.9, n: 9),
        ArmEffect(arm: .stretchBreak, policy: .reactive, effect: -0.2, lower: -0.35, upper: -0.05, level: 0.9, n: 6),
        ArmEffect(arm: .noNudge, policy: .reactive, effect: 0, lower: 0, upper: 0, level: 0.9, n: 20),
    ]

    nonisolated static var allSessions: [SessionRecord] {
        [realSession(daysAgo: 3), realSession(daysAgo: 2, focusedUntil: 70, hr: 68), realSession(daysAgo: 1, hr: 70),
         replaySession(), recordedSession(), demoDirectorSession(), scriptedLegacySession()]
    }

    nonisolated static var runs: [SARTRunRecord] {
        [SARTHistoryTests.run(daysAgo: 6, commissions: 2), SARTHistoryTests.run(daysAgo: 3, origin: .simulator),
         SARTHistoryTests.run(daysAgo: 1, commissions: 0), SARTHistoryTests.run(daysAgo: 0.5, origin: .sample)]
    }

    private func build(includeDemo: Bool = false, sessions: [SessionRecord] = allSessions,
                       runs: [SARTRunRecord] = runs) -> DoctorReportContent {
        DoctorReportBuilder.build(.init(sessions: sessions, realArmEffects: Self.effects, demoArmEffects: Self.effects,
                                        sartRuns: runs, includeDemo: includeDemo,
                                        now: Date(timeIntervalSince1970: 1_800_000_000)))
    }

    // MARK: Filtering

    func testDemoSessionClassification() {
        XCTAssertFalse(DoctorReportBuilder.isDemoSession(Self.realSession(daysAgo: 1)))
        XCTAssertTrue(DoctorReportBuilder.isDemoSession(Self.replaySession()))
        XCTAssertTrue(DoctorReportBuilder.isDemoSession(Self.recordedSession()))
        XCTAssertTrue(DoctorReportBuilder.isDemoSession(Self.demoDirectorSession()))
        XCTAssertTrue(DoctorReportBuilder.isDemoSession(Self.scriptedLegacySession()))
    }

    func testDefaultIncludesOnlyRealSessions() throws {
        let c = build()
        XCTAssertEqual(c.realSessionCount, 3)
        XCTAssertEqual(c.demoSessionCount, 4)
        XCTAssertFalse(c.includesDemo)
        let overview = try XCTUnwrap(c.section("overview"))
        XCTAssertFalse(overview.isDemoData)
        guard case let .keyValues(kv)? = overview.blocks.first(where: { if case .keyValues = $0 { true } else { false } }) else {
            return XCTFail("Overview needs key values")
        }
        XCTAssertEqual(kv.first { $0.key == "Sessions" }?.value, "3")
        XCTAssertEqual(kv.first { $0.key == "Total time" }?.value, "6 min")
        XCTAssertFalse(c.allText.lowercased().contains("demo data\n"), "No section is marked demo data by default")
        XCTAssertTrue(c.sections.allSatisfy { !$0.isDemoData })
        XCTAssertTrue(c.dataScope.contains("4 demo sessions left out"), c.dataScope)
        // SAMPLE SART runs stay out by default; device and Simulator runs are in (the taps are real).
        XCTAssertFalse(c.allText.contains("SAMPLE"))
        XCTAssertTrue(c.allText.contains("Simulator"))
        XCTAssertTrue(c.dataScope.contains("3 attention-check (SART) runs"), c.dataScope)
    }

    func testIncludeDemoMarksEverySessionDerivedSection() throws {
        let c = build(includeDemo: true)
        XCTAssertTrue(c.includesDemo)
        XCTAssertTrue(try XCTUnwrap(c.section("overview")).isDemoData)
        XCTAssertTrue(try XCTUnwrap(c.section("heartRate")).isDemoData)
        XCTAssertTrue(try XCTUnwrap(c.section("whatHelped")).isDemoData)
        XCTAssertTrue(try XCTUnwrap(c.section("selfReports")).isDemoData)
        XCTAssertTrue(try XCTUnwrap(c.section("sart")).isDemoData, "SAMPLE runs are demo data")
        XCTAssertFalse(try XCTUnwrap(c.section("notes")).isDemoData)
        XCTAssertTrue(c.allText.contains("SAMPLE (demo data)"))
        XCTAssertTrue(c.dataScope.contains("marked \u{201C}demo data\u{201D}"))
    }

    func testNoRealSessionsSaysSoPlainly() throws {
        let c = build(sessions: [Self.replaySession(), Self.recordedSession()], runs: [])
        let overview = try XCTUnwrap(c.section("overview"))
        guard case let .paragraph(t)? = overview.blocks.first else { return XCTFail() }
        XCTAssertTrue(t.hasPrefix("No real sessions have been recorded yet."), t)
        XCTAssertTrue(c.allText.contains("No attention-check runs yet."))
        XCTAssertTrue(c.allText.contains("No real sessions yet, so there are no results."))
        XCTAssertEqual(c.realSessionCount, 0)
    }

    // MARK: Content

    func testAllSectionsPresentInOrder() {
        let c = build()
        XCTAssertEqual(c.sections.map(\.id), ["overview", "sart", "heartRate", "whatHelped", "selfReports", "notes"])
        XCTAssertEqual(c.sections.map(\.number), [1, 2, 3, 4, 5, 6])
        XCTAssertTrue(c.allText.contains(DoctorReportContent.title))
        XCTAssertTrue(c.allText.contains(DoctorReportContent.statement))
        XCTAssertEqual(DoctorReportContent.statement,
                       "This is not a diagnosis. FocusLoop is a wellness tool that measures attention signals with a phone camera; it cannot diagnose any condition. Measurements can be affected by lighting, motion and device placement.")
    }

    func testSARTSectionHasTableTrendAndVerifiedReferences() throws {
        let s = try XCTUnwrap(build().section("sart"))
        let table = s.blocks.compactMap { if case let .table(t) = $0 { t } else { nil } }.first
        XCTAssertEqual(table?.rows.count, 3)
        XCTAssertEqual(table?.headers.first, "Date")
        let text = build().allText
        XCTAssertTrue(text.contains("Taps on 3 went from 100% in the first run to 0% in the latest."), text)
        if SARTReferenceData.ranges.isEmpty {
            XCTAssertTrue(text.contains(SARTReferenceData.noVerifiedSourceNote))
        } else {
            XCTAssertTrue(text.contains("Typical range in published studies of adults"))
            XCTAssertTrue(text.contains("Cheyne et al., 2009"))
        }
    }

    func testWhatHelpedUsesPlainReadings() {
        let text = build().allText
        XCTAssertTrue(text.contains("Seems to help"))
        XCTAssertTrue(text.contains("Still learning"), "Interval crossing 0 must read as still learning")
        XCTAssertTrue(text.contains("Seems not to help"))
        XCTAssertFalse(text.contains("No nudge (control)"), "The control arm is not a row")
        XCTAssertEqual(DoctorReportBuilder.reading(Self.effects[1]), "Still learning")
    }

    func testSelfReportsCountAnswers() throws {
        // Each real fixture session has one "not focused: tired" answer.
        var s = Self.realSession(daysAgo: 1)
        s.events += [.experienceSample(t: 70, focused: true, reason: nil), .experienceSample(t: 90, focused: false, reason: .stressed)]
        let c = build(sessions: [s])
        guard case let .keyValues(kv)? = try XCTUnwrap(c.section("selfReports")).blocks.last else { return XCTFail() }
        XCTAssertEqual(kv.first { $0.key == "Answers" }?.value, "3")
        XCTAssertEqual(kv.first { $0.key == "Focused" }?.value, "1 (33%)")
        XCTAssertEqual(kv.first { $0.key == "Not focused: tired" }?.value, "1 (33%)")
        XCTAssertEqual(kv.first { $0.key == "Not focused: stressed" }?.value, "1 (33%)")
    }

    func testHeartRateHasCaveat() throws {
        let text = build().allText
        XCTAssertTrue(text.contains("less accurate for some skin tones"))
        XCTAssertTrue(text.contains("Median resting heart rate"))
    }

    func testNoBannedWordsInAnyVariant() {
        for c in [build(), build(includeDemo: true), build(sessions: [], runs: []),
                  build(sessions: [Self.replaySession()], runs: [])] {
            XCTAssertEqual(DoctorReportWording.violations(in: c.allText), [], c.allText)
        }
        XCTAssertEqual(DoctorReportWording.violations(in: DoctorReportSheet.intro), [])
    }

    func testBannedWordScannerCatchesTerms() {
        XCTAssertEqual(DoctorReportWording.violations(in: "Symptoms of a disorder"), ["disorder", "symptom"])
        XCTAssertEqual(DoctorReportWording.violations(in: "Seek treatment"), ["treat"])
        XCTAssertEqual(DoctorReportWording.violations(in: "Abnormal and normal ranges"), ["normal", "abnormal"])
        XCTAssertEqual(DoctorReportWording.violations(in: "A diagnosis is not given"), ["diagnos"])
        XCTAssertEqual(DoctorReportWording.violations(in: DoctorReportContent.statement), [], "The required statement is exempt")
    }

    // MARK: PDF

    func testPDFHasPagesFooterAndStatement() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("doctor-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let content = build(includeDemo: true, sessions: Self.allSessions + (0..<20).map { Self.realSession(daysAgo: Double($0) * 0.3) })
        let url = try DoctorReportExporter.exportPDF(content, directory: dir)
        XCTAssertTrue(url.lastPathComponent.hasPrefix("FocusLoop-"), "Delete all data sweeps FocusLoop-* temp files")
        XCTAssertTrue(url.lastPathComponent.hasSuffix("-with-demo-data.pdf"))
        let pdf = try XCTUnwrap(PDFDocument(url: url))
        XCTAssertGreaterThanOrEqual(pdf.pageCount, 2)
        let page1 = try XCTUnwrap(pdf.page(at: 0))
        XCTAssertEqual(page1.bounds(for: .mediaBox).size, CGSize(width: 612, height: 792), "US Letter")
        if let text = pdf.string, !text.isEmpty {   // ImageRenderer keeps text as text
            XCTAssertTrue(text.contains("Page 1 of \(pdf.pageCount)"))
            XCTAssertTrue(text.contains("Page \(pdf.pageCount) of \(pdf.pageCount)"))
            XCTAssertTrue(text.replacingOccurrences(of: "\n", with: " ").contains("This is not a diagnosis."))
            XCTAssertEqual(DoctorReportWording.violations(in: text.replacingOccurrences(of: "\n", with: " ")
                .replacingOccurrences(of: "  ", with: " ")).filter { $0 != "diagnos" }, [])
        }
        Self.writePreviews(pdf, name: "doctor-report")
    }

    func testEmptyReportStillRendersOnePage() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("doctor-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = try DoctorReportExporter.exportPDF(build(sessions: [], runs: []), directory: dir)
        let pdf = try XCTUnwrap(PDFDocument(url: url))
        XCTAssertGreaterThanOrEqual(pdf.pageCount, 1)
        Self.writePreviews(pdf, name: "doctor-report-empty")
    }

    /// With `TEST_RUNNER_FL_PREVIEW_DIR` set for xcodebuild, saves the PDF and PNGs of its pages for review.
    static func writePreviews(_ pdf: PDFDocument, name: String) {
        guard let dir = ProcessInfo.processInfo.environment["FL_PREVIEW_DIR"] else { return }
        let base = URL(fileURLWithPath: dir)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        pdf.write(to: base.appendingPathComponent("\(name).pdf"))
        for i in 0..<pdf.pageCount {
            guard let page = pdf.page(at: i) else { continue }
            let img = page.thumbnail(of: CGSize(width: 1224, height: 1584), for: .mediaBox)
            try? img.pngData()?.write(to: base.appendingPathComponent("\(name)-p\(i + 1).png"))
        }
    }
}
