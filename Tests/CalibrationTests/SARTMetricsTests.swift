import XCTest
@testable import FocusLoop

final class SARTMetricsTests: XCTestCase {

    /// Hand-built trial: onset = index × 1.15 s; `rt` in ms (nil = no tap).
    private func trial(_ i: Int, _ digit: Int, rt: Double?, valid: Bool = true, complete: Bool = true) -> SARTTrial {
        let onset = Double(i) * 1.15
        return SARTTrial(index: i, digit: digit, fontSize: 72, isNoGo: digit == 3, onset: onset, maskOnset: onset + 0.25,
                         tapTime: rt.map { onset + $0 / 1000 }, end: complete ? onset + 1.15 : nil, timingValid: valid)
    }

    /// 0 go 300 · 1 go 400 · 2 no-go tapped · 3 go 500 (after the error) · 4 go missed · 5 no-go withheld ·
    /// 6 go 50 ms (anticipation) · 7 go 350 with disrupted timing · 8 go 300.
    private var handBuilt: [SARTTrial] {
        [trial(0, 1, rt: 300), trial(1, 2, rt: 400), trial(2, 3, rt: 250), trial(3, 4, rt: 500), trial(4, 5, rt: nil),
         trial(5, 3, rt: nil), trial(6, 6, rt: 50), trial(7, 7, rt: 350, valid: false), trial(8, 8, rt: 300)]
    }

    func testCountsAndRates() {
        let m = SARTMetrics.compute(handBuilt)
        XCTAssertEqual(m.trialCount, 9)
        XCTAssertEqual(m.goCount, 7)
        XCTAssertEqual(m.noGoCount, 2)
        XCTAssertEqual(m.commissionCount, 1)
        XCTAssertEqual(m.omissionCount, 1)
        XCTAssertEqual(m.commissionRate ?? -1, 0.5, accuracy: 1e-12)
        XCTAssertEqual(m.omissionRate ?? -1, 1.0 / 7.0, accuracy: 1e-12)
    }

    func testAnticipationsAreCountedAndExcludedFromRT() {
        let m = SARTMetrics.compute(handBuilt)
        XCTAssertEqual(m.anticipationCount, 1)
        // Usable RTs: 300, 400, 500, 300 (50 ms is an anticipation, 350 had disrupted timing).
        XCTAssertEqual(m.rtCount, 4)
        XCTAssertEqual(m.meanGoRTms ?? -1, 375, accuracy: 1e-9)
    }

    func testRTVariabilityIsSampleSDAndCV() {
        let m = SARTMetrics.compute(handBuilt)
        // Squared deviations 5625 + 625 + 15625 + 5625 = 27500; / (n - 1 = 3).
        let sd = (27500.0 / 3).squareRoot()
        XCTAssertEqual(m.rtSDms ?? -1, sd, accuracy: 1e-9)
        XCTAssertEqual(m.rtCV ?? -1, sd / 375, accuracy: 1e-12)
    }

    func testPostErrorSlowing() {
        let m = SARTMetrics.compute(handBuilt)
        // After the commission: 500. After correct go taps: 400 (after trial 0) and 300 (after trial 7) -> 350.
        XCTAssertEqual(m.postErrorSlowingMs ?? -1, 150, accuracy: 1e-9)
    }

    func testNoErrorsMeansNoPostErrorSlowing() {
        let m = SARTMetrics.compute([trial(0, 1, rt: 300), trial(1, 3, rt: nil), trial(2, 2, rt: 320)])
        XCTAssertNil(m.postErrorSlowingMs)
        XCTAssertEqual(m.commissionRate, 0)
        XCTAssertEqual(m.omissionRate, 0)
    }

    func testAnticipationBoundaryIs100ms() {
        XCTAssertTrue(SARTMetrics.isAnticipation(trial(0, 1, rt: 99.9)))
        XCTAssertFalse(SARTMetrics.isAnticipation(trial(0, 1, rt: 100)))
        XCTAssertFalse(SARTMetrics.isAnticipation(trial(0, 3, rt: 50)), "No-go taps are commissions, not anticipations")
    }

    func testIncompleteTrialsAreIgnored() {
        let m = SARTMetrics.compute([trial(0, 1, rt: 300), trial(1, 2, rt: nil, complete: false)])
        XCTAssertEqual(m.trialCount, 1)
        XCTAssertEqual(m.omissionCount, 0)
    }

    func testEmptyRunHasNoRatesOrRT() {
        let m = SARTMetrics.compute([])
        XCTAssertNil(m.commissionRate)
        XCTAssertNil(m.omissionRate)
        XCTAssertNil(m.meanGoRTms)
        XCTAssertNil(m.rtSDms)
        XCTAssertNil(m.rtCV)
    }

    func testSingleRTHasMeanButNoVariability() {
        let m = SARTMetrics.compute([trial(0, 1, rt: 300)])
        XCTAssertEqual(m.meanGoRTms, 300)
        XCTAssertNil(m.rtSDms)
        XCTAssertNil(m.rtCV)
    }

    func testRunRecordComputesMetricsAndKnowsStandardTiming() throws {
        let run = SARTRunRecord(origin: .simulator, config: SARTConfig(), trials: handBuilt)
        XCTAssertEqual(run.metrics, SARTMetrics.compute(handBuilt))
        XCTAssertTrue(run.usesStandardTiming)
        XCTAssertTrue(run.origin.isRealTaps)
        XCTAssertFalse(SARTRunOrigin.sample.isRealTaps)
        let decoded = try JSONDecoder().decode(SARTRunRecord.self, from: JSONEncoder().encode(run))
        XCTAssertEqual(decoded, run)
    }

    func testTrendChangeUsesFirstAndLatestRun() {
        let a = SARTRunRecord(date: Date(timeIntervalSince1970: 0), config: SARTConfig(),
                              trials: [trial(0, 3, rt: 300), trial(1, 3, rt: nil)])          // 50%
        let b = SARTRunRecord(date: Date(timeIntervalSince1970: 100), config: SARTConfig(),
                              trials: [trial(0, 3, rt: nil), trial(1, 3, rt: nil)])          // 0%
        XCTAssertEqual(SARTTrend.change([b, a], { $0.commissionRate }) ?? 1, -0.5, accuracy: 1e-12)
        XCTAssertNil(SARTTrend.change([a], { $0.commissionRate }))
    }

    func testReferenceRangesAreSourcedAndPhrased() {
        for r in SARTReferenceData.ranges {
            XCTAssertLessThan(r.low, r.high)
            XCTAssertFalse(r.citation.isEmpty)
            let s = SARTReferenceData.sentence(r, ourTrials: 160)
            XCTAssertTrue(s.hasPrefix("Typical range in published studies of"), s)
            XCTAssertTrue(s.contains("225 trials"), "Must note that the published task length differs")
        }
        let c = SARTReferenceData.range(.commissionRate)
        XCTAssertNotNil(c)
        XCTAssertEqual(c.map { SARTReferenceData.sentence($0, ourTrials: 225) },
                       "Typical range in published studies of adults: 20\u{2013}67% (Cheyne et al., 2009).")
    }

    func testCopyAvoidsBannedWording() {
        var text = [SARTCopy.title, SARTCopy.intro, SARTCopy.personalNote, SARTCopy.emptyTitle, SARTCopy.emptyBody,
                    SARTReferenceData.noVerifiedSourceNote]
        text += SARTCopy.explainers.flatMap { [$0.title, $0.text] }
        text += SARTReferenceData.ranges.map { SARTReferenceData.sentence($0, ourTrials: 160) + $0.derivation }
        XCTAssertEqual(DoctorReportWording.violations(in: text.joined(separator: "\n")), [])
    }
}
