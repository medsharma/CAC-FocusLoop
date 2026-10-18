import CoreGraphics
import XCTest
@testable import FocusLoop

/// Logic tests for the UI module's formatting / mapping helpers and the preview driver.
final class UITestsLogic: XCTestCase {

    // MARK: Formatting

    func testTimer() {
        XCTAssertEqual(UIFormat.timer(0), "0:00")
        XCTAssertEqual(UIFormat.timer(65.9), "1:05")
        XCTAssertEqual(UIFormat.timer(3723), "1:02:03")
        XCTAssertEqual(UIFormat.timer(-5), "0:00")
        XCTAssertEqual(UIFormat.timer(.nan), "0:00")
    }

    func testBackIn() {
        XCTAssertEqual(UIFormat.backIn(2.84), "Back in 2.8 s")
        XCTAssertEqual(UIFormat.backIn(0), "Back in 0.0 s")
        XCTAssertEqual(UIFormat.backIn(250), "Back in 99.9 s")
        XCTAssertEqual(UIFormat.backIn(-1), "Back in 0.0 s")
    }

    func testPercentPointsBpmCorrelation() {
        XCTAssertEqual(UIFormat.percent(0.724), "72%")
        XCTAssertEqual(UIFormat.percent(0.7246, decimals: 1), "72.5%")
        XCTAssertEqual(UIFormat.signedPoints(0.12), "+12 pts")
        XCTAssertEqual(UIFormat.signedPoints(-0.04), "\u{2212}4 pts")
        XCTAssertEqual(UIFormat.signedPoints(0.001), "0 pts")
        XCTAssertEqual(UIFormat.bpm(71.6), "72")
        XCTAssertEqual(UIFormat.bpm(nil), "--")
        XCTAssertEqual(UIFormat.bpm(0), "--")
        XCTAssertEqual(UIFormat.correlation(0.618), "r = 0.62")
        XCTAssertEqual(UIFormat.correlation(-0.3), "r = \u{2212}0.30")
        XCTAssertEqual(UIFormat.correlation(nil), "r = n/a")
    }

    func testDegreesMinutesHour() {
        XCTAssertEqual(UIFormat.degrees(.pi / 2), "90\u{00B0}")
        XCTAssertEqual(UIFormat.minutes(42.4), "42 min")
        XCTAssertEqual(UIFormat.minutes(65), "1 h 05 min")
        XCTAssertEqual(UIFormat.hour(0), "12 AM")
        XCTAssertEqual(UIFormat.hour(9), "9 AM")
        XCTAssertEqual(UIFormat.hour(12), "12 PM")
        XCTAssertEqual(UIFormat.hour(19), "7 PM")
    }

    // MARK: Refocus banner

    func testRefocusBannerVisibilityWindow() {
        XCTAssertNil(RefocusBanner.value(lastRefocusSeconds: nil, lastRefocusAt: nil, elapsed: 10))
        XCTAssertEqual(RefocusBanner.value(lastRefocusSeconds: 2.8, lastRefocusAt: 10, elapsed: 10), 2.8)
        XCTAssertEqual(RefocusBanner.value(lastRefocusSeconds: 2.8, lastRefocusAt: 10, elapsed: 13.9), 2.8)
        XCTAssertNil(RefocusBanner.value(lastRefocusSeconds: 2.8, lastRefocusAt: 10, elapsed: 14.0))
        XCTAssertNil(RefocusBanner.value(lastRefocusSeconds: 2.8, lastRefocusAt: 10, elapsed: 9))
    }

    // MARK: Orb / halo

    func testRiskHaloMonotonicAndBounded() {
        let ps: [Float] = [0, 0.2, 0.5, 0.8, 1, 1.5, -1]
        let o = ps.map { RiskHalo.opacity(probability: $0) }
        XCTAssertEqual(o[0], 0.04, accuracy: 1e-9)
        XCTAssertEqual(o[4], 0.65, accuracy: 1e-9)
        XCTAssertEqual(o[5], o[4]); XCTAssertEqual(o[6], o[0])
        for i in 1..<5 { XCTAssertGreaterThan(o[i], o[i - 1]) }
        XCTAssertEqual(RiskHalo.scale(probability: 0), 1.05, accuracy: 1e-9)
        XCTAssertEqual(RiskHalo.scale(probability: 1), 1.45, accuracy: 1e-9)
    }

    func testOrbMotionBlend() {
        XCTAssertEqual(OrbMotion.blend(probabilities: [1, 0, 0, 0]), .of(.focused))
        XCTAssertEqual(OrbMotion.blend(probabilities: []), .of(.focused))
        let stressed = OrbMotion.blend(probabilities: [0, 0, 0, 1])
        XCTAssertEqual(stressed.period, OrbMotion.of(.stressed).period, accuracy: 1e-9)
        XCTAssertLessThan(stressed.period, OrbMotion.of(.focused).period)
        XCTAssertGreaterThan(OrbMotion.blend(probabilities: [0, 1, 0, 0]).jitter, 0)
        XCTAssertGreaterThan(OrbMotion.blend(probabilities: [0, 0, 1, 0]).sink, 0)
        let half = OrbMotion.blend(probabilities: [0.5, 0, 0.5, 0])
        XCTAssertEqual(half.sink, OrbMotion.of(.drowsy).sink / 2, accuracy: 1e-9)
    }

    // MARK: Segments / stats

    func testSegmenter() {
        let pts: [(t: Double, state: AttentionState)] = [(0, .focused), (1, .focused), (2, .distracted), (3, .distracted), (4, .focused)]
        let s = StateSegmenter.segments(pts, end: 6)
        XCTAssertEqual(s.count, 3)
        XCTAssertEqual(s[0], .init(start: 0, end: 2, state: .focused))
        XCTAssertEqual(s[1], .init(start: 2, end: 4, state: .distracted))
        XCTAssertEqual(s[2], .init(start: 4, end: 6, state: .focused))
        let f = StateSegmenter.fractions(s)
        XCTAssertEqual(f[0], 4.0 / 6, accuracy: 1e-9)
        XCTAssertEqual(f[1], 2.0 / 6, accuracy: 1e-9)
        XCTAssertTrue(StateSegmenter.segments([]).isEmpty)
    }

    func testStats() {
        XCTAssertEqual(UIStats.pearson([1, 2, 3, 4], [2, 4, 6, 8])!, 1, accuracy: 1e-9)
        XCTAssertEqual(UIStats.pearson([1, 2, 3, 4], [8, 6, 4, 2])!, -1, accuracy: 1e-9)
        XCTAssertNil(UIStats.pearson([1, 2], [1, 2]))
        XCTAssertNil(UIStats.pearson([1, 1, 1], [1, 2, 3]))
        XCTAssertEqual(UIStats.minMax([2, 4, 6]), [0, 0.5, 1])
        XCTAssertEqual(UIStats.minMax([3, 3]), [0.5, 0.5])
        XCTAssertEqual(UIStats.median([3, 1, 2]), 2)
        XCTAssertEqual(UIStats.median([4, 1, 2, 3]), 2.5)
        XCTAssertNil(UIStats.median([]))
        XCTAssertEqual(UIStats.ema([0, 10], alpha: 0.5), [0, 5])
    }

    // MARK: Effects axis

    func testEffectAxisDomainAndVerdict() {
        XCTAssertEqual(EffectAxis.domain([]), -0.2...0.2)
        let e = ArmEffect(arm: .haptic, policy: .reactive, effect: 0.3, lower: 0.1, upper: 0.52, level: 0.9, n: 12)
        let d = EffectAxis.domain([e])
        XCTAssertEqual(d.upperBound, 0.6, accuracy: 1e-9)
        XCTAssertEqual(d.lowerBound, -d.upperBound)
        XCTAssertEqual(EffectAxis.verdict(e), "Likely helps you")
        XCTAssertEqual(EffectAxis.verdict(ArmEffect(arm: .chime, policy: .reactive, effect: 0, lower: -0.2, upper: 0.2, level: 0.9, n: 9)), "Still learning")
        XCTAssertEqual(EffectAxis.verdict(ArmEffect(arm: .chime, policy: .reactive, effect: -0.3, lower: -0.5, upper: -0.1, level: 0.9, n: 9)), "Likely not for you")
        XCTAssertEqual(EffectAxis.verdict(ArmEffect(arm: .chime, policy: .reactive, effect: 0.3, lower: 0.1, upper: 0.5, level: 0.9, n: 1)), "Too few tries yet")
        // Guardrail: the experimental arm is never described as helping.
        XCTAssertEqual(EffectAxis.verdict(ArmEffect(arm: .lightPulse, policy: .reactive, effect: 0.4, lower: 0.2, upper: 0.6, level: 0.9, n: 30)), "Experimental, under test")
    }

    // MARK: Face mesh / gauges

    func testFaceMeshMapperFitsAndMirrors() {
        let rect = CGRect(x: 0, y: 0, width: 200, height: 100)
        let m = FaceMeshMapper.map([CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 1)], into: rect, mirror: false)
        XCTAssertEqual(m[0], CGPoint(x: 50, y: 0))
        XCTAssertEqual(m[1], CGPoint(x: 150, y: 100))
        let mm = FaceMeshMapper.map([CGPoint(x: 0, y: 0.5)], into: rect, mirror: true)
        XCTAssertEqual(mm[0], CGPoint(x: 150, y: 50))
    }

    func testSyntheticFaceMeshRespondsToBlinkAndJaw() {
        var open = SyntheticFaceMesh.Inputs()
        var closed = SyntheticFaceMesh.Inputs(); closed.blinkL = 1; closed.jaw = 1
        open.blinkL = 0
        let a = SyntheticFaceMesh.shape(open, t: 0), b = SyntheticFaceMesh.shape(closed, t: 0)
        func height(_ loop: [CGPoint]) -> CGFloat { (loop.map(\.y).max() ?? 0) - (loop.map(\.y).min() ?? 0) }
        // loops: 3 rings, eyeL, eyeR, mouth
        XCTAssertLessThan(height(b.loops[3]), height(a.loops[3]) * 0.3)
        XCTAssertGreaterThan(height(b.loops[5]), height(a.loops[5]) * 2)
        for p in a.loops.flatMap({ $0 }) + a.vertices {
            XCTAssertTrue((0...1).contains(p.x) && (0...1).contains(p.y))
        }
    }

    func testGaugeScaleClamps() {
        XCTAssertEqual(GaugeScale.blinkRate(20), 0.5, accuracy: 1e-9)
        XCTAssertEqual(GaugeScale.blinkRate(400), 1)
        XCTAssertEqual(GaugeScale.perclos(-1), 0)
        XCTAssertEqual(GaugeScale.headAngle(0), 0.5, accuracy: 1e-9)
        XCTAssertEqual(GaugeScale.clamp01(.nan), 0)
    }

    func testBreathPhase() {
        let (e0, inhale0, left0) = BreathPhase.at(0, breathsPerMinute: 6)
        XCTAssertEqual(e0, 0, accuracy: 1e-9); XCTAssertTrue(inhale0); XCTAssertEqual(left0, 5, accuracy: 1e-9)
        let (e5, inhale5, _) = BreathPhase.at(5, breathsPerMinute: 6)
        XCTAssertEqual(e5, 1, accuracy: 1e-9); XCTAssertFalse(inhale5)
        let (e10, _, _) = BreathPhase.at(10, breathsPerMinute: 6)
        XCTAssertEqual(e10, 0, accuracy: 1e-9)
    }

    func testFlagsPersistence() {
        let d = UserDefaults(suiteName: "UITestsLogic.flags")!
        d.removePersistentDomain(forName: "UITestsLogic.flags")
        XCTAssertEqual(UIPrefs.loadFlags(d), FeatureFlags())
        var f = FeatureFlags(); f.demoDirector = true; f.forceReplay = true
        UIPrefs.saveFlags(f, d)
        XCTAssertEqual(UIPrefs.loadFlags(d), f)
    }

    func testPrivacyCopyFollowsGuardrails() {
        let text = (PrivacyView.storedCategories.flatMap { [$0.1, $0.2] } + PrivacyView.limitations).joined(separator: " ").lowercased()
        for banned in ["diagnose ", "clinically proven", "reads your brain", "detect adhd", " first "] {
            XCTAssertFalse(text.contains(banned), banned)
        }
        XCTAssertEqual(PrivacyView.limitations.count, 5)
        XCTAssertEqual(PrivacyView.storedCategories.count, 11)
    }

    // MARK: Preview driver (SYNTHETIC replay; preview-only heuristics)

    @MainActor
    func testPreviewDriverFillsModelAndBuildsReport() throws {
        let model = LiveSessionModel()
        let d = ReplayPreviewDriver(model: model)
        try XCTSkipIf(d.stream == nil || d.timeline == nil, "replay / brain fixtures not bundled")
        d.simulate(seconds: 180, studyMode: true)
        XCTAssertTrue(model.isReplay)
        XCTAssertEqual(model.elapsed, 180, accuracy: 0.2)
        XCTAssertEqual(model.recentFrames.count, 100)
        XCTAssertEqual(model.stateHistory.count, 180)
        XCTAssertEqual(model.forecastLine.count, 15)
        XCTAssertNotNil(model.heartRate)
        XCTAssertFalse(model.posteriors.isEmpty)
        XCTAssertEqual(model.phase, .finished)

        // The preview heuristic should broadly follow the SYNTHETIC ground-truth labels.
        let labels = d.stream!.labels
        var agree = 0, n = 0
        for e in model.stateHistory {
            let i = Int((e.t * 10).rounded()) - 1
            guard i >= 0, i < labels.count, let truth = labels[i] else { continue }
            n += 1; if truth == e.state { agree += 1 }
        }
        XCTAssertGreaterThan(Double(agree) / Double(n), 0.75, "preview heuristic agreement \(agree)/\(n)")

        let report = PreviewReportBuilder.build(record: d.record(lectureID: "sample_lecture"), timeline: d.timeline,
                                                lectureTitle: "Sample", effects: model.effects, nudgeCount: d.nudgeCount,
                                                preemptiveCount: d.preemptiveCount)
        XCTAssertTrue(report.isReplay)
        XCTAssertTrue(report.contentIsMock)
        XCTAssertFalse(report.segments.isEmpty)
        XCTAssertFalse(report.refocusTimes.isEmpty)
        XCTAssertEqual(report.predictedDrift.count, report.actualDrift.count)
        XCTAssertNotNil(report.driftCorrelation)
        XCTAssertEqual(report.biggestDriftActivations?.count, 7)
        XCTAssertTrue(report.insights.allSatisfy { !$0.text.isEmpty })
        XCTAssertTrue(report.insights.first?.text.contains("SYNTHETIC") ?? false)
    }

    @MainActor
    func testPreviewServicesSeedSyntheticHistory() async {
        let s = PreviewUIServices(model: LiveSessionModel())
        await s.seedingFinished()
        let list = s.listSessions()
        XCTAssertEqual(list.count, 6)
        XCTAssertTrue(list.allSatisfy(\.isReplay))
        XCTAssertGreaterThanOrEqual(list[0].startedAt, list[1].startedAt)
        let summary = await s.aggregateSummary()
        XCTAssertEqual(summary?.sessionCount, 6)
        XCTAssertEqual(summary?.isReplay, true)
        XCTAssertEqual(s.profileSeries().points.count, 6)
        let r = await s.report(for: list[0].id)
        XCTAssertNotNil(r)
    }
}
