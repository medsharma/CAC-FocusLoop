import XCTest
@testable import FocusLoop

final class ForecastingTests: XCTestCase {

    private func frames(_ n: Int, start: Double = 0, _ edit: (inout FeatureFrame, Int) -> Void = { _, _ in }) -> [FeatureFrame] {
        (0..<n).map { i in
            var f = FeatureFrame(t: start + Double(i) / 10)
            f.faceDetected = 1; f.inWorkZone = 1; f.blinkRate = 14; f.heartRate = 70; f.heartRateQuality = 0.8
            edit(&f, i)
            return f
        }
    }

    // MARK: Buffer / window handling

    func testBufferKeepsLast200FramesAnd20Content() {
        var b = ForecastWindowBuffer()
        XCTAssertFalse(b.isWarm)
        for f in frames(250) { b.append(f) }
        XCTAssertTrue(b.isWarm)
        XCTAssertEqual(b.frames.count, 200)
        XCTAssertEqual(b.frames.first!.t, 5.0, accuracy: 1e-9)
        XCTAssertEqual(b.frames.last!.t, 24.9, accuracy: 1e-9)
        for i in 0..<25 { b.append(content: ContentFeatures(networks: Array(repeating: Float(i), count: 7), engagement: 0.5,
                                                           driftRisk: 0, secondsToValley: 60, inValley: false)) }
        XCTAssertEqual(b.content.count, 20)
        XCTAssertEqual(b.content.first!.networks[0], 5)
        b.clearContent()
        XCTAssertTrue(b.content.isEmpty)
        b.reset()
        XCTAssertTrue(b.frames.isEmpty)
    }

    func testShortWindowIsFrontPaddedWithOldestFrame() {
        let f = frames(50) { f, i in f.perclos = Float(i) }
        let fixed = ForecastWindowBuffer.fixedFrames(f)
        XCTAssertEqual(fixed.count, 200)
        XCTAssertEqual(fixed[0].perclos, 0)
        XCTAssertEqual(fixed[149].perclos, 0)
        XCTAssertEqual(fixed[150].perclos, 0)
        XCTAssertEqual(fixed[199].perclos, 49)
        let m = ForecastWindowBuffer.frameMatrix(f)
        XCTAssertEqual(m.count, 200 * 42)
        let perclosIndex = FeatureFrame.featureNames.firstIndex(of: "perclos")!
        XCTAssertEqual(m[199 * 42 + perclosIndex], 49)
        // too long -> newest 200
        let long = ForecastWindowBuffer.fixedFrames(frames(300))
        XCTAssertEqual(long.first!.t, 10.0, accuracy: 1e-9)
        // empty
        XCTAssertTrue(ForecastWindowBuffer.fixedFrames([]).isEmpty)
        XCTAssertEqual(ForecastWindowBuffer.frameMatrix([]).count, 200 * 42)
        XCTAssertNil(ForecastWindowBuffer.contentMatrix([]))
        XCTAssertEqual(ForecastWindowBuffer.contentMatrix([.none])!.count, 20 * 11)
    }

    // MARK: Heuristic

    func testHeuristicIsMonotoneInEverySignal() {
        let h = HeuristicLapseForecaster()
        let base = HeuristicLapseForecaster.Signals()
        let p0 = h.probability(base, usesContent: true)
        let bumps: [(inout HeuristicLapseForecaster.Signals) -> Void] = [
            { $0.blinkRateTrend += 8 }, { $0.gazeDispersion += 0.1 }, { $0.offZoneDwell += 3 },
            { $0.perclosSlope += 0.05 }, { $0.heartRateDelta += 8 }, { $0.driftRisk += 1 }, { $0.valleyProximity += 0.5 },
        ]
        for (i, bump) in bumps.enumerated() {
            var s = base
            var prev = p0
            for _ in 0..<5 {
                bump(&s)
                let p = h.probability(s, usesContent: true)
                XCTAssertGreaterThanOrEqual(p, prev, "signal \(i) not monotone")
                prev = p
            }
            XCTAssertGreaterThan(prev, p0, "signal \(i) has no effect")
        }
    }

    func testHeuristicRisesWithDriftingFramesAndContent() {
        let h = HeuristicLapseForecaster()
        let calm = frames(200)
        let drifting = frames(200) { f, i in
            f.blinkRate = 14 + Float(i) * 0.06
            f.perclos = Float(i) * 0.0005
            f.gazeYaw = i > 100 ? 0.15 * sin(Float(i)) : 0
            f.offZoneDwell = i > 180 ? Float(i - 180) / 10 : 0
            f.heartRateDelta = i > 150 ? 8 : 0
        }
        let p1 = h.forecast(frames: calm, content: [], t: 20).probability
        let p2 = h.forecast(frames: drifting, content: [], t: 20).probability
        XCTAssertGreaterThan(p2, p1)
        let valley = ContentFeatures(networks: Array(repeating: 0, count: 7), engagement: 0.2, driftRisk: 1.5,
                                     secondsToValley: 2, inValley: false)
        let r = h.forecast(frames: calm, content: Array(repeating: valley, count: 20), t: 20)
        XCTAssertTrue(r.usedContent)
        XCTAssertEqual(r.source, "heuristic")
        XCTAssertGreaterThan(r.probability, p1)
        XCTAssertFalse(h.forecast(frames: calm, content: [], t: 20).usedContent)
        // empty input must not crash
        XCTAssertTrue((0...1).contains(h.forecast(frames: [], content: [], t: 0).probability))
    }

    // MARK: Personal threshold

    func testThresholdConvergesToTargetAlertRate() {
        var th = PersonalThreshold(initial: 0.5, targetAlertRate: 0.10, feedbackWeight: 0)
        var rng = SplitMix(seed: 7)
        for _ in 0..<6000 { th.observe(probability: rng.nextFloat(), lapseOccurred: false) }
        // uniform predictions -> threshold near the 90th percentile
        XCTAssertEqual(th.value, 0.9, accuracy: 0.05)
        var recent = PersonalThreshold(initial: th.value, targetAlertRate: 0.1, feedbackWeight: 0)
        for _ in 0..<2000 { recent.observe(probability: rng.nextFloat(), lapseOccurred: false) }
        XCTAssertEqual(recent.alertRate, 0.1, accuracy: 0.03)
    }

    func testThresholdOutcomeFeedbackAndBounds() {
        var a = PersonalThreshold(initial: 0.5)
        var b = PersonalThreshold(initial: 0.5)
        a.observe(probability: 0.7, lapseOccurred: false)  // false alarm -> up more
        b.observe(probability: 0.7, lapseOccurred: true)   // hit
        XCTAssertGreaterThan(a.value, b.value)
        var c = PersonalThreshold(initial: 0.5)
        var d = PersonalThreshold(initial: 0.5)
        c.observe(probability: 0.2, lapseOccurred: true)   // miss -> down more
        d.observe(probability: 0.2, lapseOccurred: false)
        XCTAssertLessThan(c.value, d.value)
        var e = PersonalThreshold(initial: 0.5)
        for _ in 0..<5000 { e.observe(probability: 1, lapseOccurred: false) }
        XCTAssertLessThanOrEqual(e.value, e.upperBound)
        XCTAssertEqual(a.hits + a.falseAlarms, a.alerts)
    }

    func testThresholdCodableAndDemoDirector() throws {
        var th = PersonalThreshold(initial: 0.4)
        th.observe(probability: 0.9, lapseOccurred: true)
        let back = try PersonalThreshold.restore(from: th.serialized())
        XCTAssertEqual(back, th)
        XCTAssertEqual(th.effective(demoDirector: false), th.value)
        XCTAssertLessThan(th.effective(demoDirector: true), th.value)  // DEMONSTRATION ONLY
        let h = HeuristicLapseForecaster(threshold: th)
        let normal = h.threshold
        h.demoDirector = true
        XCTAssertLessThan(h.threshold, normal)
    }

    // MARK: Risk stream

    func testRiskStreamRunsAheadAndRisesIntoValley() throws {
        let s = LapseRiskStream()
        let risk = LapseRisk(t: 100, probability: 0.3, threshold: 0.5, usedContent: false, source: "heuristic")
        let line = s.update(risk: risk)
        XCTAssertEqual(line.count, 16)
        XCTAssertEqual(line.first!.t, 100)
        XCTAssertEqual(line.last!.t, 115, accuracy: 1e-9)
        XCTAssertTrue(line.allSatisfy { (0...1).contains($0.p) })
        // no content: relaxes monotonically toward the base rate
        XCTAssertGreaterThan(line[1].p, line.last!.p)
        // MOCK sample lecture: valley starts at 69 s -> line from 58 s climbs
        let tl = try BrainContentTimeline(named: "sample_lecture")
        s.reset()
        let lc = s.update(risk: LapseRisk(t: 10, probability: 0.15, threshold: 0.5, usedContent: true, source: "model"),
                          timeline: tl, playhead: 58)
        XCTAssertGreaterThan(lc.last!.p, lc.first!.p)
        let free = LapseRiskStream().update(risk: LapseRisk(t: 10, probability: 0.15, threshold: 0.5, usedContent: true,
                                                            source: "model"))
        XCTAssertGreaterThan(lc.last!.p, free.last!.p)
    }

    func testRiskStreamTrendSlope() {
        let s = LapseRiskStream()
        for i in 0..<6 {
            s.update(risk: LapseRisk(t: Double(i), probability: 0.1 + 0.05 * Float(i), threshold: 0.5, usedContent: false,
                                     source: "model"))
        }
        XCTAssertGreaterThan(s.trendSlope(), 0)
    }

    // MARK: Core ML model (skips if not bundled)

    func testCoreMLForecasterLoadsAndPredicts() throws {
        guard let m = CoreMLLapseForecaster(bundle: .main) else {
            throw XCTSkip("LapseForecaster models are not bundled")
        }
        XCTAssertTrue(m.hasContentModel || m.hasFeaturesOnlyModel)
        let fr = frames(200)
        let r1 = m.forecast(frames: fr, content: [], t: 20)
        XCTAssertTrue((0...1).contains(r1.probability))
        XCTAssertEqual(r1.source, "model")
        XCTAssertFalse(r1.usedContent)
        if m.hasContentModel {
            let tl = try BrainContentTimeline(named: "sample_lecture")
            let content = (0..<20).map { tl.features(at: Double(50 + $0)) }
            let r2 = m.forecast(frames: fr, content: content, t: 20)
            XCTAssertTrue(r2.usedContent)
            XCTAssertTrue((0...1).contains(r2.probability))
        }
        // buffer path + short window
        for f in frames(30) { m.ingest(frame: f) }
        XCTAssertEqual(m.forecast(t: 3).source, "model")
        // threshold adapts through observe()
        let before = m.threshold
        m.observe(prediction: LapseRisk(t: 3, probability: 0.99, threshold: before, usedContent: false, source: "model"),
                  lapseOccurred: false)
        XCTAssertNotEqual(m.threshold, before)
    }
}

/// Deterministic RNG for tests.
private struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func nextFloat() -> Float { Float(next() >> 40) / Float(1 << 24) }
}
