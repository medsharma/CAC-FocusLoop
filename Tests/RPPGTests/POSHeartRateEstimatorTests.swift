import XCTest
@testable import FocusLoop

/// All signals here are SYNTHETIC (see `RPPGSynth`). Nothing in this file is a real measurement.
final class POSHeartRateEstimatorTests: XCTestCase {

    @discardableResult
    private func run(_ est: POSHeartRateEstimator, _ samples: [RPPGSynth.Sample], timeOffset: Double = 0,
                     motion: (Double) -> Float = { _ in 0 },
                     onUpdate: ((HeartRateSample) -> Void)? = nil) -> [HeartRateSample] {
        var updates: [HeartRateSample] = []
        var lastT: Double?
        for s in samples {
            let t = s.t + timeOffset
            est.ingest(rgb: s.rgb, t: t, headMotion: motion(t))
            if let l = est.latest, l.t != lastT {
                lastT = l.t
                updates.append(l)
                onUpdate?(l)
            }
        }
        return updates
    }

    // MARK: BPM recovery (SYNTHETIC)

    func testRecoversKnownBPMWithinThree() {
        var table: [String] = []
        for bpm in RPPGSynth.bpms {
            let est = POSHeartRateEstimator()
            let updates = run(est, RPPGSynth.generate(bpm: bpm))
            guard let last = updates.last else { return XCTFail("no estimate for \(bpm)") }
            XCTAssertEqual(Double(last.bpm), bpm, accuracy: 3, "final estimate for \(bpm) BPM")
            for u in updates where u.t >= 12 {
                XCTAssertEqual(Double(u.bpm), bpm, accuracy: 3, "estimate at t=\(u.t) for \(bpm) BPM")
            }
            XCTAssertGreaterThan(last.quality, 0.7, "clean pulse should be high quality")
            let mae = updates.map { abs(Double($0.bpm) - bpm) }.reduce(0, +) / Double(updates.count)
            table.append(String(format: "RPPG_SYNTH_RESULT bpm=%.0f est=%.2f mae=%.2f snr_db=%.2f quality=%.3f n=%d",
                                bpm, last.bpm, mae, last.snrDB, last.quality, updates.count))
        }
        // Parsed by ml/rppg_reference (swift_results_SYNTHETIC.json) for the Python cross-check.
        for line in table { print(line) }
    }

    func testNilUntilTenSecondsBuffered() {
        let est = POSHeartRateEstimator()
        run(est, RPPGSynth.generate(bpm: 72, duration: 9.5))
        XCTAssertNil(est.latest)
        run(est, RPPGSynth.generate(bpm: 72, duration: 1.5), timeOffset: 9.5)
        XCTAssertNotNil(est.latest)
    }

    func testUpdatesAboutOnceASecond() {
        let est = POSHeartRateEstimator()
        let updates = run(est, RPPGSynth.generate(bpm: 72, duration: 25))
        XCTAssertGreaterThanOrEqual(updates.count, 14)
        XCTAssertLessThanOrEqual(updates.count, 17)
        for (a, b) in zip(updates, updates.dropFirst()) {
            XCTAssertEqual(b.t - a.t, 1, accuracy: 0.05)
        }
    }

    // MARK: Quality

    func testQualityLowForPureNoise() {
        for seed in UInt64(1)...UInt64(5) {
            let est = POSHeartRateEstimator()
            let updates = run(est, RPPGSynth.generate(bpm: 72, pulseAmplitude: 0, seed: seed))
            let meanQ = updates.map(\.quality).reduce(0, +) / Float(max(updates.count, 1))
            XCTAssertLessThan(meanQ, 0.3, "pure noise (seed \(seed)) should be low quality")
            XCTAssertLessThan(updates.last?.snrDB ?? 0, 0)
        }
    }

    func testQualityHighForCleanPulse() {
        let est = POSHeartRateEstimator()
        let updates = run(est, RPPGSynth.generate(bpm: 66, noiseSigma: 0.0002))
        XCTAssertGreaterThan(updates.last?.quality ?? 0, 0.9)
        XCTAssertGreaterThan(updates.last?.snrDB ?? 0, 4)
    }

    // MARK: Motion gating

    func testMotionGatingTriggersAndHoldsLastGoodValue() {
        let est = POSHeartRateEstimator()
        // 0-20 s calm at 72 BPM; 20-25 s head motion (1.2 rad/s) with a strong 2.2 Hz (132 BPM)
        // motion artefact on all channels; 25-35 s calm again.
        var samples = RPPGSynth.generate(bpm: 72, duration: 35)
        for i in samples.indices where samples[i].t >= 20 && samples[i].t < 25 {
            let a = Float(1 + 0.03 * sin(2 * Double.pi * 2.2 * samples[i].t))
            samples[i].rgb *= SIMD3<Float>(a, a * 1.01, a * 0.98)
        }
        var heldBefore: Float?
        var flaggedUpdates: [HeartRateSample] = []
        var afterUpdates: [HeartRateSample] = []
        run(est, samples, motion: { $0 >= 20 && $0 < 25 ? 1.2 : 0 }) { u in
            if u.t < 20 { heldBefore = u.bpm }
            if u.t >= 20 && u.t < 25 { flaggedUpdates.append(u) }
            if u.t >= 27.5 { afterUpdates.append(u) }
        }
        XCTAssertFalse(flaggedUpdates.isEmpty)
        for u in flaggedUpdates {
            XCTAssertTrue(u.motionFlagged, "update at \(u.t) should be motion-flagged")
            XCTAssertEqual(u.bpm, heldBefore, "flagged reading should hold the last good BPM")
        }
        XCTAssertFalse(afterUpdates.isEmpty)
        XCTAssertTrue(afterUpdates.allSatisfy { !$0.motionFlagged }, "flag clears 2 s after motion stops")
    }

    func testMotionFlagWithoutPriorGoodValueHasZeroQuality() {
        let est = POSHeartRateEstimator()
        run(est, RPPGSynth.generate(bpm: 72, duration: 12), motion: { _ in 2 })
        XCTAssertEqual(est.latest?.motionFlagged, true)
        XCTAssertEqual(est.latest?.quality, 0)
    }

    // MARK: Baseline

    func testBaselineEstablishedFromCalmGoodReadings() {
        let est = POSHeartRateEstimator()
        XCTAssertNil(est.restingBaseline)
        run(est, RPPGSynth.generate(bpm: 60, duration: 40))
        let b = try? XCTUnwrap(est.restingBaseline)
        XCTAssertEqual(Double(b ?? 0), 60, accuracy: 3)
        XCTAssertEqual(est.latest?.baselineBPM, est.restingBaseline)
    }

    func testPersistedBaselineMovesTowardCalmReadings() {
        let est = POSHeartRateEstimator(restingBaseline: 80)
        run(est, RPPGSynth.generate(bpm: 66, duration: 40))
        let b = est.restingBaseline ?? 80
        XCTAssertLessThan(b, 80)
        XCTAssertGreaterThan(b, 66)
        est.reset()
        XCTAssertEqual(est.restingBaseline, b, "reset() keeps the persisted baseline")
    }

    func testBaselineNotUpdatedWhenNotCalm() {
        let est = POSHeartRateEstimator(restingBaseline: 80)
        // 0.4 rad/s: below the motion gate (not flagged) but above the calm threshold.
        run(est, RPPGSynth.generate(bpm: 66, duration: 40), motion: { _ in 0.4 })
        XCTAssertEqual(est.restingBaseline, 80)
        XCTAssertEqual(est.latest?.motionFlagged, false)
    }

    // MARK: Waveform, gaps

    func testWaveformIsLastThreeSecondsNormalised() throws {
        let est = POSHeartRateEstimator()
        run(est, RPPGSynth.generate(bpm: 72, duration: 15))
        let wf = try XCTUnwrap(est.latest?.waveform)
        XCTAssertEqual(wf.count, 90)
        XCTAssertTrue(wf.allSatisfy { $0 >= -1 && $0 <= 1 })
        XCTAssertEqual(wf.map { abs($0) }.max() ?? 0, 1, accuracy: 1e-5)
        // ~3.6 cycles of 1.2 Hz in 3 s -> 6-8 zero crossings.
        let crossings = zip(wf, wf.dropFirst()).filter { ($0 < 0) != ($1 < 0) }.count
        XCTAssertTrue((6...8).contains(crossings), "crossings \(crossings)")
    }

    func testGapRestartsBuffer() {
        let est = POSHeartRateEstimator()
        run(est, RPPGSynth.generate(bpm: 72, duration: 12))
        let before = est.latest
        XCTAssertNotNil(before)
        // 3 s gap (face lost), then 5 s of signal: not enough for a new estimate yet.
        run(est, RPPGSynth.generate(bpm: 72, duration: 5), timeOffset: 15)
        XCTAssertEqual(est.latest, before, "no new estimate until 10 s of fresh signal")
        run(est, RPPGSynth.generate(bpm: 72, duration: 6), timeOffset: 20.03)
        XCTAssertGreaterThan(est.latest?.t ?? 0, 25)
    }

    func testIgnoresInvalidSamples() {
        let est = POSHeartRateEstimator()
        est.ingest(rgb: SIMD3<Float>(.nan, 0.3, 0.2), t: 0, headMotion: 0)
        est.ingest(rgb: SIMD3<Float>(0, 0, 0), t: 0.1, headMotion: 0)
        run(est, RPPGSynth.generate(bpm: 96, duration: 15))
        XCTAssertEqual(Double(est.latest?.bpm ?? 0), 96, accuracy: 3)
    }

    func testHandlesSixtyHertzInput() {
        // ARKit can deliver 60 fps; the estimator resamples to 30 Hz.
        var samples: [RPPGSynth.Sample] = []
        let a = RPPGSynth.generate(bpm: 84, duration: 20)
        for (s0, s1) in zip(a, a.dropFirst()) {
            samples.append(s0)
            samples.append(RPPGSynth.Sample(t: (s0.t + s1.t) / 2, rgb: (s0.rgb + s1.rgb) / 2))
        }
        let est = POSHeartRateEstimator()
        run(est, samples)
        XCTAssertEqual(Double(est.latest?.bpm ?? 0), 84, accuracy: 3)
    }

    // MARK: DSP building blocks

    func testButterworthSectionsHaveMinus3dBAtCutoff() {
        let hp = RPPGBiquad.butterworthHighPass(cutoff: 0.7, fs: 30)
        let lp = RPPGBiquad.butterworthLowPass(cutoff: 3.0, fs: 30)
        XCTAssertEqual(hp.gain(at: 0.7, fs: 30), 1 / 2.0.squareRoot(), accuracy: 1e-6)
        XCTAssertEqual(lp.gain(at: 3.0, fs: 30), 1 / 2.0.squareRoot(), accuracy: 1e-6)
        XCTAssertLessThan(hp.gain(at: 0.1, fs: 30), 0.03)
        XCTAssertLessThan(lp.gain(at: 10, fs: 30), 0.1)
        XCTAssertEqual(hp.gain(at: 1.5, fs: 30) * lp.gain(at: 1.5, fs: 30), 1, accuracy: 0.1)
    }

    func testDetrendRemovesLine() {
        var x = (0..<100).map { 3 + 0.5 * Double($0) }
        RPPGDSP.detrendLinear(&x)
        XCTAssertLessThan(x.map(abs).max() ?? 1, 1e-9)
    }

    // MARK: Performance

    func testIngestIsFast() {
        let samples = RPPGSynth.generate(bpm: 72, duration: 60)
        let est = POSHeartRateEstimator()
        let start = CFAbsoluteTimeGetCurrent()
        run(est, samples)
        let perSampleMs = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(samples.count)
        print(String(format: "RPPG_PERF ingest_ms_per_sample=%.4f (60 s @ 30 Hz incl. 50 spectral updates)", perSampleMs))
        // Budget: one camera frame is 33 ms; we allow 0.5 ms average even in an unoptimised Debug build.
        XCTAssertLessThan(perSampleMs, 0.5)
    }

    func testIngestPerformanceMetric() {
        let samples = RPPGSynth.generate(bpm: 72, duration: 30)
        measure {
            let est = POSHeartRateEstimator()
            for s in samples { est.ingest(rgb: s.rgb, t: s.t, headMotion: 0) }
        }
    }
}
