import XCTest
@testable import FocusLoop

// MARK: - Helpers

/// Drives a SARTEngine with a simulated 60 Hz display and an injected tap policy.
private func runSART(config: SARTConfig, origin: Double = 1000, fps: Double = 60,
                     tapRT: (SARTTrial) -> Double?) -> SARTEngine {
    let engine = SARTEngine(config: config, clockOrigin: origin)
    var t = origin + 0.5
    var tapped = Set<Int>()
    var guardCount = 0
    var pendingTaps: [Double] = []
    while !engine.isFinished && guardCount < 1_000_000 {
        engine.tick(frameTime: t)
        if let i = engine.currentTrialIndex, !tapped.contains(i) {
            tapped.insert(i)
            if let rt = tapRT(engine.trials[i]) { pendingTaps.append((origin + engine.trials[i].onset + rt)) }
        }
        let next = t + 1 / fps
        while let p = pendingTaps.first, p < next { engine.tap(at: p); pendingTaps.removeFirst() }
        t = next
        guardCount += 1
    }
    return engine
}

private func frames(from start: Double, to end: Double) -> [FeatureFrame] {
    stride(from: start, to: end, by: 0.1).map { t in
        var f = FeatureFrame(t: (t * 10).rounded() / 10)
        f.faceDetected = 1
        return f
    }
}

private func goTrial(_ i: Int, onset: Double, rt: Double?, nogo: Bool = false) -> SARTTrial {
    SARTTrial(index: i, digit: nogo ? 3 : 5, fontSize: 72, isNoGo: nogo, onset: onset, maskOnset: onset + 0.25,
              tapTime: rt.map { onset + $0 }, end: onset + 1.15)
}

// MARK: - SART engine

final class SARTEngineTests: XCTestCase {
    func testSequenceNoGoRatioAndSizes() {
        var rng = CalibrationSplitMix64(seed: 42)
        let seq = SARTEngine.makeSequence(count: 900, using: &rng)
        XCTAssertEqual(seq.count, 900)
        XCTAssertEqual(seq.filter { $0.digit == 3 }.count, 100)   // exactly 1/9 in full blocks
        XCTAssertEqual(Set(seq.map(\.digit)), Set(1...9))
        XCTAssertEqual(Set(seq.map(\.fontSize)), Set(SARTConfig().fontSizes))
        for i in 1..<seq.count { XCTAssertNotEqual(seq[i].digit, seq[i - 1].digit, "no immediate repeat at \(i)") }

        var rng2 = CalibrationSplitMix64(seed: 1)
        let def = SARTEngine.makeSequence(count: SARTConfig().trialCount, using: &rng2)
        let ratio = Double(def.filter { $0.digit == 3 }.count) / Double(def.count)
        XCTAssertEqual(ratio, 1.0 / 9.0, accuracy: 0.01)
    }

    func testDefaultLengthAboutThreeMinutes() {
        let c = SARTConfig()
        XCTAssertEqual(c.trialCount, 160)
        XCTAssertEqual(c.estimatedDuration, 184, accuracy: 1e-9)
        XCTAssertEqual(SARTConfig.practice.trialCount, 18)
        XCTAssertTrue(SARTConfig.practice.isPractice)
    }

    func testSeededSequenceIsReproducible() {
        var c = SARTConfig(); c.seed = 9
        XCTAssertEqual(SARTEngine(config: c).sequence.map(\.digit), SARTEngine(config: c).sequence.map(\.digit))
    }

    func testTimingRecordsOnSessionClock() {
        var c = SARTConfig(); c.trialCount = 27; c.seed = 3
        let e = runSART(config: c, fps: 60) { _ in nil }
        XCTAssertTrue(e.isFinished)
        XCTAssertEqual(e.trials.count, 27)
        XCTAssertEqual(e.trials[0].onset, 0.5, accuracy: 1e-9)   // session time = media - origin
        for tr in e.trials {
            // Stimulus lasts 250 ms quantised to the 60 Hz frame grid (15 frames).
            XCTAssertEqual(tr.stimulusDuration!, 0.25, accuracy: 1.0 / 60 / 2 + 1e-6)
            XCTAssertTrue(tr.timingValid)
            XCTAssertEqual(tr.end! - tr.onset, 1.15, accuracy: 1.0 / 60 / 2 + 1e-6)
        }
        for i in 1..<e.trials.count { XCTAssertEqual(e.trials[i].onset, e.trials[i - 1].end!, accuracy: 1e-12) }
        XCTAssertEqual(e.display, .finished)
    }

    func testTimingAt120Hz() {
        var c = SARTConfig(); c.trialCount = 9; c.seed = 4
        let e = runSART(config: c, fps: 120) { _ in nil }
        for tr in e.trials { XCTAssertEqual(tr.stimulusDuration!, 0.25, accuracy: 1.0 / 120 / 2 + 1e-6) }
    }

    func testScoringWithInjectedTaps() {
        var c = SARTConfig(); c.trialCount = 90; c.seed = 11
        // Tap every go trial at 350 ms except every 10th; also tap on every no-go (commission).
        let e = runSART(config: c) { tr in
            if tr.isNoGo { return 0.3 }
            return tr.index % 10 == 0 ? nil : 0.35
        }
        let r = e.result
        let goMissed = r.trials.filter { !$0.isNoGo && $0.index % 10 == 0 }.count
        XCTAssertEqual(r.noGoCount, 10)
        XCTAssertEqual(r.commissionCount, 10)
        XCTAssertEqual(r.omissionCount, goMissed)
        XCTAssertEqual(r.meanRT!, 0.35, accuracy: 1e-6)
        XCTAssertEqual(r.rtCV!, 0, accuracy: 1e-6)
        for tr in r.trials where tr.tapped { XCTAssertEqual(tr.rtMs!, tr.isNoGo ? 300 : 350, accuracy: 1e-6) }
    }

    func testTapAttributionAndFirstTapOnly() {
        var c = SARTConfig(); c.trialCount = 3; c.seed = 5
        let e = SARTEngine(config: c, clockOrigin: 0)
        e.tap(at: 0.1)                         // before any trial: ignored
        e.tick(frameTime: 1.0)                 // trial 0 onset at 1.0
        e.tap(at: 1.3); e.tap(at: 1.4)         // only first counts
        var t = 1.0
        while e.currentTrialIndex == 0 { t += 1.0 / 60; e.tick(frameTime: t) }
        e.tap(at: 2.0)                          // late-delivered tap stamped in trial 0's window: already tapped
        XCTAssertEqual(e.trials[0].rt!, 0.3, accuracy: 1e-9)
        XCTAssertNil(e.trials[1].tapTime)
        e.tap(at: t + 0.2)
        XCTAssertEqual(e.trials[1].rt!, 0.2, accuracy: 1e-9)
    }

    func testFrameGapInvalidatesTrial() {
        var c = SARTConfig(); c.trialCount = 2; c.seed = 5
        let e = SARTEngine(config: c)
        e.tick(frameTime: 0)
        e.tick(frameTime: 0.5)                 // 500 ms hitch
        XCTAssertFalse(e.trials[0].timingValid)
    }

    func testFinishCallbackAndCSV() throws {
        var c = SARTConfig(); c.trialCount = 9; c.seed = 2
        var finished: SARTResult?
        let e = SARTEngine(config: c, clockOrigin: 0)
        e.onFinish = { finished = $0 }
        var t = 0.0
        while !e.isFinished { e.tick(frameTime: t); if let i = e.currentTrialIndex, !e.trials[i].isNoGo { e.tap(at: e.trials[i].onset + 0.4) }; t += 1.0 / 60 }
        XCTAssertEqual(finished?.trials.count, 9)
        let csv = SARTTrialCSV.make(finished!.trials, isSynthetic: true)
        let lines = csv.split(separator: "\n").map(String.init)
        XCTAssertTrue(lines[0].hasPrefix("#") && lines[0].contains("SYNTHETIC"))
        XCTAssertEqual(lines[1], "trial,onset_s,digit,is_nogo,tapped,rt_ms,commission,omission")
        XCTAssertEqual(lines.count, 11)
        let nogo = lines.dropFirst(2).first { $0.split(separator: ",", omittingEmptySubsequences: false)[3] == "1" }!
        XCTAssertEqual(nogo.split(separator: ",", omittingEmptySubsequences: false).count, 8)
        XCTAssertTrue(nogo.hasSuffix(",3,1,0,,0,0"))
        let url = try SARTTrialCSV.write(finished!.trials, isSynthetic: false)
        XCTAssertEqual(url.lastPathComponent, "FocusLoop_sart_trials.csv")
        XCTAssertTrue(try String(contentsOf: url, encoding: .utf8).hasPrefix("# FocusLoop SART trials v1 (REAL device data)"))
    }
}

// MARK: - CV + labeler

final class SARTLabelerTests: XCTestCase {
    func testCoefficientOfVariation() {
        XCTAssertNil(CalibrationStats.coefficientOfVariation([0.4]))
        XCTAssertEqual(CalibrationStats.coefficientOfVariation([0.3, 0.3, 0.3])!, 0, accuracy: 1e-12)
        // mean 0.4, sample sd = sqrt(((0.1)^2 + 0 + (0.1)^2)/2) = 0.1 -> CV 0.25
        XCTAssertEqual(CalibrationStats.coefficientOfVariation([0.3, 0.4, 0.5])!, 0.25, accuracy: 1e-12)
        XCTAssertEqual(CalibrationStats.percentile([1, 2, 3, 4, 5], 0.8)!, 4.2, accuracy: 1e-12)
        XCTAssertEqual(CalibrationStats.percentile([7], 0.8)!, 7)
    }

    func testRollingCV() {
        var l = SARTLabeler(); l.cvWindowTrials = 4
        let trials = [0.3, 0.3, 0.3, 0.3, 0.6, 0.2].enumerated().map { goTrial($0.offset, onset: Double($0.offset) * 1.15, rt: $0.element) }
        let cv = l.rollingCV(trials)
        XCTAssertNil(cv[0]); XCTAssertNil(cv[1])
        XCTAssertEqual(cv[2]!, 0, accuracy: 1e-12)
        XCTAssertEqual(cv[3]!, 0, accuracy: 1e-12)
        XCTAssertEqual(cv[5]!, CalibrationStats.coefficientOfVariation([0.3, 0.3, 0.6, 0.2])!, accuracy: 1e-12)
        l.cvWindowTrials = 20   // clamped to 8
        XCTAssertEqual(l.rollingCV(trials)[5]!, CalibrationStats.coefficientOfVariation([0.3, 0.3, 0.3, 0.3, 0.6, 0.2])!, accuracy: 1e-12)
    }

    /// 120 trials (~138 s): steady RTs with small deterministic jitter, a volatile stretch around trials
    /// 60-70, and one commission at trial 100.
    private func constructedTrials() -> [SARTTrial] {
        (0..<120).map { i in
            let onset = 2.0 + Double(i) * 1.15
            if i == 100 { return goTrial(i, onset: onset, rt: 0.25, nogo: true) }   // commission
            let base = 0.35 + 0.01 * Double(i % 3)
            let volatile = (60...70).contains(i) ? (i % 2 == 0 ? 0.25 : -0.15) : 0
            return goTrial(i, onset: onset, rt: base + volatile)
        }
    }

    func testLabelerOnConstructedTrials() {
        let trials = constructedTrials()
        let fr = frames(from: 0, to: 150)
        let sid = UUID()
        let w = SARTLabeler().label(trials: trials, frames: fr, sessionID: sid)
        XCTAssertFalse(w.isEmpty)
        let lapses = w.filter(\.isLapse), focused = w.filter { !$0.isLapse }
        XCTAssertFalse(lapses.isEmpty); XCTAssertFalse(focused.isEmpty)

        // Commission at trial 100 -> a lapse window covering the 10 s before its onset.
        let cOnset = trials[100].onset
        XCTAssertTrue(lapses.contains { $0.start <= cOnset - 10 + 1e-9 && $0.end >= cOnset - 1e-9 })
        // Volatile stretch -> a lapse window overlapping trials 60-70.
        let vs = trials[60].onset, ve = trials[70].end!
        XCTAssertTrue(lapses.contains { $0.start < ve && $0.end > vs })

        for x in lapses { XCTAssertEqual(x.label, .distracted); XCTAssertEqual(x.source, .sart) }
        for x in focused {
            XCTAssertEqual(x.label, .focused)
            XCTAssertEqual(x.end - x.start, 10, accuracy: 1e-9)
            // focused windows never overlap lapse windows nor contain errors
            XCTAssertFalse(lapses.contains { $0.start < x.end && $0.end > x.start })
            XCTAssertFalse(trials.contains { $0.isError && $0.onset >= x.start && $0.onset < x.end })
        }
        // Lapse windows don't overlap one another (merged) and frames lie inside windows.
        for (a, b) in zip(lapses, lapses.dropFirst()) { XCTAssertLessThan(a.end, b.start + 1e-9) }
        for x in w {
            XCTAssertEqual(x.sessionID, sid)
            XCTAssertTrue(x.frames.allSatisfy { $0.t >= x.start && $0.t < x.end })
            XCTAssertGreaterThanOrEqual(x.frames.count, 50)
        }
        // Windows are clipped to the SART span.
        XCTAssertTrue(w.allSatisfy { $0.start >= trials[0].onset - 1e-9 && $0.end <= trials.last!.end! + 1e-9 })
    }

    func testLabelerNeedsFrames() {
        XCTAssertTrue(SARTLabeler().label(trials: constructedTrials(), frames: [], sessionID: UUID()).isEmpty)
        XCTAssertTrue(SARTLabeler().label(trials: [], frames: frames(from: 0, to: 10), sessionID: UUID()).isEmpty)
    }

    func testLabelerOnEngineOutput() {
        var c = SARTConfig(); c.trialCount = 90; c.seed = 21
        let e = runSART(config: c, origin: 500) { tr in tr.isNoGo ? (tr.index > 60 ? 0.3 : nil) : 0.3 }
        let r = e.result
        let w = SARTLabeler().label(trials: r.trials, frames: frames(from: 0, to: r.end! + 1), sessionID: UUID())
        XCTAssertTrue(w.contains(where: \.isLapse))
        XCTAssertTrue(w.contains { !$0.isLapse })
    }
}

// MARK: - Experience sampling

final class ExperienceSamplerTests: XCTestCase {
    func testRateLimitingAndJitter() {
        let s = ExperienceSampler(pacing: PacingConfig(), sessionStart: 0, seed: 1)
        let interval = PacingConfig().experienceSamplingInterval
        XCTAssertGreaterThanOrEqual(s.nextEligibleAt, interval)
        XCTAssertLessThanOrEqual(s.nextEligibleAt, interval * 1.5)
        var prompts: [Double] = []
        for t in stride(from: 0.0, through: 3 * 3600, by: 1) where s.shouldPrompt(at: t, nudgeActive: false) {
            prompts.append(t)
            _ = s.answer(focused: true, reason: nil, frames: [], sessionID: UUID())
        }
        XCTAssertGreaterThan(prompts.count, 10)
        let gaps = zip(prompts, prompts.dropFirst()).map { $1 - $0 }
        XCTAssertTrue(gaps.allSatisfy { $0 >= interval })
        XCTAssertTrue(gaps.allSatisfy { $0 <= interval * 1.5 + 1 })
        XCTAssertGreaterThan(Set(gaps).count, 1, "timing is randomised")
    }

    func testNoPromptDuringNudgeOrWhilePendingOrDisabled() {
        var p = PacingConfig(); p.experienceSamplingInterval = 100
        let s = ExperienceSampler(pacing: p, jitterFraction: 0, promptTimeout: 30, seed: 2)
        XCTAssertFalse(s.shouldPrompt(at: 99, nudgeActive: false))
        XCTAssertFalse(s.shouldPrompt(at: 100, nudgeActive: true))
        XCTAssertTrue(s.shouldPrompt(at: 101, nudgeActive: false))
        XCTAssertEqual(s.pendingPromptAt, 101)
        XCTAssertFalse(s.shouldPrompt(at: 120, nudgeActive: false), "pending prompt blocks")
        XCTAssertEqual(s.pendingPromptAt, 101)
        XCTAssertFalse(s.shouldPrompt(at: 140, nudgeActive: false), "expired after timeout but not yet due")
        XCTAssertNil(s.pendingPromptAt)
        XCTAssertFalse(s.shouldPrompt(at: 200, nudgeActive: false))
        XCTAssertTrue(s.shouldPrompt(at: 201, nudgeActive: false), "due again one interval after the last prompt")
        s.dismiss()
        s.isEnabled = false
        XCTAssertFalse(s.shouldPrompt(at: 10_000, nudgeActive: false))
    }

    func testAnswerProducesPrecedingWindow() {
        var p = PacingConfig(); p.experienceSamplingInterval = 60
        let s = ExperienceSampler(pacing: p, jitterFraction: 0, seed: 3)
        let fr = frames(from: 0, to: 100)
        XCTAssertNil(s.answer(focused: true, reason: nil, frames: fr, sessionID: UUID()), "no pending prompt")
        XCTAssertTrue(s.shouldPrompt(at: 60, nudgeActive: false))
        let w = s.answer(focused: false, reason: .tired, frames: fr, sessionID: UUID())!
        XCTAssertEqual(w.start, 50); XCTAssertEqual(w.end, 60)
        XCTAssertEqual(w.label, .drowsy); XCTAssertTrue(w.isLapse); XCTAssertEqual(w.source, .experienceSampling)
        XCTAssertEqual(w.frames.count, 100)
        XCTAssertNil(s.pendingPromptAt)
        XCTAssertTrue(s.shouldPrompt(at: 120, nudgeActive: false))
        let f = s.answer(focused: true, reason: nil, frames: fr, sessionID: UUID())!
        XCTAssertEqual(f.label, .focused); XCTAssertFalse(f.isLapse)
        let st = ExperienceSampler.window(focused: false, reason: .stressed, promptTime: 30, frames: fr, sessionID: UUID())
        XCTAssertEqual(st.label, .stressed)
        XCTAssertEqual(ExperienceSampler.window(focused: false, reason: nil, promptTime: 30, frames: fr, sessionID: UUID()).label, .distracted)
    }
}

// MARK: - CSV sink

final class CSVLabelSinkTests: XCTestCase {
    private func replayHeader() throws -> String {
        let url = try XCTUnwrap(Bundle.main.url(forResource: "synthetic_session", withExtension: "replay.csv"))
        let text = try String(contentsOf: url, encoding: .utf8)
        return try XCTUnwrap(text.split(separator: "\n").first { !$0.hasPrefix("#") }.map(String.init))
    }

    private func window(label: AttentionState, lapse: Bool, from: Double, n: Int) -> LabeledWindow {
        let fr = (0..<n).map { k -> FeatureFrame in
            var f = FeatureFrame(t: from + Double(k) * 0.1)
            f.faceDetected = 1; f.perclos = 0.12345; f.heartRate = 71.5
            return f
        }
        return LabeledWindow(sessionID: UUID(), start: from, end: from + Double(n) * 0.1, label: label, isLapse: lapse,
                             source: .sart, frames: fr)
    }

    func testCSVHeaderMatchesReplayPlusExtras() throws {
        let replay = try replayHeader()
        let csv = CSVLabelSink.csv([window(label: .focused, lapse: false, from: 0, n: 3)], isSynthetic: false)
        let lines = csv.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines[0], "# FocusLoop export v1 (REAL device data)")
        XCTAssertEqual(lines[1], replay + ",session_id,source,window_id")
        XCTAssertTrue(lines[1].hasPrefix(replay + ","))
        XCTAssertEqual(lines.count, 5)
    }

    func testRowsAndSyntheticFlag() throws {
        let a = window(label: .focused, lapse: false, from: 10, n: 4)
        let b = window(label: .distracted, lapse: true, from: 20, n: 2)
        let csv = CSVLabelSink.csv([a, b], isSynthetic: true)
        let lines = csv.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines[0], "# FocusLoop export v1 (SYNTHETIC replay)")
        let rows = lines.dropFirst(2).map { $0.split(separator: ",", omittingEmptySubsequences: false).map(String.init) }
        XCTAssertEqual(rows.count, 6)
        let width = CSVLabelSink.columns.count
        XCTAssertEqual(width, 3 + 42 + 3)
        for r in rows { XCTAssertEqual(r.count, width) }
        XCTAssertEqual(rows[0][0], "10.000"); XCTAssertEqual(rows[0][1], "focused"); XCTAssertEqual(rows[0][2], "0")
        XCTAssertEqual(rows[4][1], "distracted"); XCTAssertEqual(rows[4][2], "1")
        let perclos = 3 + FeatureFrame.featureNames.firstIndex(of: "perclos")!
        XCTAssertEqual(rows[0][perclos], "0.1235")
        XCTAssertEqual(rows[0][width - 3], a.sessionID.uuidString)
        XCTAssertEqual(rows[0][width - 2], "sart")
        XCTAssertEqual(rows[5][width - 1], b.id.uuidString)
        // Leading 45 columns parse as the replay format.
        let replayLike = ([lines[0], lines[1]] + lines.dropFirst(2)).map { l -> String in
            l.hasPrefix("#") ? l : l.split(separator: ",", omittingEmptySubsequences: false).prefix(45).joined(separator: ",")
        }.joined(separator: "\n")
        let parsed = try ReplayStream.parse(replayLike)
        XCTAssertTrue(parsed.isSynthetic)
        XCTAssertEqual(parsed.frames.count, 6)
        XCTAssertEqual(parsed.frames[0].heartRate, 71.5, accuracy: 1e-4)
        XCTAssertEqual(parsed.lapse, [false, false, false, false, true, true])
    }

    func testPersistenceAndExport() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = dir.appendingPathComponent("labeled_windows.json")
        let sink = CSVLabelSink(storeURL: store)
        sink.add(window(label: .focused, lapse: false, from: 0, n: 5))
        sink.add(contentsOf: [window(label: .drowsy, lapse: true, from: 5, n: 5)])
        XCTAssertEqual(sink.windows.count, 2)
        let reloaded = CSVLabelSink(storeURL: store)
        XCTAssertEqual(reloaded.windows, sink.windows)
        let url = try reloaded.exportCSV()
        XCTAssertEqual(url.deletingLastPathComponent().standardizedFileURL, FileManager.default.temporaryDirectory.standardizedFileURL)
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.hasPrefix(CSVLabelSink.realHeaderComment))
        XCTAssertEqual(text.split(separator: "\n").count, 12)
        reloaded.isSyntheticData = true
        XCTAssertTrue(try String(contentsOf: reloaded.exportCSV(), encoding: .utf8).hasPrefix(CSVLabelSink.syntheticHeaderComment))
        reloaded.removeAll()
        XCTAssertTrue(CSVLabelSink(storeURL: store).windows.isEmpty)
        try? FileManager.default.removeItem(at: dir)
    }
}

// MARK: - Work zone

final class WorkZoneCalibrationTests: XCTestCase {
    func testDefaultBuilder() {
        let center: [(yaw: Float, pitch: Float)] = [(0.30, -0.10), (0.31, -0.11), (0.29, -0.09), (2.0, 2.0)] // outlier
        let corners: [[(yaw: Float, pitch: Float)]] = [
            [(0.05, 0.05), (0.06, 0.06), (0.04, 0.04)],
            [(0.55, 0.06)], [(0.56, -0.25)], [(0.02, -0.26)],
        ]
        let z = CalibrationMedianWorkZoneBuilder().build(center: center, corners: corners, isPhoneScreen: true)
        XCTAssertEqual(z.centerYaw, 0.305, accuracy: 1e-5)     // median of 4 -> mean of middle two
        XCTAssertEqual(z.centerPitch, -0.095, accuracy: 1e-5)
        XCTAssertEqual(z.halfWidthYaw, 0.305 - 0.02 + 0.05, accuracy: 1e-5)
        XCTAssertEqual(z.halfWidthPitch, 0.26 - 0.095 + 0.05, accuracy: 1e-5)
        XCTAssertTrue(z.isPhoneScreen)
    }

    func testBuilderFallsBackToDefault() {
        let z = CalibrationMedianWorkZoneBuilder().build(center: [], corners: [], isPhoneScreen: false)
        XCTAssertEqual(z.centerYaw, WorkZone.default.centerYaw)
        XCTAssertEqual(z.halfWidthYaw, WorkZone.default.halfWidthYaw)
        XCTAssertEqual(z.halfWidthPitch, WorkZone.default.halfWidthPitch)
    }

    func testRunCollectsAllTargetsWithSimulator() {
        var run = WorkZoneCalibrationRun()
        var sim = CalibrationGazeSimulator(isPhoneScreen: false, seed: 3)
        var t = 0.0
        while !run.isFinished && t < 60 {
            run.ingest(sim.reading(for: run.currentTarget!, at: t))
            t += 1.0 / 30
        }
        XCTAssertTrue(run.isFinished)
        XCTAssertEqual(t, 5 * 2.0, accuracy: 0.2)
        for s in run.samples { XCTAssertGreaterThanOrEqual(s.count, 40) }
        let z = run.makeWorkZone(builder: CalibrationMedianWorkZoneBuilder(), isPhoneScreen: false)
        let nominalCenter = sim.nominal(.center)
        XCTAssertEqual(z.centerYaw, nominalCenter.yaw, accuracy: 0.02)
        XCTAssertEqual(z.halfWidthYaw, 0.26 + 0.05, accuracy: 0.03)
        XCTAssertEqual(z.halfWidthPitch, 0.16 + 0.05, accuracy: 0.03)
    }

    func testRunRepeatsTargetWithoutSamples() {
        var run = WorkZoneCalibrationRun(dwell: 2, settle: 0.4, minSamplesPerTarget: 5)
        run.ingest(CalibrationGazeReading(t: 0, yaw: 0, pitch: 0))
        run.ingest(CalibrationGazeReading(t: 2.1, yaw: 0, pitch: 0))   // face lost for the whole dwell
        XCTAssertEqual(run.targetIndex, 0)
        XCTAssertEqual(run.progress, 0, accuracy: 1e-9)
    }

    func testFrameAdapterAndStore() throws {
        var f = FeatureFrame(t: 3); f.faceDetected = 1; f.gazeYaw = 0.1; f.headYaw = 0.2; f.gazePitch = -0.05; f.headPitch = 0.1
        let r = try XCTUnwrap(CalibrationGazeReading(frame: f))
        XCTAssertEqual(r.yaw, 0.3, accuracy: 1e-6); XCTAssertEqual(r.pitch, 0.05, accuracy: 1e-6)
        f.faceDetected = 0
        XCTAssertNil(CalibrationGazeReading(frame: f))

        let defaults = try XCTUnwrap(UserDefaults(suiteName: "calib-test-\(UUID().uuidString)"))
        let store = WorkZoneStore(defaults: defaults)
        XCTAssertNil(store.load(isPhoneScreen: false))
        XCTAssertTrue(store.current(isPhoneScreen: true).isPhoneScreen)
        let z = WorkZone(centerYaw: 0.3, centerPitch: -0.1, halfWidthYaw: 0.3, halfWidthPitch: 0.2,
                         calibratedAt: Date(timeIntervalSince1970: 1000), isPhoneScreen: false)
        store.save(z)
        XCTAssertEqual(store.load(isPhoneScreen: false), z)
        XCTAssertNil(store.load(isPhoneScreen: true))
        store.clear()
        XCTAssertNil(store.load(isPhoneScreen: false))
    }
}
