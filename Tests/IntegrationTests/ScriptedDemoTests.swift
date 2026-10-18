import XCTest
@testable import FocusLoop

/// The one-tap scripted demo (SYNTHETIC replay + Demo Director, DEMONSTRATION ONLY): runs the real session
/// engine in Study Mode on `demo_session.replay.csv`, locked to the "Travel deep inside a leaf" lecture clock
/// (a fake player advances the playhead and pauses for recall cards, as the real one does), and checks that
/// every demo beat fires, in order, with the current bundled models AND with the rule-based fallback.
/// Everything here is SYNTHETIC and proves only that the demo choreography works.
@MainActor
final class ScriptedDemoTests: XCTestCase {
    enum Stack: String { case currentModels, heuristic }

    struct Run {
        var record: SessionRecord
        var decisions: [NudgeDecision]
        var trial: ThompsonTrialEngine
        var initialPosteriors: [ArmPosterior]
    }

    static let lectureID = "leaf-photosynthesis-01"

    private func runScriptedDemo(_ stack: Stack) async throws -> Run {
        let model = LiveSessionModel()
        let trial = ThompsonTrialEngine(priors: NudgePriorsFile.load()?.priors ?? [], seed: 7)
        let initial = trial.posteriors()
        let demo = DemoScriptPolicy(base: trial, pacing: trial)
        let policy = SensingOnlyPolicy(base: demo)
        let head = LogisticPersonalHead()
        let labels = CSVLabelSink(storeURL: nil, isSyntheticData: true)
        var flags = FeatureFlags()
        flags.demoDirector = true          // DEMONSTRATION ONLY
        flags.experienceSampling = false

        let classifier: BlendedAttentionClassifier
        let forecaster: LapseForecaster
        switch stack {
        case .currentModels:
            classifier = BlendedAttentionClassifier.makeDefault(personalHead: head)
            XCTAssertTrue(classifier.usesModel, "the bundled attention-state model should load")
            let f = try XCTUnwrap(CoreMLLapseForecaster(bundle: .main), "the bundled lapse forecaster should load")
            f.demoDirector = true
            forecaster = f
        case .heuristic:
            classifier = BlendedAttentionClassifier(base: nil, personalHead: head)
            let f = HeuristicLapseForecaster()
            f.demoDirector = true
            forecaster = f
        }
        let thresholdBefore = forecaster.threshold

        let stream = try ReplayStream.load("demo_session")
        XCTAssertTrue(stream.isSynthetic, "the scripted demo replay must say SYNTHETIC")
        var saved: SessionRecord?
        let services = EngineServices(
            flags: flags, classifier: classifier, personalHead: head, forecaster: forecaster,
            policy: policy, pacing: demo, logger: InMemoryTrialLogger(), labels: labels,
            makeFrameSource: { [model] _ in LectureClockReplaySource(stream: stream, pollInterval: 0.002) { model.playhead } },
            makeNudge: { arm, lecture, playhead in NudgeFactory.make(arm: arm, lecture: lecture, playhead: playhead) },
            persistSession: { saved = $0 }, demoScript: demo)
        let engine = SessionEngine(model: model, services: services)
        let lecture = try XCTUnwrap(StudyLectureLibrary.lectures().first { $0.id == Self.lectureID })
        let timeline = try BrainContentTimeline(named: Self.lectureID)

        engine.start(lecture: lecture, timeline: timeline)
        XCTAssertTrue(model.isReplay)
        XCTAssertTrue(model.demoDirector)

        // Fake player: advances the lecture 0.1 s per step (only as fast as the engine keeps up) and pauses
        // while a recall card is up, then answers it, exactly like Study Mode does.
        var playhead = 0.0
        var cardTicks = 0
        let deadline = Date().addingTimeInterval(120)
        model.actions.setPlayhead(0)
        while playhead < lecture.durationSeconds, Date() < deadline {
            if case let .card(card)? = model.activePresentation, card.kind == .recall {
                cardTicks += 1
                if cardTicks > 15 { model.actions.answerRecall(card.correctIndex ?? 0); cardTicks = 0 }
                try await Task.sleep(for: .milliseconds(2))
                continue
            }
            if model.elapsed < playhead - 0.25 {
                try await Task.sleep(for: .milliseconds(1))
                continue
            }
            playhead = min(lecture.durationSeconds, playhead + 0.1)
            model.actions.setPlayhead(playhead)
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertGreaterThanOrEqual(playhead, lecture.durationSeconds, "the lecture should play to the end")
        try await Task.sleep(for: .milliseconds(100))
        engine.stop()
        let rec = try XCTUnwrap(saved)
        XCTAssertEqual(forecaster.threshold, thresholdBefore, "a demo session must not adapt the forecaster threshold")
        XCTAssertEqual(head.labelCount, 0)
        XCTAssertTrue(labels.windows.isEmpty)
        let decisions = rec.events.compactMap { if case let .nudge(d) = $0 { return d } else { return nil } }
        Self.printTrace(stack, rec)
        return Run(record: rec, decisions: decisions, trial: trial, initialPosteriors: initial)
    }

    /// Per-second trace (state, risk vs threshold, events) for tuning the replay.
    static func printTrace(_ stack: Stack, _ rec: SessionRecord) {
        var lines: [String] = []
        for tick in rec.ticks {
            lines.append(String(format: "%5.0f %@ %.2f", tick.t, String(tick.state.rawValue.prefix(4)), tick.lapseRisk))
        }
        print("SCRIPTED-DEMO[\(stack.rawValue)] ticks: " + lines.joined(separator: " | "))
        let ev = rec.events.compactMap { e -> String? in
            switch e {
            case let .nudge(d): return String(format: "%.0f:%@:%@", d.t, d.policy.rawValue, d.arm.rawValue)
            case let .lapseDetected(t, s): return String(format: "%.0f:lapse:%@", t, s.rawValue)
            case let .refocused(t, a): return String(format: "%.0f:back:%.1f", t, a)
            case let .riskAlert(t, p): return String(format: "%.0f:alert:%.2f", t, p)
            default: return nil
            }
        }
        print("SCRIPTED-DEMO[\(stack.rawValue)] events: " + ev.joined(separator: " "))
    }

    /// Asserts the demo beats in order: a pre-emptive nudge before the first lapse, then a distracted nudge
    /// (putMeDown or ambientGlow), then stretchBreak, then pacedBreathing, plus at least one refocus.
    private func assertDemoBeats(_ run: Run, _ stack: Stack, file: StaticString = #filePath, line: UInt = #line) {
        let rec = run.record
        let firstLapse = rec.events.compactMap { if case let .lapseDetected(t, _) = $0 { return t } else { return nil } }.first
        let lapseT = firstLapse ?? .infinity
        let sequence = run.decisions.map { "\($0.policy.rawValue):\($0.arm.rawValue)@\(Int($0.t))" }
        print("SCRIPTED-DEMO[\(stack.rawValue)] arms: \(sequence.joined(separator: " → "))")

        XCTAssertTrue(run.decisions.contains { $0.policy == .preemptive && $0.t < lapseT },
                      "[\(stack)] a pre-emptive nudge must fire before the first lapse (\(lapseT)): \(sequence)", file: file, line: line)
        var stage = 0
        let wanted: [Set<NudgeArm>] = [[.putMeDown, .ambientGlow], [.stretchBreak], [.pacedBreathing]]
        for d in run.decisions where d.policy == .reactive && stage < wanted.count {
            if wanted[stage].contains(d.arm) { stage += 1 }
        }
        XCTAssertEqual(stage, wanted.count, "[\(stack)] expected putMeDown/ambientGlow → stretchBreak → pacedBreathing in order: \(sequence)",
                       file: file, line: line)
        let refocus = rec.events.compactMap { if case let .refocused(_, s) = $0 { return s } else { return nil } }
        XCTAssertFalse(refocus.isEmpty, "[\(stack)] expected at least one refocus (\"Back in X.X s\")", file: file, line: line)
        print("SCRIPTED-DEMO[\(stack.rawValue)] back in: \(refocus.map { String(format: "%.1f", $0) })")

        // DEMONSTRATION ONLY decisions are logged as deterministic and never reach the bandit.
        XCTAssertTrue(run.decisions.allSatisfy { $0.randomizationProbability == 1 && $0.armProbabilities == [$0.arm.rawValue: 1] },
                      file: file, line: line)
        XCTAssertTrue(run.decisions.allSatisfy(DemoScriptPolicy.isScripted), file: file, line: line)
        XCTAssertTrue(run.trial.records.isEmpty, "demo decisions must not enter the trial", file: file, line: line)
        XCTAssertEqual(run.trial.posteriors(), run.initialPosteriors, "demo outcomes must not move the posteriors", file: file, line: line)
        XCTAssertTrue(rec.isReplay, file: file, line: line)
        XCTAssertTrue(rec.contentIsMock, file: file, line: line)
        XCTAssertEqual(rec.lectureID, Self.lectureID, file: file, line: line)
    }

    func testScriptedDemoBeatsWithCurrentModels() async throws {
        let run = try await runScriptedDemo(.currentModels)
        assertDemoBeats(run, .currentModels)
    }

    func testScriptedDemoBeatsWithHeuristicFallback() async throws {
        let run = try await runScriptedDemo(.heuristic)
        assertDemoBeats(run, .heuristic)
    }

    // MARK: Pieces

    func testDemoReplayCoversTheLecture() throws {
        let stream = try ReplayStream.load("demo_session")
        let lecture = try XCTUnwrap(StudyLectureLibrary.lectures().first { $0.id == Self.lectureID })
        XCTAssertTrue(stream.isSynthetic)
        XCTAssertGreaterThanOrEqual(stream.duration, lecture.durationSeconds - 0.05)
        XCTAssertEqual(stream.frames.first?.t, 0)
        // Ground truth contains every scripted state.
        XCTAssertEqual(Set(stream.labels.compactMap { $0 }), Set(AttentionState.allCases))
    }

    func testDemoScriptArms() {
        func ctx(_ state: AttentionState, pickup: Bool = false, study: Bool = true, valley: Float? = 10) -> NudgeContext {
            NudgeContext(t: 0, state: state, lapseRisk: 0.5, predictedEngagement: 0.5, sessionMinute: 0, hourOfDay: 12,
                         minutesSinceLastNudge: 99, studyMode: study, phonePickedUp: pickup, upcomingValleyInSeconds: valley)
        }
        XCTAssertEqual(DemoScriptPolicy.scriptedArm(policy: .preemptive, context: ctx(.focused), recallUnlocked: true), .recallQuestion)
        XCTAssertEqual(DemoScriptPolicy.scriptedArm(policy: .preemptive, context: ctx(.focused), recallUnlocked: false), .preemptiveGlow)
        XCTAssertEqual(DemoScriptPolicy.scriptedArm(policy: .preemptive, context: ctx(.focused, study: false), recallUnlocked: true), .preemptiveGlow)
        XCTAssertEqual(DemoScriptPolicy.scriptedArm(policy: .preemptive, context: ctx(.focused, valley: nil), recallUnlocked: true), .preemptiveGlow)
        XCTAssertEqual(DemoScriptPolicy.scriptedArm(policy: .reactive, context: ctx(.distracted, pickup: true), recallUnlocked: false), .putMeDown)
        XCTAssertEqual(DemoScriptPolicy.scriptedArm(policy: .reactive, context: ctx(.distracted), recallUnlocked: false), .ambientGlow)
        XCTAssertEqual(DemoScriptPolicy.scriptedArm(policy: .reactive, context: ctx(.drowsy), recallUnlocked: false), .stretchBreak)
        XCTAssertEqual(DemoScriptPolicy.scriptedArm(policy: .reactive, context: ctx(.stressed), recallUnlocked: false), .pacedBreathing)
        XCTAssertNil(DemoScriptPolicy.scriptedArm(policy: .reactive, context: ctx(.focused), recallUnlocked: false))
    }

    func testDemoPolicyUsesDemoPacingAndNeverTouchesTheTrial() throws {
        let trial = ThompsonTrialEngine(seed: 3)
        let before = trial.posteriors()
        let demo = DemoScriptPolicy(base: trial, pacing: trial)
        demo.beginSession(enabled: true, lecture: nil)
        demo.setPacing(.demo)
        demo.resetSession()
        func ctx(_ t: Double, _ s: AttentionState) -> NudgeContext {
            NudgeContext(t: t, state: s, lapseRisk: 0.9, predictedEngagement: 0.5, sessionMinute: 0, hourOfDay: 12,
                         minutesSinceLastNudge: 99, studyMode: false, phonePickedUp: false, upcomingValleyInSeconds: nil)
        }
        // Reactive debounce (1.5 s of sustained evidence), then the cooldown (12 s).
        demo.noteState(.drowsy, t: 10)
        XCTAssertNil(demo.decide(policy: .reactive, context: ctx(11, .drowsy)))
        let d = try XCTUnwrap(demo.decide(policy: .reactive, context: ctx(12, .drowsy)))
        XCTAssertEqual(d.arm, .stretchBreak)
        XCTAssertEqual(d.randomizationProbability, 1)
        XCTAssertEqual(d.armProbabilities, [NudgeArm.stretchBreak.rawValue: 1])
        XCTAssertNil(demo.decide(policy: .preemptive, context: ctx(20, .focused)), "cooldown applies")
        XCTAssertEqual(demo.decide(policy: .preemptive, context: ctx(24.5, .focused))?.arm, .preemptiveGlow)

        // Study Mode: each unlocked recall question is asked once, then the script falls back to the glow;
        // no pre-emptive nudge before a full forecast window.
        let lecture = try XCTUnwrap(StudyLectureLibrary.lectures().first { $0.id == Self.lectureID })
        demo.beginSession(enabled: true, lecture: lecture)
        func studyCtx(_ t: Double) -> NudgeContext {
            NudgeContext(t: t, state: .focused, lapseRisk: 0.9, predictedEngagement: 0.5, sessionMinute: 0, hourOfDay: 12,
                         minutesSinceLastNudge: 99, studyMode: true, phonePickedUp: false, upcomingValleyInSeconds: 8)
        }
        XCTAssertNil(demo.decide(policy: .preemptive, context: studyCtx(5)), "warm-up: wait for a full forecast window")
        demo.notePlayhead(30)
        XCTAssertEqual(demo.decide(policy: .preemptive, context: studyCtx(30))?.arm, .preemptiveGlow, "no question unlocked yet")
        demo.notePlayhead(65)
        XCTAssertEqual(demo.decide(policy: .preemptive, context: studyCtx(65))?.arm, .recallQuestion)
        demo.notePlayhead(80)
        XCTAssertEqual(demo.decide(policy: .preemptive, context: studyCtx(80))?.arm, .preemptiveGlow, "already asked")
        demo.notePlayhead(105)
        XCTAssertEqual(demo.decide(policy: .preemptive, context: studyCtx(105))?.arm, .recallQuestion, "next question")
        demo.record(outcome: NudgeOutcome(decisionID: d.id, t: 15, reward: 1, refocused: true, refocusSeconds: 3,
                                          dismissed: false, lapseAvoided: nil), for: d)
        XCTAssertEqual(trial.posteriors(), before, "demo outcomes never reach the trial")
        XCTAssertTrue(trial.records.isEmpty, "demo decisions never enter the trial")
        XCTAssertEqual(trial.pacing, PacingConfig(), "demo pacing never re-paces the trial engine")

        // Off: everything goes to the trial as before.
        demo.beginSession(enabled: false, lecture: nil)
        demo.noteState(.distracted, t: 0)
        let real = try XCTUnwrap(demo.decide(policy: .reactive, context: ctx(5, .distracted)))
        XCTAssertLessThan(real.randomizationProbability, 1)
        XCTAssertFalse(DemoScriptPolicy.isScripted(real))
        XCTAssertEqual(trial.records.count, 1)
    }

    func testLectureClockReplayFollowsThePlayhead() async throws {
        let stream = try ReplayStream.load("demo_session")
        var playhead = 0.0
        let src = LectureClockReplaySource(stream: stream) { playhead }
        final class Box { var frames: [FeatureFrame] = [] }
        let box = Box()
        let collect = Task { @MainActor in for await f in src.frames { box.frames.append(f) } }
        src.poll()
        XCTAssertEqual(src.nextIndex, 1)
        playhead = 2.0; src.poll()
        XCTAssertEqual(src.nextIndex, 21)
        src.poll()                                   // paused video: nothing new
        XCTAssertEqual(src.nextIndex, 21)
        playhead = 1.0; src.poll()                   // seek back: replays from 1.0 s, session clock keeps going
        XCTAssertEqual(src.nextIndex, 11)
        playhead = 100; src.poll()                   // seek forward: jumps close to the new position
        XCTAssertEqual(src.nextIndex, 1001)
        src.stop()
        await collect.value
        let got = box.frames
        XCTAssertEqual(got.map(\.t), (0..<got.count).map { Double($0) / 10 }, "frame times are a monotonic 10 Hz session clock")
        XCTAssertEqual(got.last?.blinkRate, stream.frames[1000].blinkRate)
    }

    func testSessionHints() {
        let face = FeatureFrame(t: 0, vector: [1] + Array(repeating: 0, count: FeatureFrame.featureNames.count - 1))
        let noFace = FeatureFrame(t: 0, vector: Array(repeating: 0, count: FeatureFrame.featureNames.count))
        let missing = Array(repeating: noFace, count: 40)
        XCTAssertEqual(SessionHintsOverlay.hint(isReplay: false, running: true, cameraDenied: true, recentFrames: []), .cameraDenied)
        XCTAssertEqual(SessionHintsOverlay.hint(isReplay: false, running: true, cameraDenied: false, recentFrames: missing), .faceMissing)
        XCTAssertNil(SessionHintsOverlay.hint(isReplay: false, running: true, cameraDenied: false, recentFrames: missing + [face]))
        XCTAssertNil(SessionHintsOverlay.hint(isReplay: false, running: true, cameraDenied: false, recentFrames: Array(missing.prefix(20))))
        XCTAssertNil(SessionHintsOverlay.hint(isReplay: true, running: true, cameraDenied: true, recentFrames: missing))
    }
}
