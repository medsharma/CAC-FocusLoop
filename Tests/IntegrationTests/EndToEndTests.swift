import XCTest
@testable import FocusLoop

/// Full pipeline on the SYNTHETIC replay stream + MOCK sample lecture, with every real module wired in
/// (classifier, forecaster, trial engine, reward tracker, nudges, report builder, insights).
/// Numbers here are SYNTHETIC and only prove the loop runs end to end.
@MainActor
final class EndToEndTests: XCTestCase {
    @MainActor private final class FastReplayServices {
        static func make(policy: ThompsonTrialEngine, labels: LabelSink, onPersist: @escaping (SessionRecord) -> Void) -> EngineServices {
            let head = LogisticPersonalHead()
            var flags = FeatureFlags()
            flags.experienceSampling = false
            return EngineServices(
                flags: flags, classifier: BlendedAttentionClassifier.makeDefault(personalHead: head), personalHead: head,
                forecaster: CoreMLLapseForecaster(bundle: .main) ?? HeuristicLapseForecaster(),
                policy: policy, pacing: policy, logger: InMemoryTrialLogger(), labels: labels,
                makeFrameSource: { _ in
                    ReplayFeatureSource(stream: try! ReplayStream.load(), speed: 12, loop: false)
                },
                makeNudge: { arm, lecture, playhead in NudgeFactory.make(arm: arm, lecture: lecture, playhead: playhead) },
                persistSession: onPersist)
        }
    }

    func testStudyModeSessionRunsFullLoop() async throws {
        let model = LiveSessionModel()
        let policy = ThompsonTrialEngine(priors: NudgePriorsFile.load()?.priors ?? [], seed: 42)
        let labels = CSVLabelSink(storeURL: nil, isSyntheticData: true)
        var saved: SessionRecord?
        let services = FastReplayServices.make(policy: policy, labels: labels) { saved = $0 }
        let engine = SessionEngine(model: model, services: services)
        let timeline = try BrainContentTimeline(named: "sample_lecture")
        let lecture = StudyLectureLibrary.lecture(id: "sample_lecture")

        engine.start(lecture: lecture, timeline: timeline)
        XCTAssertTrue(model.isReplay, "Replay must be flagged SYNTHETIC")
        // 180 s of replay at 12x ≈ 15 s wall clock.
        let deadline = Date().addingTimeInterval(40)
        while model.elapsed < 178, Date() < deadline { try await Task.sleep(for: .milliseconds(200)) }
        engine.stop()
        let rec = try XCTUnwrap(saved)

        // Sense + understand + predict at 1 Hz.
        XCTAssertGreaterThan(rec.ticks.count, 150)
        let states = Set(rec.ticks.map(\.state))
        XCTAssertTrue(states.contains(.focused))
        XCTAssertGreaterThanOrEqual(states.count, 3, "classifier should see several scripted states: \(states)")
        XCTAssertTrue(rec.ticks.contains { $0.lapseRisk > 0 })
        XCTAssertTrue(rec.ticks.allSatisfy { $0.playhead != nil }, "Study Mode ticks carry the lecture playhead")

        // Intervene + measure + learn.
        let decisions = rec.events.compactMap { if case let .nudge(d) = $0 { return d } else { return nil } }
        let outcomes = rec.events.compactMap { if case let .outcome(o) = $0 { return o } else { return nil } }
        let refocus = rec.events.filter { if case .refocused = $0 { return true } else { return false } }
        let valleys = rec.events.filter { if case .valleyEntered = $0 { return true } else { return false } }
        XCTAssertTrue(decisions.contains { $0.policy == .reactive }, "expected a reactive decision")
        XCTAssertTrue(decisions.allSatisfy { $0.randomizationProbability > 0 && $0.randomizationProbability <= 1 })
        XCTAssertFalse(outcomes.isEmpty, "rewards should be recorded")
        XCTAssertFalse(refocus.isEmpty, "expected at least one refocus")
        XCTAssertFalse(valleys.isEmpty, "MOCK lecture has valleys")
        print("E2E (SYNTHETIC): ticks=\(rec.ticks.count) states=\(states) decisions=\(decisions.map { "\($0.policy.rawValue):\($0.arm.rawValue)" }) outcomes=\(outcomes.count) refocus=\(refocus.count)")

        // Report: brain-vs-actual overlay, cortex frame, Personal Trial, insights.
        let report = ReportBuilder.build(record: rec, timeline: timeline, lectureTitle: lecture?.title,
                                         effects: policy.effectEstimates())
        XCTAssertTrue(report.isReplay)
        XCTAssertTrue(report.contentIsMock)
        XCTAssertFalse(report.predictedDrift.isEmpty)
        XCTAssertFalse(report.actualDrift.isEmpty)
        XCTAssertNotNil(report.driftCorrelation)
        XCTAssertEqual(report.biggestDriftActivations?.count, 7)
        XCTAssertGreaterThan(report.segments.count, 2)
        let insights = await TemplateInsightGenerator().insights(for: AttentionProfileBuilder.summary(for: rec,
                                                                  armEffects: policy.effectEstimates()))
        XCTAssertTrue((2...4).contains(insights.count))
        XCTAssertTrue(insights.allSatisfy { InsightGuardrails.violations(in: $0.text).isEmpty })
        print("E2E report (SYNTHETIC/MOCK): focused=\(report.focusedFraction) r=\(report.driftCorrelation ?? .nan) refocus=\(report.refocusTimes)")
    }
}
