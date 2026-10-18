import XCTest
@testable import FocusLoop

/// Regression tests for integration fixes found in code review. Data here is SYNTHETIC / MOCK.
@MainActor
final class ReviewFixesTests: XCTestCase {

    // MARK: Engine

    private func makeServices(policy: ThompsonTrialEngine, speed: Double = 12,
                              onPersist: @escaping (SessionRecord) -> Void = { _ in }) -> EngineServices {
        let head = LogisticPersonalHead()
        var flags = FeatureFlags()
        flags.experienceSampling = false
        return EngineServices(
            flags: flags, classifier: BlendedAttentionClassifier.makeDefault(personalHead: head), personalHead: head,
            forecaster: HeuristicLapseForecaster(), policy: policy, pacing: policy, logger: InMemoryTrialLogger(),
            labels: CSVLabelSink(storeURL: nil, isSyntheticData: true),
            makeFrameSource: { _ in ReplayFeatureSource(stream: try! ReplayStream.load(), speed: speed, loop: false) },
            makeNudge: { arm, lecture, playhead in NudgeFactory.make(arm: arm, lecture: lecture, playhead: playhead) },
            persistSession: onPersist)
    }

    /// Free study previews the MOCK sample timeline in the UI, but the session itself must not use it.
    func testFreeStudyKeepsPreviewContentOutOfTheSession() async throws {
        let model = LiveSessionModel()
        let policy = ThompsonTrialEngine(priors: [], seed: 1)
        let engine = SessionEngine(model: model, services: makeServices(policy: policy))
        let preview = try BrainContentTimeline(named: "sample_lecture")
        engine.start(lecture: nil, timeline: preview)
        XCTAssertFalse(model.studyMode)
        XCTAssertNotNil(model.timeline, "the Science view still previews the MOCK timeline")
        let deadline = Date().addingTimeInterval(20)
        while model.elapsed < 60, Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
        let collected = await engine.finishAndCollect()
        let rec = try XCTUnwrap(collected)
        XCTAssertNil(rec.lectureID)
        XCTAssertFalse(rec.contentIsMock)
        XCTAssertFalse(rec.ticks.isEmpty)
        XCTAssertTrue(rec.ticks.allSatisfy { $0.playhead == nil && $0.predictedEngagement == nil })
        XCTAssertFalse(rec.events.contains { if case .valleyEntered = $0 { return true } else { return false } })
    }

    /// Risk alerts are logged once per alert episode, not every second.
    func testRiskAlertsAreLoggedOnRisingEdgeOnly() async throws {
        let model = LiveSessionModel()
        let policy = ThompsonTrialEngine(priors: [], seed: 2)
        let engine = SessionEngine(model: model, services: makeServices(policy: policy))
        let tl = try BrainContentTimeline(named: "sample_lecture")
        let lecture = StudyLectureLibrary.lecture(id: "sample_lecture")
        engine.start(lecture: lecture, timeline: tl)
        XCTAssertTrue(model.studyMode, "studyMode is set before the frame source is created")
        let deadline = Date().addingTimeInterval(30)
        while model.elapsed < 178, Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
        let collected = await engine.finishAndCollect()
        let rec = try XCTUnwrap(collected)
        let alerts = rec.events.compactMap { e -> Double? in if case let .riskAlert(t, _) = e { return t } else { return nil } }
        // Consecutive alert events are never 1 s apart (that would be per-second logging).
        for (a, b) in zip(alerts, alerts.dropFirst()) { XCTAssertGreaterThan(b - a, 1.5) }
    }

    /// `finishAndCollect` waits for the pipeline to resolve pending rewards, so the record has every outcome.
    func testFinishAndCollectRecordsEveryResolvedOutcome() async throws {
        let model = LiveSessionModel()
        let policy = ThompsonTrialEngine(priors: [], seed: 3)
        let engine = SessionEngine(model: model, services: makeServices(policy: policy))
        let tl = try BrainContentTimeline(named: "sample_lecture")
        let lecture = StudyLectureLibrary.lecture(id: "sample_lecture")
        engine.start(lecture: lecture, timeline: tl)
        let deadline = Date().addingTimeInterval(30)
        while model.elapsed < 178, Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
        let collected = await engine.finishAndCollect()
        let rec = try XCTUnwrap(collected)
        let decided = Set(rec.events.compactMap { e -> UUID? in if case let .nudge(d) = e { return d.id } else { return nil } })
        let resolved = Set(rec.events.compactMap { e -> UUID? in if case let .outcome(o) = e { return o.decisionID } else { return nil } })
        XCTAssertFalse(decided.isEmpty)
        XCTAssertEqual(decided, resolved, "every decision's outcome reaches the record")
        XCTAssertEqual(policy.records.filter { $0.outcome == nil }.count, 0)
    }

    // MARK: Thread safety

    func testTrialEngineToleratesConcurrentReadersAndWriters() async {
        let engine = ThompsonTrialEngine(priors: [], pacing: { var p = PacingConfig(); p.cooldown = 0; p.reactiveDebounce = 0; return p }(), seed: 4)
        let writer = Task.detached {
            for i in 0..<400 {
                let ctx = NudgeContext(t: Double(i), state: .distracted, lapseRisk: 0.5, predictedEngagement: 0.5,
                                       sessionMinute: 1, hourOfDay: 10, minutesSinceLastNudge: 1, studyMode: false,
                                       phonePickedUp: false, upcomingValleyInSeconds: nil)
                engine.noteState(.distracted, t: Double(i))
                if let d = engine.decide(policy: .reactive, context: ctx) {
                    engine.record(outcome: NudgeOutcome(decisionID: d.id, t: Double(i) + 1, reward: i % 2 == 0 ? 1 : 0,
                                                        refocused: i % 2 == 0, refocusSeconds: 3, dismissed: false,
                                                        lapseAvoided: nil), for: d)
                }
            }
        }
        for _ in 0..<50 {
            _ = engine.posteriors()
            _ = try? engine.serialized()
            _ = engine.records.count
        }
        await writer.value
        XCTAssertFalse(engine.records.isEmpty)
        XCTAssertNoThrow(try ThompsonTrialEngine(priors: []).restore(from: engine.serialized()))
    }

    // MARK: Exports

    func testFeatureStreamRecordingAndManifest() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("fl-streams-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let defaults = UserDefaults(suiteName: "fl-streams-\(UUID().uuidString)")!
        let store = FeatureStreamStore(directory: dir, defaults: defaults)
        XCTAssertNil(store.begin(sessionID: UUID(), startedAt: Date(), isSynthetic: false, lectureID: nil), "off by default")
        store.isEnabled = true

        let rec = try XCTUnwrap(store.begin(sessionID: UUID(), startedAt: Date(), isSynthetic: true, lectureID: "black-holes-01"))
        let frames = try ReplayStream.load().frames.prefix(120)
        for f in frames { rec.append(f, playhead: f.t + 5) }
        rec.finish(experienceSamples: [(t: 3, focused: false, reason: .tired)])

        let files = try store.prepareExport()
        let manifest = try XCTUnwrap(files.first { $0.lastPathComponent == FeatureStreamStore.manifestName })
        let rows = try String(contentsOf: manifest, encoding: .utf8).split(separator: "\n").map(String.init)
        XCTAssertEqual(rows.first, FeatureStreamStore.manifestColumns.joined(separator: ","))
        XCTAssertEqual(rows.count, 2)
        let cells = rows[1].split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        XCTAssertEqual(cells[5], "black-holes-01.brain.json")
        XCTAssertEqual(cells[7], "1", "SYNTHETIC replay streams are flagged synthetic in the manifest")
        XCTAssertTrue(cells[4].hasSuffix(".es.csv"))

        let csv = try String(contentsOf: dir.appendingPathComponent(cells[2]), encoding: .utf8).split(separator: "\n")
        XCTAssertTrue(csv[0].contains("SYNTHETIC"))
        XCTAssertEqual(String(csv[1]), FeatureStreamStore.sessionColumns.joined(separator: ","))
        XCTAssertEqual(csv.count, 2 + frames.count)
        XCTAssertEqual(csv[2].split(separator: ",", omittingEmptySubsequences: false).count, FeatureStreamStore.sessionColumns.count)

        store.removeAll()
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.path))
    }

    func testHeartRateCSVForWatchValidation() {
        let start = Date(timeIntervalSince1970: 1_800_000_000)
        let report = SessionReport(sessionID: UUID(), startedAt: start, duration: 30, lectureID: nil, lectureTitle: nil,
                                   contentIsMock: false, isReplay: true, segments: [], focusedFraction: 1, refocusTimes: [],
                                   medianRefocusSeconds: nil, heartRate: [.init(t: 1, value: 70), .init(t: 2, value: 71.5)],
                                   restingHeartRate: nil, predictedDrift: [], actualDrift: [], driftCorrelation: nil,
                                   biggestDriftPlayhead: nil, biggestDriftActivations: nil, nudgeCount: 0, preemptiveCount: 0,
                                   effects: [], insights: [])
        let lines = UIHeartRateCSV.make(report).split(separator: "\n").map(String.init)
        XCTAssertTrue(lines[0].contains("SYNTHETIC"))
        XCTAssertEqual(lines[1], "timestamp,bpm")
        XCTAssertEqual(lines[2], "1800000001.000,70.00")
        XCTAssertEqual(lines.count, 4)
    }

    func testDemoPacingKeepsTheEightMinuteSamplingLimit() {
        XCTAssertGreaterThanOrEqual(PacingConfig.demo.experienceSamplingInterval, 8 * 60)
    }
}
