import AVFoundation
import CoreMotion
import Foundation
import HealthKit
import Observation
import SwiftUI

/// Everything FocusLoop learns about the user: the trial engine, the personal head, the forecaster's
/// personal threshold and the labelled windows. Real sessions and SYNTHETIC Replay sessions use separate
/// sets, persisted under separate keys, so Replay never leaks into the real Personal Trial, the personal
/// model or the REAL label export (spec 1.9: synthetic outputs must be labelled and kept apart).
@MainActor
final class LearnerSet {
    let isSynthetic: Bool
    let personalHead = LogisticPersonalHead()
    let policy: ThompsonTrialEngine
    /// DEMONSTRATION ONLY: Demo Director's fixed nudges; never feeds `policy`.
    let demoScript: DemoScriptPolicy
    /// What the engine calls: sensing-only (study condition B) → demo script → Thompson trial.
    let sensingWrapper: SensingOnlyPolicy
    let forecaster: LapseForecaster
    let labels: CSVLabelSink

    init(isSynthetic: Bool) {
        self.isSynthetic = isSynthetic
        policy = ThompsonTrialEngine(bundle: .main)
        demoScript = DemoScriptPolicy(base: policy, pacing: policy)
        sensingWrapper = SensingOnlyPolicy(base: demoScript)
        forecaster = CoreMLLapseForecaster(bundle: .main) ?? HeuristicLapseForecaster()
        labels = CSVLabelSink(storeURL: isSynthetic ? Self.syntheticLabelsURL : CSVLabelSink.defaultStoreURL,
                              isSyntheticData: isSynthetic)
    }

    static var syntheticLabelsURL: URL? {
        CSVLabelSink.defaultStoreURL?.deletingLastPathComponent().appendingPathComponent("labeled_windows_SYNTHETIC.json")
    }

    /// Model-weights key for this set (`<key>.SYNTHETIC` for Replay).
    func key(_ base: String) -> String { isSynthetic ? "\(base).SYNTHETIC" : base }
}

/// The real DI container behind the UI (`UIServices`). Owns persistence, the trial engine, models,
/// the session engine and every module's views. Real sensors on device; the SYNTHETIC Replay stream in the
/// Simulator (or when Force Replay is on).
@MainActor
@Observable
final class AppServices: UIServices {
    let model: LiveSessionModel

    var flags: FeatureFlags {
        didSet {
            UIPrefs.saveFlags(flags)
            applyEffectiveFlags()
        }
    }
    var replayIsForced: Bool { !LiveFeatureFrameSource.isSupported }
    /// The next session will run on the SYNTHETIC Replay stream.
    var willUseReplay: Bool { effectiveFlags.forceReplay || !LiveFeatureFrameSource.isSupported }

    /// DEMONSTRATION ONLY: the one-tap scripted demo (SYNTHETIC) is prepared or running. It turns Demo Director
    /// on and forces the scripted replay for that session only; the saved flags are never changed.
    var scriptedDemoActive = false
    static let scriptedDemoLectureID = "leaf-photosynthesis-01"
    static let scriptedDemoStream = "demo_session"

    /// The flags the next session runs with: the saved flags, plus the scripted demo's session-only overrides.
    var effectiveFlags: FeatureFlags {
        var f = flags
        if scriptedDemoActive {
            f.demoDirector = true
            f.forceReplay = true
            f.experienceSampling = false
        }
        return f
    }

    func applyEffectiveFlags() {
        let f = effectiveFlags
        model.demoDirector = f.demoDirector
        engineServices?.flags = f
        forecasterDemo(f.demoDirector)
    }

    /// Sensing only: every decision is forced to the no-nudge control (the study's sensing-only condition).
    var sensingOnly: Bool {
        get { UserDefaults.standard.bool(forKey: Self.sensingOnlyKey) }
        set {
            UserDefaults.standard.set(newValue, forKey: Self.sensingOnlyKey)
            real.sensingWrapper.enabled = newValue
            replay.sensingWrapper.enabled = newValue
        }
    }
    static let sensingOnlyKey = "focusloop.sensingOnly"

    var recordFeatureStreams: Bool {
        get { streams.isEnabled }
        set { streams.isEnabled = newValue }
    }

    @ObservationIgnored let store: SwiftDataSessionStore
    @ObservationIgnored let insights = InsightService()
    @ObservationIgnored let library = LectureLibrary()
    @ObservationIgnored private let streams = FeatureStreamStore()
    @ObservationIgnored var real: LearnerSet
    @ObservationIgnored var replay: LearnerSet
    /// The set the current engine is wired to.
    @ObservationIgnored var active: LearnerSet
    @ObservationIgnored private var engineServices: EngineServices?
    @ObservationIgnored var engine: SessionEngine!
    @ObservationIgnored var reportCache: [UUID: SessionReport] = [:]
    @ObservationIgnored private var permissions: [UIPermission: UIPermissionStatus] = [:]
    @ObservationIgnored private var currentKind: UISessionKind = .focus
    @ObservationIgnored private var auxSource: LiveFeatureFrameSource?
    @ObservationIgnored private var lifecycleTokens: [NSObjectProtocol] = []
    /// Demo readiness screen's short camera probe (device only) and the cached bundle checks.
    @ObservationIgnored var readinessProbe: ReadinessProbe?
    @ObservationIgnored var bundleReadinessCache: [DemoReadinessItem]?

    init(model: LiveSessionModel) {
        self.model = model
        let flags = UIPrefs.loadFlags()
        self.flags = flags
        do { store = try SwiftDataSessionStore() } catch {
            // Fall back to memory so the app still works; nothing is lost that was ever saved.
            store = try! SwiftDataSessionStore(inMemory: true)
        }
        real = LearnerSet(isSynthetic: false)
        replay = LearnerSet(isSynthetic: true)
        active = real
        for set in [real, replay] {
            restoreWeights(set)
            set.sensingWrapper.enabled = UserDefaults.standard.bool(forKey: Self.sensingOnlyKey)
        }
        active = willUseReplay || willUseRecordedVideo ? replay : real
        model.demoDirector = flags.demoDirector
        forecasterDemo(flags.demoDirector)
        buildEngine()
        observeAppLifecycle()
        importLegacySARTRunIfNeeded()
        #if DEBUG
        applySARTDebugLaunchArguments()
        #endif
    }

    /// Backgrounding mid-session pauses it and stops the sensors (AR session included); returning resumes it.
    private func observeAppLifecycle() {
        let nc = NotificationCenter.default
        lifecycleTokens = [
            nc.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.engine.appDidEnterBackground()
                    self?.pauseReadinessProbe(true)
                }
            },
            nc.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.engine.appWillEnterForeground()
                    self?.pauseReadinessProbe(false)
                }
            },
        ]
    }

    private func buildEngine() {
        let set = active
        let classifier = BlendedAttentionClassifier.makeDefault(personalHead: set.personalHead)
        let logger = InMemoryTrialLogger()
        let services = EngineServices(
            flags: effectiveFlags, classifier: classifier, personalHead: set.personalHead, forecaster: set.forecaster,
            policy: set.sensingWrapper, pacing: set.demoScript, logger: logger, labels: set.labels,
            makeFrameSource: { [weak self] forceReplay in self?.makeFrameSource(forceReplay: forceReplay) ?? Self.replaySource() },
            makeNudge: { arm, lecture, playhead in NudgeFactory.make(arm: arm, lecture: lecture, playhead: playhead) },
            persistSession: { [weak self] r in self?.persist(r) },
            streams: streams, demoScript: set.demoScript)
        engineServices = services
        engine = SessionEngine(model: model, services: services)
    }

    /// Wires the engine to the real or the SYNTHETIC learner set for the next session (RECORDED-video sessions
    /// use the SYNTHETIC/demo set too: a replayed clip must not train the real learners).
    func selectLearners(replay useReplay: Bool) {
        let set = useReplay ? replay : real
        guard set !== active else { return }
        if engine.isRunning, let r = engine.stopAndCollect() { persist(r) }
        active = set
        buildEngine()
    }

    private func learners(synthetic: Bool) -> LearnerSet { synthetic ? replay : real }

    private func forecasterDemo(_ on: Bool) {
        for set in [real, replay] {
            (set.forecaster as? CoreMLLapseForecaster)?.demoDirector = on
            (set.forecaster as? HeuristicLapseForecaster)?.demoDirector = on
        }
    }

    // MARK: Sources

    private func makeFrameSource(forceReplay: Bool) -> FeatureFrameSource {
        if scriptedDemoActive, model.studyMode {
            // SYNTHETIC scripted demo: the choreographed replay follows the lecture playhead frame by frame.
            let stream = (try? ReplayStream.load(Self.scriptedDemoStream))
                ?? ReplayStream(frames: [], labels: [], lapse: [], isSynthetic: true)
            return LectureClockReplaySource(stream: stream) { [weak model] in model?.playhead ?? 0 }
        }
        if willUseRecordedVideo, let url = recordedVideoURL {
            // RECORDED: the user's own video drives the real perception pipeline (Simulator too).
            return makeRecordedVideoSource(url: url)
        }
        if !forceReplay, LiveFeatureFrameSource.isSupported {
            let s = LiveFeatureFrameSource(workZone: WorkZoneStore().current(isPhoneScreen: model.studyMode),
                                           restingBaseline: weightsFloat(PersistenceWeightKey.heartRateBaseline))
            s.onBaselineUpdate = { [weak self] b in
                Task { @MainActor in self?.saveFloat(b, key: PersistenceWeightKey.heartRateBaseline) }
            }
            return s
        }
        return Self.replaySource()
    }

    static func replaySource() -> FeatureFrameSource {
        let stream = (try? ReplayStream.load()) ?? ReplayStream(frames: [], labels: [], lapse: [], isSynthetic: true)
        return ReplayFeatureSource(stream: stream)
    }

    // MARK: Permissions

    func permissionStatus(_ p: UIPermission) -> UIPermissionStatus {
        if let s = permissions[p] { return s }
        switch p {
        case .camera:
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: return .granted
            case .denied, .restricted: return .denied
            default: return .notDetermined
            }
        case .motion:
            guard CMMotionActivityManager.isActivityAvailable() else { return .unavailable }
            switch CMMotionActivityManager.authorizationStatus() {
            case .authorized: return .granted
            case .denied, .restricted: return .denied
            default: return .notDetermined
            }
        case .health:
            return HKHealthStore.isHealthDataAvailable() ? .notDetermined : .unavailable
        }
    }

    func requestPermission(_ p: UIPermission) async -> UIPermissionStatus {
        let status: UIPermissionStatus
        switch p {
        case .camera:
            status = await AVCaptureDevice.requestAccess(for: .video) ? .granted : .denied
        case .motion:
            guard CMMotionActivityManager.isActivityAvailable() else { status = .unavailable; break }
            let mgr = CMMotionActivityManager()
            status = await withCheckedContinuation { cont in
                mgr.queryActivityStarting(from: Date().addingTimeInterval(-60), to: Date(), to: .main) { _, error in
                    cont.resume(returning: error == nil ? .granted : .denied)
                }
            }
        case .health:
            guard HKHealthStore.isHealthDataAvailable(), let hr = HKObjectType.quantityType(forIdentifier: .heartRate) else {
                status = .unavailable; break
            }
            do { try await HKHealthStore().requestAuthorization(toShare: [], read: [hr]); status = .granted }
            catch { status = .denied }
        }
        permissions[p] = status
        return status
    }

    // MARK: Sessions

    func startSession(_ kind: UISessionKind) {
        currentKind = kind
        selectLearners(replay: willUseReplay || willUseRecordedVideo)
        switch kind {
        case .focus, .science:
            // Free study: the science view previews the MOCK sample content alongside the session
            // (display only; the engine keeps it out of the forecaster, trial and record).
            engine.start(lecture: nil, timeline: try? BrainContentTimeline(named: "sample_lecture"))
        case .study(let id):
            let lecture = lectures().first { $0.id == id }
            engine.start(lecture: lecture, timeline: library.timeline(for: id) ?? (try? BrainContentTimeline(named: id)))
        }
    }

    func stopSession() async -> UUID? {
        let set = active
        defer { endScriptedDemo() }
        guard let rec = await engine.finishAndCollect() else { return nil }
        guard currentKind != .science, rec.duration >= 5 else { saveWeights(set); return nil }
        persist(rec)
        return rec.id
    }

    private func persist(_ rec: SessionRecord) {
        try? store.save(rec)
        reportCache[rec.id] = nil
        saveWeights(learners(synthetic: usesDemoLearners(rec)))
    }

    /// SYNTHETIC Replay and RECORDED-video sessions belong to the SYNTHETIC/demo learner set.
    func usesDemoLearners(_ r: SessionRecord) -> Bool { r.isReplay || r.isRecorded }

    func lectures() -> [LectureInfo] { StudyLectureLibrary.lectures() }

    func listSessions() -> [UISessionSummary] {
        let records = (try? store.sessions()) ?? []
        let effects = effectsProvider()
        return records.sorted { $0.startedAt > $1.startedAt }.map { r in
            let rep = cachedReport(r, effects: effects)
            return UISessionSummary(id: r.id, startedAt: r.startedAt, duration: r.duration,
                                    focusedFraction: rep.focusedFraction, medianRefocusSeconds: rep.medianRefocusSeconds,
                                    lectureTitle: rep.lectureTitle, isReplay: r.isReplay, contentIsMock: r.contentIsMock)
        }
    }

    func timeline(for lectureID: String?) -> ContentTimeline? {
        guard let id = lectureID else { return nil }
        return library.timeline(for: id) ?? (try? BrainContentTimeline(named: id))
    }

    /// Per-arm effects of the real or the SYNTHETIC trial, computed at most once per call site (the IPW
    /// bootstrap is not free, and History builds one report per session).
    private func effectsProvider() -> (Bool) -> [ArmEffect] {
        var memo: [Bool: [ArmEffect]] = [:]
        return { [unowned self] synthetic in
            if let m = memo[synthetic] { return m }
            let e = self.learners(synthetic: synthetic).policy.effectEstimates()
            memo[synthetic] = e
            return e
        }
    }

    private func cachedReport(_ r: SessionRecord, effects: (Bool) -> [ArmEffect]) -> SessionReport {
        if let c = reportCache[r.id] { return c }
        let title = r.lectureID.flatMap { id in lectures().first { $0.id == id }?.title }
        // A Replay report shows the SYNTHETIC trial's estimates (labelled SYNTHETIC); a real report only real ones.
        let rep = ReportBuilder.build(record: r, timeline: timeline(for: r.lectureID), lectureTitle: title,
                                      effects: effects(usesDemoLearners(r)),
                                      restingHeartRate: weightsFloat(PersistenceWeightKey.heartRateBaseline).map(Double.init))
        reportCache[r.id] = rep
        return rep
    }

    func report(for sessionID: UUID) async -> SessionReport? {
        guard var r = try? store.session(id: sessionID) else { return nil }
        if r.insights.isEmpty {
            r.insights = await insights.insights(for: r, armEffects: learners(synthetic: usesDemoLearners(r)).policy.effectEstimates())
            try? store.save(r)
            reportCache[r.id] = nil
        }
        return cachedReport(r, effects: effectsProvider())
    }

    func reportPreview(for sessionID: UUID) -> SessionReport? {
        guard let r = try? store.session(id: sessionID) else { return nil }
        return cachedReport(r, effects: effectsProvider())
    }

    func aggregateSummary() async -> AggregateSummary? {
        let s = (try? store.sessions()) ?? []
        guard !s.isEmpty else { return nil }
        // Real effects whenever real sessions exist; the SYNTHETIC trial only for an all-Replay history.
        let synthetic = !s.contains { !usesDemoLearners($0) }
        return AttentionProfileBuilder.summary(sessions: s, armEffects: learners(synthetic: synthetic).policy.effectEstimates())
    }

    func profileSeries() -> ProfileSeries {
        let records = ((try? store.sessions()) ?? []).sorted { $0.startedAt < $1.startedAt }
        let effects = effectsProvider()
        let reps = records.map { cachedReport($0, effects: effects) }
        var byHour = Array<[Double]>(repeating: [], count: 24)
        for r in reps { byHour[Calendar.current.component(.hour, from: r.startedAt)].append(r.focusedFraction) }
        return ProfileSeries(points: reps.map {
            .init(date: $0.startedAt, focusedFraction: $0.focusedFraction, medianRefocusSeconds: $0.medianRefocusSeconds,
                  restingHeartRate: $0.restingHeartRate, minutes: $0.duration / 60)
        }, focusByHour: byHour.map { $0.isEmpty ? nil : $0.reduce(0, +) / Double($0.count) },
           isSynthetic: reps.contains { $0.isReplay })
    }

    func profileInsights() async -> [Insight] {
        guard let s = await aggregateSummary() else { return [] }
        return await insights.insights(for: s)
    }

    func faceMeshPoints() -> [CGPoint]? {
        if let rec = engine.currentSource as? RecordedVideoFrameSource { return rec.currentMesh() }
        return (engine.currentSource as? LiveFeatureFrameSource)?.currentMesh()
    }

    // MARK: Data

    func exportReport(_ report: SessionReport) async throws -> URL {
        let cortex = await cortexSnapshot(for: report)
        return try ReportExporter.exportPDF(ReportExportPage(report: report, cortexImage: cortex).frame(width: 612), report: report)
    }

    func exportReportImage(_ report: SessionReport) async throws -> URL {
        let cortex = await cortexSnapshot(for: report)
        return try ReportExporter.exportPNG(ReportExportPage(report: report, cortexImage: cortex), report: report,
                                            width: 612, colorScheme: .light)
    }

    /// A live Metal view can't be captured by `ImageRenderer`, so exports get an offscreen still of the
    /// predicted response at the biggest drift moment.
    private func cortexSnapshot(for report: SessionReport) async -> UIImage? {
        guard let acts = report.biggestDriftActivations else { return nil }
        return await CortexSnapshotter.snapshot(activations: acts)
    }

    /// REAL windows when there are any; otherwise the SYNTHETIC Replay windows (CSV header says SYNTHETIC).
    func exportLabelsCSV() throws -> URL {
        let sink = real.labels.windows.isEmpty && !replay.labels.windows.isEmpty ? replay.labels : real.labels
        return try sink.exportCSV()
    }

    func exportFeatureStreams() throws -> [URL] { try streams.prepareExport() }

    func deleteAllData() throws {
        _ = engine.stopAndCollect()
        scriptedDemoActive = false
        stopReadinessProbe()
        flags = FeatureFlags()            // before the defaults wipe, so no flags key survives it
        try store.deleteAll()
        real.labels.removeAll()
        replay.labels.removeAll()
        streams.removeAll()
        ReportExporter.removeExports()
        try? RecordedVideoLibrary.removeImported()   // the demo recording the user imported (optional)
        insights.clearCache()
        reportCache = [:]
        real = LearnerSet(isSynthetic: false)
        replay = LearnerSet(isSynthetic: true)
        active = willUseReplay || willUseRecordedVideo ? replay : real
        buildEngine()
        model.posteriors = active.policy.posteriors()
        model.effects = []
    }

    // MARK: Weights

    private func restoreWeights(_ set: LearnerSet) {
        if let d = try? store.modelWeights(key: set.key(PersistenceWeightKey.personalHead)) { try? set.personalHead.restore(from: d) }
        if let d = try? store.modelWeights(key: set.key(PersistenceWeightKey.banditState)) { try? set.policy.restore(from: d) }
        if let d = try? store.modelWeights(key: set.key(PersistenceWeightKey.forecasterThreshold)),
           let t = try? PersonalThreshold.restore(from: d) {
            (set.forecaster as? CoreMLLapseForecaster)?.restoreThreshold(t)
            (set.forecaster as? HeuristicLapseForecaster)?.personalThreshold = t
        }
    }

    func saveWeights(_ set: LearnerSet) {
        if let d = try? set.personalHead.serialized() { try? store.saveModelWeights(d, key: set.key(PersistenceWeightKey.personalHead)) }
        if let d = try? set.policy.serialized() { try? store.saveModelWeights(d, key: set.key(PersistenceWeightKey.banditState)) }
        let threshold = (set.forecaster as? CoreMLLapseForecaster)?.personalThreshold
            ?? (set.forecaster as? HeuristicLapseForecaster)?.personalThreshold
        if let d = try? threshold?.serialized() {
            try? store.saveModelWeights(d, key: set.key(PersistenceWeightKey.forecasterThreshold))
        }
    }

    func weightsFloat(_ key: String) -> Float? {
        guard let d = try? store.modelWeights(key: key) else { return nil }
        return try? JSONDecoder().decode(Float.self, from: d)
    }

    private func saveFloat(_ v: Float, key: String) {
        if let d = try? JSONEncoder().encode(v) { try? store.saveModelWeights(d, key: key) }
    }

    // MARK: Slots

    func calibrationView(isPhoneScreen: Bool, onFinish: @escaping () -> Void) -> AnyView? {
        let owner = UUID()
        return AnyView(WorkZoneCalibrationView(isPhoneScreen: isPhoneScreen, samples: { [weak self] in
            self?.calibrationStream(owner: owner) ?? AsyncStream { $0.finish() }
        }, onComplete: { [weak self] _ in
            self?.stopAux(owner: owner)
            onFinish()
        })
        // Leaving the screen early (Close, or the next onboarding step) must turn the camera off too.
        .onDisappear { [weak self] in self?.stopAux(owner: owner) })
    }

    /// Live gaze for calibration (device only). The Simulator path uses the view's SYNTHETIC simulate button.
    private func calibrationStream(owner: UUID) -> AsyncStream<CalibrationGazeReading> {
        guard LiveFeatureFrameSource.isSupported else { return AsyncStream { $0.finish() } }
        let s = LiveFeatureFrameSource(workZone: .default, restingBaseline: nil)
        startAux(s, owner: owner)
        return CalibrationGazeReading.stream(from: s.frames)
    }

    @ObservationIgnored private var auxOwner: UUID?

    func startAux(_ s: LiveFeatureFrameSource, owner: UUID) {
        stopAux()
        auxSource = s
        auxOwner = owner
        s.start()
    }

    /// Stops the calibration/SART camera source; with `owner`, only if that screen still owns it.
    func stopAux(owner: UUID? = nil) {
        if let owner, owner != auxOwner { return }
        auxSource?.stop(); auxSource = nil; auxOwner = nil
    }

    func sartView(onFinish: @escaping () -> Void) -> AnyView? {
        // Record features during the SART on device so its trials become labelled windows.
        var collected: [FeatureFrame] = []
        var collectTask: Task<Void, Never>?
        var origin: Double = 0
        let sessionID = UUID()
        let owner = UUID()
        var recording: FeatureStreamStore.Recording?
        if LiveFeatureFrameSource.isSupported {
            let s = LiveFeatureFrameSource(workZone: WorkZoneStore().current(isPhoneScreen: true), restingBaseline: nil)
            origin = ProcessInfo.processInfo.systemUptime
            startAux(s, owner: owner)
            let rec = streams.begin(sessionID: sessionID, startedAt: Date(), isSynthetic: false, lectureID: nil, kind: "sart")
            recording = rec
            collectTask = Task { for await f in s.frames { collected.append(f); rec?.append(f, playhead: nil) } }
        }
        return AnyView(SARTView(sessionClockOrigin: origin, onSkip: { [weak self] in
            // Skipping keeps nothing: no labels, no trials (the recording is closed by onDisappear).
            collectTask?.cancel()
            self?.stopAux(owner: owner)
            onFinish()
        }) { [weak self] result in
            guard let self else { return }
            collectTask?.cancel()
            self.stopAux(owner: owner)
            // SART labels come only from live frames (none in the Simulator), so they are always REAL.
            let windows = SARTLabeler().label(trials: result.trials, frames: collected, sessionID: sessionID)
            self.real.labels.add(contentsOf: windows)
            self.saveSARTRun(result, sensorsRecorded: !collected.isEmpty)
            if let rec = recording {
                recording = nil
                rec.finish(experienceSamples: [])
                self.streams.attachSART(SARTTrialCSV.make(result.trials, isSynthetic: false), to: rec)
            }
            onFinish()
        }
        .onDisappear { [weak self] in
            collectTask?.cancel()
            recording?.finish(experienceSamples: [])
            recording = nil
            self?.stopAux(owner: owner)
        })
    }
    static let sartKey = "sartTrials"

    private func exportSARTCSV() throws -> URL {
        guard let d = try store.modelWeights(key: Self.sartKey) else { throw ExportError.noSART }
        let trials = try JSONDecoder().decode([SARTTrial].self, from: d)
        return try SARTTrialCSV.write(trials, isSynthetic: !LiveFeatureFrameSource.isSupported)
    }
    enum ExportError: LocalizedError {
        case noSART
        var errorDescription: String? { "Run the SART once (Settings → Calibration) before exporting its trials." }
    }

    func studyModeView(lecture: LectureInfo, model: LiveSessionModel, onFinish: @escaping () -> Void) -> AnyView? {
        guard let tl = timeline(for: lecture.id) else { return nil }
        let scripted = scriptedDemoActive
        // Study Mode renders card nudges itself; every other presentation (glows, light pulse, breathing,
        // stretch) is hosted on top by the shared nudge layer. The scripted demo (SYNTHETIC) closes itself
        // when the lecture ends (which saves the session and opens its report) and lets an unanswered card
        // continue on its own, so the whole demo runs hands-free.
        return AnyView(ZStack {
            StudyModeView(model: model, lecture: lecture, timeline: tl, autoplay: true,
                          closeWhenFinished: scripted, autoContinueCardsAfter: scripted ? Self.scriptedCardSeconds : nil,
                          onClose: onFinish)
            NudgeHostLayer(model: model, services: self, excludeCards: true)
            SessionHintsOverlay(model: model, services: self)
        })
    }
    /// Scripted demo: an unanswered card continues by itself after this many seconds.
    static let scriptedCardSeconds: Double = 8

    func nudgeView(_ presentation: NudgePresentation, model: LiveSessionModel) -> AnyView? {
        let fullScreen: Bool
        switch presentation {
        case .breathing, .stretch, .lightPulse: fullScreen = true
        case .glow, .card: fullScreen = false
        }
        let opaque: Bool
        switch presentation {
        case .breathing, .stretch: opaque = true
        default: opaque = false
        }
        return AnyView(NudgePresentationView(presentation: presentation, heartRate: model.heartRate,
                                             bpmHistory: model.heartRateHistory,
                                             onDismiss: { model.actions.dismissNudge() },
                                             onChoice: { model.actions.answerRecall($0) })
            // Breathing and the stretch break take the whole screen on an opaque backdrop (over Study Mode's
            // video and scrubber they were unreadable); the light pulse stays a translucent overlay.
            .frame(maxWidth: fullScreen ? .infinity : nil, maxHeight: fullScreen ? .infinity : nil)
            .background { if opaque { FLColor.background.opacity(0.97).ignoresSafeArea() } }
            // Full-screen nudges cover the screen's own badges, so they carry SYNTHETIC / DEMO themselves.
            .overlay(alignment: .top) { if fullScreen { ProvenanceBadges(model: model) } })
    }

    func breathingView(model: LiveSessionModel, onClose: @escaping () -> Void) -> AnyView? {
        // Full-screen Quiet Dark chrome: solid black, the guide centred, provenance tags pinned top-centre.
        AnyView(ZStack {
            FLColor.bg.ignoresSafeArea()
            BreathingGuideView(targetBreathsPerMinute: 6, heartRate: model.heartRate,
                               bpmHistory: model.heartRateHistory, onDone: onClose)
                .padding(.horizontal, FLSpacing.gutter)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .overlay(alignment: .top) { ProvenanceBadges(model: model) })
    }

    func experienceSampleView(model: LiveSessionModel) -> AnyView? {
        AnyView(ExperienceSamplePrompt(onAnswer: { focused, reason in model.actions.answerExperienceSample(focused, reason) },
                                       onDismiss: { model.experiencePromptVisible = false }))
    }

    func debugMenuItems() -> AnyView? {
        AnyView(DebugMenuExtras(services: self, exportSART: { [weak self] in
            guard let self else { throw ExportError.noSART }
            return try self.exportSARTCSV()
        }))
    }
}

/// SYNTHETIC (Replay) and DEMO badges for full-screen module views that have none of their own.
private struct ProvenanceBadges: View {
    let model: LiveSessionModel
    var body: some View {
        if model.isReplay || model.demoDirector || model.isRecorded {
            FLBadgeRow(synthetic: model.isReplay, demo: model.demoDirector, compact: true)
                .padding(.top, 8)
                .allowsHitTesting(false)
        }
    }
}

/// Extra debug rows: brain kit demo, Study Mode debug root, sensing-only toggle, SART CSV export.
private struct DebugMenuExtras: View {
    @Bindable var services: AppServices
    let exportSART: () throws -> URL

    var body: some View {
        NavigationLink("Brain cortex demo (debug)") { BrainCortexDemoView() }
        NavigationLink("Study Mode debug") { StudyModeDebugRoot(showPicker: true) }
        Toggle("Sensing only (no nudges, study condition B)", isOn: $services.sensingOnly)
        NavigationLink("Use my recorded video (RECORDED)") {
            List { RecordedVideoSettingsSection(controller: services) }
                .listStyle(.insetGrouped)
                .navigationTitle("Recorded video")
        }
        .accessibilityIdentifier("settings.recordedVideo")
        CSVExportButton("Export SART trials (CSV)", makeFile: exportSART)
    }
}

/// Wraps the trial engine. When `enabled` (sensing-only study condition), every decision is the no-nudge
/// control with randomisation probability 1 and is NOT fed back into the bandit.
final class SensingOnlyPolicy: InterventionPolicy {
    let base: InterventionPolicy
    var enabled = false
    init(base: InterventionPolicy) { self.base = base }

    func decide(policy: NudgePolicyKind, context: NudgeContext) -> NudgeDecision? {
        guard let d = base.decide(policy: policy, context: context) else { return nil }
        guard enabled else { return d }
        return NudgeDecision(t: d.t, policy: d.policy, context: d.context, arm: .noNudge, randomizationProbability: 1,
                             armProbabilities: [NudgeArm.noNudge.rawValue: 1])
    }
    func record(outcome: NudgeOutcome, for decision: NudgeDecision) {
        guard !enabled || decision.randomizationProbability < 1 else { return }
        base.record(outcome: outcome, for: decision)
    }
    func posteriors() -> [ArmPosterior] { base.posteriors() }
    func effectEstimates() -> [ArmEffect] { base.effectEstimates() }
    func serialized() throws -> Data { try base.serialized() }
    func restore(from data: Data) throws { try base.restore(from: data) }
}
