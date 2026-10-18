import Foundation

/// Runs one FocusLoop session: Sense → Understand → Predict → Intervene → Measure → Learn.
///
/// Threading: all pipeline state lives on `SessionPipeline` (an actor). It consumes the 10 Hz
/// `FeatureFrame` stream, runs the classifier and forecaster at 1 Hz, and hands UI-facing
/// results to `SessionEngine` on the main actor, which is the only writer of `LiveSessionModel`.
@MainActor
final class SessionEngine {
    let model: LiveSessionModel
    private let services: EngineServices
    private var pipeline: SessionPipeline?
    private var source: FeatureFrameSource?
    private var consumeTask: Task<Void, Never>?
    private var hrTask: Task<Void, Never>?
    private var activeNudge: Nudge?
    private var record = SessionRecord()
    private var streamRecording: FeatureStreamStore.Recording?
    private var lecture: LectureInfo?
    private lazy var presenter = ModelNudgePresenter(model: model)
    /// Elapsed time when the Study Mode player last reported a playhead (to detect a stalled player).
    private var lastPlayheadUpdate: Double = -1
    /// True while the session is paused because the app went to the background.
    private(set) var autoPaused = false

    init(model: LiveSessionModel, services: EngineServices) {
        self.model = model
        self.services = services
        installActions()
    }

    var isRunning: Bool { model.phase == .running || model.phase == .paused }
    var currentSource: FeatureFrameSource? { source }

    // MARK: Lifecycle

    func start(lecture: LectureInfo?, timeline: ContentTimeline?) {
        if isRunning { _ = stopAndCollect() }
        self.lecture = lecture
        let flags = services.flags
        let studyMode = lecture != nil
        // Set before creating the source: the live source picks the phone-screen work zone in Study Mode.
        model.studyMode = studyMode
        let src = services.makeFrameSource(flags.forceReplay)
        source = src
        // RECORDED video of a real person (demonstration): real signals, but the same clip replays many times,
        // so like Demo Director it never trains anything (personal head, labels, forecaster threshold).
        let recorded = src is RecordedVideoFrameSource
        // DEMONSTRATION ONLY pacing when Demo Director is on.
        let pacing = flags.demoDirector ? PacingConfig.demo : PacingConfig()
        // DEMONSTRATION ONLY: Demo Director sessions get fixed demo nudges (not Thompson sampling) and learn
        // nothing (no bandit, personal head, label store or forecaster-threshold updates).
        services.demoScript?.beginSession(enabled: flags.demoDirector, lecture: studyMode ? lecture : nil)
        autoPaused = false

        model.phase = .running
        model.sessionID = UUID()
        model.startedAt = Date()
        model.elapsed = 0
        model.isReplay = src.isSynthetic
        model.isRecorded = recorded
        model.demoDirector = flags.demoDirector
        model.timeline = timeline
        model.playhead = 0
        model.stateHistory = []; model.riskHistory = []; model.events = []
        model.heartRateHistory = []; model.recentFrames = []; model.forecastLine = []
        model.estimate = nil; model.lapseRisk = nil; model.heartRate = nil; model.latestFrame = nil
        model.lapseElapsed = nil; model.lastRefocusSeconds = nil; model.lastRefocusAt = nil
        model.activeNudge = nil; model.activePresentation = nil; model.lastDecision = nil
        model.experiencePromptVisible = false
        model.posteriors = services.policy.posteriors()
        model.effects = services.policy.effectEstimates()
        lastPlayheadUpdate = -1

        // Free study only *previews* the (MOCK) sample timeline in the Science view. The student is not
        // watching it, so its content features must not reach the forecaster, the trial context or the
        // session record (they would be unrelated, unlabelled MOCK influence on real numbers).
        let sessionTimeline = studyMode ? timeline : nil
        record = SessionRecord(id: model.sessionID, startedAt: model.startedAt, lectureID: studyMode ? lecture?.id : nil,
                               contentIsMock: sessionTimeline?.isMock ?? false, isReplay: src.isSynthetic,
                               sourceKind: recorded ? RecordedVideoFrameSource.sourceKind
                                   : (flags.demoDirector ? DoctorReportBuilder.demoDirectorSourceKind : nil))

        // A replayed recording is not new data: never add it to the retraining feature streams.
        streamRecording = recorded ? nil : services.streams?.begin(sessionID: model.sessionID, startedAt: model.startedAt,
                                                  isSynthetic: src.isSynthetic,
                                                  lectureID: studyMode ? (lecture?.id ?? timeline?.lectureID) : nil)
        let rec = streamRecording

        let p = SessionPipeline(components: services.pipelineComponents, pacing: pacing, timeline: sessionTimeline,
                                studyMode: studyMode, sessionID: model.sessionID,
                                experienceSampling: flags.experienceSampling && !recorded,
                                learning: !flags.demoDirector && !recorded)
        pipeline = p
        src.start()

        consumeTask = Task { [weak self] in
            for await frame in src.frames {
                guard let self, !Task.isCancelled else { break }
                guard self.model.phase == .running else { continue }
                self.model.elapsed = frame.t
                // Free study previews the content timeline alongside the session; in Study Mode the
                // player drives the playhead (fallback: follow session time if the player stalls).
                if let tl = self.model.timeline, tl.duration > 0,
                   !self.model.studyMode || frame.t - self.lastPlayheadUpdate > 3 {
                    let ph = self.model.studyMode ? min(tl.duration, self.model.playhead + 0.1)
                                                  : frame.t.truncatingRemainder(dividingBy: tl.duration)
                    self.model.playhead = ph
                    await p.setPlayhead(ph)
                }
                rec?.append(frame, playhead: studyMode ? self.model.playhead : nil)
                if let out = await p.ingest(frame) { self.apply(out) }
                self.model.latestFrame = frame
                self.model.faceDetected = frame.faceDetected > 0.5
                self.model.recentFrames.append(frame)
                if self.model.recentFrames.count > 100 { self.model.recentFrames.removeFirst(self.model.recentFrames.count - 100) }
            }
        }
        hrTask = Task { [weak self] in
            for await hr in src.heartRate {
                guard let self, !Task.isCancelled else { break }
                self.model.heartRate = hr
                self.model.heartRateHistory.append((hr.t, hr.bpm))
                if self.model.heartRateHistory.count > 600 { self.model.heartRateHistory.removeFirst() }
            }
        }
    }

    /// Stops the session, waits for the pipeline to resolve pending rewards (so the last nudge's outcome
    /// reaches the trial AND the record), and returns the record (nil if nothing was running). Does not persist.
    func finishAndCollect() async -> SessionRecord? {
        let p = pipeline
        guard var rec = stopAndCollect(flushPipeline: false) else { return nil }
        if let p {
            let outs = await p.finish(at: rec.duration)
            let events = outs.map { SessionEvent.outcome($0) }
            rec.events.append(contentsOf: events)
            // A new session may have started while we waited; don't write into its model.
            if !isRunning, model.sessionID == rec.id {
                model.events.append(contentsOf: events)
                model.posteriors = services.policy.posteriors()
                model.effects = services.policy.effectEstimates()
            }
        }
        return rec
    }

    /// Stops the session and returns its record (nil if nothing was running). Does not persist.
    /// Pending rewards are resolved in the background (use `finishAndCollect()` to wait for them).
    func stopAndCollect(flushPipeline: Bool = true) -> SessionRecord? {
        consumeTask?.cancel(); consumeTask = nil
        hrTask?.cancel(); hrTask = nil
        source?.stop(); source = nil
        activeNudge?.cancel(); activeNudge = nil
        model.activeNudge = nil; model.activePresentation = nil
        model.experiencePromptVisible = false
        if let r = streamRecording {
            streamRecording = nil
            r.finish(experienceSamples: model.events.compactMap {
                if case let .experienceSample(t, focused, reason) = $0 { return (t, focused, reason) } else { return nil }
            })
        }
        guard isRunning else { return nil }
        model.phase = .finished
        record.duration = model.elapsed
        if flushPipeline, let p = pipeline {
            let t = model.elapsed
            Task { _ = await p.finish(at: t) }
        }
        pipeline = nil
        record.events = model.events
        model.posteriors = services.policy.posteriors()
        model.effects = services.policy.effectEstimates()
        return record
    }

    func stop() {
        if let r = stopAndCollect() { services.persistSession(r) }
    }

    // MARK: Background / foreground

    /// The app left the foreground mid-session: pause the session and stop the sensors (the AR session and
    /// Core Motion are paused; Replay holds its stream), so nothing runs in the background and no gap is
    /// misread as a lapse.
    func appDidEnterBackground() {
        guard model.phase == .running else { return }
        autoPaused = true
        model.phase = .paused
        (source as? SuspendableFrameSource)?.suspend()
    }

    /// Back in the foreground: restart the sensors and resume a session that was paused by backgrounding
    /// (a session the user paused stays paused).
    func appWillEnterForeground() {
        guard autoPaused else { return }
        autoPaused = false
        (source as? SuspendableFrameSource)?.resume()
        if model.phase == .paused { model.phase = .running }
    }

    // MARK: Pipeline output → UI

    private func apply(_ out: SessionPipeline.Output) {
        if let e = out.estimate {
            model.estimate = e
            model.stateHistory.append(e)
            if model.stateHistory.count > 900 { model.stateHistory.removeFirst() }
        }
        if let r = out.risk {
            model.lapseRisk = r
            model.riskHistory.append(r)
            if model.riskHistory.count > 900 { model.riskHistory.removeFirst() }
        }
        model.forecastLine = out.forecastLine
        model.lapseElapsed = out.lapseElapsed
        if let s = out.refocusSeconds {
            model.lastRefocusSeconds = s
            model.lastRefocusAt = model.elapsed
        }
        if let tick = out.tick { record.ticks.append(tick) }
        model.events.append(contentsOf: out.events)
        if out.endNudge { endNudge() }
        if let d = out.decision { present(decision: d) }
        if out.outcomeRecorded || out.decision != nil {
            model.posteriors = services.policy.posteriors()
            model.effects = services.policy.effectEstimates()
        }
        if out.promptExperienceSample, model.activeNudge == nil { model.experiencePromptVisible = true }
    }

    private func present(decision d: NudgeDecision) {
        model.lastDecision = d
        guard d.arm != .noNudge else { return }
        activeNudge?.cancel()
        let n = services.makeNudge(d.arm, lecture, model.playhead)
        activeNudge = n
        model.activeNudge = d.arm
        n.start(presenter: presenter)
    }

    private func endNudge() {
        activeNudge?.cancel(); activeNudge = nil
        model.activeNudge = nil
        model.activePresentation = nil
    }

    // MARK: Actions from UI

    private func installActions() {
        var a = LiveSessionModel.Actions()
        a.stop = { [weak self] in self?.stop() }
        a.pause = { [weak self] in if self?.model.phase == .running { self?.model.phase = .paused } }
        a.resume = { [weak self] in
            guard let self, self.model.phase == .paused else { return }
            if self.autoPaused { self.autoPaused = false; (self.source as? SuspendableFrameSource)?.resume() }
            self.model.phase = .running
        }
        a.dismissNudge = { [weak self] in
            guard let self else { return }
            let p = self.pipeline
            Task { await p?.noteDismissal() }
            self.endNudge()
        }
        a.answerRecall = { [weak self] _ in self?.endNudge() }
        a.answerExperienceSample = { [weak self] focused, reason in
            guard let self else { return }
            self.model.experiencePromptVisible = false
            let t = self.model.elapsed
            self.model.events.append(.experienceSample(t: t, focused: focused, reason: reason))
            let p = self.pipeline
            Task { await p?.label(focused: focused, reason: reason, at: t) }
        }
        a.setPlayhead = { [weak self] s in
            guard let self else { return }
            self.model.playhead = s
            self.lastPlayheadUpdate = self.model.elapsed
            let p = self.pipeline
            Task { await p?.setPlayhead(s) }
        }
        model.actions = a
    }
}

/// A frame source that can hold its sensors while the app is in the background and pick up again after.
protocol SuspendableFrameSource: AnyObject {
    func suspend()
    func resume()
}

extension ReplayFeatureSource: SuspendableFrameSource {
    func suspend() { pause() }
}

/// The lecture-clocked SYNTHETIC replay follows the video, which pauses on its own in the background.
extension LectureClockReplaySource: SuspendableFrameSource {
    func suspend() {}
    func resume() {}
}
