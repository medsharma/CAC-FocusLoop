import Foundation

/// The per-session perception → decision pipeline. Runs off the main thread (actor isolation).
/// Consumes 10 Hz frames; classifier + forecaster run at 1 Hz.
actor SessionPipeline {
    struct Output {
        var estimate: StateEstimate?
        var risk: LapseRisk?
        var forecastLine: [(t: Double, p: Float)] = []
        var tick: SessionTick?
        var events: [SessionEvent] = []
        var decision: NudgeDecision?
        var outcomeRecorded = false
        var endNudge = false
        var lapseElapsed: Double?
        var refocusSeconds: Double?
        var promptExperienceSample = false
    }

    private let c: PipelineComponents
    private let timeline: ContentTimeline?
    private let studyMode: Bool
    private let sessionID: UUID
    private let rewards: RewardTracker
    private let riskStream = LapseRiskStream()
    private let sampler: ExperienceSampler
    /// False for Demo Director sessions (DEMONSTRATION ONLY): nothing learns from them. The personal head,
    /// the label store and the forecaster's personal threshold are left untouched (the demo policy already
    /// keeps its fixed decisions away from the bandit).
    private let learning: Bool

    private var frames: [FeatureFrame] = []
    private var content: [ContentFeatures] = []
    private var lastSecond = -1
    private var playhead: Double = 0
    private var lastEstimate: StateEstimate?

    // Lapse / refocus bookkeeping (a state must be sustained 2 s to count as a change).
    private var stableState: AttentionState = .focused
    private var candidate: AttentionState = .focused
    private var candidateSince: Double = 0
    private var lapseStart: Double?

    private var pending: [UUID: NudgeDecision] = [:]
    private var activeDecision: UUID?
    private var predictions: [LapseRisk] = []
    private var lapseOnsets: [Double] = []
    private var wasInValley = false
    private var wasPickedUp = false
    private var wasAlert = false

    init(components: PipelineComponents, pacing: PacingConfig, timeline: ContentTimeline?, studyMode: Bool,
         sessionID: UUID, experienceSampling: Bool, learning: Bool = true) {
        c = components
        self.learning = learning
        self.timeline = timeline
        self.studyMode = studyMode
        self.sessionID = sessionID
        rewards = RewardTracker(pacing: pacing)
        sampler = ExperienceSampler(pacing: pacing, isEnabled: experienceSampling)
        c.pacing?.setPacing(pacing)
        c.pacing?.resetSession()
    }

    func setPlayhead(_ t: Double) {
        playhead = t
        c.demoScript?.notePlayhead(t)
    }

    func noteDismissal() {
        rewards.noteDismissal(decisionID: activeDecision)
    }

    func dismissExperienceSample() { sampler.dismiss() }

    /// Experience-sampling answer → labelled window + personal-head update.
    func label(focused: Bool, reason: NotFocusedReason?, at t: Double) {
        let state: AttentionState = focused ? .focused : (reason?.state ?? .distracted)
        if !focused { lapseOnsets.append(t); rewards.noteLapse(t: t) }
        let window = sampler.answer(focused: focused, reason: reason, frames: frames, sessionID: sessionID)
        guard learning else { return }   // DEMONSTRATION ONLY sessions never train the personal model
        if let w = window { c.labels.add(w) }
        if let e = lastEstimate, !e.embedding.isEmpty { c.personalHead?.update(embedding: e.embedding, label: state) }
    }

    /// Ends the session: resolves pending rewards so the trial learns from the last nudge too.
    func finish(at t: Double) -> [NudgeOutcome] {
        let outs = rewards.flush(t: t)
        for o in outs {
            if let d = pending[o.decisionID] { c.policy.record(outcome: o, for: d) }
            c.logger.log(outcome: o)
        }
        pending.removeAll()
        return outs
    }

    /// Returns UI output once per second (nil for the other 9 frames).
    func ingest(_ f: FeatureFrame) -> Output? {
        frames.append(f)
        let maxFrames = Int(FeatureFrame.forecastWindowSeconds * FeatureFrame.rateHz)
        if frames.count > maxFrames { frames.removeFirst(frames.count - maxFrames) }
        let second = Int(f.t)
        guard second != lastSecond else { return nil }
        lastSecond = second
        return step(t: f.t, frame: f)
    }

    private func step(t: Double, frame: FeatureFrame) -> Output {
        var out = Output()

        // Content features for this second (predicted brain response to the content, average brain).
        let cf = timeline?.features(at: playhead) ?? .none
        content.append(cf)
        if content.count > Int(FeatureFrame.forecastWindowSeconds) { content.removeFirst() }
        if timeline != nil {
            if cf.inValley && !wasInValley { out.events.append(.valleyEntered(t: t, playhead: playhead)) }
            wasInValley = cf.inValley
        }

        // Understand.
        let est = c.classifier.classify(window: Array(frames.suffix(Int(FeatureFrame.stateWindowSeconds * FeatureFrame.rateHz))))
        lastEstimate = est
        out.estimate = est
        c.pacing?.noteState(est.state, t: t)

        // Predict.
        let risk = c.forecaster.forecast(frames: frames, content: timeline == nil ? [] : content, t: t)
        out.risk = risk
        predictions.append(risk)
        out.forecastLine = riskStream.update(risk: risk, timeline: timeline, playhead: timeline == nil ? nil : playhead)
        resolveMaturedPredictions(now: t)

        // Measure lapses / refocus.
        trackState(est.state, t: t, out: &out)
        let picked = frame.phonePickup > 0.5
        if picked && !wasPickedUp { out.events.append(.phonePickup(t: t)) }
        wasPickedUp = picked

        // Rewards.
        for o in rewards.observe(state: est.state, t: t) {
            if let d = pending.removeValue(forKey: o.decisionID) { c.policy.record(outcome: o, for: d) }
            c.logger.log(outcome: o)
            out.events.append(.outcome(o))
            out.outcomeRecorded = true
            if o.decisionID == activeDecision { activeDecision = nil; out.endNudge = true }
        }
        // A reactive nudge ends as soon as the student is focused again (breathing/stretch run their course).
        if est.state == .focused, let id = activeDecision, let d = pending[id], d.policy == .reactive,
           d.arm != .pacedBreathing, d.arm != .stretchBreak {
            out.endNudge = true
        }

        // Intervene (one nudge at a time; the trial engine applies debounce + cooldown + randomisation).
        if pending.isEmpty, let d = decide(est: est, risk: risk, frame: frame, t: t) {
            c.logger.log(decision: d)
            rewards.track(d)
            pending[d.id] = d
            activeDecision = d.arm == .noNudge ? nil : d.id
            out.decision = d
            out.events.append(.nudge(d))
            out.endNudge = false
        }
        // Log alerts on the rising edge only (one event per alert episode, not one per second).
        if risk.isAlert && !wasAlert { out.events.append(.riskAlert(t: t, probability: risk.probability)) }
        wasAlert = risk.isAlert

        // Experience sampling (labels for the personal model).
        if sampler.shouldPrompt(at: t, nudgeActive: activeDecision != nil) { out.promptExperienceSample = true }

        out.lapseElapsed = lapseStart.map { t - $0 }
        out.tick = SessionTick(t: t, state: est.state, probabilities: est.probabilities, lapseRisk: risk.probability,
                               heartRate: frame.heartRate > 0 ? frame.heartRate : nil,
                               heartRateQuality: frame.heartRate > 0 ? frame.heartRateQuality : nil,
                               playhead: timeline != nil ? playhead : nil,
                               predictedEngagement: timeline != nil ? cf.engagement : nil,
                               predictedDriftRisk: timeline != nil ? cf.driftRisk : nil)
        return out
    }

    private func trackState(_ s: AttentionState, t: Double, out: inout Output) {
        if s != candidate { candidate = s; candidateSince = t }
        guard candidate != stableState, t - candidateSince >= 2 else { return }
        let from = stableState
        stableState = candidate
        out.events.append(.stateChanged(t: candidateSince, from: from, to: stableState))
        if stableState != .focused, lapseStart == nil {
            lapseStart = candidateSince
            lapseOnsets.append(candidateSince)
            rewards.noteLapse(t: candidateSince)
            out.events.append(.lapseDetected(t: candidateSince, state: stableState))
        } else if stableState == .focused, let ls = lapseStart {
            let secs = candidateSince - ls
            out.refocusSeconds = secs
            out.events.append(.refocused(t: candidateSince, afterSeconds: secs))
            lapseStart = nil
        }
    }

    private func decide(est: StateEstimate, risk: LapseRisk, frame: FeatureFrame, t: Double) -> NudgeDecision? {
        let upcoming = timeline?.upcomingValley(after: playhead, within: 30).map { Float($0.start - playhead) }
        let last = c.logger.decisions.last?.t
        let ctx = NudgeContext(t: t, state: est.state, lapseRisk: risk.probability,
                               predictedEngagement: timeline?.engagement(at: playhead) ?? 0.5,
                               sessionMinute: Float(t / 60), hourOfDay: Calendar.current.component(.hour, from: Date()),
                               minutesSinceLastNudge: last.map { Float((t - $0) / 60) } ?? 99,
                               studyMode: studyMode, phonePickedUp: frame.phonePickup > 0.5,
                               upcomingValleyInSeconds: upcoming)
        if est.state != .focused { return c.policy.decide(policy: .reactive, context: ctx) }
        // Pre-emptive: focused now, but the forecaster says a lapse is coming.
        // Not before the forecaster has its full 20 s input window: early "risk" is just padding.
        guard risk.isAlert, lapseStart == nil, t >= FeatureFrame.forecastWindowSeconds else { return nil }
        return c.policy.decide(policy: .preemptive, context: ctx)
    }

    private func resolveMaturedPredictions(now: Double) {
        while let first = predictions.first, now - first.t >= first.horizon {
            predictions.removeFirst()
            let lapsed = lapseOnsets.contains { $0 > first.t && $0 <= first.t + first.horizon }
            // Demo sessions (DEMONSTRATION ONLY) run on a lowered threshold; never adapt the real one from them.
            if learning { c.forecaster.observe(prediction: first, lapseOccurred: lapsed) }
        }
        lapseOnsets.removeAll { now - $0 > 120 }
    }
}
