import Foundation

// PREVIEW-ONLY. Everything in this file is a stand-in so the UI animates in the Simulator before the
// real session engine (classifier, forecaster, MRT engine) is wired in by the orchestrator.
// The state / risk / bandit numbers here come from simple heuristics over the SYNTHETIC replay stream;
// they are never reported as results and every screen shows the SYNTHETIC badge (`model.isReplay`).

/// Simple rule-based stand-ins for the classifier, forecaster and bandit. PREVIEW-ONLY.
struct UIPreviewHeuristics {
    private var smoothed: [Float] = [0.85, 0.05, 0.05, 0.05]
    private var lastYawn: Float = 0, lastNod: Float = 0
    private var drowsyEvidenceUntil: Double = -1

    /// Probabilities in `AttentionState.allCases` order from the last ~3 s of frames.
    mutating func classify(window: [FeatureFrame]) -> [Float] {
        guard let f = window.last else { return smoothed }
        let recent = window.suffix(30)
        func mean(_ k: KeyPath<FeatureFrame, Float>) -> Float { recent.map { $0[keyPath: k] }.reduce(0, +) / Float(recent.count) }

        // New yawn / nod events (trailing-60 s counts going up) are drowsiness evidence for 12 s.
        if f.yawnCount > lastYawn || f.nodCount > lastNod || f.yawnActive > 0.5 || f.headPitch < -0.2 {
            drowsyEvidenceUntil = f.t + 12
        }
        lastYawn = f.yawnCount; lastNod = f.nodCount

        let distracted = min(1, mean(\.gazeDeviation) / 0.3) + mean(\.phonePickup) + min(1, mean(\.offZoneDwell) / 3)
        let drowsy: Float = (f.t < drowsyEvidenceUntil ? 1.3 : 0) + max(0, mean(\.perclos) - 0.15) * 4
        let stressed = max(0, min(1.2, (mean(\.heartRateDelta) - 5) / 7)) + max(0, min(1, (mean(\.browTension) - 0.25) / 0.2))
        let logits: [Float] = [1.4, 3 * distracted, 3 * drowsy, 2.6 * stressed]
        let mx = logits.max() ?? 0
        let ex = logits.map { exp($0 - mx) }
        let s = ex.reduce(0, +)
        let p = ex.map { $0 / s }
        smoothed = zip(smoothed, p).map { $0 + 0.55 * ($1 - $0) }
        let total = smoothed.reduce(0, +)
        smoothed = smoothed.map { $0 / total }
        return smoothed
    }

    /// Stand-in P(lapse within 15 s) from current non-focus, its trend, and the content's predicted drift.
    static func risk(pNotFocused: Float, trend: Float, content: ContentFeatures?) -> Float {
        var z: Float = -2.4 + 3.2 * pNotFocused + 4 * max(0, trend)
        if let c = content {
            z += 0.55 * c.driftRisk + 1.1 * (1 - c.engagement)
            if c.secondsToValley < 10 { z += 0.9 * (1 - c.secondsToValley / 10) }
            if c.inValley { z += 0.4 }
        }
        return 1 / (1 + exp(-z))
    }
}

/// Fills a `LiveSessionModel` from `ReplayStream.load()` + `BrainContentTimeline(named: "sample_lecture")`
/// at 10 Hz. PREVIEW-ONLY: the real engine replaces this at integration.
@MainActor
public final class ReplayPreviewDriver {
    public let model: LiveSessionModel
    public private(set) var stream: ReplayStream?
    public private(set) var timeline: BrainContentTimeline?
    /// Playback speed multiplier (1 = real time).
    public var speed: Double = 1
    public private(set) var isRunning = false
    public private(set) var ticks: [SessionTick] = []
    public private(set) var nudgeCount = 0
    public private(set) var preemptiveCount = 0

    /// Shared across sessions so the Personal Trial intervals narrow as you "study more".
    var bandit: UIPreviewBandit

    private var task: Task<Void, Never>?
    private var index = 0
    private var lap = 0.0
    private var heuristics = UIPreviewHeuristics()
    private var lapseStart: Double?
    private var nonFocusedSince: Double?
    private var lastNudgeAt: Double = -100
    private var pending: (decision: NudgeDecision, until: Double)?
    private var presentationUntil: Double = 0
    private var lastState: AttentionState = .focused
    private var rng = UIPreviewRNG(seed: 7)

    public init(model: LiveSessionModel, bandit: UIPreviewBandit? = nil) {
        self.model = model
        self.bandit = bandit ?? UIPreviewBandit()
        stream = try? ReplayStream.load()
        timeline = try? BrainContentTimeline(named: "sample_lecture")
    }

    // MARK: Lifecycle

    /// Resets the model and starts real-time playback (or just resets when `realtime` is false).
    public func start(studyMode: Bool, startOffset: Double = 0, realtime: Bool = true) {
        stop()
        reset(studyMode: studyMode, startOffset: startOffset)
        guard realtime else { return }
        isRunning = true
        task = Task { [weak self] in
            let clock = ContinuousClock()
            var next = clock.now
            while !Task.isCancelled {
                guard let self else { return }
                self.step()
                next = next.advanced(by: .milliseconds(Int(100 / max(0.1, self.speed))))
                try? await clock.sleep(until: next)
            }
        }
    }

    public func pause() { task?.cancel(); task = nil; isRunning = false; model.phase = .paused }

    public func resume() {
        guard !isRunning, model.phase == .paused else { return }
        model.phase = .running
        isRunning = true
        task = Task { [weak self] in
            let clock = ContinuousClock()
            var next = clock.now
            while !Task.isCancelled {
                guard let self else { return }
                self.step()
                next = next.advanced(by: .milliseconds(Int(100 / max(0.1, self.speed))))
                try? await clock.sleep(until: next)
            }
        }
    }

    public func stop() {
        task?.cancel(); task = nil; isRunning = false
        if model.phase == .running || model.phase == .paused { model.phase = .finished }
        model.activePresentation = nil
        model.activeNudge = nil
    }

    /// Runs `seconds` of replay synchronously (for building SYNTHETIC history / report fixtures).
    public func simulate(seconds: Double, studyMode: Bool, startOffset: Double = 0) {
        start(studyMode: studyMode, startOffset: startOffset, realtime: false)
        for _ in 0..<Int(seconds * FeatureFrame.rateHz) { step() }
        stop()
    }

    /// Snapshot of what was recorded (numbers only).
    public func record(lectureID: String?) -> SessionRecord {
        SessionRecord(id: model.sessionID, startedAt: model.startedAt, duration: model.elapsed, lectureID: lectureID,
                      contentIsMock: timeline?.isMock ?? false, isReplay: true, ticks: ticks, events: model.events)
    }

    private func reset(studyMode: Bool, startOffset: Double) {
        let m = model
        m.sessionID = UUID(); m.startedAt = Date(); m.elapsed = 0
        m.phase = .running; m.isReplay = true; m.studyMode = studyMode
        m.latestFrame = nil; m.recentFrames = []; m.heartRate = nil; m.heartRateHistory = []
        m.faceDetected = false; m.estimate = nil; m.stateHistory = []; m.lapseRisk = nil; m.riskHistory = []
        m.forecastLine = []; m.lapseElapsed = nil; m.lastRefocusSeconds = nil; m.lastRefocusAt = nil
        m.timeline = timeline; m.playhead = 0; m.activeNudge = nil; m.activePresentation = nil
        m.lastDecision = nil; m.events = []
        m.posteriors = bandit.posteriors(); m.effects = bandit.effects()
        m.experiencePromptVisible = false
        ticks = []; nudgeCount = 0; preemptiveCount = 0
        heuristics = UIPreviewHeuristics()
        lapseStart = nil; nonFocusedSince = nil; lastNudgeAt = -100; pending = nil; presentationUntil = 0
        lastState = .focused
        let n = stream?.frames.count ?? 0
        index = n > 0 ? Int(startOffset * FeatureFrame.rateHz) % n : 0
        lap = 0
        installActions()
    }

    private func installActions() {
        var a = LiveSessionModel.Actions()
        a.start = { [weak self] in self?.resume() }
        a.pause = { [weak self] in self?.pause() }
        a.resume = { [weak self] in self?.resume() }
        a.stop = { [weak self] in self?.stop() }
        a.dismissNudge = { [weak self] in
            guard let self else { return }
            self.model.activePresentation = nil
            self.model.activeNudge = nil
            if let p = self.pending { self.finish(p.decision, refocusSeconds: nil, dismissed: true) }
        }
        a.answerRecall = { [weak self] _ in
            self?.model.activePresentation = nil
            self?.model.activeNudge = nil
        }
        a.answerExperienceSample = { [weak self] _, _ in self?.model.experiencePromptVisible = false }
        a.setPlayhead = { [weak self] s in self?.model.playhead = s }
        model.actions = a
    }

    // MARK: Tick

    /// Advances one 10 Hz frame.
    public func step() {
        guard let stream, !stream.frames.isEmpty else { return }
        if index >= stream.frames.count { index = 0; lap += stream.duration }
        var f = stream.frames[index]
        index += 1
        let t = model.elapsed + 0.1
        f.t = t
        let m = model
        m.elapsed = t
        m.latestFrame = f
        m.faceDetected = f.faceDetected > 0.5
        var recent = m.recentFrames
        recent.append(f)
        if recent.count > 100 { recent.removeFirst(recent.count - 100) }
        m.recentFrames = recent
        if let tl = timeline, tl.duration > 0 {
            m.playhead = t.truncatingRemainder(dividingBy: tl.duration)
        }

        // 1 Hz: HR, state, risk, nudges.
        let tenth = Int((t * 10).rounded())
        if tenth % 10 == 0 { secondTick(frame: f, recent: recent) }

        if model.activePresentation != nil, t > presentationUntil {
            m.activePresentation = nil
            m.activeNudge = nil
        }
    }

    private func secondTick(frame f: FeatureFrame, recent: [FeatureFrame]) {
        let m = model, t = f.t
        if f.heartRate > 0 {
            let hr = ReplayFeatureSource.syntheticHR(for: f)
            m.heartRate = hr
            var h = m.heartRateHistory
            h.append((t, hr.bpm))
            if h.count > 600 { h.removeFirst(h.count - 600) }
            m.heartRateHistory = h
        }

        let probs = heuristics.classify(window: recent)
        let est = StateEstimate(t: t, probabilities: probs, source: .heuristic)
        m.estimate = est
        m.stateHistory.append(est)
        let state = est.state

        // Refocus bookkeeping.
        if state != lastState {
            m.events.append(.stateChanged(t: t, from: lastState, to: state))
            if lastState == .focused { lapseStart = t; m.events.append(.lapseDetected(t: t, state: state)) }
            if state == .focused, let s = lapseStart {
                let secs = t - s
                m.lastRefocusSeconds = secs
                m.lastRefocusAt = t
                m.events.append(.refocused(t: t, afterSeconds: secs))
                lapseStart = nil
                if let p = pending { finish(p.decision, refocusSeconds: t - p.decision.t, dismissed: false) }
            }
            lastState = state
        }
        m.lapseElapsed = lapseStart.map { t - $0 }
        nonFocusedSince = state == .focused ? nil : (nonFocusedSince ?? t)

        // Risk + forecast line.
        let content = timeline.map { $0.features(at: m.playhead) }
        let pNot = 1 - est.probability(of: .focused)
        let hist = m.stateHistory
        let trend: Float = hist.count > 3 ? pNot - (1 - hist[hist.count - 4].probability(of: .focused)) : 0
        let p = UIPreviewHeuristics.risk(pNotFocused: pNot, trend: trend, content: content)
        let threshold: Float = m.demoDirector ? 0.45 : 0.6
        let risk = LapseRisk(t: t, probability: p, threshold: threshold, usedContent: content != nil, source: "heuristic")
        m.lapseRisk = risk
        m.riskHistory.append(risk)
        if let tl = timeline {
            m.forecastLine = (1...15).map { k in
                let c = tl.features(at: (m.playhead + Double(k)).truncatingRemainder(dividingBy: max(1, tl.duration)))
                let ahead = UIPreviewHeuristics.risk(pNotFocused: pNot * Float(exp(-Double(k) / 8)), trend: 0, content: c)
                let w = Float(k) / 15
                return (t + Double(k), p * (1 - w) + ahead * w)
            }
        } else {
            m.forecastLine = (1...15).map { (t + Double($0), p) }
        }

        // Pending outcome window (20 s).
        if let pd = pending, t > pd.until { finish(pd.decision, refocusSeconds: nil, dismissed: false) }

        // Decide nudges (stand-in for the MRT engine).
        let debounce = m.demoDirector ? 1.5 : 3
        let cooldown = m.demoDirector ? 12.0 : 20.0
        if pending == nil, t - lastNudgeAt > cooldown {
            if state != .focused, let since = nonFocusedSince, t - since >= debounce {
                decide(policy: .reactive, state: state, t: t, risk: p, content: content, pickup: f.phonePickup > 0.5)
            } else if state == .focused, risk.isAlert {
                decide(policy: .preemptive, state: state, t: t, risk: p, content: content, pickup: false)
            }
        }

        m.ticks1Hz(t: t, est: est, risk: p, hr: m.heartRate, content: content, into: &ticks)
    }

    private func decide(policy: NudgePolicyKind, state: AttentionState, t: Double, risk: Float,
                        content: ContentFeatures?, pickup: Bool) {
        var arms = policy == .preemptive ? NudgeArm.preemptiveArms : NudgeArm.reactiveArms(for: state)
        if !model.studyMode { arms.removeAll { $0 == .recapCard || $0 == .recallQuestion } }
        if !pickup { arms.removeAll { $0 == .putMeDown } }
        guard !arms.isEmpty else { return }
        let key: AttentionState? = policy == .preemptive ? nil : state
        let (arm, probs) = bandit.choose(state: key, policy: policy, arms: arms, rng: &rng)
        let ctx = NudgeContext(t: t, state: state, lapseRisk: risk, predictedEngagement: content?.engagement ?? 0.5,
                               sessionMinute: Float(t / 60), hourOfDay: Calendar.current.component(.hour, from: Date()),
                               minutesSinceLastNudge: Float((t - lastNudgeAt) / 60), studyMode: model.studyMode,
                               phonePickedUp: pickup, upcomingValleyInSeconds: content.map(\.secondsToValley))
        let d = NudgeDecision(t: t, policy: policy, context: ctx, arm: arm,
                              randomizationProbability: probs[arm.rawValue] ?? 0, armProbabilities: probs)
        model.lastDecision = d
        model.events.append(.nudge(d))
        lastNudgeAt = t
        pending = (d, t + 20)
        nudgeCount += 1
        if policy == .preemptive { preemptiveCount += 1 }
        model.activeNudge = arm == .noNudge ? nil : arm
        model.activePresentation = Self.presentation(for: arm, studyMode: model.studyMode)
        presentationUntil = t + (arm == .stretchBreak || arm == .pacedBreathing ? 12 : 7)
    }

    private func finish(_ d: NudgeDecision, refocusSeconds: Double?, dismissed: Bool) {
        let refocused = refocusSeconds != nil
        let avoided: Bool? = d.policy == .preemptive ? (lapseStart == nil) : nil
        var reward = d.policy == .preemptive ? ((avoided ?? false) ? 1.0 : 0.0) : (refocused ? 1.0 : 0.0)
        if dismissed { reward = max(0, reward - 0.5) }
        let o = NudgeOutcome(decisionID: d.id, t: model.elapsed, reward: reward, refocused: refocused,
                             refocusSeconds: refocusSeconds, dismissed: dismissed, lapseAvoided: avoided)
        model.events.append(.outcome(o))
        bandit.record(state: d.policy == .preemptive ? nil : d.context.state, policy: d.policy, arm: d.arm, reward: reward)
        model.posteriors = bandit.posteriors()
        model.effects = bandit.effects()
        pending = nil
    }

    static func presentation(for arm: NudgeArm, studyMode: Bool) -> NudgePresentation? {
        switch arm {
        case .ambientGlow: .glow(intensity: 0.8, preemptive: false)
        case .haptic, .chime: .glow(intensity: 0.35, preemptive: false)
        case .putMeDown: .card(NudgeCard(kind: .putMeDown, title: "Put me down",
                                          body: "Your study session is still running. Set the phone back down and pick up where you left off."))
        case .recapCard: .card(NudgeCard(kind: .recap, title: "Quick recap",
                                          body: "The last few minutes covered how light energy is captured and stored. Ready to keep going?"))
        case .stretchBreak: .stretch(seconds: 60)
        case .lightPulse: .lightPulse
        case .pacedBreathing: .breathing(targetBreathsPerMinute: 6)
        case .preemptiveGlow: .glow(intensity: 0.45, preemptive: true)
        case .recallQuestion: .card(NudgeCard(kind: .recall, title: "Quick check",
                                               body: "Which network is most linked to mind-wandering?",
                                               choices: ["Visual", "Default Mode", "Somatomotor"], correctIndex: 1))
        case .noNudge: nil
        }
    }
}

private extension LiveSessionModel {
    func ticks1Hz(t: Double, est: StateEstimate, risk: Float, hr: HeartRateSample?, content: ContentFeatures?,
                  into ticks: inout [SessionTick]) {
        ticks.append(SessionTick(t: t, state: est.state, probabilities: est.probabilities, lapseRisk: risk,
                                 heartRate: hr?.bpm, heartRateQuality: hr?.quality,
                                 playhead: timeline == nil ? nil : playhead,
                                 predictedEngagement: content?.engagement, predictedDriftRisk: content?.driftRisk))
    }
}

/// Beta-Bernoulli Thompson-sampling stand-in. PREVIEW-ONLY (the Trial agent owns the real engine).
public struct UIPreviewBandit {
    struct Key: Hashable { var state: AttentionState?; var policy: NudgePolicyKind; var arm: NudgeArm }
    private var ab: [Key: (a: Double, b: Double, n: Int)] = [:]

    public init() {}

    private func prior(_ arm: NudgeArm) -> (Double, Double) {
        arm == .noNudge ? (1.5, 2.5) : (2, 2)
    }

    mutating func choose(state: AttentionState?, policy: NudgePolicyKind, arms: [NudgeArm],
                         rng: inout UIPreviewRNG) -> (NudgeArm, [String: Double]) {
        // Thompson sampling with a 10% control floor; probabilities estimated by Monte Carlo.
        var wins: [NudgeArm: Int] = [:]
        let draws = 200
        for _ in 0..<draws {
            let best = arms.max { a, b in sample(state, policy, a, &rng) < sample(state, policy, b, &rng) }!
            wins[best, default: 0] += 1
        }
        var probs: [String: Double] = [:]
        let control = arms.contains(.noNudge) ? 0.1 : 0
        for a in arms {
            let base = Double(wins[a] ?? 0) / Double(draws) * (1 - control)
            probs[a.rawValue] = base + (a == .noNudge ? control : 0)
        }
        let u = rng.uniform()
        var acc = 0.0
        for a in arms {
            acc += probs[a.rawValue] ?? 0
            if u <= acc { return (a, probs) }
        }
        return (arms.last!, probs)
    }

    private func sample(_ s: AttentionState?, _ p: NudgePolicyKind, _ a: NudgeArm, _ rng: inout UIPreviewRNG) -> Double {
        let k = Key(state: s, policy: p, arm: a)
        let (pa, pb) = prior(a)
        let v = ab[k] ?? (pa, pb, 0)
        // Normal approximation to the Beta draw (fine for a preview).
        let mean = v.a / (v.a + v.b)
        let sd = (v.a * v.b / ((v.a + v.b) * (v.a + v.b) * (v.a + v.b + 1))).squareRoot()
        return mean + sd * rng.normal()
    }

    mutating func record(state: AttentionState?, policy: NudgePolicyKind, arm: NudgeArm, reward: Double) {
        let k = Key(state: state, policy: policy, arm: arm)
        let (pa, pb) = prior(arm)
        var v = ab[k] ?? (pa, pb, 0)
        v.a += reward; v.b += 1 - reward; v.n += 1
        ab[k] = v
    }

    func posteriors() -> [ArmPosterior] {
        var out: [ArmPosterior] = []
        for s in AttentionState.allCases where s != .focused {
            for a in NudgeArm.reactiveArms(for: s) {
                let (pa, pb) = prior(a)
                let v = ab[Key(state: s, policy: .reactive, arm: a)] ?? (pa, pb, 0)
                out.append(ArmPosterior(state: s, policy: .reactive, arm: a, alpha: v.a, beta: v.b, pulls: v.n))
            }
        }
        for a in NudgeArm.preemptiveArms {
            let (pa, pb) = prior(a)
            let v = ab[Key(state: nil, policy: .preemptive, arm: a)] ?? (pa, pb, 0)
            out.append(ArmPosterior(state: nil, policy: .preemptive, arm: a, alpha: v.a, beta: v.b, pulls: v.n))
        }
        return out
    }

    /// Difference of posterior means vs the control arm, 90% normal-approximation interval. PREVIEW-ONLY.
    func effects() -> [ArmEffect] {
        let post = posteriors()
        var out: [ArmEffect] = []
        var seen = Set<String>()
        for p in post where p.arm != .noNudge && p.pulls > 0 {
            guard let ctrl = post.first(where: { $0.arm == .noNudge && $0.policy == p.policy && $0.state == p.state }) else { continue }
            func v(_ x: ArmPosterior) -> Double { x.alpha * x.beta / ((x.alpha + x.beta) * (x.alpha + x.beta) * (x.alpha + x.beta + 1)) }
            let e = p.mean - ctrl.mean, half = 1.645 * (v(p) + v(ctrl)).squareRoot()
            let id = "\(p.policy.rawValue).\(p.arm.rawValue)"
            guard !seen.contains(id) else { continue }
            seen.insert(id)
            out.append(ArmEffect(arm: p.arm, policy: p.policy, effect: e, lower: e - half, upper: e + half, level: 0.9, n: p.pulls))
        }
        return out.sorted { $0.effect > $1.effect }
    }
}

/// Deterministic SplitMix64 RNG for previews.
public struct UIPreviewRNG {
    private var s: UInt64
    public init(seed: UInt64) { s = seed }
    mutating func next() -> UInt64 {
        s &+= 0x9E3779B97F4A7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func uniform() -> Double { Double(next() >> 11) / Double(1 << 53) }
    mutating func normal() -> Double {
        let u1 = max(1e-12, uniform()), u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }
}
