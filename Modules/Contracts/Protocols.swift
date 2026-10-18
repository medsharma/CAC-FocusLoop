import Foundation

// FROZEN CONTRACT (Phase 0). Changes require orchestrator approval and a broadcast to all agents.
//
// Threading model:
//   * Sources deliver samples on their own queue via `AsyncStream`.
//   * The session engine (App, Phase 2) consumes streams on a dedicated perception actor and
//     publishes to `@MainActor @Observable` view models. No UI work off the main thread.
//   * Components below are plain classes/structs and are NOT required to be thread-safe unless
//     stated; the session engine calls each from a single actor.

// MARK: - Sensing

/// ARKit face tracking (device) or replay/mock face samples. Camera rate (~30-60 Hz).
public protocol FaceSignalSource: AnyObject {
    /// True if this source can run here (ARKit face tracking supported, or always for mocks).
    var isAvailable: Bool { get }
    var samples: AsyncStream<FaceSample> { get }
    func start() throws
    func stop()
}

/// Core Motion pickup / tilt detection (or mock).
public protocol MotionSource: AnyObject {
    var samples: AsyncStream<MotionSample> { get }
    func start()
    func stop()
}

/// rPPG: consumes skin-ROI mean RGB at camera rate, emits BPM + quality + waveform.
public protocol HeartRateEstimator: AnyObject {
    /// Feed one camera-rate skin-colour sample. `headMotion` in rad/s gates the estimate.
    func ingest(rgb: SIMD3<Float>, t: Double, headMotion: Float)
    /// Latest estimate (updated ~1 Hz once 10 s of signal is buffered); nil until then.
    var latest: HeartRateSample? { get }
    /// Personal resting baseline in BPM, persisted across sessions by the caller.
    var restingBaseline: Float? { get set }
    func reset()
}

/// Turns face + motion + HR into 10 Hz `FeatureFrame`s (blinks, PERCLOS, yawns, nods, work zone...).
public protocol FeatureExtractor: AnyObject {
    var workZone: WorkZone { get set }
    func ingest(face: FaceSample)
    func ingest(motion: MotionSample)
    func ingest(heartRate: HeartRateSample)
    /// Returns a frame when a new 10 Hz tick is due at `t`, else nil.
    func frame(at t: Double) -> FeatureFrame?
    func reset()
}

/// A ready-made 10 Hz `FeatureFrame` stream (Replay / Demo). Bypasses FaceSignalSource + extractor.
public protocol FeatureFrameSource: AnyObject {
    var frames: AsyncStream<FeatureFrame> { get }
    /// Optional pulse waveform/HR alongside replayed frames, for the science view.
    var heartRate: AsyncStream<HeartRateSample> { get }
    var isSynthetic: Bool { get }
    func start()
    func stop()
}

// MARK: - Understanding

/// 4-state classifier over a 10 s window (100 frames). Called at 1 Hz.
public protocol AttentionClassifier: AnyObject {
    /// Window of the most recent frames, oldest first. May be shorter than 100 at session start.
    func classify(window: [FeatureFrame]) -> StateEstimate
}

/// On-device logistic-regression head over the classifier embedding, trained from the user's labels.
public protocol PersonalHead: AnyObject {
    var labelCount: Int { get }
    /// Probabilities in `AttentionState.allCases` order, or nil if not trained enough.
    func predict(embedding: [Float]) -> [Float]?
    func update(embedding: [Float], label: AttentionState)
    func serialized() throws -> Data
    func restore(from data: Data) throws
    func reset()
}

// MARK: - Predicting

/// P(lapse within 15 s) from the previous 20 s of frames plus content features.
public protocol LapseForecaster: AnyObject {
    /// `frames`: last 20 s (200 frames), oldest first. `content`: one entry per second (20), oldest first,
    /// or empty when no lecture is playing.
    func forecast(frames: [FeatureFrame], content: [ContentFeatures], t: Double) -> LapseRisk
    /// Personalised threshold, adapted from the user's history.
    var threshold: Float { get }
    /// Inform the forecaster whether a lapse actually happened `horizon` s after a prediction.
    func observe(prediction: LapseRisk, lapseOccurred: Bool)
}

/// Thin protocol over `BrainTimeline` (see `BrainContent` module for the wrapper).
/// Predicted brain response to this content (average brain).
public protocol ContentTimeline {
    var lectureID: String { get }
    var isMock: Bool { get }
    var duration: Double { get }
    /// Display activations 0...1 for `BrainCortexView`, `BrainNetwork` order.
    func activations(at t: Double) -> [Float]
    /// Raw z-scored network values, `BrainNetwork` order.
    func networkValues(at t: Double) -> [Float]
    func engagement(at t: Double) -> Float
    func driftRisk(at t: Double) -> Float
    func valleys() -> [(start: Double, end: Double)]
    func upcomingValley(after t: Double, within horizon: Double) -> (start: Double, end: Double)?
    func features(at t: Double) -> ContentFeatures
}

// MARK: - Intervening

/// Micro-randomised trial policy (Thompson sampling with a control arm).
public protocol InterventionPolicy: AnyObject {
    /// Choose an arm. Returns nil if pacing (cooldown/debounce) forbids a nudge now.
    func decide(policy: NudgePolicyKind, context: NudgeContext) -> NudgeDecision?
    func record(outcome: NudgeOutcome, for decision: NudgeDecision)
    func posteriors() -> [ArmPosterior]
    /// IPW effect estimates of each arm vs noNudge with intervals.
    func effectEstimates() -> [ArmEffect]
    func serialized() throws -> Data
    func restore(from data: Data) throws
}

@MainActor
public protocol NudgePresenting: AnyObject {
    func present(_ presentation: NudgePresentation, arm: NudgeArm)
    func dismissNudge()
}

/// One concrete nudge (glow, haptic, chime, cards, stretch, breathing...).
@MainActor
public protocol Nudge: AnyObject {
    var arm: NudgeArm { get }
    func start(presenter: NudgePresenting)
    func cancel()
}

public protocol TrialLogger: AnyObject {
    func log(decision: NudgeDecision)
    func log(outcome: NudgeOutcome)
    var decisions: [NudgeDecision] { get }
    var outcomes: [NudgeOutcome] { get }
}

// MARK: - Learning / insights

public protocol InsightGenerator: AnyObject {
    /// 2-4 sentence insights from aggregated numbers only. Never medical advice.
    func insights(for summary: AggregateSummary) async -> [Insight]
    /// Which backend will be used on this device.
    var usesFoundationModels: Bool { get }
}

// MARK: - Persistence

@MainActor
public protocol SessionStore: AnyObject {
    func save(_ session: SessionRecord) throws
    func sessions() throws -> [SessionRecord]
    func session(id: UUID) throws -> SessionRecord?
    func saveModelWeights(_ data: Data, key: String) throws
    func modelWeights(key: String) throws -> Data?
    func deleteAll() throws
}

public protocol LabelSink: AnyObject {
    func add(_ window: LabeledWindow)
    var windows: [LabeledWindow] { get }
    /// Writes all labelled windows as CSV (one row per frame) and returns the file URL (temp dir).
    func exportCSV() throws -> URL
}
