import Foundation
import simd

// FROZEN CONTRACT (Phase 0). Changes require orchestrator approval and a broadcast to all agents.
// Network-indexed data always uses `BrainNetwork.allCases` order.

// MARK: - Attention state

public enum AttentionState: String, Codable, CaseIterable, Sendable, Identifiable {
    case focused, distracted, drowsy, stressed
    public var id: String { rawValue }
    /// Index into probability vectors (`StateEstimate.probabilities`), == `allCases` order.
    public var index: Int { Self.allCases.firstIndex(of: self)! }
    public var displayName: String { rawValue.capitalized }
}

public enum EstimateSource: String, Codable, Sendable {
    case model, heuristic, blended
}

/// Output of an `AttentionClassifier` at 1 Hz.
public struct StateEstimate: Codable, Equatable, Sendable {
    public var t: Double
    /// Probabilities in `AttentionState.allCases` order; sums to 1.
    public var probabilities: [Float]
    public var source: EstimateSource
    /// Penultimate-layer embedding (model) or engineered summary (heuristic) for the personal head.
    public var embedding: [Float]

    public init(t: Double, probabilities: [Float], source: EstimateSource, embedding: [Float] = []) {
        self.t = t
        self.probabilities = probabilities
        self.source = source
        self.embedding = embedding
    }

    public var state: AttentionState {
        let i = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
        return AttentionState.allCases[i]
    }
    public var confidence: Float { probabilities.max() ?? 0 }
    public func probability(of s: AttentionState) -> Float {
        s.index < probabilities.count ? probabilities[s.index] : 0
    }
}

// MARK: - Heart rate

/// Output of the rPPG `HeartRateEstimator`.
public struct HeartRateSample: Codable, Equatable, Sendable {
    public var t: Double
    public var bpm: Float
    /// 0...1 signal-quality index derived from spectral SNR around the peak.
    public var quality: Float
    public var snrDB: Float
    /// True when head motion made this reading unreliable (reading is held/flagged).
    public var motionFlagged: Bool
    /// Personal resting baseline (nil until enough calm, good-quality data).
    public var baselineBPM: Float?
    /// Recent band-passed pulse waveform for display, ~30 Hz, normalised to -1...1.
    public var waveform: [Float]

    public init(t: Double, bpm: Float, quality: Float, snrDB: Float, motionFlagged: Bool,
                baselineBPM: Float?, waveform: [Float]) {
        self.t = t; self.bpm = bpm; self.quality = quality; self.snrDB = snrDB
        self.motionFlagged = motionFlagged; self.baselineBPM = baselineBPM; self.waveform = waveform
    }
}

// MARK: - Raw sensor samples (Perception -> RPPG / FeatureExtractor)

/// One ARKit (or replay/mock) face sample at camera rate (~30-60 Hz). Numbers only.
public struct FaceSample: Sendable {
    public var t: Double
    public var faceDetected: Bool
    /// ARKit blendshape values keyed by `ARFaceAnchor.BlendShapeLocation.rawValue`.
    public var blendShapes: [String: Float]
    /// `ARFaceAnchor.lookAtPoint` in face space.
    public var lookAtPoint: SIMD3<Float>
    /// Head orientation relative to the camera, radians.
    public var headYaw: Float
    public var headPitch: Float
    public var headRoll: Float
    /// Mean linear RGB (0...1) of the forehead + cheek skin ROI, for rPPG. nil if ROI invalid.
    public var skinRGB: SIMD3<Float>?

    public init(t: Double, faceDetected: Bool, blendShapes: [String: Float] = [:],
                lookAtPoint: SIMD3<Float> = .zero, headYaw: Float = 0, headPitch: Float = 0,
                headRoll: Float = 0, skinRGB: SIMD3<Float>? = nil) {
        self.t = t; self.faceDetected = faceDetected; self.blendShapes = blendShapes
        self.lookAtPoint = lookAtPoint; self.headYaw = headYaw; self.headPitch = headPitch
        self.headRoll = headRoll; self.skinRGB = skinRGB
    }
}

public struct MotionSample: Sendable {
    public var t: Double
    /// True while the phone is picked up / tilted out of its resting pose.
    public var pickedUp: Bool
    /// Tilt from the resting attitude, radians.
    public var tilt: Float
    /// User-acceleration magnitude, g.
    public var accelerationMagnitude: Float

    public init(t: Double, pickedUp: Bool, tilt: Float, accelerationMagnitude: Float) {
        self.t = t; self.pickedUp = pickedUp; self.tilt = tilt; self.accelerationMagnitude = accelerationMagnitude
    }
}

/// Calibrated gaze/head-pose cone the student looks at while working.
public struct WorkZone: Codable, Equatable, Sendable {
    /// Centre direction (combined gaze+head) in radians.
    public var centerYaw: Float
    public var centerPitch: Float
    /// Half-widths of the cone, radians (from the 4 corners, plus margin).
    public var halfWidthYaw: Float
    public var halfWidthPitch: Float
    public var calibratedAt: Date
    /// True when calibrated on the phone screen itself (Study Mode) rather than a laptop.
    public var isPhoneScreen: Bool

    public init(centerYaw: Float, centerPitch: Float, halfWidthYaw: Float, halfWidthPitch: Float,
                calibratedAt: Date = Date(), isPhoneScreen: Bool = false) {
        self.centerYaw = centerYaw; self.centerPitch = centerPitch
        self.halfWidthYaw = halfWidthYaw; self.halfWidthPitch = halfWidthPitch
        self.calibratedAt = calibratedAt; self.isPhoneScreen = isPhoneScreen
    }

    /// Reasonable uncalibrated default: phone beside a laptop, ±25° x ±18° cone.
    public static let `default` = WorkZone(centerYaw: 0, centerPitch: 0, halfWidthYaw: 0.44, halfWidthPitch: 0.31,
                                           calibratedAt: .distantPast)
}

// MARK: - Content (brain-encoding layer)

/// Content features at one playhead second, from the precomputed brain timeline.
/// Predicted brain response to this content (average brain), never the student's own brain.
public struct ContentFeatures: Codable, Equatable, Sendable {
    /// 7 z-scored network values in `BrainNetwork.allCases` order.
    public var networks: [Float]
    public var engagement: Float
    public var driftRisk: Float
    /// Seconds until the next predicted valley starts (clamped to 0...60; 60 = none soon). 0 if inside one.
    public var secondsToValley: Float
    public var inValley: Bool

    public init(networks: [Float], engagement: Float, driftRisk: Float, secondsToValley: Float, inValley: Bool) {
        self.networks = networks; self.engagement = engagement; self.driftRisk = driftRisk
        self.secondsToValley = secondsToValley; self.inValley = inValley
    }

    /// Neutral features used when no lecture is playing (free study).
    public static let none = ContentFeatures(networks: Array(repeating: 0, count: 7), engagement: 0.5,
                                             driftRisk: 0, secondsToValley: 60, inValley: false)

    /// Forecaster input order: 7 networks, engagement, driftRisk, secondsToValley/60, inValley.
    public static let featureNames = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default",
                                      "engagement", "driftRisk", "secondsToValleyNorm", "inValley"]
    public var vector: [Float] {
        networks + [engagement, driftRisk, min(60, max(0, secondsToValley)) / 60, inValley ? 1 : 0]
    }
}

// MARK: - Forecasting

public struct LapseRisk: Codable, Equatable, Sendable {
    public var t: Double
    /// P(lapse within the next `horizon` seconds).
    public var probability: Float
    /// Personalised alert threshold currently in force.
    public var threshold: Float
    public var horizon: Double
    /// Whether brain-content features were used for this prediction.
    public var usedContent: Bool
    /// "model" or "heuristic".
    public var source: String

    public init(t: Double, probability: Float, threshold: Float, horizon: Double = FeatureFrame.forecastHorizonSeconds,
                usedContent: Bool, source: String) {
        self.t = t; self.probability = probability; self.threshold = threshold
        self.horizon = horizon; self.usedContent = usedContent; self.source = source
    }

    public var isAlert: Bool { probability >= threshold }
}

// MARK: - Interventions / trial

public enum NudgePolicyKind: String, Codable, CaseIterable, Sendable {
    case reactive, preemptive
}

public enum NudgeArm: String, Codable, CaseIterable, Sendable, Identifiable {
    // Distracted
    case ambientGlow, haptic, chime, putMeDown, recapCard
    // Drowsy
    case stretchBreak
    /// EXPERIMENTAL arm under test. Never claim it works.
    case lightPulse
    // Stressed
    case pacedBreathing
    // Pre-emptive
    case preemptiveGlow, recallQuestion
    // Control (no nudge) for causal estimates
    case noNudge

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .ambientGlow: "Ambient glow"
        case .haptic: "Haptic tap"
        case .chime: "Soft chime"
        case .putMeDown: "\u{201C}Put me down\u{201D} card"
        case .recapCard: "Recap card"
        case .stretchBreak: "Stretch break"
        case .lightPulse: "Cool-light pulse (experimental)"
        case .pacedBreathing: "Paced breathing"
        case .preemptiveGlow: "Pre-emptive glow"
        case .recallQuestion: "Active-recall question"
        case .noNudge: "No nudge (control)"
        }
    }

    public var isExperimental: Bool { self == .lightPulse }

    /// Arms the engine may randomise among for a reactive decision in `state`
    /// (noNudge is always included as the control arm). Further eligibility (e.g. pickup,
    /// Study Mode) is applied by the trial engine via `NudgeContext`.
    public static func reactiveArms(for state: AttentionState) -> [NudgeArm] {
        switch state {
        case .focused: []
        case .distracted: [.ambientGlow, .haptic, .chime, .putMeDown, .recapCard, .noNudge]
        case .drowsy: [.stretchBreak, .lightPulse, .noNudge]
        case .stressed: [.pacedBreathing, .noNudge]
        }
    }

    public static let preemptiveArms: [NudgeArm] = [.preemptiveGlow, .recallQuestion, .noNudge]
}

/// Context at decision time (also the feature vector for linear Thompson sampling).
public struct NudgeContext: Codable, Equatable, Sendable {
    public var t: Double
    public var state: AttentionState
    public var lapseRisk: Float
    public var predictedEngagement: Float
    public var sessionMinute: Float
    public var hourOfDay: Int
    public var minutesSinceLastNudge: Float
    public var studyMode: Bool
    public var phonePickedUp: Bool
    public var upcomingValleyInSeconds: Float?

    public init(t: Double, state: AttentionState, lapseRisk: Float, predictedEngagement: Float,
                sessionMinute: Float, hourOfDay: Int, minutesSinceLastNudge: Float, studyMode: Bool,
                phonePickedUp: Bool, upcomingValleyInSeconds: Float?) {
        self.t = t; self.state = state; self.lapseRisk = lapseRisk; self.predictedEngagement = predictedEngagement
        self.sessionMinute = sessionMinute; self.hourOfDay = hourOfDay
        self.minutesSinceLastNudge = minutesSinceLastNudge; self.studyMode = studyMode
        self.phonePickedUp = phonePickedUp; self.upcomingValleyInSeconds = upcomingValleyInSeconds
    }
}

public struct NudgeDecision: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var t: Double
    public var date: Date
    public var policy: NudgePolicyKind
    public var context: NudgeContext
    public var arm: NudgeArm
    /// Probability with which `arm` was chosen (logged for IPW / MRT analysis).
    public var randomizationProbability: Double
    /// Selection probability of every eligible arm at decision time.
    public var armProbabilities: [String: Double]

    public init(id: UUID = UUID(), t: Double, date: Date = Date(), policy: NudgePolicyKind, context: NudgeContext,
                arm: NudgeArm, randomizationProbability: Double, armProbabilities: [String: Double]) {
        self.id = id; self.t = t; self.date = date; self.policy = policy; self.context = context
        self.arm = arm; self.randomizationProbability = randomizationProbability; self.armProbabilities = armProbabilities
    }
}

public struct NudgeOutcome: Codable, Equatable, Sendable {
    public var decisionID: UUID
    public var t: Double
    /// Reward in 0...1 (1 = refocused within 20 s / lapse avoided), minus dismissal penalty, clamped at 0.
    public var reward: Double
    public var refocused: Bool
    /// Seconds from decision to sustained `focused` (nil if not within the window).
    public var refocusSeconds: Double?
    public var dismissed: Bool
    /// Pre-emptive only: true when no lapse occurred within the forecast horizon.
    public var lapseAvoided: Bool?

    public init(decisionID: UUID, t: Double, reward: Double, refocused: Bool, refocusSeconds: Double?,
                dismissed: Bool, lapseAvoided: Bool?) {
        self.decisionID = decisionID; self.t = t; self.reward = reward; self.refocused = refocused
        self.refocusSeconds = refocusSeconds; self.dismissed = dismissed; self.lapseAvoided = lapseAvoided
    }
}

/// Posterior summary for display ("live bandit posteriors").
public struct ArmPosterior: Codable, Equatable, Sendable, Identifiable {
    public var state: AttentionState?   // nil for pre-emptive arms
    public var policy: NudgePolicyKind
    public var arm: NudgeArm
    public var alpha: Double
    public var beta: Double
    public var pulls: Int
    public var id: String { "\(policy.rawValue).\(state?.rawValue ?? "pre").\(arm.rawValue)" }
    public var mean: Double { alpha / (alpha + beta) }

    public init(state: AttentionState?, policy: NudgePolicyKind, arm: NudgeArm, alpha: Double, beta: Double, pulls: Int) {
        self.state = state; self.policy = policy; self.arm = arm; self.alpha = alpha; self.beta = beta; self.pulls = pulls
    }
}

/// Per-user causal effect estimate of an arm vs the no-nudge control (IPW), with an interval.
public struct ArmEffect: Codable, Equatable, Sendable, Identifiable {
    public var arm: NudgeArm
    public var policy: NudgePolicyKind
    /// Estimated difference in P(reward) vs noNudge.
    public var effect: Double
    public var lower: Double
    public var upper: Double
    /// Interval level, e.g. 0.9.
    public var level: Double
    public var n: Int
    public var id: String { "\(policy.rawValue).\(arm.rawValue)" }

    public init(arm: NudgeArm, policy: NudgePolicyKind, effect: Double, lower: Double, upper: Double, level: Double, n: Int) {
        self.arm = arm; self.policy = policy; self.effect = effect; self.lower = lower; self.upper = upper
        self.level = level; self.n = n
    }
}

/// Beta prior per arm, as written by `ml/brain/score_nudges.py` to `nudge_priors.json`.
public struct NudgePrior: Codable, Equatable, Sendable {
    public var arm: NudgeArm
    public var alpha: Double
    public var beta: Double
    public init(arm: NudgeArm, alpha: Double, beta: Double) { self.arm = arm; self.alpha = alpha; self.beta = beta }
}

/// What a nudge asks the UI to show. Haptics/chime are played by the nudge itself.
public enum NudgePresentation: Equatable, Sendable {
    case glow(intensity: Float, preemptive: Bool)
    case card(NudgeCard)
    case breathing(targetBreathsPerMinute: Double)
    case stretch(seconds: Int)
    case lightPulse
}

public struct NudgeCard: Equatable, Sendable, Identifiable {
    public enum Kind: String, Sendable { case putMeDown, recap, recall }
    public var id = UUID()
    public var kind: Kind
    public var title: String
    public var body: String
    /// For recall questions.
    public var choices: [String]
    public var correctIndex: Int?

    public init(kind: Kind, title: String, body: String, choices: [String] = [], correctIndex: Int? = nil) {
        self.kind = kind; self.title = title; self.body = body; self.choices = choices; self.correctIndex = correctIndex
    }
}

// MARK: - Labels & sessions

public enum LabelSource: String, Codable, Sendable {
    case sart, experienceSampling, demo
}

/// Not-focused sub-reason from experience sampling.
public enum NotFocusedReason: String, Codable, CaseIterable, Sendable {
    case tired, distracted, stressed
    public var state: AttentionState {
        switch self { case .tired: .drowsy; case .distracted: .distracted; case .stressed: .stressed }
    }
}

/// A labelled window of features for retraining (exported to CSV).
public struct LabeledWindow: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var sessionID: UUID
    public var start: Double
    public var end: Double
    /// focused / distracted / drowsy / stressed.
    public var label: AttentionState
    /// True for SART-derived or "not focused" answers: a lapse label for the forecaster.
    public var isLapse: Bool
    public var source: LabelSource
    /// 10 Hz frames covering [start, end).
    public var frames: [FeatureFrame]

    public init(id: UUID = UUID(), sessionID: UUID, start: Double, end: Double, label: AttentionState,
                isLapse: Bool, source: LabelSource, frames: [FeatureFrame]) {
        self.id = id; self.sessionID = sessionID; self.start = start; self.end = end; self.label = label
        self.isLapse = isLapse; self.source = source; self.frames = frames
    }
}

public enum SessionEvent: Codable, Equatable, Sendable {
    case stateChanged(t: Double, from: AttentionState, to: AttentionState)
    case lapseDetected(t: Double, state: AttentionState)
    case refocused(t: Double, afterSeconds: Double)
    case nudge(NudgeDecision)
    case outcome(NudgeOutcome)
    case riskAlert(t: Double, probability: Float)
    case valleyEntered(t: Double, playhead: Double)
    case experienceSample(t: Double, focused: Bool, reason: NotFocusedReason?)
    case phonePickup(t: Double)

    public var t: Double {
        switch self {
        case let .stateChanged(t, _, _), let .lapseDetected(t, _), let .refocused(t, _),
             let .riskAlert(t, _), let .valleyEntered(t, _), let .experienceSample(t, _, _), let .phonePickup(t):
            t
        case let .nudge(d): d.t
        case let .outcome(o): o.t
        }
    }
}

/// 1 Hz numeric summary row kept for the report (numbers only).
public struct SessionTick: Codable, Equatable, Sendable {
    public var t: Double
    public var state: AttentionState
    public var probabilities: [Float]
    public var lapseRisk: Float
    public var heartRate: Float?
    public var heartRateQuality: Float?
    /// Lecture playhead (Study Mode) or nil.
    public var playhead: Double?
    public var predictedEngagement: Float?
    public var predictedDriftRisk: Float?

    public init(t: Double, state: AttentionState, probabilities: [Float], lapseRisk: Float, heartRate: Float?,
                heartRateQuality: Float?, playhead: Double?, predictedEngagement: Float?, predictedDriftRisk: Float?) {
        self.t = t; self.state = state; self.probabilities = probabilities; self.lapseRisk = lapseRisk
        self.heartRate = heartRate; self.heartRateQuality = heartRateQuality; self.playhead = playhead
        self.predictedEngagement = predictedEngagement; self.predictedDriftRisk = predictedDriftRisk
    }
}

/// Everything persisted about one session. Numeric only.
public struct SessionRecord: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var startedAt: Date
    public var duration: Double
    public var lectureID: String?
    /// True if the lecture brain timeline was MOCK.
    public var contentIsMock: Bool
    /// True if the session ran on the synthetic Replay stream rather than live sensors.
    public var isReplay: Bool
    public var ticks: [SessionTick]
    public var events: [SessionEvent]
    public var insights: [Insight]
    /// Where the signals came from when not the live camera: "recorded" = a RECORDED video of a real person
    /// (demonstration). nil = live sensors or (with `isReplay`) the synthetic Replay. Optional so older stored
    /// records still decode.
    public var sourceKind: String?

    public init(id: UUID = UUID(), startedAt: Date = Date(), duration: Double = 0, lectureID: String? = nil,
                contentIsMock: Bool = false, isReplay: Bool = false, ticks: [SessionTick] = [],
                events: [SessionEvent] = [], insights: [Insight] = [], sourceKind: String? = nil) {
        self.id = id; self.startedAt = startedAt; self.duration = duration; self.lectureID = lectureID
        self.contentIsMock = contentIsMock; self.isReplay = isReplay; self.ticks = ticks
        self.events = events; self.insights = insights; self.sourceKind = sourceKind
    }

    /// True for a session driven by a RECORDED video (demonstration; never trains the real learners).
    public var isRecorded: Bool { sourceKind == "recorded" }
}

// MARK: - Insights

/// Aggregated numbers only: the sole input to the AI coach. No raw features, no text from the user.
public struct AggregateSummary: Codable, Equatable, Sendable {
    public var sessionCount: Int
    public var totalMinutes: Double
    public var focusedFraction: Double
    /// Fraction of time per state, `AttentionState.allCases` order.
    public var stateFractions: [Double]
    /// Minute-of-session at which focus most often drops (nil if unknown).
    public var typicalDropMinute: Double?
    public var medianRefocusSeconds: Double?
    /// Best hours of day (0-23) by focused fraction, best first.
    public var bestHours: [Int]
    public var restingHeartRate: Double?
    public var heartRateTrendPerSession: Double?
    /// Correlation between predicted drift risk and actual not-focused time (Study Mode), nil if no lecture data.
    public var contentDriftCorrelation: Double?
    public var contentIsMock: Bool
    public var isReplay: Bool
    public var armEffects: [ArmEffect]

    public init(sessionCount: Int, totalMinutes: Double, focusedFraction: Double, stateFractions: [Double],
                typicalDropMinute: Double?, medianRefocusSeconds: Double?, bestHours: [Int],
                restingHeartRate: Double?, heartRateTrendPerSession: Double?, contentDriftCorrelation: Double?,
                contentIsMock: Bool, isReplay: Bool, armEffects: [ArmEffect]) {
        self.sessionCount = sessionCount; self.totalMinutes = totalMinutes; self.focusedFraction = focusedFraction
        self.stateFractions = stateFractions; self.typicalDropMinute = typicalDropMinute
        self.medianRefocusSeconds = medianRefocusSeconds; self.bestHours = bestHours
        self.restingHeartRate = restingHeartRate; self.heartRateTrendPerSession = heartRateTrendPerSession
        self.contentDriftCorrelation = contentDriftCorrelation; self.contentIsMock = contentIsMock
        self.isReplay = isReplay; self.armEffects = armEffects
    }
}

public struct Insight: Codable, Equatable, Sendable, Identifiable {
    public enum Source: String, Codable, Sendable { case foundationModels, template }
    public var id: UUID
    public var text: String
    public var source: Source
    public var createdAt: Date
    public init(id: UUID = UUID(), text: String, source: Source, createdAt: Date = Date()) {
        self.id = id; self.text = text; self.source = source; self.createdAt = createdAt
    }
}
