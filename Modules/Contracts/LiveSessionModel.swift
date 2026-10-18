import Foundation
import Observation

// FROZEN CONTRACT (Phase 0).
//
// The single observable model the UI renders during a session. The session engine (App layer)
// is the only writer, always on the main actor. UI screens only read it (plus call the closures
// in `actions`). Previews/tests fill it from Replay.

@MainActor
@Observable
public final class LiveSessionModel {
    public enum Phase: String, Sendable { case idle, running, paused, finished }

    public var phase: Phase = .idle
    public var sessionID = UUID()
    public var startedAt = Date()
    /// Seconds since session start.
    public var elapsed: Double = 0
    /// True when driven by the synthetic Replay stream (show a SYNTHETIC badge).
    public var isReplay = false
    /// DEMONSTRATION ONLY flag mirrored for a small on-screen indicator.
    public var demoDirector = false
    /// True when driven by a RECORDED video of a real person (not live, not synthetic; show a RECORDED badge).
    public var isRecorded = false

    // Sensing
    public var latestFrame: FeatureFrame?
    /// Last ~10 s of frames for gauges/sparklines.
    public var recentFrames: [FeatureFrame] = []
    public var heartRate: HeartRateSample?
    /// Ring of (t, bpm) for the HR trend line.
    public var heartRateHistory: [(t: Double, bpm: Float)] = []
    public var faceDetected = false

    // Understanding / predicting
    public var estimate: StateEstimate?
    public var stateHistory: [StateEstimate] = []
    public var lapseRisk: LapseRisk?
    public var riskHistory: [LapseRisk] = []
    /// Forecast line running ahead of "now": (t in the future, predicted P(lapse)).
    public var forecastLine: [(t: Double, p: Float)] = []

    // Refocus
    /// Seconds since the current lapse began (nil while focused).
    public var lapseElapsed: Double?
    /// Most recent "Back in X.X s" value, shown briefly after refocusing.
    public var lastRefocusSeconds: Double?
    public var lastRefocusAt: Double?

    // Content / Study Mode
    public var timeline: ContentTimeline?
    /// Lecture playhead in seconds (Study Mode), or session-elapsed modulo duration when the
    /// science view previews content without a lecture playing.
    public var playhead: Double = 0
    public var studyMode = false

    // Interventions / trial
    public var activeNudge: NudgeArm?
    public var activePresentation: NudgePresentation?
    public var lastDecision: NudgeDecision?
    public var posteriors: [ArmPosterior] = []
    public var effects: [ArmEffect] = []
    public var events: [SessionEvent] = []

    /// Hooks the engine installs so UI controls can drive it.
    public struct Actions {
        public var start: () -> Void = {}
        public var pause: () -> Void = {}
        public var resume: () -> Void = {}
        public var stop: () -> Void = {}
        /// User dismissed the current nudge (counts as a dismissal for reward).
        public var dismissNudge: () -> Void = {}
        /// Recall-card answer.
        public var answerRecall: (_ choice: Int) -> Void = { _ in }
        /// Experience-sampling answer.
        public var answerExperienceSample: (_ focused: Bool, _ reason: NotFocusedReason?) -> Void = { _, _ in }
        /// Study Mode playhead updates (from the AVPlayer time observer).
        public var setPlayhead: (_ seconds: Double) -> Void = { _ in }
        public init() {}
    }
    public var actions = Actions()

    /// Experience-sampling prompt is due.
    public var experiencePromptVisible = false

    public init() {}

    public var state: AttentionState { estimate?.state ?? .focused }
}
