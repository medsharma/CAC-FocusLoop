import Foundation

// FROZEN CONTRACT (Phase 0).

// MARK: - Lecture manifest

/// One bundled lecture. `Resources/Lectures/lectures.json` is `[LectureInfo]`.
/// Files sit at the bundle root: `<videoFile>` and `<id>.brain.json` (+ optional `<id>.vertices.f16`).
public struct LectureInfo: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var title: String
    public var subject: String
    /// e.g. "MIT OpenCourseWare", "Khan Academy", "Wikimedia Commons".
    public var source: String
    public var sourceURL: String
    /// SPDX-style, e.g. "CC-BY-NC-SA-4.0".
    public var license: String
    /// Full attribution line shown in-app and in the README credits.
    public var attribution: String
    /// Bundle-root file name, e.g. "photosynthesis-01.mp4". nil = no video (timer-driven playback).
    public var videoFile: String?
    public var durationSeconds: Double
    /// Short recap shown by the `recapCard` arm, keyed to time ranges.
    public var recaps: [TimedText]
    /// Active-recall questions for the pre-emptive `recallQuestion` arm.
    public var recallQuestions: [RecallQuestion]

    public init(id: String, title: String, subject: String, source: String, sourceURL: String, license: String,
                attribution: String, videoFile: String?, durationSeconds: Double, recaps: [TimedText],
                recallQuestions: [RecallQuestion]) {
        self.id = id; self.title = title; self.subject = subject; self.source = source; self.sourceURL = sourceURL
        self.license = license; self.attribution = attribution; self.videoFile = videoFile
        self.durationSeconds = durationSeconds; self.recaps = recaps; self.recallQuestions = recallQuestions
    }

    public struct TimedText: Codable, Equatable, Sendable {
        public var start: Double
        public var end: Double
        public var text: String
        public init(start: Double, end: Double, text: String) { self.start = start; self.end = end; self.text = text }
    }

    public struct RecallQuestion: Codable, Equatable, Sendable {
        /// Ask only after the playhead passes this time (the content it covers).
        public var after: Double
        public var question: String
        public var choices: [String]
        public var correctIndex: Int
        public init(after: Double, question: String, choices: [String], correctIndex: Int) {
            self.after = after; self.question = question; self.choices = choices; self.correctIndex = correctIndex
        }
    }

    /// Loads `lectures.json` from the bundle root ([] if missing).
    public static func catalog(bundle: Bundle = .main) -> [LectureInfo] {
        guard let url = bundle.url(forResource: "lectures", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONDecoder().decode([LectureInfo].self, from: data)) ?? []
    }
}

// MARK: - Session report

/// Everything the session report screen and the PDF/image export show. Built by
/// `Persistence` from a `SessionRecord` (+ content timeline + trial effects); rendered by `UI`.
public struct SessionReport: Equatable, Sendable {
    public struct Segment: Equatable, Sendable {
        public var start: Double
        public var end: Double
        public var state: AttentionState
        public init(start: Double, end: Double, state: AttentionState) { self.start = start; self.end = end; self.state = state }
    }
    public struct Point: Equatable, Sendable {
        public var t: Double
        public var value: Double
        public init(t: Double, value: Double) { self.t = t; self.value = value }
    }

    public var sessionID: UUID
    public var startedAt: Date
    public var duration: Double
    public var lectureID: String?
    public var lectureTitle: String?
    public var contentIsMock: Bool
    public var isReplay: Bool

    /// Focus timeline: contiguous state segments.
    public var segments: [Segment]
    public var focusedFraction: Double
    /// Each refocus (seconds) in session order.
    public var refocusTimes: [Double]
    public var medianRefocusSeconds: Double?
    public var heartRate: [Point]
    public var restingHeartRate: Double?

    /// "Where this lecture was predicted to lose you vs where you actually drifted".
    /// Both on the lecture playhead axis, each min-max scaled to 0...1 for overlay.
    public var predictedDrift: [Point]
    /// Actual P(not focused) smoothed (1 - P(focused)) on the same axis.
    public var actualDrift: [Point]
    /// Pearson r between the two curves (nil if no lecture or too short).
    public var driftCorrelation: Double?
    /// Playhead (lecture seconds) of the biggest actual drift moment, for the cortex snapshot.
    public var biggestDriftPlayhead: Double?
    /// `BrainCortexView` activations at that moment (BrainNetwork order).
    public var biggestDriftActivations: [Float]?

    public var nudgeCount: Int
    public var preemptiveCount: Int
    /// "Your Personal Trial": per-arm effects vs no-nudge, cumulative across sessions.
    public var effects: [ArmEffect]
    public var insights: [Insight]

    public init(sessionID: UUID, startedAt: Date, duration: Double, lectureID: String?, lectureTitle: String?,
                contentIsMock: Bool, isReplay: Bool, segments: [Segment], focusedFraction: Double,
                refocusTimes: [Double], medianRefocusSeconds: Double?, heartRate: [Point], restingHeartRate: Double?,
                predictedDrift: [Point], actualDrift: [Point], driftCorrelation: Double?,
                biggestDriftPlayhead: Double?, biggestDriftActivations: [Float]?, nudgeCount: Int,
                preemptiveCount: Int, effects: [ArmEffect], insights: [Insight]) {
        self.sessionID = sessionID; self.startedAt = startedAt; self.duration = duration; self.lectureID = lectureID
        self.lectureTitle = lectureTitle; self.contentIsMock = contentIsMock; self.isReplay = isReplay
        self.segments = segments; self.focusedFraction = focusedFraction; self.refocusTimes = refocusTimes
        self.medianRefocusSeconds = medianRefocusSeconds; self.heartRate = heartRate
        self.restingHeartRate = restingHeartRate; self.predictedDrift = predictedDrift; self.actualDrift = actualDrift
        self.driftCorrelation = driftCorrelation; self.biggestDriftPlayhead = biggestDriftPlayhead
        self.biggestDriftActivations = biggestDriftActivations; self.nudgeCount = nudgeCount
        self.preemptiveCount = preemptiveCount; self.effects = effects; self.insights = insights
    }
}
