import CoreGraphics
import Foundation
import SwiftUI

// The UI's single seam to the rest of the app. The orchestrator's DI container implements this
// (`AppServices`), wiring the session engine, persistence, insights, calibration and nudge views.
// `PreviewUIServices` (Modules/UI/Preview) implements it with the synthetic Replay stream so every
// screen runs in the Simulator before integration.
//
// Slot factories return `AnyView?`: return nil to use the UI module's built-in fallback view.

public enum UIPermission: String, CaseIterable, Sendable, Identifiable {
    case camera, motion, health
    public var id: String { rawValue }
}

public enum UIPermissionStatus: String, Sendable {
    case notDetermined, granted, denied, unavailable
}

/// What kind of session the user is starting.
public enum UISessionKind: Equatable, Sendable {
    /// Free study: phone beside the laptop.
    case focus
    /// Study Mode with a bundled lecture (playhead synced to its brain timeline).
    case study(lectureID: String)
    /// Opened from the Science view with no session running (live preview of the pipeline).
    case science
}

/// One row in History.
public struct UISessionSummary: Identifiable, Equatable, Sendable {
    public var id: UUID
    public var startedAt: Date
    public var duration: Double
    public var focusedFraction: Double
    public var medianRefocusSeconds: Double?
    public var lectureTitle: String?
    public var isReplay: Bool
    public var contentIsMock: Bool

    public init(id: UUID, startedAt: Date, duration: Double, focusedFraction: Double,
                medianRefocusSeconds: Double?, lectureTitle: String?, isReplay: Bool, contentIsMock: Bool) {
        self.id = id; self.startedAt = startedAt; self.duration = duration; self.focusedFraction = focusedFraction
        self.medianRefocusSeconds = medianRefocusSeconds; self.lectureTitle = lectureTitle
        self.isReplay = isReplay; self.contentIsMock = contentIsMock
    }
}

/// Per-session series for the Attention Profile trend charts (one point per session, oldest first).
public struct ProfileSeries: Equatable, Sendable {
    public struct Point: Identifiable, Equatable, Sendable {
        public var date: Date
        public var focusedFraction: Double
        public var medianRefocusSeconds: Double?
        public var restingHeartRate: Double?
        public var minutes: Double
        public var id: Date { date }
        public init(date: Date, focusedFraction: Double, medianRefocusSeconds: Double?, restingHeartRate: Double?, minutes: Double) {
            self.date = date; self.focusedFraction = focusedFraction; self.medianRefocusSeconds = medianRefocusSeconds
            self.restingHeartRate = restingHeartRate; self.minutes = minutes
        }
    }
    public var points: [Point]
    /// Focused fraction by hour of day (0-23), nil where there is no data.
    public var focusByHour: [Double?]
    /// True if any point came from the synthetic Replay stream.
    public var isSynthetic: Bool

    public init(points: [Point], focusByHour: [Double?] = Array(repeating: nil, count: 24), isSynthetic: Bool) {
        self.points = points; self.focusByHour = focusByHour; self.isSynthetic = isSynthetic
    }

    public static let empty = ProfileSeries(points: [], isSynthetic: false)
}

@MainActor
public protocol UIServices: AnyObject {
    // MARK: Flags & onboarding
    /// Persisted feature flags (Demo Director, Force Replay, experience sampling).
    var flags: FeatureFlags { get set }
    /// True on simulators / devices without ARKit face tracking (Replay will be used regardless).
    var replayIsForced: Bool { get }

    // MARK: Permissions
    func permissionStatus(_ p: UIPermission) -> UIPermissionStatus
    func requestPermission(_ p: UIPermission) async -> UIPermissionStatus

    // MARK: Sessions
    /// Starts a session and fills the shared `LiveSessionModel`.
    func startSession(_ kind: UISessionKind)
    /// Stops the running session, persists it, and returns its id (nil if nothing was running).
    func stopSession() async -> UUID?
    func lectures() -> [LectureInfo]
    /// Most recent first.
    func listSessions() -> [UISessionSummary]
    func report(for sessionID: UUID) async -> SessionReport?
    /// The report without waiting for coach insights (shown at once; insights fill in after). Nil = not supported.
    func reportPreview(for sessionID: UUID) -> SessionReport?
    func aggregateSummary() async -> AggregateSummary?
    func profileSeries() -> ProfileSeries
    /// Coach insights for the Attention Profile (aggregated numbers only).
    func profileInsights() async -> [Insight]
    /// Normalized (0...1, top-left origin) face-mesh landmarks for the Science view overlay; nil if none.
    /// Polled at display rate while the Science view is on screen. Never stored.
    func faceMeshPoints() -> [CGPoint]?

    // MARK: Data
    /// PDF of a report; returns a temp file URL for the share sheet.
    func exportReport(_ report: SessionReport) async throws -> URL
    /// PNG image of a report; temp file URL.
    func exportReportImage(_ report: SessionReport) async throws -> URL
    /// The session's camera heart rate as `timestamp,bpm` (Unix seconds), for `ml/rppg_validate.py --rppg`.
    func exportHeartRateCSV(_ report: SessionReport) throws -> URL
    /// Opt-in: record each session's full 10 Hz numeric feature stream for forecaster retraining.
    var recordFeatureStreams: Bool { get set }
    /// Recorded feature streams + `manifest.csv` for `ml/forecast/train.py --manifest`.
    func exportFeatureStreams() throws -> [URL]
    /// Labeled windows as CSV for retraining; temp file URL.
    func exportLabelsCSV() throws -> URL
    func deleteAllData() throws

    // MARK: Sustained attention & clinician report
    /// Every saved SART run, oldest first (numbers only).
    func sartHistory() -> [SARTRunRecord]
    /// What a clinician report would contain right now (for the share sheet's copy and demo toggle).
    func doctorReportCounts() -> DoctorReportCounts
    /// Builds the "Share with your doctor" PDF on this phone and returns its temp file URL (`FocusLoop-` prefix,
    /// swept by Delete all data). Real sessions only unless `includeDemo`. Nothing is sent anywhere.
    func makeDoctorReport(includeDemo: Bool) async throws -> URL

    // MARK: Demo (DEMONSTRATION ONLY)
    /// Prepares the one-tap scripted demo (SYNTHETIC): Demo Director for this session only, the scripted
    /// `demo_session` replay locked to the lecture clock. Returns the lecture to open in Study Mode, or nil
    /// if the demo files are missing. The next `startSession(.study)` runs the demo.
    func beginScriptedDemo() -> LectureInfo?
    /// True while a scripted demo is prepared or running (Study Mode closes itself when the lecture ends).
    var scriptedDemoActive: Bool { get }
    /// Generates about 6 SYNTHETIC past sessions (the replay run through the real pipeline) on the SYNTHETIC
    /// side only. Returns how many were added. Real data is never touched.
    func loadSampleHistory() async throws -> Int
    /// Removes the sessions added by `loadSampleHistory()`. Returns how many were removed.
    func removeSampleHistory() throws -> Int
    var sampleHistoryCount: Int { get }
    /// Live checklist for Settings → Demo → Demo readiness.
    func demoReadiness() -> [DemoReadinessItem]
    /// Starts / stops a short camera probe so the readiness screen can show face detection and HR quality.
    func startReadinessProbe()
    func stopReadinessProbe()
    /// True if the session's nudges came from Demo Director's fixed script (DEMONSTRATION ONLY).
    func sessionUsedDemoNudges(_ id: UUID) -> Bool

    // MARK: Slots (nil = use the UI module's fallback)
    /// Work-zone calibration. `isPhoneScreen` = the phone screen itself (Study Mode) instead of a laptop.
    func calibrationView(isPhoneScreen: Bool, onFinish: @escaping () -> Void) -> AnyView?
    func sartView(onFinish: @escaping () -> Void) -> AnyView?
    func studyModeView(lecture: LectureInfo, model: LiveSessionModel, onFinish: @escaping () -> Void) -> AnyView?
    /// Renders `presentation` on top of the focus/study screens (Trial agent's nudge views).
    func nudgeView(_ presentation: NudgePresentation, model: LiveSessionModel) -> AnyView?
    /// Full breathing biofeedback screen.
    func breathingView(model: LiveSessionModel, onClose: @escaping () -> Void) -> AnyView?
    /// Experience-sampling prompt ("Were you focused just now?").
    func experienceSampleView(model: LiveSessionModel) -> AnyView?
    /// Extra rows for the Debug menu.
    func debugMenuItems() -> AnyView?
    /// Live face video for the Science view / Focus screen when the session is driven by a recorded video
    /// (the frames the pipeline is analysing right now). nil for ARKit live (no preview) and Replay.
    /// Render it underneath the face-mesh overlay. Never recorded or stored by the app.
    func faceVideoView() -> AnyView?
    /// Width / height of the video `faceVideoView()` shows (it aspect-fills its frame), so the face mesh can be
    /// drawn exactly on the face. nil when there is no video.
    func faceVideoAspectRatio() -> CGFloat?
}

public extension UIServices {
    func sartHistory() -> [SARTRunRecord] { [] }
    func doctorReportCounts() -> DoctorReportCounts { DoctorReportCounts() }
    func makeDoctorReport(includeDemo: Bool) async throws -> URL { throw DoctorReportUnavailable() }
    func reportPreview(for sessionID: UUID) -> SessionReport? { nil }
    func calibrationView(isPhoneScreen: Bool, onFinish: @escaping () -> Void) -> AnyView? { nil }
    /// Laptop work-zone calibration (onboarding).
    func calibrationView(onFinish: @escaping () -> Void) -> AnyView? { calibrationView(isPhoneScreen: false, onFinish: onFinish) }
    func exportReportImage(_ report: SessionReport) async throws -> URL { try await exportReport(report) }
    func exportHeartRateCSV(_ report: SessionReport) throws -> URL { try UIHeartRateCSV.write(report) }
    var recordFeatureStreams: Bool { get { false } set {} }
    func exportFeatureStreams() throws -> [URL] { throw FeatureStreamStore.ExportError.nothingRecorded }
    func sartView(onFinish: @escaping () -> Void) -> AnyView? { nil }
    func studyModeView(lecture: LectureInfo, model: LiveSessionModel, onFinish: @escaping () -> Void) -> AnyView? { nil }
    func nudgeView(_ presentation: NudgePresentation, model: LiveSessionModel) -> AnyView? { nil }
    func breathingView(model: LiveSessionModel, onClose: @escaping () -> Void) -> AnyView? { nil }
    func experienceSampleView(model: LiveSessionModel) -> AnyView? { nil }
    func debugMenuItems() -> AnyView? { nil }
    func faceMeshPoints() -> [CGPoint]? { nil }
    func faceVideoView() -> AnyView? { nil }
    func faceVideoAspectRatio() -> CGFloat? { nil }
    func beginScriptedDemo() -> LectureInfo? { nil }
    var scriptedDemoActive: Bool { false }
    func loadSampleHistory() async throws -> Int { 0 }
    func removeSampleHistory() throws -> Int { 0 }
    var sampleHistoryCount: Int { 0 }
    func demoReadiness() -> [DemoReadinessItem] { [] }
    func startReadinessProbe() {}
    func stopReadinessProbe() {}
    func sessionUsedDemoNudges(_ id: UUID) -> Bool { false }
}

/// Inputs available for the clinician report.
public struct DoctorReportCounts: Equatable, Sendable {
    public var realSessions: Int
    public var demoSessions: Int
    public var sartRuns: Int
    public init(realSessions: Int = 0, demoSessions: Int = 0, sartRuns: Int = 0) {
        self.realSessions = realSessions; self.demoSessions = demoSessions; self.sartRuns = sartRuns
    }
}

public struct DoctorReportUnavailable: LocalizedError {
    public init() {}
    public var errorDescription: String? { "The summary can't be made in this preview." }
}

/// One row of the Demo readiness checklist (Settings → Demo).
public struct DemoReadinessItem: Identifiable, Equatable, Sendable {
    public enum Status: String, Sendable { case ok, warning, problem, info }
    public var id: String
    public var title: String
    public var detail: String
    public var status: Status
    public init(id: String, title: String, detail: String, status: Status) {
        self.id = id; self.title = title; self.detail = detail; self.status = status
    }
}

/// Persisted UI-only preferences.
public enum UIPrefs {
    public static let onboardingCompleteKey = "focusloop.ui.onboardingComplete"
    public static let flagsKey = "focusloop.featureFlags"

    public static func loadFlags(_ defaults: UserDefaults = .standard) -> FeatureFlags {
        guard let d = defaults.data(forKey: flagsKey), let f = try? JSONDecoder().decode(FeatureFlags.self, from: d) else {
            return FeatureFlags()
        }
        return f
    }

    public static func saveFlags(_ f: FeatureFlags, _ defaults: UserDefaults = .standard) {
        if let d = try? JSONEncoder().encode(f) { defaults.set(d, forKey: flagsKey) }
    }
}

/// `timestamp,bpm` CSV of a report's heart-rate series (Unix seconds), the input format of
/// `ml/rppg_validate.py --rppg`. Written to the temp dir with the `FocusLoop` prefix (swept by Delete all).
public enum UIHeartRateCSV {
    public static func make(_ report: SessionReport) -> String {
        var s = report.isReplay ? "# FocusLoop camera heart rate (SYNTHETIC replay)\n" : "# FocusLoop camera heart rate (REAL device data)\n"
        s += "timestamp,bpm\n"
        let t0 = report.startedAt.timeIntervalSince1970
        for p in report.heartRate { s += String(format: "%.3f,%.2f\n", t0 + p.t, p.value) }
        return s
    }

    public static func write(_ report: SessionReport, directory: URL = FileManager.default.temporaryDirectory) throws -> URL {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd-HHmm"
        let url = directory.appendingPathComponent("FocusLoop-heart-rate-\(f.string(from: report.startedAt))\(report.isReplay ? "-SYNTHETIC" : "").csv")
        try make(report).write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}

/// Maps points normalized to an upright video (0...1, top-left origin) into a view that shows that video aspect-filled.
public enum FLAspectFill {
    public static func map(_ pts: [CGPoint], videoAspect: CGFloat, into size: CGSize) -> [CGPoint] {
        guard videoAspect > 0, size.width > 0, size.height > 0 else { return [] }
        let viewAspect = size.width / size.height
        // Displayed video size that covers the view.
        let w = viewAspect > videoAspect ? size.width : size.height * videoAspect
        let h = viewAspect > videoAspect ? size.width / videoAspect : size.height
        let ox = (size.width - w) / 2, oy = (size.height - h) / 2
        return pts.map { CGPoint(x: ox + $0.x * w, y: oy + $0.y * h) }
    }
}
