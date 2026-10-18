import AVFoundation
import CoreGraphics
import CoreMotion
import Foundation
import HealthKit
import Observation
import SwiftUI

/// `UIServices` backed by the SYNTHETIC Replay stream and in-memory storage. PREVIEW-ONLY:
/// the orchestrator replaces it with the real `AppServices` (session engine, SwiftData, insights...).
@MainActor
@Observable
public final class PreviewUIServices: UIServices {
    public let model: LiveSessionModel
    public let driver: ReplayPreviewDriver

    public var flags: FeatureFlags {
        didSet { UIPrefs.saveFlags(flags); model.demoDirector = flags.demoDirector }
    }
    public var replayIsForced: Bool { true }

    private var records: [UUID: (record: SessionRecord, report: SessionReport)] = [:]
    private var order: [UUID] = []
    private var currentLectureID: String?
    private var currentKind: UISessionKind = .focus
    private var seeded = false
    private var seedTask: Task<Void, Never>?

    /// Awaits the background SYNTHETIC history seeding (tests / launch arguments).
    public func seedingFinished() async { await seedTask?.value }
    private var permissions: [UIPermission: UIPermissionStatus] = [:]

    public init(model: LiveSessionModel, seedHistory: Bool = true) {
        self.model = model
        self.flags = UIPrefs.loadFlags()
        self.driver = ReplayPreviewDriver(model: model)
        model.demoDirector = flags.demoDirector
        if seedHistory { seedTask = Task { @MainActor [weak self] in await self?.seedSyntheticHistory() } }
    }

    // MARK: Permissions

    public func permissionStatus(_ p: UIPermission) -> UIPermissionStatus {
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

    public func requestPermission(_ p: UIPermission) async -> UIPermissionStatus {
        let status: UIPermissionStatus
        switch p {
        case .camera:
            status = await AVCaptureDevice.requestAccess(for: .video) ? .granted : .denied
        case .motion:
            guard CMMotionActivityManager.isActivityAvailable() else { status = .unavailable; break }
            // Querying activity triggers the system prompt.
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
            do {
                try await HKHealthStore().requestAuthorization(toShare: [], read: [hr])
                status = .granted
            } catch { status = .denied }
        }
        permissions[p] = status
        return status
    }

    // MARK: Sessions

    public func startSession(_ kind: UISessionKind) {
        currentKind = kind
        switch kind {
        case .focus: currentLectureID = nil; driver.start(studyMode: false)
        case .study(let id): currentLectureID = id; driver.start(studyMode: true)
        case .science: currentLectureID = nil; driver.start(studyMode: false)
        }
        model.demoDirector = flags.demoDirector
    }

    public func stopSession() async -> UUID? {
        guard model.phase == .running || model.phase == .paused else { return nil }
        driver.stop()
        // Science-view previews aren't study sessions; too-short sessions aren't worth a report.
        guard currentKind != .science, model.elapsed >= 5 else { return nil }
        let rec = driver.record(lectureID: currentLectureID)
        store(rec, nudges: driver.nudgeCount, preemptive: driver.preemptiveCount, effects: driver.bandit.effects())
        return rec.id
    }

    public func lectures() -> [LectureInfo] {
        let c = LectureInfo.catalog()
        return c.isEmpty ? [Self.sampleLecture] : c
    }

    /// Fallback when `Resources/Lectures/lectures.json` isn't bundled yet: the MOCK sample timeline.
    public static let sampleLecture = LectureInfo(
        id: "sample_lecture", title: "Sample lecture", subject: "Brain-kit fixture", source: "FocusLoop",
        sourceURL: "", license: "N/A", attribution: "MOCK brain timeline bundled with the brain kit (no video).",
        videoFile: nil, durationSeconds: 180, recaps: [], recallQuestions: [])

    public func listSessions() -> [UISessionSummary] {
        order.reversed().compactMap { id in
            guard let r = records[id]?.report else { return nil }
            return UISessionSummary(id: id, startedAt: r.startedAt, duration: r.duration, focusedFraction: r.focusedFraction,
                                    medianRefocusSeconds: r.medianRefocusSeconds, lectureTitle: r.lectureTitle,
                                    isReplay: r.isReplay, contentIsMock: r.contentIsMock)
        }
    }

    public func report(for sessionID: UUID) async -> SessionReport? { records[sessionID]?.report }

    /// Most recent report (used by the `-screen report` launch argument).
    public var latestReportID: UUID? { order.last }

    public func aggregateSummary() async -> AggregateSummary? {
        let reps = order.compactMap { records[$0]?.report }
        guard !reps.isEmpty else { return nil }
        let total = reps.reduce(0) { $0 + $1.duration }
        var fr = Array(repeating: 0.0, count: 4)
        for r in reps { for (i, f) in StateSegmenter.fractions(r.segments).enumerated() { fr[i] += f * r.duration } }
        fr = fr.map { $0 / max(1, total) }
        let refocus = reps.flatMap(\.refocusTimes)
        let rs = reps.compactMap(\.restingHeartRate)
        let corr = reps.compactMap(\.driftCorrelation)
        let hours = Dictionary(grouping: reps) { Calendar.current.component(.hour, from: $0.startedAt) }
            .mapValues { g in g.map(\.focusedFraction).reduce(0, +) / Double(g.count) }
            .sorted { $0.value > $1.value }.map(\.key)
        return AggregateSummary(sessionCount: reps.count, totalMinutes: total / 60, focusedFraction: fr[0],
                                stateFractions: fr, typicalDropMinute: nil, medianRefocusSeconds: UIStats.median(refocus),
                                bestHours: Array(hours.prefix(3)), restingHeartRate: UIStats.median(rs),
                                heartRateTrendPerSession: rs.count > 1 ? (rs.last! - rs.first!) / Double(rs.count - 1) : nil,
                                contentDriftCorrelation: corr.isEmpty ? nil : corr.reduce(0, +) / Double(corr.count),
                                contentIsMock: reps.contains { $0.contentIsMock }, isReplay: reps.contains { $0.isReplay },
                                armEffects: driver.bandit.effects())
    }

    public func profileSeries() -> ProfileSeries {
        let reps = order.compactMap { records[$0]?.report }
        var byHour = Array<[Double]>(repeating: [], count: 24)
        for r in reps { byHour[Calendar.current.component(.hour, from: r.startedAt)].append(r.focusedFraction) }
        return ProfileSeries(points: reps.map {
            .init(date: $0.startedAt, focusedFraction: $0.focusedFraction, medianRefocusSeconds: $0.medianRefocusSeconds,
                  restingHeartRate: $0.restingHeartRate, minutes: $0.duration / 60)
        }, focusByHour: byHour.map { $0.isEmpty ? nil : $0.reduce(0, +) / Double($0.count) },
           isSynthetic: reps.contains { $0.isReplay })
    }

    public func profileInsights() async -> [Insight] {
        guard let s = await aggregateSummary() else { return [] }
        var out: [Insight] = []
        let pre = s.isReplay ? "Across \(s.sessionCount) SYNTHETIC replay sessions, " : "Across \(s.sessionCount) sessions, "
        out.append(Insight(text: pre + "you were focused \(UIFormat.percent(s.focusedFraction)) of the time.", source: .template))
        if let h = s.bestHours.first {
            out.append(Insight(text: "Your steadiest focus so far was around \(UIFormat.hour(h)).", source: .template))
        }
        if let best = s.armEffects.first(where: { !$0.arm.isExperimental && $0.n >= 3 }), best.lower > 0 {
            out.append(Insight(text: "\(best.arm.displayName) is the nudge most likely to bring you back so far.", source: .template))
        } else {
            out.append(Insight(text: "It's still too early to say which nudge works best for you; intervals narrow as you study more.",
                               source: .template))
        }
        return out
    }

    // MARK: Data

    public func exportReport(_ report: SessionReport) async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("FocusLoop-report\(report.isReplay ? "-SYNTHETIC" : "").pdf")
        let renderer = ImageRenderer(content: ReportExportPage(report: report).frame(width: 612))
        renderer.render { size, draw in
            var box = CGRect(origin: .zero, size: size)
            guard let ctx = CGContext(url as CFURL, mediaBox: &box, nil) else { return }
            ctx.beginPDFPage(nil); draw(ctx); ctx.endPDFPage(); ctx.closePDF()
        }
        return url
    }

    public struct PreviewError: LocalizedError {
        public var errorDescription: String? { "Label export is available once calibration and SART are connected." }
    }

    public func exportLabelsCSV() throws -> URL { throw PreviewError() }

    public func deleteAllData() throws {
        records = [:]; order = []
        UserDefaults.standard.removeObject(forKey: UIPrefs.flagsKey)
        flags = FeatureFlags()
        driver.bandit = UIPreviewBandit()
    }

    // MARK: Face mesh (stylised SYNTHETIC landmarks in Replay)

    public func faceMeshPoints() -> [CGPoint]? { nil }

    // MARK: Private

    private func store(_ rec: SessionRecord, nudges: Int, preemptive: Int, effects: [ArmEffect]) {
        let title = rec.lectureID.map { id in lectures().first { $0.id == id }?.title ?? id }
        let report = PreviewReportBuilder.build(record: rec, timeline: driver.timeline, lectureTitle: title,
                                                effects: effects, nudgeCount: nudges, preemptiveCount: preemptive)
        records[rec.id] = (rec, report)
        order.append(rec.id)
    }

    /// Runs the replay through the preview heuristics a few times (fast, offline) so History and the
    /// report have SYNTHETIC content. Every resulting session is flagged `isReplay`.
    private func seedSyntheticHistory() async {
        guard !seeded else { return }
        seeded = true
        let scratch = LiveSessionModel()
        let d = ReplayPreviewDriver(model: scratch, bandit: driver.bandit)
        let cal = Calendar.current
        let plan: [(daysAgo: Int, hour: Int, offset: Double, study: Bool)] =
            [(6, 16, 40, false), (5, 20, 90, false), (4, 9, 0, true), (3, 15, 120, false), (2, 10, 0, true), (1, 19, 60, false)]
        for p in plan {
            d.simulate(seconds: 180, studyMode: p.study, startOffset: p.offset)
            var rec = d.record(lectureID: p.study ? Self.sampleLecture.id : nil)
            let day = cal.date(byAdding: .day, value: -p.daysAgo, to: Date()) ?? Date()
            rec.startedAt = cal.date(bySettingHour: p.hour, minute: 12, second: 0, of: day) ?? day
            store(rec, nudges: d.nudgeCount, preemptive: d.preemptiveCount, effects: d.bandit.effects())
            await Task.yield()   // keep the UI responsive while seeding
        }
        driver.bandit = d.bandit
    }
}
