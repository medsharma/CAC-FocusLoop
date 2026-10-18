import AVFoundation
import Foundation
import UIKit

// Demo support (DEMONSTRATION ONLY): the one-tap scripted demo (SYNTHETIC), the SYNTHETIC sample history,
// the Demo readiness checklist and Demo Director session detection. Nothing here touches real data.

extension AppServices {

    // MARK: Scripted demo (SYNTHETIC)

    /// Prepares the one-tap scripted demo: Demo Director on for this session only, the scripted
    /// `demo_session` replay (SYNTHETIC) locked to the "Travel deep inside a leaf" lecture clock. The session
    /// runs on the SYNTHETIC learner set, and Demo Director's fixed nudges keep it out of every learner.
    func beginScriptedDemo() -> LectureInfo? {
        guard let lecture = lectures().first(where: { $0.id == Self.scriptedDemoLectureID }),
              timeline(for: lecture.id) != nil,
              (try? ReplayStream.load(Self.scriptedDemoStream)) != nil else { return nil }
        if engine.isRunning, let r = engine.stopAndCollect() { try? store.save(r) }
        scriptedDemoActive = true
        applyEffectiveFlags()
        return lecture
    }

    /// Back to the saved flags once the scripted demo session has ended.
    func endScriptedDemo() {
        guard scriptedDemoActive else { return }
        scriptedDemoActive = false
        applyEffectiveFlags()
    }

    /// True when the session's nudges came from Demo Director's fixed script (DEMONSTRATION ONLY).
    func sessionUsedDemoNudges(_ id: UUID) -> Bool {
        guard let r = try? store.session(id: id) else { return false }
        return r.events.contains { if case let .nudge(d) = $0 { return DemoScriptPolicy.isScripted(d) } else { return false } }
    }

    // MARK: Sample history (SYNTHETIC)

    static let sampleHistoryKey = "focusloop.demo.sampleHistoryIDs"

    private var sampleIDs: [UUID] {
        get { (UserDefaults.standard.stringArray(forKey: Self.sampleHistoryKey) ?? []).compactMap(UUID.init(uuidString:)) }
        set { UserDefaults.standard.set(newValue.map(\.uuidString), forKey: Self.sampleHistoryKey) }
    }

    var sampleHistoryCount: Int { sampleIDs.count }

    enum SampleHistoryError: LocalizedError {
        case sessionRunning, missingReplay
        var errorDescription: String? {
            switch self {
            case .sessionRunning: "End the current session, then try again."
            case .missingReplay: "The demo replay files are missing from this build."
            }
        }
    }

    /// One planned sample session: when it happened and what it replays.
    struct SamplePlan {
        var daysAgo: Int
        var hour: Int
        var minute: Int
        var study: Bool
        var useDemoStream: Bool
        var seconds: Double
        var offsetFrames: Int
    }

    static let samplePlans: [SamplePlan] = [
        SamplePlan(daysAgo: 6, hour: 16, minute: 10, study: true, useDemoStream: true, seconds: 172, offsetFrames: 0),
        SamplePlan(daysAgo: 5, hour: 19, minute: 40, study: false, useDemoStream: false, seconds: 300, offsetFrames: 0),
        SamplePlan(daysAgo: 4, hour: 17, minute: 5, study: true, useDemoStream: false, seconds: 172, offsetFrames: 150),
        SamplePlan(daysAgo: 2, hour: 20, minute: 15, study: false, useDemoStream: false, seconds: 240, offsetFrames: 600),
        SamplePlan(daysAgo: 1, hour: 15, minute: 30, study: true, useDemoStream: true, seconds: 172, offsetFrames: 0),
        SamplePlan(daysAgo: 0, hour: 9, minute: 45, study: false, useDemoStream: false, seconds: 200, offsetFrames: 1200),
    ]

    /// Runs the SYNTHETIC replay streams through the real pipeline (classifier, forecaster, SYNTHETIC trial,
    /// reward tracker) quickly and saves about 6 past sessions over the last week, on the SYNTHETIC side
    /// only: the records are `isReplay` (badged SYNTHETIC everywhere) and only the SYNTHETIC trial learns
    /// from them. The personal head, label store and forecaster threshold are not updated.
    func loadSampleHistory() async throws -> Int {
        guard !engine.isRunning else { throw SampleHistoryError.sessionRunning }
        guard let demo = try? ReplayStream.load(Self.scriptedDemoStream), let free = try? ReplayStream.load(),
              !demo.frames.isEmpty, !free.frames.isEmpty else { throw SampleHistoryError.missingReplay }
        let set = replay
        let lecture = lectures().first { $0.id == Self.scriptedDemoLectureID }
        let tl = timeline(for: Self.scriptedDemoLectureID)
        let classifier = BlendedAttentionClassifier.makeDefault(personalHead: set.personalHead)
        var ids = sampleIDs
        let cal = Calendar.current
        let now = Date()
        for plan in Self.samplePlans {
            let day = cal.date(byAdding: .day, value: -plan.daysAgo, to: cal.startOfDay(for: now)) ?? now
            var start = cal.date(bySettingHour: plan.hour, minute: plan.minute, second: 0, of: day) ?? day
            if start > now.addingTimeInterval(-plan.seconds - 600) { start = now.addingTimeInterval(-plan.seconds - 3 * 3600) }
            let study = plan.study && lecture != nil && tl != nil
            let components = PipelineComponents(classifier: classifier, personalHead: set.personalHead,
                                                forecaster: set.forecaster, policy: set.policy, pacing: set.policy,
                                                logger: InMemoryTrialLogger(), labels: set.labels)
            let rec = await Self.runSample(stream: plan.useDemoStream ? demo : free, seconds: plan.seconds,
                                           offsetFrames: plan.offsetFrames, startedAt: start,
                                           lectureID: study ? lecture?.id : nil, timeline: study ? tl : nil,
                                           components: components)
            try store.save(rec)
            ids.append(rec.id)
            sampleIDs = ids
        }
        saveWeights(set)
        reportCache = [:]
        insights.clearCache()
        return Self.samplePlans.count
    }

    /// Feeds `seconds` of a replay stream through a fresh `SessionPipeline` and assembles its record.
    static func runSample(stream: ReplayStream, seconds: Double, offsetFrames: Int, startedAt: Date, lectureID: String?,
                          timeline: ContentTimeline?, components: PipelineComponents) async -> SessionRecord {
        let id = UUID()
        let pipeline = SessionPipeline(components: components, pacing: PacingConfig(), timeline: timeline,
                                       studyMode: timeline != nil, sessionID: id, experienceSampling: false,
                                       learning: false)
        var rec = SessionRecord(id: id, startedAt: startedAt, lectureID: lectureID,
                                contentIsMock: timeline?.isMock ?? false, isReplay: true)
        let n = Int(seconds * FeatureFrame.rateHz)
        var events: [SessionEvent] = []
        for k in 0..<n {
            var f = stream.frames[(offsetFrames + k) % stream.frames.count]
            f.t = Double(k) / FeatureFrame.rateHz
            if let tl = timeline { await pipeline.setPlayhead(min(tl.duration, f.t)) }
            if let out = await pipeline.ingest(f) {
                if let tick = out.tick { rec.ticks.append(tick) }
                events.append(contentsOf: out.events)
            }
        }
        rec.duration = Double(n) / FeatureFrame.rateHz
        events.append(contentsOf: await pipeline.finish(at: rec.duration).map { SessionEvent.outcome($0) })
        rec.events = events
        return rec
    }

    /// Deletes the sample sessions and rebuilds the SYNTHETIC trial without their decisions. Real data and
    /// any other Replay sessions are untouched.
    func removeSampleHistory() throws -> Int {
        let ids = sampleIDs
        guard !ids.isEmpty else { return 0 }
        var decisionIDs = Set<UUID>()
        for id in ids {
            guard let r = try store.session(id: id) else { continue }
            for e in r.events { if case let .nudge(d) = e { decisionIDs.insert(d.id) } }
        }
        let removed = try store.deleteSessions(ids: ids)
        // Replaying the remaining (decision, outcome) pairs onto the priors gives exactly the posteriors the
        // SYNTHETIC trial would have without the sample sessions.
        let rebuilt = ThompsonTrialEngine(bundle: .main)
        for r in replay.policy.records where !decisionIDs.contains(r.decision.id) {
            if let o = r.outcome { rebuilt.record(outcome: o, for: r.decision) }
        }
        try replay.policy.restore(from: rebuilt.serialized())
        saveWeights(replay)
        sampleIDs = []
        reportCache = [:]
        insights.clearCache()
        model.posteriors = active.policy.posteriors()
        return removed
    }

    // MARK: Demo readiness

    func startReadinessProbe() {
        guard readinessProbe == nil, LiveFeatureFrameSource.isSupported, !engine.isRunning,
              AVCaptureDevice.authorizationStatus(for: .video) == .authorized else { return }
        let source = LiveFeatureFrameSource(workZone: WorkZoneStore().current(isPhoneScreen: true),
                                            restingBaseline: weightsFloat(PersistenceWeightKey.heartRateBaseline))
        let probe = ReadinessProbe(source: source)
        readinessProbe = probe
        startAux(source, owner: probe.owner)
        probe.tasks = [
            Task { @MainActor [weak probe] in
                for await f in source.frames { probe?.latestFrame = f; probe?.latestFrameAt = Date() }
            },
            Task { @MainActor [weak probe] in
                for await h in source.heartRate { probe?.latestHR = h; probe?.latestHRAt = Date() }
            },
        ]
    }

    func stopReadinessProbe() {
        guard let probe = readinessProbe else { return }
        probe.tasks.forEach { $0.cancel() }
        stopAux(owner: probe.owner)
        readinessProbe = nil
    }

    func pauseReadinessProbe(_ paused: Bool) {
        guard let s = readinessProbe?.source else { return }
        paused ? s.suspend() : s.resume()
    }

    func demoReadiness() -> [DemoReadinessItem] {
        var items: [DemoReadinessItem] = []
        let supported = LiveFeatureFrameSource.isSupported

        switch permissionStatus(.camera) {
        case .granted:
            items.append(.init(id: "camera", title: "Camera permission", detail: "Allowed", status: .ok))
        case .denied:
            items.append(.init(id: "camera", title: "Camera permission",
                               detail: "Denied. Open iPhone Settings → FocusLoop → Camera.", status: .problem))
        case .notDetermined:
            items.append(.init(id: "camera", title: "Camera permission",
                               detail: "Not asked yet. Start any session to see the prompt.", status: .warning))
        case .unavailable:
            items.append(.init(id: "camera", title: "Camera permission", detail: "No camera here", status: .info))
        }

        items.append(supported
            ? .init(id: "arkit", title: "Face tracking (TrueDepth)", detail: "Supported on this device", status: .ok)
            : .init(id: "arkit", title: "Face tracking (TrueDepth)",
                    detail: "Not available here, so every session uses the demo replay (use the scripted demo).",
                    status: .info))

        items.append(faceItem(supported: supported))
        items.append(heartRateItem(supported: supported))

        let zones = WorkZoneStore()
        let laptop = zones.load(isPhoneScreen: false) != nil, phone = zones.load(isPhoneScreen: true) != nil
        let missing = [laptop ? nil : "laptop", phone ? nil : "phone screen (Study Mode)"].compactMap { $0 }
        items.append(.init(id: "zone", title: "Work zone calibrated",
                           detail: missing.isEmpty ? "Laptop and phone screen" : "Missing: \(missing.joined(separator: ", ")). Settings → Calibration.",
                           status: missing.isEmpty ? .ok : (supported ? .warning : .info)))

        if bundleReadinessCache == nil { bundleReadinessCache = Self.bundleChecks(lectures: lectures()) }
        items.append(contentsOf: bundleReadinessCache ?? [])

        let f = flags
        items.append(f.demoDirector
            ? .init(id: "demo", title: "Demo Director", detail: "On: fixed demo nudges, nothing learns", status: .ok)
            : .init(id: "demo", title: "Demo Director",
                    detail: "Off: nudges are randomized by the trial (the scripted demo turns it on by itself)", status: .info))
        if f.forceReplay {
            items.append(.init(id: "replay", title: "Force Replay", detail: "On: sessions use demo data, not the camera",
                               status: supported ? .warning : .info))
        }
        if f.experienceSampling {
            items.append(.init(id: "es", title: "\u{201C}Were you focused?\u{201D} prompts",
                               detail: "On: a prompt can interrupt a take. Turn off for filming.", status: .warning))
        }
        if sensingOnly {
            items.append(.init(id: "sensing", title: "Sensing only", detail: "On: no nudges will show. Turn off in Debug.",
                               status: .problem))
        }
        let thermal = ProcessInfo.processInfo.thermalState
        items.append(.init(id: "thermal", title: "Phone temperature", detail: ThermalStatus.label(thermal),
                           status: thermal == .serious || thermal == .critical ? .warning : .ok))
        if ProcessInfo.processInfo.isLowPowerModeEnabled {
            items.append(.init(id: "power", title: "Low Power Mode", detail: "On: tracking may run slower", status: .warning))
        }
        return items
    }

    private func faceItem(supported: Bool) -> DemoReadinessItem {
        guard supported else {
            return .init(id: "face", title: "Face detected now", detail: "Not available (no face tracking here)", status: .info)
        }
        if engine.isRunning, !model.isReplay {
            return model.faceDetected
                ? .init(id: "face", title: "Face detected now", detail: "Yes (live session)", status: .ok)
                : .init(id: "face", title: "Face detected now", detail: "Can't see your face", status: .warning)
        }
        guard let probe = readinessProbe else {
            return .init(id: "face", title: "Face detected now", detail: "Allow the camera to check", status: .warning)
        }
        guard let f = probe.latestFrame, let at = probe.latestFrameAt, Date().timeIntervalSince(at) < 1.5 else {
            return .init(id: "face", title: "Face detected now", detail: "Starting the camera\u{2026}", status: .info)
        }
        return f.faceDetected > 0.5
            ? .init(id: "face", title: "Face detected now", detail: "Yes", status: .ok)
            : .init(id: "face", title: "Face detected now", detail: "Can't see your face. Face the phone.", status: .warning)
    }

    private func heartRateItem(supported: Bool) -> DemoReadinessItem {
        let sample: HeartRateSample?
        if engine.isRunning, !model.isReplay { sample = model.heartRate } else { sample = readinessProbe?.latestHR }
        guard supported else {
            return .init(id: "hr", title: "Heart-rate signal", detail: "Not available (camera heart rate needs the device)", status: .info)
        }
        guard let h = sample else {
            return .init(id: "hr", title: "Heart-rate signal", detail: "No reading yet. Sit still in even light for about 15 s.", status: .warning)
        }
        let bpm = Int(h.bpm.rounded())
        if h.quality >= 0.6 { return .init(id: "hr", title: "Heart-rate signal", detail: "Good (\(bpm) BPM)", status: .ok) }
        if h.quality >= 0.3 { return .init(id: "hr", title: "Heart-rate signal", detail: "Fair (\(bpm) BPM). Hold still, check the light.", status: .warning) }
        return .init(id: "hr", title: "Heart-rate signal", detail: "Weak. Face the light, hold still.", status: .warning)
    }

    /// Checks of the app bundle that can't change while the app runs (cached).
    static func bundleChecks(lectures: [LectureInfo], bundle: Bundle = .main) -> [DemoReadinessItem] {
        var items: [DemoReadinessItem] = []
        let real = lectures.filter { $0.videoFile != nil }
        let missing = real.filter { l in
            let video = StudyLectureLibrary.videoURL(for: l, bundle: bundle) != nil
            let brain = bundle.url(forResource: l.id, withExtension: "brain.json") != nil
            return !(video && brain)
        }
        let mock = real.contains { (try? BrainContentTimeline(named: $0.id))?.isMock ?? false }
        let titles = missing.map { (l: LectureInfo) in l.title }.joined(separator: ", ")
        items.append(missing.isEmpty
            ? .init(id: "lectures", title: "Lecture files",
                    detail: "\(real.count) lectures with video and brain timelines\(mock ? " (brain predictions from a placeholder preview model)" : "")", status: .ok)
            : .init(id: "lectures", title: "Lecture files", detail: "Missing video or brain file: \(titles)", status: .problem))
        let demoReplay = (try? ReplayStream.load(scriptedDemoStream, bundle: bundle)) != nil
        items.append(.init(id: "demoReplay", title: "Scripted demo replay",
                           detail: demoReplay ? "demo_session.replay.csv present" : "demo_session.replay.csv missing",
                           status: demoReplay ? .ok : .problem))

        let stateLoaded = CoreMLAttentionClassifier(bundle: bundle) != nil
        let forecaster = CoreMLLapseForecaster(bundle: bundle)
        let stateSynthetic = modelIsSynthetic("AttentionStateModel", bundle: bundle)
        let forecastSynthetic = forecaster?.metadata?.synthetic ?? modelIsSynthetic(CoreMLLapseForecaster.contentModelName, bundle: bundle)
        func detail(_ loaded: Bool, _ synthetic: Bool?, fallback: String) -> String {
            guard loaded else { return "Not bundled: using the \(fallback)" }
            switch synthetic {
            case true?: return "Core ML model loaded, trained on generated demo data"
            case false?: return "Core ML model loaded, trained on real data"
            case nil: return "Core ML model loaded"
            }
        }
        items.append(.init(id: "stateModel", title: "Attention-state model",
                           detail: detail(stateLoaded, stateSynthetic, fallback: "rule-based heuristic"),
                           status: stateLoaded ? .ok : .warning))
        items.append(.init(id: "forecastModel", title: "Lapse forecaster",
                           detail: detail(forecaster != nil, forecastSynthetic, fallback: "rule-based heuristic"),
                           status: forecaster != nil ? .ok : .warning))
        return items
    }

    /// Reads `<name>.json` model metadata: `SYNTHETIC` / `synthetic` flags or `training_data`.
    static func modelIsSynthetic(_ name: String, bundle: Bundle) -> Bool? {
        guard let url = bundle.url(forResource: name, withExtension: "json"), let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let b = obj["SYNTHETIC"] as? Bool { return b }
        if let b = obj["synthetic"] as? Bool { return b }
        if let s = obj["training_data"] as? String { return s.uppercased().contains("SYNTHETIC") }
        return nil
    }
}

/// The readiness screen's short-lived camera probe (device only; never stored).
@MainActor
final class ReadinessProbe {
    let source: LiveFeatureFrameSource
    let owner = UUID()
    var tasks: [Task<Void, Never>] = []
    var latestFrame: FeatureFrame?
    var latestFrameAt: Date?
    var latestHR: HeartRateSample?
    var latestHRAt: Date?
    init(source: LiveFeatureFrameSource) { self.source = source }
}
