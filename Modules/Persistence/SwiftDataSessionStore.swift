import Foundation
import SwiftData

/// SwiftData-backed `SessionStore`. Main-actor only (uses the container's `mainContext`).
///
///     let store = try SwiftDataSessionStore()                 // on-disk, app
///     let store = try SwiftDataSessionStore(inMemory: true)   // tests / previews
///
/// `save(_:)` is an upsert keyed by `SessionRecord.id`. It also explodes the record's `.nudge` /
/// `.outcome` events into queryable `StoredDecision` / `StoredOutcome` rows and its insights into
/// `StoredInsight` rows (replacing any rows previously saved for that session).
///
/// `deleteAll()` is the Privacy screen's "Delete all data". It removes:
///   1. every row of every SwiftData model type (`PersistenceContainer.modelTypes`);
///   2. every file in the temporary directory whose name starts with `FocusLoop` (case-insensitive):
///      PDF/PNG report exports (`ReportExporter`) and CSV label exports. Other modules that write
///      temp files MUST use that prefix so they are covered;
///   3. every `UserDefaults` key starting with `focusloop.` (feature flags, onboarding state, etc.).
///      Other modules MUST namespace their defaults keys with that prefix.
@MainActor
public final class SwiftDataSessionStore: SessionStore {
    public let container: ModelContainer
    public var context: ModelContext { container.mainContext }
    let defaults: UserDefaults
    let tempDirectory: URL

    /// Prefix for UserDefaults keys wiped by `deleteAll()`.
    public static let defaultsKeyPrefix = "focusloop."
    /// Prefix (case-insensitive) for temp-dir files wiped by `deleteAll()`.
    public static let tempFilePrefix = "focusloop"

    public init(container: ModelContainer, defaults: UserDefaults = .standard,
                tempDirectory: URL = FileManager.default.temporaryDirectory) {
        self.container = container
        self.defaults = defaults
        self.tempDirectory = tempDirectory
    }

    public convenience init(inMemory: Bool = false, defaults: UserDefaults = .standard,
                            tempDirectory: URL = FileManager.default.temporaryDirectory) throws {
        self.init(container: try PersistenceContainer.make(inMemory: inMemory), defaults: defaults,
                  tempDirectory: tempDirectory)
    }

    // MARK: SessionStore

    public func save(_ session: SessionRecord) throws {
        let sid = session.id
        let ticksBlob = try PersistenceBlobCodec.encode(session.ticks)
        let eventsBlob = try PersistenceBlobCodec.encode(session.events)
        let insightsBlob = try PersistenceBlobCodec.encode(session.insights)
        let focused = Self.focusedSeconds(ticks: session.ticks, duration: session.duration)

        if let existing = try storedSession(id: sid) {
            existing.startedAt = session.startedAt
            existing.duration = session.duration
            existing.lectureID = session.lectureID
            existing.contentIsMock = session.contentIsMock
            existing.isReplay = session.isReplay
            existing.sourceKind = session.sourceKind
            existing.focusedSeconds = focused
            existing.tickCount = session.ticks.count
            existing.eventCount = session.events.count
            existing.ticksBlob = ticksBlob
            existing.eventsBlob = eventsBlob
            existing.insightsBlob = insightsBlob
        } else {
            context.insert(StoredSession(id: sid, startedAt: session.startedAt, duration: session.duration,
                                         lectureID: session.lectureID, contentIsMock: session.contentIsMock,
                                         isReplay: session.isReplay, focusedSeconds: focused,
                                         tickCount: session.ticks.count, eventCount: session.events.count,
                                         ticksBlob: ticksBlob, eventsBlob: eventsBlob, insightsBlob: insightsBlob,
                                         sourceKind: session.sourceKind))
        }

        // Replace the exploded rows for this session.
        try deleteRows(StoredDecision.self, where: #Predicate { $0.sessionID == sid })
        try deleteRows(StoredOutcome.self, where: #Predicate { $0.sessionID == sid })
        try deleteRows(StoredInsight.self, where: #Predicate { $0.sessionID == sid })
        for event in session.events {
            switch event {
            case let .nudge(d): try insertDecision(d, sessionID: sid)
            case let .outcome(o): try insertOutcome(o, sessionID: sid)
            default: break
            }
        }
        for i in session.insights { try insertInsight(i, sessionID: sid) }
        try context.save()
    }

    public func sessions() throws -> [SessionRecord] {
        let d = FetchDescriptor<StoredSession>(sortBy: [SortDescriptor(\.startedAt, order: .reverse)])
        return try context.fetch(d).map(Self.record(from:))
    }

    public func session(id: UUID) throws -> SessionRecord? {
        try storedSession(id: id).map(Self.record(from:))
    }

    /// Deletes the given sessions and their exploded decision / outcome / insight rows. Returns how many
    /// sessions were found and removed.
    @discardableResult
    public func deleteSessions(ids: [UUID]) throws -> Int {
        var removed = 0
        for sid in ids {
            if let row = try storedSession(id: sid) { context.delete(row); removed += 1 }
            try deleteRows(StoredDecision.self, where: #Predicate { $0.sessionID == sid })
            try deleteRows(StoredOutcome.self, where: #Predicate { $0.sessionID == sid })
            try deleteRows(StoredInsight.self, where: #Predicate { $0.sessionID == sid })
        }
        try context.save()
        return removed
    }

    public func saveModelWeights(_ data: Data, key: String) throws {
        let d = FetchDescriptor<StoredModelWeights>(predicate: #Predicate { $0.key == key })
        if let row = try context.fetch(d).first {
            row.data = data
            row.updatedAt = Date()
        } else {
            context.insert(StoredModelWeights(key: key, data: data))
        }
        try context.save()
    }

    public func modelWeights(key: String) throws -> Data? {
        var d = FetchDescriptor<StoredModelWeights>(predicate: #Predicate { $0.key == key })
        d.fetchLimit = 1
        return try context.fetch(d).first?.data
    }

    /// When the weights under `key` were last saved.
    public func modelWeightsUpdatedAt(key: String) throws -> Date? {
        var d = FetchDescriptor<StoredModelWeights>(predicate: #Predicate { $0.key == key })
        d.fetchLimit = 1
        return try context.fetch(d).first?.updatedAt
    }

    public func deleteAll() throws {
        try deleteRows(StoredSession.self)
        try deleteRows(StoredDecision.self)
        try deleteRows(StoredOutcome.self)
        try deleteRows(StoredLabelWindow.self)
        try deleteRows(StoredModelWeights.self)
        try deleteRows(StoredInsight.self)
        try deleteRows(StoredSARTRun.self)
        try context.save()
        Self.deleteTempFiles(in: tempDirectory)
        Self.deleteDefaults(defaults)
    }

    // MARK: Trial log (queryable)

    /// Logs a decision outside of a session save (e.g. live, as it happens). Idempotent by id.
    public func log(decision: NudgeDecision, sessionID: UUID? = nil) throws {
        try insertDecision(decision, sessionID: sessionID)
        try context.save()
    }

    /// Logs an outcome outside of a session save. Idempotent by decision id.
    public func log(outcome: NudgeOutcome, sessionID: UUID? = nil) throws {
        try insertOutcome(outcome, sessionID: sessionID)
        try context.save()
    }

    /// All logged decisions, oldest first (e.g. to rebuild bandit state or recompute IPW effects).
    public func decisions(policy: NudgePolicyKind? = nil) throws -> [NudgeDecision] {
        var d = FetchDescriptor<StoredDecision>(sortBy: [SortDescriptor(\.date), SortDescriptor(\.t)])
        if let raw = policy?.rawValue { d.predicate = #Predicate { $0.policy == raw } }
        return try context.fetch(d).map { try PersistenceBlobCodec.decode(NudgeDecision.self, from: $0.decisionBlob) }
    }

    /// All logged outcomes, oldest first.
    public func outcomes() throws -> [NudgeOutcome] {
        let d = FetchDescriptor<StoredOutcome>(sortBy: [SortDescriptor(\.t)])
        return try context.fetch(d).map {
            NudgeOutcome(decisionID: $0.decisionID, t: $0.t, reward: $0.reward, refocused: $0.refocused,
                         refocusSeconds: $0.refocusSeconds, dismissed: $0.dismissed, lapseAvoided: $0.lapseAvoided)
        }
    }

    // MARK: Labelled windows

    public func saveLabelWindow(_ w: LabeledWindow) throws {
        let id = w.id
        try deleteRows(StoredLabelWindow.self, where: #Predicate { $0.id == id })
        context.insert(StoredLabelWindow(id: w.id, sessionID: w.sessionID, start: w.start, end: w.end,
                                         label: w.label.rawValue, isLapse: w.isLapse, source: w.source.rawValue,
                                         frameCount: w.frames.count, createdAt: Date(),
                                         framesBlob: try PersistenceFrameCodec.encode(w.frames)))
        try context.save()
    }

    public func labelWindows() throws -> [LabeledWindow] {
        let d = FetchDescriptor<StoredLabelWindow>(sortBy: [SortDescriptor(\.createdAt), SortDescriptor(\.start)])
        return try context.fetch(d).map { row in
            LabeledWindow(id: row.id, sessionID: row.sessionID, start: row.start, end: row.end,
                          label: AttentionState(rawValue: row.label) ?? .focused, isLapse: row.isLapse,
                          source: LabelSource(rawValue: row.source) ?? .experienceSampling,
                          frames: try PersistenceFrameCodec.decode(row.framesBlob))
        }
    }

    // MARK: SART runs (sustained-attention profile)

    /// Saves (upserts by id) one completed SART run.
    public func saveSARTRun(_ run: SARTRunRecord) throws {
        let id = run.id
        try deleteRows(StoredSARTRun.self, where: #Predicate { $0.id == id })
        context.insert(StoredSARTRun(id: run.id, date: run.date, origin: run.origin.rawValue,
                                     trialCount: run.trials.count, runBlob: try PersistenceBlobCodec.encode(run)))
        try context.save()
    }

    /// Every saved SART run, oldest first. Rows that fail to decode are skipped.
    public func sartRuns() throws -> [SARTRunRecord] {
        let d = FetchDescriptor<StoredSARTRun>(sortBy: [SortDescriptor(\.date)])
        return try context.fetch(d).compactMap { try? PersistenceBlobCodec.decode(SARTRunRecord.self, from: $0.runBlob) }
    }

    /// Deletes SART runs by origin (e.g. the DEBUG `.sample` runs). Returns how many were removed.
    @discardableResult
    public func deleteSARTRuns(origin: SARTRunOrigin) throws -> Int {
        let raw = origin.rawValue
        let rows = try context.fetch(FetchDescriptor<StoredSARTRun>(predicate: #Predicate { $0.origin == raw }))
        for r in rows { context.delete(r) }
        try context.save()
        return rows.count
    }

    // MARK: Insights

    /// Saves cross-session coach insights (sessionID nil) or per-session ones. Upsert by id.
    public func saveInsights(_ insights: [Insight], sessionID: UUID? = nil) throws {
        for i in insights { try insertInsight(i, sessionID: sessionID) }
        try context.save()
    }

    /// Most recent first.
    public func insights(limit: Int? = nil) throws -> [Insight] {
        var d = FetchDescriptor<StoredInsight>(sortBy: [SortDescriptor(\.createdAt, order: .reverse)])
        d.fetchLimit = limit
        return try context.fetch(d).map {
            Insight(id: $0.id, text: $0.text, source: Insight.Source(rawValue: $0.source) ?? .template,
                    createdAt: $0.createdAt)
        }
    }

    // MARK: Inventory

    public func inventory() throws -> StorageInventory {
        try StorageInventory.compute(context: context, defaults: defaults, tempDirectory: tempDirectory)
    }

    // MARK: Helpers

    func storedSession(id: UUID) throws -> StoredSession? {
        var d = FetchDescriptor<StoredSession>(predicate: #Predicate { $0.id == id })
        d.fetchLimit = 1
        return try context.fetch(d).first
    }

    static func record(from row: StoredSession) throws -> SessionRecord {
        SessionRecord(id: row.id, startedAt: row.startedAt, duration: row.duration, lectureID: row.lectureID,
                      contentIsMock: row.contentIsMock, isReplay: row.isReplay,
                      ticks: try PersistenceBlobCodec.decode([SessionTick].self, from: row.ticksBlob),
                      events: try PersistenceBlobCodec.decode([SessionEvent].self, from: row.eventsBlob),
                      insights: try PersistenceBlobCodec.decode([Insight].self, from: row.insightsBlob),
                      sourceKind: row.sourceKind)
    }

    /// Time-weighted focused seconds: each tick covers [t, next t) (the last one up to `duration`, max 1 s).
    static func focusedSeconds(ticks: [SessionTick], duration: Double) -> Double {
        let s = ticks.sorted { $0.t < $1.t }
        var total = 0.0
        for (i, tick) in s.enumerated() where tick.state == .focused {
            let end = i + 1 < s.count ? s[i + 1].t : max(tick.t, min(duration, tick.t + 1))
            total += max(0, end - tick.t)
        }
        return total
    }

    private func insertDecision(_ d: NudgeDecision, sessionID: UUID?) throws {
        let id = d.id
        try deleteRows(StoredDecision.self, where: #Predicate { $0.id == id })
        context.insert(StoredDecision(id: d.id, sessionID: sessionID, t: d.t, date: d.date, policy: d.policy.rawValue,
                                      arm: d.arm.rawValue, state: d.context.state.rawValue,
                                      randomizationProbability: d.randomizationProbability,
                                      decisionBlob: try PersistenceBlobCodec.encode(d)))
    }

    private func insertOutcome(_ o: NudgeOutcome, sessionID: UUID?) throws {
        let id = o.decisionID
        try deleteRows(StoredOutcome.self, where: #Predicate { $0.decisionID == id })
        context.insert(StoredOutcome(decisionID: o.decisionID, sessionID: sessionID, t: o.t, reward: o.reward,
                                     refocused: o.refocused, refocusSeconds: o.refocusSeconds, dismissed: o.dismissed,
                                     lapseAvoided: o.lapseAvoided))
    }

    private func insertInsight(_ i: Insight, sessionID: UUID?) throws {
        let id = i.id
        try deleteRows(StoredInsight.self, where: #Predicate { $0.id == id })
        context.insert(StoredInsight(id: i.id, sessionID: sessionID, text: i.text, source: i.source.rawValue,
                                     createdAt: i.createdAt))
    }

    /// Fetch-and-delete (not batch delete) so pending inserts in the context are handled too.
    private func deleteRows<T: PersistentModel>(_ type: T.Type, where predicate: Predicate<T>? = nil) throws {
        for row in try context.fetch(FetchDescriptor<T>(predicate: predicate)) { context.delete(row) }
    }

    static func exportFiles(in dir: URL) -> [URL] {
        let fm = FileManager.default
        let items = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return items.filter { $0.lastPathComponent.lowercased().hasPrefix(tempFilePrefix) }
    }

    static func deleteTempFiles(in dir: URL) {
        for url in exportFiles(in: dir) { try? FileManager.default.removeItem(at: url) }
    }

    static func focusLoopDefaultsKeys(_ defaults: UserDefaults) -> [String] {
        defaults.dictionaryRepresentation().keys.filter { $0.hasPrefix(defaultsKeyPrefix) }.sorted()
    }

    static func deleteDefaults(_ defaults: UserDefaults) {
        for key in focusLoopDefaultsKeys(defaults) { defaults.removeObject(forKey: key) }
    }
}
