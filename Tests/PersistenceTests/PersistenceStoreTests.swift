import SwiftData
import XCTest
@testable import FocusLoop

@MainActor
final class PersistenceStoreTests: XCTestCase {
    private var defaults: UserDefaults!
    private var suiteName: String!
    private var tempDir: URL!

    override func setUp() async throws {
        suiteName = "PersistenceStoreTests-\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("persist-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        defaults.removePersistentDomain(forName: suiteName)
        try? FileManager.default.removeItem(at: tempDir)
    }

    private func makeStore() throws -> SwiftDataSessionStore {
        try SwiftDataSessionStore(inMemory: true, defaults: defaults, tempDirectory: tempDir)
    }

    func testContainerIsInMemoryAndLocalOnly() throws {
        let container = try PersistenceContainer.make(inMemory: true)
        let config = try XCTUnwrap(container.configurations.first)
        XCTAssertTrue(config.isStoredInMemoryOnly)
        XCTAssertNil(config.cloudKitContainerIdentifier)
        XCTAssertEqual(Set(container.schema.entities.map(\.name)),
                       ["StoredSession", "StoredDecision", "StoredOutcome", "StoredLabelWindow",
                        "StoredModelWeights", "StoredInsight", "StoredSARTRun"])
    }

    func testSessionRoundTripAndUpsert() throws {
        let store = try makeStore()
        let rec = PersistenceFixtures.record()
        try store.save(rec)

        XCTAssertEqual(try store.session(id: rec.id), rec)
        XCTAssertEqual(try store.sessions(), [rec])
        XCTAssertNil(try store.session(id: UUID()))

        // Trial log exploded into queryable rows.
        XCTAssertEqual(try store.decisions().count, 2)
        XCTAssertEqual(try store.decisions(policy: .preemptive).map(\.arm), [.preemptiveGlow])
        XCTAssertEqual(try store.outcomes().map(\.refocusSeconds), [8])
        XCTAssertEqual(try store.insights().map(\.text), rec.insights.map(\.text))

        // Upsert: same id, changed content -> still one session, rows replaced not duplicated.
        var edited = rec
        edited.duration = 61
        edited.events.removeAll { if case .nudge = $0 { return true } else { return false } }
        try store.save(edited)
        XCTAssertEqual(try store.sessions(), [edited])
        XCTAssertEqual(try store.decisions().count, 0)
        XCTAssertEqual(try store.outcomes().count, 1)
        XCTAssertEqual(try store.insights().count, 1)

        // Second session, newest first.
        let later = PersistenceFixtures.record(startedAt: rec.startedAt.addingTimeInterval(3600))
        try store.save(later)
        XCTAssertEqual(try store.sessions().map(\.id), [later.id, rec.id])
    }

    func testNonFiniteValuesSurviveRoundTrip() throws {
        let store = try makeStore()
        var rec = PersistenceFixtures.record()
        rec.ticks[0].lapseRisk = .nan
        rec.ticks[1].heartRate = .infinity
        try store.save(rec)
        let back = try XCTUnwrap(try store.session(id: rec.id))
        XCTAssertTrue(back.ticks[0].lapseRisk.isNaN)
        XCTAssertEqual(back.ticks[1].heartRate, .infinity)
    }

    func testModelWeightsRoundTripAndOverwrite() throws {
        let store = try makeStore()
        XCTAssertNil(try store.modelWeights(key: PersistenceWeightKey.banditState))
        try store.saveModelWeights(Data([1, 2, 3]), key: PersistenceWeightKey.banditState)
        try store.saveModelWeights(Data([9]), key: PersistenceWeightKey.personalHead)
        try store.saveModelWeights(Data([4, 5]), key: PersistenceWeightKey.banditState)
        XCTAssertEqual(try store.modelWeights(key: PersistenceWeightKey.banditState), Data([4, 5]))
        XCTAssertEqual(try store.modelWeights(key: PersistenceWeightKey.personalHead), Data([9]))
        XCTAssertEqual(try store.context.fetch(FetchDescriptor<StoredModelWeights>()).count, 2)
    }

    func testLabelWindowRoundTripIsBitExact() throws {
        let store = try makeStore()
        let w = PersistenceFixtures.labelWindow(sessionID: UUID())
        try store.saveLabelWindow(w)
        XCTAssertEqual(try store.labelWindows(), [w])
        try store.saveLabelWindow(w)   // idempotent by id
        XCTAssertEqual(try store.labelWindows().count, 1)
    }

    func testLiveTrialLogging() throws {
        let store = try makeStore()
        let d = PersistenceFixtures.decision(t: 5, policy: .reactive, arm: .chime)
        try store.log(decision: d)
        try store.log(outcome: NudgeOutcome(decisionID: d.id, t: 12, reward: 0, refocused: false, refocusSeconds: nil,
                                            dismissed: true, lapseAvoided: nil))
        XCTAssertEqual(try store.decisions(), [d])
        XCTAssertEqual(try store.outcomes().first?.dismissed, true)
    }

    func testDeleteAllWipesEverything() throws {
        let store = try makeStore()
        let rec = PersistenceFixtures.record()
        try store.save(rec)
        try store.saveLabelWindow(PersistenceFixtures.labelWindow(sessionID: rec.id))
        try store.saveModelWeights(Data([1]), key: PersistenceWeightKey.workZone)
        try store.saveInsights([Insight(text: "cross-session", source: .template)])
        defaults.set(true, forKey: "focusloop.flags.demoDirector")
        defaults.set("x", forKey: "focusloop.onboarding")
        defaults.set(1, forKey: "unrelated.key")
        let export = tempDir.appendingPathComponent("FocusLoop-Report-test.pdf")
        let csv = tempDir.appendingPathComponent("focusloop-labels.csv")
        let other = tempDir.appendingPathComponent("someone-else.txt")
        for url in [export, csv, other] { try Data([0]).write(to: url) }

        let before = try store.inventory()
        XCTAssertGreaterThan(before.totalBytes, 0)
        XCTAssertEqual(before.pendingExportFiles, 2)

        try store.deleteAll()

        XCTAssertEqual(try store.sessions(), [])
        XCTAssertEqual(try store.labelWindows(), [])
        XCTAssertEqual(try store.decisions(), [])
        XCTAssertEqual(try store.outcomes(), [])
        XCTAssertEqual(try store.insights(), [])
        XCTAssertNil(try store.modelWeights(key: PersistenceWeightKey.workZone))
        func count<T: PersistentModel>(_ type: T.Type) throws -> Int {
            try store.context.fetchCount(FetchDescriptor<T>())
        }
        XCTAssertEqual(try count(StoredSession.self), 0)
        XCTAssertEqual(try count(StoredDecision.self), 0)
        XCTAssertEqual(try count(StoredOutcome.self), 0)
        XCTAssertEqual(try count(StoredLabelWindow.self), 0)
        XCTAssertEqual(try count(StoredModelWeights.self), 0)
        XCTAssertEqual(try count(StoredInsight.self), 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: export.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: csv.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: other.path))
        XCTAssertNil(defaults.object(forKey: "focusloop.flags.demoDirector"))
        XCTAssertNil(defaults.object(forKey: "focusloop.onboarding"))
        XCTAssertEqual(defaults.integer(forKey: "unrelated.key"), 1)

        let after = try store.inventory()
        XCTAssertEqual(after.entries.map(\.count), Array(repeating: 0, count: StorageInventory.Category.allCases.count))
        XCTAssertEqual(after.pendingExportFiles, 0)
    }

    func testInventoryCountsAndNoMediaCategories() throws {
        let store = try makeStore()
        let rec = PersistenceFixtures.record()
        try store.save(rec)
        try store.save(PersistenceFixtures.record())
        try store.saveLabelWindow(PersistenceFixtures.labelWindow(sessionID: rec.id))
        try store.saveModelWeights(Data(count: 100), key: PersistenceWeightKey.personalHead)
        defaults.set(true, forKey: "focusloop.flags")

        let inv = try store.inventory()
        XCTAssertEqual(inv.entry(.sessions)?.count, 2)
        XCTAssertEqual(inv.entry(.sessionTicks)?.count, 120)
        XCTAssertEqual(inv.entry(.sessionEvents)?.count, 2 * rec.events.count)
        XCTAssertEqual(inv.entry(.trialDecisions)?.count, 4)
        XCTAssertEqual(inv.entry(.trialOutcomes)?.count, 2)
        XCTAssertEqual(inv.entry(.labelWindows)?.count, 1)
        XCTAssertEqual(inv.entry(.labelFrames)?.count, 30)
        XCTAssertEqual(inv.entry(.modelWeights)?.count, 1)
        XCTAssertGreaterThanOrEqual(inv.entry(.modelWeights)?.bytes ?? 0, 100)
        XCTAssertEqual(inv.entry(.insights)?.count, 2)
        XCTAssertEqual(inv.entry(.preferences)?.count, 1)
        XCTAssertTrue(inv.entries.allSatisfy { $0.count == 0 || $0.bytes > 0 })

        // Proof by construction: categories are numeric / generated text / preferences only.
        XCTAssertEqual(inv.videoBytes, 0)
        XCTAssertEqual(inv.imageBytes, 0)
        XCTAssertEqual(Set(StorageInventory.storedKinds.map(\.rawValue)), ["numeric", "generatedText", "preferences"])
        XCTAssertEqual(inv.entries.map(\.category), StorageInventory.Category.allCases)
        XCTAssertEqual(StorageInventory.Category.allCases.filter { $0.kind == .numeric }.count, 9)
    }

    func testHistoryQueries() throws {
        let store = try makeStore()
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let now = Date(timeIntervalSince1970: 1_800_000_000)   // Fri 2027-01-15 08:00 UTC
        let today = cal.startOfDay(for: now)
        let r1 = PersistenceFixtures.record(startedAt: today.addingTimeInterval(3600))          // today
        let r2 = PersistenceFixtures.record(startedAt: today.addingTimeInterval(-86_400 + 60))  // yesterday
        let r3 = PersistenceFixtures.record(startedAt: today.addingTimeInterval(-20 * 86_400))  // weeks ago
        for r in [r1, r2, r3] { try store.save(r) }
        let q = SessionHistoryQueries(store: store)

        XCTAssertEqual(try q.lastSessions(2).map(\.id), [r1.id, r2.id])
        let focusedSeconds = 50.0   // fixture: 10 of 60 ticks distracted
        XCTAssertEqual(try q.lastSessions(1).first?.focusedSeconds, focusedSeconds)
        XCTAssertEqual(try q.lastSessions(1).first?.focusedFraction ?? 0, 50.0 / 60, accuracy: 1e-9)
        XCTAssertEqual(try q.lastRecords(1), [r1])

        let week = try q.sessionsThisWeek(now: now, calendar: cal)
        XCTAssertEqual(Set(week.map(\.id)), [r1.id, r2.id])

        let days = try q.focusedMinutesPerDay(days: 7, now: now, calendar: cal)
        XCTAssertEqual(days.count, 7)
        XCTAssertEqual(days.last?.day, today)
        XCTAssertEqual(days.last?.focusedMinutes ?? 0, focusedSeconds / 60, accuracy: 1e-9)
        XCTAssertEqual(days[5].sessionCount, 1)
        XCTAssertTrue(days[5].includesSynthetic)
        XCTAssertEqual(days.prefix(5).map(\.focusedMinutes), [0, 0, 0, 0, 0])
    }
}
