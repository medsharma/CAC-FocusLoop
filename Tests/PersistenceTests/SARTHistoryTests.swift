import SwiftData
import XCTest
@testable import FocusLoop

@MainActor
final class SARTHistoryTests: XCTestCase {
    private var defaults: UserDefaults!
    private var suiteName: String!
    private var tempDir: URL!

    override func setUp() async throws {
        suiteName = "SARTHistoryTests-\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("sart-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        defaults.removePersistentDomain(forName: suiteName)
        try? FileManager.default.removeItem(at: tempDir)
    }

    private func makeStore() throws -> SwiftDataSessionStore {
        try SwiftDataSessionStore(inMemory: true, defaults: defaults, tempDirectory: tempDir)
    }

    nonisolated static func run(daysAgo: Double, origin: SARTRunOrigin = .device, commissions: Int = 1) -> SARTRunRecord {
        var trials: [SARTTrial] = []
        for i in 0..<18 {
            let digit = i % 9 + 1
            let onset = Double(i) * 1.15
            let noGo = digit == 3
            let tapped = noGo ? (i / 9) < commissions : true
            trials.append(SARTTrial(index: i, digit: digit, fontSize: 72, isNoGo: noGo, onset: onset,
                                    maskOnset: onset + 0.25, tapTime: tapped ? onset + 0.3 + Double(i % 4) * 0.02 : nil,
                                    end: onset + 1.15))
        }
        var config = SARTConfig()
        config.trialCount = trials.count
        return SARTRunRecord(date: Date(timeIntervalSince1970: 1_800_000_000 - daysAgo * 86_400), origin: origin,
                             config: config, trials: trials)
    }

    func testEveryRunIsKeptOldestFirstWithMetrics() throws {
        let store = try makeStore()
        let a = Self.run(daysAgo: 3, commissions: 2), b = Self.run(daysAgo: 1, origin: .simulator, commissions: 0)
        let c = Self.run(daysAgo: 2)
        for r in [a, b, c] { try store.saveSARTRun(r) }
        let runs = try store.sartRuns()
        XCTAssertEqual(runs.map(\.id), [a.id, c.id, b.id])
        XCTAssertEqual(runs.first, a, "Config, trials, origin and metrics round-trip exactly")
        XCTAssertEqual(runs.last?.origin, .simulator)
        XCTAssertEqual(runs.first?.metrics.commissionRate, 1)
        XCTAssertEqual(runs.last?.metrics.commissionRate, 0)
    }

    func testSaveIsAnUpsertByID() throws {
        let store = try makeStore()
        var r = Self.run(daysAgo: 1)
        try store.saveSARTRun(r)
        r.sensorsRecorded = true
        try store.saveSARTRun(r)
        XCTAssertEqual(try store.sartRuns(), [r])
    }

    func testDeleteAllRemovesRuns() throws {
        let store = try makeStore()
        try store.saveSARTRun(Self.run(daysAgo: 1))
        try store.saveSARTRun(Self.run(daysAgo: 2))
        try store.deleteAll()
        XCTAssertEqual(try store.sartRuns(), [])
        XCTAssertEqual(try store.inventory().entry(.sartRuns)?.count, 0)
    }

    func testDeleteByOriginOnlyRemovesSamples() throws {
        let store = try makeStore()
        let real = Self.run(daysAgo: 2)
        try store.saveSARTRun(real)
        try store.saveSARTRun(Self.run(daysAgo: 1, origin: .sample))
        XCTAssertEqual(try store.deleteSARTRuns(origin: .sample), 1)
        XCTAssertEqual(try store.sartRuns(), [real])
    }

    func testInventoryListsRunsAsNumbers() throws {
        let store = try makeStore()
        try store.saveSARTRun(Self.run(daysAgo: 1))
        try store.saveSARTRun(Self.run(daysAgo: 2))
        let inv = try store.inventory()
        let e = try XCTUnwrap(inv.entry(.sartRuns))
        XCTAssertEqual(e.count, 2)
        XCTAssertGreaterThan(e.bytes, 0)
        XCTAssertEqual(StorageInventory.Category.sartRuns.kind, .numeric)
        XCTAssertTrue(PrivacyView.storedCategories.contains { $0.1.contains("SART") && $0.2.contains("Numbers only") })
    }

    func testWeightsTimestampForLegacyImport() throws {
        let store = try makeStore()
        XCTAssertNil(try store.modelWeightsUpdatedAt(key: "sartTrials"))
        try store.saveModelWeights(Data([1]), key: "sartTrials")
        XCTAssertNotNil(try store.modelWeightsUpdatedAt(key: "sartTrials"))
    }
}
