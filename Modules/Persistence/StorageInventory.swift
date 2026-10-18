import Foundation
import SwiftData

/// What FocusLoop keeps on this device, for the Privacy screen ("an exact list of what is stored
/// locally"). Every category is numeric (or short coach text generated from aggregated numbers);
/// the type system has no image/video/audio category at all, so `videoBytes`/`imageBytes` are 0
/// by construction and the categories list proves it.
public struct StorageInventory: Equatable, Sendable {
    public enum ContentKind: String, Sendable, CaseIterable {
        /// Numbers only (timestamps, probabilities, features, counters, model parameters).
        case numeric
        /// Short sentences generated on-device from aggregated numbers (AI coach insights).
        case generatedText
        /// App preferences (flags, onboarding progress).
        case preferences
    }

    public enum Category: String, Sendable, CaseIterable, Identifiable {
        case sessions, sessionTicks, sessionEvents, trialDecisions, trialOutcomes
        case labelWindows, labelFrames, sartRuns, modelWeights, insights, preferences
        public var id: String { rawValue }

        public var title: String {
            switch self {
            case .sessions: "Sessions"
            case .sessionTicks: "1-second attention summaries"
            case .sessionEvents: "Session events"
            case .trialDecisions: "Personal-trial decisions"
            case .trialOutcomes: "Personal-trial outcomes"
            case .labelWindows: "Labelled windows"
            case .labelFrames: "Labelled feature frames (10 Hz)"
            case .sartRuns: "Sustained-attention (SART) runs"
            case .modelWeights: "Learned model settings"
            case .insights: "Coach insights"
            case .preferences: "App preferences"
            }
        }

        public var detail: String {
            switch self {
            case .sessions: "Start time, duration, lecture ID, MOCK/SYNTHETIC flags"
            case .sessionTicks: "State probabilities, lapse risk, heart rate, playhead (numbers)"
            case .sessionEvents: "State changes, refocus times, nudges, risk alerts (numbers)"
            case .trialDecisions: "Which nudge was chosen, context numbers, randomisation probability"
            case .trialOutcomes: "Reward, refocus seconds, dismissed flag"
            case .labelWindows: "Label, start/end time, source"
            case .labelFrames: "42 numeric features per frame (blinks, gaze angles, head pose, heart rate)"
            case .sartRuns: "Date, task settings, per-trial digit, tap and reaction time, computed scores (numbers)"
            case .modelWeights: "Personal head, bandit state, thresholds, heart-rate baseline, work zone"
            case .insights: "Short text generated from aggregated numbers only"
            case .preferences: "Feature flags and onboarding progress (focusloop.* keys)"
            }
        }

        public var kind: ContentKind {
            switch self {
            case .insights: .generatedText
            case .preferences: .preferences
            default: .numeric
            }
        }
    }

    public struct Entry: Equatable, Sendable, Identifiable {
        public var category: Category
        public var count: Int
        /// Approximate bytes on disk (blob sizes plus fixed-size scalar fields).
        public var bytes: Int
        public var id: String { category.id }
    }

    public var entries: [Entry]
    /// User-initiated report/CSV exports currently in the temp dir (deleted by Delete all data).
    public var pendingExportFiles: Int
    public var pendingExportBytes: Int

    /// Always 0: no model, blob or category can hold video. Shown as "0 bytes of video stored or sent."
    public let videoBytes = 0
    /// Always 0: no stored category holds images. (A report the user explicitly exports may embed a
    /// rendered cortex picture; it lives only in the temp dir until shared, see `pendingExportBytes`.)
    public let imageBytes = 0

    public var totalBytes: Int { entries.reduce(0) { $0 + $1.bytes } }
    public func entry(_ c: Category) -> Entry? { entries.first { $0.category == c } }

    /// Only these content kinds exist; there is no image/video/audio kind to report.
    public static let storedKinds = ContentKind.allCases

    @MainActor
    public static func compute(context: ModelContext, defaults: UserDefaults = .standard,
                               tempDirectory: URL = FileManager.default.temporaryDirectory) throws -> StorageInventory {
        let sessions = try context.fetch(FetchDescriptor<StoredSession>())
        let decisions = try context.fetch(FetchDescriptor<StoredDecision>())
        let outcomes = try context.fetch(FetchDescriptor<StoredOutcome>())
        let windows = try context.fetch(FetchDescriptor<StoredLabelWindow>())
        let weights = try context.fetch(FetchDescriptor<StoredModelWeights>())
        let insights = try context.fetch(FetchDescriptor<StoredInsight>())
        let sart = try context.fetch(FetchDescriptor<StoredSARTRun>())

        let uuid = 16, double = 8, int = 8, bool = 1, date = 8
        var e: [Entry] = []
        e.append(Entry(category: .sessions, count: sessions.count,
                       bytes: sessions.reduce(0) { $0 + uuid + date + 2 * double + 2 * bool + 2 * int
                           + ($1.lectureID?.utf8.count ?? 0) + $1.insightsBlob.count }))
        e.append(Entry(category: .sessionTicks, count: sessions.reduce(0) { $0 + $1.tickCount },
                       bytes: sessions.reduce(0) { $0 + $1.ticksBlob.count }))
        e.append(Entry(category: .sessionEvents, count: sessions.reduce(0) { $0 + $1.eventCount },
                       bytes: sessions.reduce(0) { $0 + $1.eventsBlob.count }))
        e.append(Entry(category: .trialDecisions, count: decisions.count,
                       bytes: decisions.reduce(0) { $0 + 2 * uuid + double + date + double + $1.policy.utf8.count
                           + $1.arm.utf8.count + $1.state.utf8.count + $1.decisionBlob.count }))
        e.append(Entry(category: .trialOutcomes, count: outcomes.count,
                       bytes: outcomes.count * (2 * uuid + 3 * double + 3 * bool)))
        e.append(Entry(category: .labelWindows, count: windows.count,
                       bytes: windows.reduce(0) { $0 + 2 * uuid + 2 * double + bool + int + date
                           + $1.label.utf8.count + $1.source.utf8.count }))
        e.append(Entry(category: .labelFrames, count: windows.reduce(0) { $0 + $1.frameCount },
                       bytes: windows.reduce(0) { $0 + $1.framesBlob.count }))
        e.append(Entry(category: .sartRuns, count: sart.count,
                       bytes: sart.reduce(0) { $0 + uuid + date + int + $1.origin.utf8.count + $1.runBlob.count }))
        e.append(Entry(category: .modelWeights, count: weights.count,
                       bytes: weights.reduce(0) { $0 + $1.key.utf8.count + $1.data.count + date }))
        e.append(Entry(category: .insights, count: insights.count,
                       bytes: insights.reduce(0) { $0 + uuid + date + $1.text.utf8.count + $1.source.utf8.count }))

        let keys = SwiftDataSessionStore.focusLoopDefaultsKeys(defaults)
        let prefBytes = keys.reduce(0) { acc, k in
            let v = defaults.object(forKey: k)
            let size = v.flatMap { try? PropertyListSerialization.data(fromPropertyList: $0, format: .binary,
                                                                       options: 0).count } ?? 0
            return acc + k.utf8.count + size
        }
        e.append(Entry(category: .preferences, count: keys.count, bytes: prefBytes))

        let files = SwiftDataSessionStore.exportFiles(in: tempDirectory)
        let fileBytes = files.reduce(0) { $0 + ((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
        return StorageInventory(entries: e, pendingExportFiles: files.count, pendingExportBytes: fileBytes)
    }

    public init(entries: [Entry], pendingExportFiles: Int, pendingExportBytes: Int) {
        self.entries = entries
        self.pendingExportFiles = pendingExportFiles
        self.pendingExportBytes = pendingExportBytes
    }
}
