import Foundation
import SwiftData

// SwiftData models for FocusLoop (iOS 17). Numbers only: there is deliberately no model, attribute
// or blob that holds an image, video frame, audio or face mesh. Enum values are stored as raw strings
// so they are queryable with `#Predicate`.
//
// Bulky per-session series (1 Hz ticks, events, insights) are stored as LZFSE-compressed JSON blobs
// on `StoredSession`; the trial log (decisions/outcomes) is additionally exploded into queryable rows.

/// One finished session. `ticksBlob`/`eventsBlob`/`insightsBlob` hold compressed JSON of the
/// corresponding `SessionRecord` arrays (see `PersistenceBlobCodec`).
@Model
public final class StoredSession {
    @Attribute(.unique) public var id: UUID
    public var startedAt: Date
    public var duration: Double
    public var lectureID: String?
    /// True if the lecture brain timeline was MOCK.
    public var contentIsMock: Bool
    /// True if the session ran on the SYNTHETIC replay stream.
    public var isReplay: Bool
    /// Seconds spent in `focused` (from ticks), denormalised for fast history queries.
    public var focusedSeconds: Double
    public var tickCount: Int
    public var eventCount: Int
    public var ticksBlob: Data
    public var eventsBlob: Data
    public var insightsBlob: Data
    /// `SessionRecord.sourceKind` ("recorded" for a RECORDED-video demonstration; nil otherwise).
    public var sourceKind: String?

    public init(id: UUID, startedAt: Date, duration: Double, lectureID: String?, contentIsMock: Bool,
                isReplay: Bool, focusedSeconds: Double, tickCount: Int, eventCount: Int,
                ticksBlob: Data, eventsBlob: Data, insightsBlob: Data, sourceKind: String? = nil) {
        self.sourceKind = sourceKind
        self.id = id; self.startedAt = startedAt; self.duration = duration; self.lectureID = lectureID
        self.contentIsMock = contentIsMock; self.isReplay = isReplay; self.focusedSeconds = focusedSeconds
        self.tickCount = tickCount; self.eventCount = eventCount
        self.ticksBlob = ticksBlob; self.eventsBlob = eventsBlob; self.insightsBlob = insightsBlob
    }
}

/// Queryable micro-randomised-trial decision (one row per `NudgeDecision`).
@Model
public final class StoredDecision {
    @Attribute(.unique) public var id: UUID
    public var sessionID: UUID?
    public var t: Double
    public var date: Date
    /// `NudgePolicyKind.rawValue`.
    public var policy: String
    /// `NudgeArm.rawValue`.
    public var arm: String
    /// `AttentionState.rawValue` at decision time.
    public var state: String
    public var randomizationProbability: Double
    /// Compressed JSON of the full `NudgeDecision` (context + arm probabilities).
    public var decisionBlob: Data

    public init(id: UUID, sessionID: UUID?, t: Double, date: Date, policy: String, arm: String, state: String,
                randomizationProbability: Double, decisionBlob: Data) {
        self.id = id; self.sessionID = sessionID; self.t = t; self.date = date; self.policy = policy
        self.arm = arm; self.state = state; self.randomizationProbability = randomizationProbability
        self.decisionBlob = decisionBlob
    }
}

/// Queryable trial outcome (one row per `NudgeOutcome`), keyed by its decision.
@Model
public final class StoredOutcome {
    @Attribute(.unique) public var decisionID: UUID
    public var sessionID: UUID?
    public var t: Double
    public var reward: Double
    public var refocused: Bool
    public var refocusSeconds: Double?
    public var dismissed: Bool
    public var lapseAvoided: Bool?

    public init(decisionID: UUID, sessionID: UUID?, t: Double, reward: Double, refocused: Bool,
                refocusSeconds: Double?, dismissed: Bool, lapseAvoided: Bool?) {
        self.decisionID = decisionID; self.sessionID = sessionID; self.t = t; self.reward = reward
        self.refocused = refocused; self.refocusSeconds = refocusSeconds; self.dismissed = dismissed
        self.lapseAvoided = lapseAvoided
    }
}

/// A labelled window of 10 Hz numeric feature frames (`PersistenceFrameCodec` binary, compressed).
@Model
public final class StoredLabelWindow {
    @Attribute(.unique) public var id: UUID
    public var sessionID: UUID
    public var start: Double
    public var end: Double
    /// `AttentionState.rawValue`.
    public var label: String
    public var isLapse: Bool
    /// `LabelSource.rawValue`.
    public var source: String
    public var frameCount: Int
    public var createdAt: Date
    public var framesBlob: Data

    public init(id: UUID, sessionID: UUID, start: Double, end: Double, label: String, isLapse: Bool,
                source: String, frameCount: Int, createdAt: Date, framesBlob: Data) {
        self.id = id; self.sessionID = sessionID; self.start = start; self.end = end; self.label = label
        self.isLapse = isLapse; self.source = source; self.frameCount = frameCount; self.createdAt = createdAt
        self.framesBlob = framesBlob
    }
}

/// Opaque learned state keyed by name (see `PersistenceWeightKey`).
@Model
public final class StoredModelWeights {
    @Attribute(.unique) public var key: String
    public var data: Data
    public var updatedAt: Date

    public init(key: String, data: Data, updatedAt: Date = Date()) {
        self.key = key; self.data = data; self.updatedAt = updatedAt
    }
}

/// AI-coach insight text (generated from aggregated numbers only). `sessionID` nil = cross-session.
@Model
public final class StoredInsight {
    @Attribute(.unique) public var id: UUID
    public var sessionID: UUID?
    public var text: String
    /// `Insight.Source.rawValue`.
    public var source: String
    public var createdAt: Date

    public init(id: UUID, sessionID: UUID?, text: String, source: String, createdAt: Date) {
        self.id = id; self.sessionID = sessionID; self.text = text; self.source = source; self.createdAt = createdAt
    }
}

/// One completed SART run (sustained-attention profile). `runBlob` is compressed JSON of `SARTRunRecord`:
/// the date, task settings, per-trial digit / tap / reaction time, and the computed scores. Numbers only.
@Model
public final class StoredSARTRun {
    @Attribute(.unique) public var id: UUID
    public var date: Date
    /// `SARTRunOrigin.rawValue` ("device", "simulator", or "sample" for DEBUG sample runs).
    public var origin: String
    public var trialCount: Int
    public var runBlob: Data

    public init(id: UUID, date: Date, origin: String, trialCount: Int, runBlob: Data) {
        self.id = id; self.date = date; self.origin = origin; self.trialCount = trialCount; self.runBlob = runBlob
    }
}

/// Well-known keys for `SessionStore.saveModelWeights(_:key:)`. Any string works; these are conventions.
public enum PersistenceWeightKey {
    public static let personalHead = "personalHead"
    public static let banditState = "banditState"
    public static let forecasterThreshold = "forecasterThreshold"
    public static let heartRateBaseline = "heartRateBaseline"
    public static let workZone = "workZone"
}

/// Builds the `ModelContainer`. Always local: CloudKit is explicitly off, so nothing leaves the device.
public enum PersistenceContainer {
    /// Every persistent model type. `StorageInventory` covers exactly these.
    public static let modelTypes: [any PersistentModel.Type] = [
        StoredSession.self, StoredDecision.self, StoredOutcome.self,
        StoredLabelWindow.self, StoredModelWeights.self, StoredInsight.self, StoredSARTRun.self,
    ]

    public static var schema: Schema { Schema(modelTypes) }

    /// - Parameter inMemory: true for tests and previews (`isStoredInMemoryOnly`); each in-memory
    ///   container gets a unique name so containers never share state.
    public static func make(inMemory: Bool = false) throws -> ModelContainer {
        let config = ModelConfiguration(inMemory ? "FocusLoop-mem-\(UUID().uuidString)" : "FocusLoop",
                                        schema: schema,
                                        isStoredInMemoryOnly: inMemory,
                                        allowsSave: true,
                                        groupContainer: .none,
                                        cloudKitDatabase: .none)
        return try ModelContainer(for: schema, configurations: [config])
    }
}

// MARK: - Codecs

public enum PersistenceError: Error, Equatable {
    case compressionFailed
    case corruptFrameBlob
}

/// JSON + LZFSE. Non-finite floats are encoded as strings so a stray NaN never fails a save.
public enum PersistenceBlobCodec {
    private static let nonFinite = (pos: "inf", neg: "-inf", nan: "nan")

    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        let enc = JSONEncoder()
        enc.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: nonFinite.pos,
                                                                  negativeInfinity: nonFinite.neg, nan: nonFinite.nan)
        return try compress(enc.encode(value))
    }

    public static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let dec = JSONDecoder()
        dec.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: nonFinite.pos,
                                                                    negativeInfinity: nonFinite.neg, nan: nonFinite.nan)
        return try dec.decode(type, from: decompress(data))
    }

    public static func compress(_ data: Data) throws -> Data {
        if data.isEmpty { return data }
        do { return try (data as NSData).compressed(using: .lzfse) as Data }
        catch { throw PersistenceError.compressionFailed }
    }

    public static func decompress(_ data: Data) throws -> Data {
        if data.isEmpty { return data }
        return try (data as NSData).decompressed(using: .lzfse) as Data
    }
}

/// Compact numeric frame blob: per frame, little-endian Float64 `t` then the 42 `FeatureFrame.vector`
/// Float32 values; the whole buffer is LZFSE-compressed. Header: "FLF1" + UInt16 column count.
public enum PersistenceFrameCodec {
    private static let magic: [UInt8] = Array("FLF1".utf8)

    public static func encode(_ frames: [FeatureFrame]) throws -> Data {
        let cols = FeatureFrame.featureNames.count
        var out = Data(magic)
        withUnsafeBytes(of: UInt16(cols).littleEndian) { out.append(contentsOf: $0) }
        out.reserveCapacity(6 + frames.count * (8 + 4 * cols))
        for f in frames {
            withUnsafeBytes(of: f.t.bitPattern.littleEndian) { out.append(contentsOf: $0) }
            for v in f.vector {
                withUnsafeBytes(of: v.bitPattern.littleEndian) { out.append(contentsOf: $0) }
            }
        }
        return try PersistenceBlobCodec.compress(out)
    }

    public static func decode(_ data: Data) throws -> [FeatureFrame] {
        let raw = [UInt8](try PersistenceBlobCodec.decompress(data))
        guard raw.count >= 6, Array(raw[0..<4]) == magic else { throw PersistenceError.corruptFrameBlob }
        let cols = Int(UInt16(raw[4]) | UInt16(raw[5]) << 8)
        guard cols == FeatureFrame.featureNames.count else { throw PersistenceError.corruptFrameBlob }
        let stride = 8 + 4 * cols
        let body = raw.count - 6
        guard body % stride == 0 else { throw PersistenceError.corruptFrameBlob }
        func u64(_ o: Int) -> UInt64 { (0..<8).reduce(0) { $0 | UInt64(raw[o + $1]) << (8 * UInt64($1)) } }
        func u32(_ o: Int) -> UInt32 { (0..<4).reduce(0) { $0 | UInt32(raw[o + $1]) << (8 * UInt32($1)) } }
        var frames: [FeatureFrame] = []
        frames.reserveCapacity(body / stride)
        var o = 6
        while o < raw.count {
            let t = Double(bitPattern: u64(o))
            var v = [Float](repeating: 0, count: cols)
            for c in 0..<cols { v[c] = Float(bitPattern: u32(o + 8 + 4 * c)) }
            frames.append(FeatureFrame(t: t, vector: v))
            o += stride
        }
        return frames
    }
}
