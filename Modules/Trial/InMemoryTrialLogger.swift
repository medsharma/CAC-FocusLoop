import Foundation

/// Simple in-memory `TrialLogger`. Thread-safe (lock-protected) so the UI can read while the engine writes.
public final class InMemoryTrialLogger: TrialLogger, @unchecked Sendable {
    private let lock = NSLock()
    private var _decisions: [NudgeDecision] = []
    private var _outcomes: [NudgeOutcome] = []

    public init() {}

    public func log(decision: NudgeDecision) {
        lock.lock(); defer { lock.unlock() }
        _decisions.append(decision)
    }

    public func log(outcome: NudgeOutcome) {
        lock.lock(); defer { lock.unlock() }
        _outcomes.append(outcome)
    }

    public var decisions: [NudgeDecision] {
        lock.lock(); defer { lock.unlock() }
        return _decisions
    }

    public var outcomes: [NudgeOutcome] {
        lock.lock(); defer { lock.unlock() }
        return _outcomes
    }

    /// Decisions paired with their outcomes (nil while pending), in decision order.
    public var records: [TrialRecord] {
        lock.lock(); defer { lock.unlock() }
        let byID = Dictionary(_outcomes.map { ($0.decisionID, $0) }, uniquingKeysWith: { _, b in b })
        return _decisions.map { TrialRecord(decision: $0, outcome: byID[$0.id]) }
    }

    public func removeAll() {
        lock.lock(); defer { lock.unlock() }
        _decisions.removeAll()
        _outcomes.removeAll()
    }
}
