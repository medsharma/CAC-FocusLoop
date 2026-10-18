import Foundation
@testable import FocusLoop

/// Deterministic content timeline for tests (MOCK).
final class PersistenceTestTimeline: ContentTimeline {
    let lectureID = "test-lecture"
    let isMock = true
    let duration: Double = 600
    let drift: (Double) -> Float
    init(drift: @escaping (Double) -> Float) { self.drift = drift }

    func activations(at t: Double) -> [Float] { (0..<7).map { Float($0) / 10 + Float(t.truncatingRemainder(dividingBy: 1000)) / 10_000 } }
    func networkValues(at t: Double) -> [Float] { activations(at: t) }
    func engagement(at t: Double) -> Float { 1 - drift(t) }
    func driftRisk(at t: Double) -> Float { drift(t) }
    func valleys() -> [(start: Double, end: Double)] { [] }
    func upcomingValley(after t: Double, within horizon: Double) -> (start: Double, end: Double)? { nil }
    func features(at t: Double) -> ContentFeatures {
        ContentFeatures(networks: networkValues(at: t), engagement: engagement(at: t), driftRisk: driftRisk(at: t),
                        secondsToValley: 60, inValley: false)
    }
}

enum PersistenceFixtures {
    static func probs(focused p: Float) -> [Float] {
        let rest = (1 - p) / 3
        return [p, rest, rest, rest]
    }

    static func tick(_ t: Double, _ state: AttentionState, pFocused: Float? = nil, hr: Float? = nil,
                     hrQ: Float? = nil, playhead: Double? = nil) -> SessionTick {
        let p = pFocused ?? (state == .focused ? 0.9 : 0.1)
        return SessionTick(t: t, state: state, probabilities: probs(focused: p), lapseRisk: 0.2, heartRate: hr,
                           heartRateQuality: hrQ, playhead: playhead, predictedEngagement: nil, predictedDriftRisk: nil)
    }

    static func decision(t: Double, policy: NudgePolicyKind, arm: NudgeArm, state: AttentionState = .distracted) -> NudgeDecision {
        let ctx = NudgeContext(t: t, state: state, lapseRisk: 0.6, predictedEngagement: 0.4, sessionMinute: Float(t / 60),
                               hourOfDay: 15, minutesSinceLastNudge: 2, studyMode: true, phonePickedUp: false,
                               upcomingValleyInSeconds: 12)
        return NudgeDecision(t: t, date: Date(timeIntervalSince1970: 1_800_000_000 + t), policy: policy, context: ctx,
                             arm: arm, randomizationProbability: 0.3, armProbabilities: arm == .noNudge ? ["noNudge": 0.3] : [arm.rawValue: 0.3, "noNudge": 0.1])
    }

    /// A 60 s SYNTHETIC-style record with every event kind.
    static func record(id: UUID = UUID(), startedAt: Date = Date(timeIntervalSince1970: 1_800_000_000)) -> SessionRecord {
        let ticks = (0..<60).map { i -> SessionTick in
            let state: AttentionState = (20..<30).contains(i) ? .distracted : .focused
            return tick(Double(i), state, hr: 70 + Float(i % 5), hrQ: 0.8, playhead: Double(i) + 10)
        }
        let d1 = decision(t: 22, policy: .reactive, arm: .haptic)
        let d2 = decision(t: 40, policy: .preemptive, arm: .preemptiveGlow, state: .focused)
        let o1 = NudgeOutcome(decisionID: d1.id, t: 30, reward: 1, refocused: true, refocusSeconds: 8, dismissed: false,
                              lapseAvoided: nil)
        let events: [SessionEvent] = [
            .stateChanged(t: 20, from: .focused, to: .distracted),
            .lapseDetected(t: 21, state: .distracted),
            .nudge(d1), .outcome(o1),
            .refocused(t: 30, afterSeconds: 10),
            .riskAlert(t: 39, probability: 0.7),
            .nudge(d2),
            .valleyEntered(t: 41, playhead: 51),
            .experienceSample(t: 50, focused: false, reason: .tired),
            .phonePickup(t: 55),
        ]
        return SessionRecord(id: id, startedAt: startedAt, duration: 60, lectureID: "test-lecture", contentIsMock: true,
                             isReplay: true, ticks: ticks, events: events,
                             insights: [Insight(text: "SYNTHETIC: you refocused in 10 s.", source: .template,
                                                createdAt: Date(timeIntervalSince1970: 1_800_000_100))])
    }

    static func labelWindow(sessionID: UUID) -> LabeledWindow {
        let frames = (0..<30).map { i -> FeatureFrame in
            FeatureFrame(t: 100 + Double(i) / 10, vector: (0..<42).map { Float($0) * 0.01 + Float(i) })
        }
        return LabeledWindow(sessionID: sessionID, start: 100, end: 103, label: .drowsy, isLapse: true, source: .sart,
                             frames: frames)
    }
}
