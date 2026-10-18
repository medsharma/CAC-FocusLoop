import XCTest
@testable import FocusLoop

final class RewardTrackerTests: XCTestCase {
    private func decision(arm: NudgeArm = .haptic, policy: NudgePolicyKind = .reactive, t: Double = 100) -> NudgeDecision {
        NudgeDecision(t: t, policy: policy, context: TrialTestSupport.context(t: t), arm: arm,
                      randomizationProbability: 0.5,
                      armProbabilities: arm == .noNudge ? ["haptic": 0.5, "noNudge": 0.5] : [arm.rawValue: 0.5, "noNudge": 0.5])
    }

    /// Feeds 1 Hz states starting at `from`; returns all emitted outcomes.
    private func feed(_ tracker: RewardTracker, _ states: [AttentionState], from: Double, step: Double = 1) -> [NudgeOutcome] {
        var out: [NudgeOutcome] = []
        for (i, s) in states.enumerated() { out += tracker.observe(state: s, t: from + Double(i) * step) }
        return out
    }

    func testReactiveSuccessWithRefocusSeconds() {
        let tr = RewardTracker()
        let d = decision()
        tr.track(d)
        // distracted 100..105, focused from 106 on; sustained 2 s at 108.
        let out = feed(tr, [.distracted, .distracted, .distracted, .distracted, .distracted, .distracted, .focused, .focused, .focused],
                       from: 100)
        XCTAssertEqual(out.count, 1)
        let o = out[0]
        XCTAssertEqual(o.decisionID, d.id)
        XCTAssertEqual(o.reward, 1)
        XCTAssertTrue(o.refocused)
        XCTAssertEqual(o.refocusSeconds, 6)
        XCTAssertEqual(o.t, 108)
        XCTAssertFalse(o.dismissed)
        XCTAssertNil(o.lapseAvoided)
        XCTAssertTrue(tr.pendingDecisions.isEmpty)
    }

    func testBriefFocusBlipDoesNotCount() {
        let tr = RewardTracker()
        tr.track(decision())
        // 1 s focused blip at 103, then distracted; then sustained focus starting at 110.
        var states: [AttentionState] = [.distracted, .distracted, .distracted, .focused, .distracted]
        states += Array(repeating: .distracted, count: 5) + [.focused, .focused, .focused]
        let out = feed(tr, states, from: 100)
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].refocusSeconds, 10)
        XCTAssertEqual(out[0].reward, 1)
    }

    func testReactiveFailureAfterWindow() {
        let tr = RewardTracker()
        tr.track(decision())
        let out = feed(tr, Array(repeating: .distracted, count: 25), from: 100)
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].reward, 0)
        XCTAssertFalse(out[0].refocused)
        XCTAssertNil(out[0].refocusSeconds)
        XCTAssertEqual(out[0].t, 121)   // first observation past the 20 s window
    }

    func testFocusStartingInsideWindowCanCompleteJustAfterIt() {
        let tr = RewardTracker()
        tr.track(decision())
        // focus starts at 119.5 (inside the 20 s window), sustained 2 s at 121.5.
        var out: [NudgeOutcome] = []
        out += tr.observe(state: .distracted, t: 110)
        out += tr.observe(state: .focused, t: 119.5)
        out += tr.observe(state: .focused, t: 120.5)
        XCTAssertTrue(out.isEmpty)
        out += tr.observe(state: .focused, t: 121.5)
        XCTAssertEqual(out.first?.reward, 1)
        XCTAssertEqual(out.first?.refocusSeconds ?? 0, 19.5, accuracy: 1e-9)
    }

    func testDismissalPenalty() {
        let tr = RewardTracker()
        let d = decision(arm: .putMeDown)
        tr.track(d)
        tr.noteDismissal()
        let out = feed(tr, [.focused, .focused, .focused], from: 101)
        XCTAssertEqual(out.first?.reward, 0.5)   // 1 - 0.5 penalty
        XCTAssertEqual(out.first?.dismissed, true)
        // Dismissal on a failure clamps at 0.
        let tr2 = RewardTracker()
        let d2 = decision(arm: .chime)
        tr2.track(d2)
        tr2.noteDismissal(decisionID: d2.id)
        let out2 = feed(tr2, Array(repeating: .distracted, count: 22), from: 100)
        XCTAssertEqual(out2.first?.reward, 0)
        XCTAssertEqual(out2.first?.dismissed, true)
    }

    func testNoNudgeControlIsTrackedIdentically() {
        let tr = RewardTracker()
        let d = decision(arm: .noNudge)
        tr.track(d)
        tr.noteDismissal()   // nothing to dismiss for the control: ignored
        let out = feed(tr, [.distracted, .distracted, .focused, .focused, .focused], from: 100)
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].reward, 1)
        XCTAssertEqual(out[0].refocusSeconds, 2)
        XCTAssertFalse(out[0].dismissed)
    }

    func testPreemptiveLapseAvoided() {
        let tr = RewardTracker()
        let d = decision(arm: .preemptiveGlow, policy: .preemptive)
        tr.track(d)
        // A 1 s wobble is not a lapse; outcome at the 15 s horizon.
        var states = Array(repeating: AttentionState.focused, count: 16)
        states[5] = .distracted
        let out = feed(tr, states, from: 100)
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].reward, 1)
        XCTAssertEqual(out[0].lapseAvoided, true)
        XCTAssertEqual(out[0].t, 115)
        XCTAssertFalse(out[0].refocused)
    }

    func testPreemptiveLapseOccurredEmitsEarly() {
        let tr = RewardTracker()
        tr.track(decision(arm: .recallQuestion, policy: .preemptive))
        let out = feed(tr, [.focused, .focused, .drowsy, .drowsy, .drowsy], from: 100)
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].reward, 0)
        XCTAssertEqual(out[0].lapseAvoided, false)
        XCTAssertEqual(out[0].t, 104)
    }

    func testExplicitLapseAndFlush() {
        let tr = RewardTracker()
        let pre = decision(arm: .noNudge, policy: .preemptive, t: 0)
        let re = decision(arm: .haptic, policy: .reactive, t: 0)
        tr.track(pre)
        tr.track(re)
        tr.noteLapse(t: 4)
        let out = tr.observe(state: .focused, t: 5)
        XCTAssertEqual(out.map(\.decisionID), [pre.id])
        XCTAssertEqual(out.first?.lapseAvoided, false)
        // Session ends 1 s into the reactive focus run: not sustained -> failure.
        let flushed = tr.flush(t: 6)
        XCTAssertEqual(flushed.map(\.decisionID), [re.id])
        XCTAssertEqual(flushed.first?.reward, 0)
        XCTAssertTrue(tr.pendingDecisions.isEmpty)
    }

    func testOverlappingDecisionsResolveIndependently() {
        let tr = RewardTracker(pacing: .demo)
        let a = decision(arm: .haptic, t: 0)
        let b = decision(arm: .noNudge, t: 12)
        tr.track(a)
        _ = feed(tr, Array(repeating: .distracted, count: 12), from: 0)
        tr.track(b)
        let out = feed(tr, Array(repeating: .distracted, count: 4) + [.focused, .focused, .focused], from: 12)
        // a: window [0, 20], focus starts at 16 -> success at 18 with refocus 16 s.
        // b: focus starts at 16 -> refocus 4 s.
        XCTAssertEqual(Set(out.map(\.decisionID)), [a.id, b.id])
        XCTAssertEqual(out.first { $0.decisionID == a.id }?.refocusSeconds, 16)
        XCTAssertEqual(out.first { $0.decisionID == b.id }?.refocusSeconds, 4)
    }
}
