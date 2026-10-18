import SwiftUI
import XCTest
@testable import FocusLoop

@MainActor
private final class RecordingPresenter: NudgePresenting {
    var presented: [(NudgePresentation, NudgeArm)] = []
    var dismissals = 0
    func present(_ presentation: NudgePresentation, arm: NudgeArm) { presented.append((presentation, arm)) }
    func dismissNudge() { dismissals += 1 }
}

@MainActor
final class NudgeTests: XCTestCase {
    private let lecture = LectureInfo(
        id: "test", title: "Test", subject: "Bio", source: "Test", sourceURL: "", license: "CC-BY-4.0",
        attribution: "Test", videoFile: nil, durationSeconds: 300,
        recaps: [.init(start: 0, end: 60, text: "Recap A"), .init(start: 60, end: 120, text: "Recap B")],
        recallQuestions: [.init(after: 30, question: "Q1?", choices: ["a", "b"], correctIndex: 1),
                          .init(after: 90, question: "Q2?", choices: ["x", "y", "z"], correctIndex: 2)])

    func testFactoryBuildsEveryArm() {
        for arm in NudgeArm.allCases {
            XCTAssertEqual(NudgeFactory.make(arm: arm, lecture: lecture, playhead: 50).arm, arm)
        }
    }

    func testPresentationsPerArm() {
        let expected: [NudgeArm: NudgePresentation] = [
            .ambientGlow: .glow(intensity: 0.7, preemptive: false),
            .preemptiveGlow: .glow(intensity: 0.4, preemptive: true),
            .stretchBreak: .stretch(seconds: 60),
            .lightPulse: .lightPulse,
            .pacedBreathing: .breathing(targetBreathsPerMinute: 6),
        ]
        for (arm, presentation) in expected {
            let p = RecordingPresenter()
            let nudge = NudgeFactory.make(arm: arm)
            nudge.start(presenter: p)
            XCTAssertEqual(p.presented.count, 1)
            XCTAssertEqual(p.presented.first?.0, presentation)
            XCTAssertEqual(p.presented.first?.1, arm)
            nudge.cancel()
            XCTAssertEqual(p.dismissals, 1)
        }
        let p = RecordingPresenter()
        NudgeFactory.make(arm: .noNudge).start(presenter: p)
        XCTAssertTrue(p.presented.isEmpty)
    }

    func testCards() {
        guard case let .card(put) = PutMeDownNudge().presentation else { return XCTFail() }
        XCTAssertEqual(put.kind, .putMeDown)

        XCTAssertEqual(RecapCardNudge.card(lecture: lecture, playhead: 70).body, "Recap B")
        XCTAssertEqual(RecapCardNudge.card(lecture: lecture, playhead: 10).body, "Recap A")
        XCTAssertEqual(RecapCardNudge.card(lecture: lecture, playhead: 200).body, "Recap B")   // latest passed
        XCTAssertEqual(RecapCardNudge.card(lecture: lecture, playhead: 10).kind, .recap)
        XCTAssertFalse(RecapCardNudge.card(lecture: nil, playhead: 10).body.isEmpty)

        let early = RecallQuestionNudge.card(lecture: lecture, playhead: 10)
        XCTAssertTrue(early.choices.isEmpty)   // nothing covered yet: generic prompt
        let q1 = RecallQuestionNudge.card(lecture: lecture, playhead: 45)
        XCTAssertEqual(q1.body, "Q1?"); XCTAssertEqual(q1.correctIndex, 1); XCTAssertEqual(q1.kind, .recall)
        let q2 = RecallQuestionNudge.card(lecture: lecture, playhead: 95)
        XCTAssertEqual(q2.body, "Q2?"); XCTAssertEqual(q2.choices, ["x", "y", "z"]); XCTAssertEqual(q2.correctIndex, 2)
    }

    func testChimeBufferAndHapticPattern() throws {
        let buf = try XCTUnwrap(ChimeNudge.makeBuffer())
        XCTAssertEqual(Int(buf.frameLength), Int(44_100 * 1.4))
        let data = try XCTUnwrap(buf.floatChannelData?[0])
        let peak = (0..<Int(buf.frameLength)).map { abs(data[$0]) }.max()!
        XCTAssertGreaterThan(peak, 0.05)
        XCTAssertLessThan(peak, 0.5)               // soft
        XCTAssertLessThan(abs(data[Int(buf.frameLength) - 1]), 0.001)   // faded out, no click
        XCTAssertNoThrow(try HapticNudge.makePattern())
        // Starting sensory nudges must be safe in the Simulator (no-op / silent failure).
        let p = RecordingPresenter()
        let h = HapticNudge(); h.start(presenter: p); h.cancel()
        XCTAssertTrue(p.presented.isEmpty)
    }

    func testViewHelpers() {
        // 6 breaths/min: 10 s cycle, inhale 4 s, exhale 6 s.
        let a = BreathingGuideView.phase(elapsed: 2, breathsPerMinute: 6)
        XCTAssertTrue(a.inhale); XCTAssertEqual(a.secondsLeft, 2, accuracy: 1e-9)
        let b = BreathingGuideView.phase(elapsed: 7, breathsPerMinute: 6)
        XCTAssertFalse(b.inhale); XCTAssertEqual(b.secondsLeft, 3, accuracy: 1e-9)
        XCTAssertEqual(BreathingGuideView.phase(elapsed: 4, breathsPerMinute: 6).level, 1, accuracy: 1e-9)
        XCTAssertEqual(BreathingGuideView.phase(elapsed: 10, breathsPerMinute: 6).level, 0, accuracy: 1e-9)
        XCTAssertEqual(StretchCountdownView.promptIndex(elapsed: 0, total: 60), 0)
        XCTAssertEqual(StretchCountdownView.promptIndex(elapsed: 31, total: 60), 2)
        XCTAssertEqual(StretchCountdownView.promptIndex(elapsed: 60, total: 60), 3)

        // Views instantiate and render a body without crashing.
        let presentations: [NudgePresentation] = [.glow(intensity: 0.5, preemptive: true), .lightPulse,
                                                  .card(PutMeDownNudge.card), .breathing(targetBreathsPerMinute: 6),
                                                  .stretch(seconds: 60)]
        for p in presentations {
            let host = UIHostingController(rootView: NudgePresentationView(presentation: p, bpmHistory: [(0, 70), (1, 71)],
                                                                           onDismiss: {}))
            host.view.frame = CGRect(x: 0, y: 0, width: 390, height: 844)
            host.view.layoutIfNeeded()
        }
    }
}
