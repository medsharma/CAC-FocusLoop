import Foundation

// Concrete nudges. Each is a small @MainActor object: `start` asks the presenter to show something
// (or plays a sound/haptic itself), `cancel` tears it down. Copy is wellness-framed and gentle.

/// Base class for nudges that only hand a presentation to the UI.
@MainActor
public class PresentationNudge: Nudge {
    public let arm: NudgeArm
    public let presentation: NudgePresentation
    private weak var presenter: NudgePresenting?

    public init(arm: NudgeArm, presentation: NudgePresentation) {
        self.arm = arm
        self.presentation = presentation
    }

    public func start(presenter: NudgePresenting) {
        self.presenter = presenter
        presenter.present(presentation, arm: arm)
    }

    public func cancel() {
        presenter?.dismissNudge()
        presenter = nil
    }
}

/// Distracted: a soft ambient glow around the screen edge.
@MainActor
public final class AmbientGlowNudge: PresentationNudge {
    public init(intensity: Float = 0.7) {
        super.init(arm: .ambientGlow, presentation: .glow(intensity: intensity, preemptive: false))
    }
}

/// Pre-emptive: a softer, slower glow shown when lapse risk is rising.
@MainActor
public final class PreemptiveGlowNudge: PresentationNudge {
    public init(intensity: Float = 0.4) {
        super.init(arm: .preemptiveGlow, presentation: .glow(intensity: intensity, preemptive: true))
    }
}

/// Distracted + phone picked up: a gentle "put me down" card.
@MainActor
public final class PutMeDownNudge: PresentationNudge {
    public static let card = NudgeCard(kind: .putMeDown, title: "Put me down?",
                                       body: "Your study session is still going. Set the phone down and pick up where you left off.")
    public init() {
        super.init(arm: .putMeDown, presentation: .card(Self.card))
    }
}

/// Distracted in Study Mode: a short recap of the lecture section at the playhead.
@MainActor
public final class RecapCardNudge: PresentationNudge {
    public init(lecture: LectureInfo?, playhead: Double) {
        super.init(arm: .recapCard, presentation: .card(Self.card(lecture: lecture, playhead: playhead)))
    }

    /// The recap covering `playhead`, else the most recent one already passed, else a generic prompt.
    public static func card(lecture: LectureInfo?, playhead: Double) -> NudgeCard {
        let recaps = lecture?.recaps ?? []
        let current = recaps.first { $0.start <= playhead && playhead < $0.end }
            ?? recaps.filter { $0.end <= playhead }.max { $0.end < $1.end }
        let body = current?.text ?? "Take a breath. What was the last idea in this lecture, in your own words?"
        return NudgeCard(kind: .recap, title: "Quick recap", body: body)
    }
}

/// Drowsy: a 60-second stand-and-stretch countdown.
@MainActor
public final class StretchBreakNudge: PresentationNudge {
    public init(seconds: Int = 60) {
        super.init(arm: .stretchBreak, presentation: .stretch(seconds: seconds))
    }
}

/// Drowsy: a bright, cool-light pulse.
/// EXPERIMENTAL arm under test: the trial measures whether it helps this user. Never claim it works.
@MainActor
public final class LightPulseNudge: PresentationNudge {
    public init() {
        super.init(arm: .lightPulse, presentation: .lightPulse)
    }
}

/// Stressed: guided breathing at about 6 breaths/min with live heart-rate biofeedback.
@MainActor
public final class PacedBreathingNudge: PresentationNudge {
    public init(breathsPerMinute: Double = 6) {
        super.init(arm: .pacedBreathing, presentation: .breathing(targetBreathsPerMinute: breathsPerMinute))
    }
}

/// Pre-emptive in Study Mode: an active-recall question about content already covered
/// (the latest `RecallQuestion` with `after <= playhead`).
@MainActor
public final class RecallQuestionNudge: PresentationNudge {
    public init(lecture: LectureInfo?, playhead: Double) {
        super.init(arm: .recallQuestion, presentation: .card(Self.card(lecture: lecture, playhead: playhead)))
    }

    public static func card(lecture: LectureInfo?, playhead: Double) -> NudgeCard {
        let q = (lecture?.recallQuestions ?? []).filter { $0.after <= playhead }.max { $0.after < $1.after }
        guard let q else {
            return NudgeCard(kind: .recall, title: "Quick check",
                             body: "What is the main idea of the last minute? Say it to yourself in one sentence.")
        }
        return NudgeCard(kind: .recall, title: "Quick check", body: q.question, choices: q.choices,
                         correctIndex: q.choices.indices.contains(q.correctIndex) ? q.correctIndex : nil)
    }
}

/// Control arm: does nothing (the decision and its outcome are still logged for causal estimates).
@MainActor
public final class NoNudge: Nudge {
    public let arm: NudgeArm = .noNudge
    public init() {}
    public func start(presenter: NudgePresenting) {}
    public func cancel() {}
}

/// Builds the concrete nudge for an arm.
@MainActor
public enum NudgeFactory {
    public static func make(arm: NudgeArm, lecture: LectureInfo? = nil, playhead: Double = 0) -> Nudge {
        switch arm {
        case .ambientGlow: AmbientGlowNudge()
        case .haptic: HapticNudge()
        case .chime: ChimeNudge()
        case .putMeDown: PutMeDownNudge()
        case .recapCard: RecapCardNudge(lecture: lecture, playhead: playhead)
        case .stretchBreak: StretchBreakNudge()
        case .lightPulse: LightPulseNudge()
        case .pacedBreathing: PacedBreathingNudge()
        case .preemptiveGlow: PreemptiveGlowNudge()
        case .recallQuestion: RecallQuestionNudge(lecture: lecture, playhead: playhead)
        case .noNudge: NoNudge()
        }
    }
}
