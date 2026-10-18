import Foundation

/// Dependency container for the session engine. Built by `AppServices`; every component is a protocol so
/// the engine runs against real sources on device, Replay in the Simulator, or mocks in tests.
@MainActor
final class EngineServices {
    var flags: FeatureFlags
    let classifier: AttentionClassifier
    let personalHead: PersonalHead?
    let forecaster: LapseForecaster
    let policy: InterventionPolicy
    let pacing: TrialPacingControl?
    let logger: TrialLogger
    let labels: LabelSink
    let makeFrameSource: (_ forceReplay: Bool) -> FeatureFrameSource
    let makeNudge: (NudgeArm, LectureInfo?, Double) -> Nudge
    let persistSession: (SessionRecord) -> Void
    /// Opt-in 10 Hz feature-stream recording for forecaster retraining (nil = never record).
    let streams: FeatureStreamStore?
    /// DEMONSTRATION ONLY: the fixed-nudge wrapper inside `policy`, switched on for Demo Director sessions.
    let demoScript: DemoScriptPolicy?

    init(flags: FeatureFlags, classifier: AttentionClassifier, personalHead: PersonalHead?, forecaster: LapseForecaster,
         policy: InterventionPolicy, pacing: TrialPacingControl?, logger: TrialLogger, labels: LabelSink,
         makeFrameSource: @escaping (Bool) -> FeatureFrameSource,
         makeNudge: @escaping (NudgeArm, LectureInfo?, Double) -> Nudge,
         persistSession: @escaping (SessionRecord) -> Void, streams: FeatureStreamStore? = nil,
         demoScript: DemoScriptPolicy? = nil) {
        self.flags = flags; self.classifier = classifier; self.personalHead = personalHead
        self.forecaster = forecaster; self.policy = policy; self.pacing = pacing; self.logger = logger
        self.labels = labels; self.makeFrameSource = makeFrameSource; self.makeNudge = makeNudge
        self.persistSession = persistSession
        self.streams = streams
        self.demoScript = demoScript
    }

    var pipelineComponents: PipelineComponents {
        PipelineComponents(classifier: classifier, personalHead: personalHead, forecaster: forecaster, policy: policy,
                           pacing: pacing, logger: logger, labels: labels, demoScript: demoScript)
    }
}

/// Components handed to `SessionPipeline`. While a session runs they are touched only from the pipeline actor.
struct PipelineComponents {
    let classifier: AttentionClassifier
    let personalHead: PersonalHead?
    let forecaster: LapseForecaster
    let policy: InterventionPolicy
    let pacing: TrialPacingControl?
    let logger: TrialLogger
    let labels: LabelSink
    var demoScript: DemoScriptPolicy? = nil
}

/// Default presenter: writes the presentation into the model; the UI's nudge host renders it.
@MainActor
final class ModelNudgePresenter: NudgePresenting {
    private weak var model: LiveSessionModel?
    init(model: LiveSessionModel) { self.model = model }
    func present(_ presentation: NudgePresentation, arm: NudgeArm) {
        model?.activePresentation = presentation
        model?.activeNudge = arm
    }
    func dismissNudge() {
        model?.activePresentation = nil
        model?.activeNudge = nil
    }
}
