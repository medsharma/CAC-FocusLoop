import Foundation

/// PREVIEW / DEBUG ONLY (DEMONSTRATION). Stands in for the session engine so Study Mode can be
/// seen on its own: every value it writes is SYNTHETIC (derived from the MOCK content timeline,
/// not from any sensor), and it sets `model.isReplay = true` so the UI shows a SYNTHETIC badge.
/// The real engine replaces all of this by installing its own `model.actions`.
@MainActor
final class StudyModeDemoDriver {
    let model: LiveSessionModel
    let timeline: ContentTimeline
    private(set) var provider: StudyContentProvider
    private var lastCardPlayhead: Double = -.infinity
    private var recappedValleys: Set<Int> = []

    init(model: LiveSessionModel, lecture: LectureInfo, timeline: ContentTimeline) {
        self.model = model
        self.timeline = timeline
        self.provider = StudyContentProvider(lecture: lecture)
        model.isReplay = true
        model.studyMode = true
        model.timeline = timeline
        model.phase = .running
        var actions = LiveSessionModel.Actions()
        actions.setPlayhead = { [weak self] t in self?.tick(playhead: t) }
        actions.answerRecall = { [weak self] _ in self?.model.activePresentation = nil }
        actions.dismissNudge = { [weak self] in self?.model.activePresentation = nil }
        model.actions = actions
    }

    func tick(playhead t: Double) {
        model.playhead = t
        model.elapsed = t
        let valleys = timeline.valleys()
        let valleyIndex = StudyScrubberGeometry.valleyIndex(containing: t, in: valleys)
        // SYNTHETIC risk: logistic of the content drift-risk a few seconds ahead.
        let z = timeline.driftRisk(at: t + 6)
        let p = Float(1 / (1 + exp(-Double(z) * 1.4 + 1.0)))
        model.lapseRisk = LapseRisk(t: t, probability: p, threshold: 0.6, usedContent: true,
                                    source: "SYNTHETIC demo driver")
        let drifting = valleyIndex != nil && p > 0.7
        let probs: [Float] = drifting ? [0.22, 0.62, 0.10, 0.06] : [0.78, 0.12, 0.06, 0.04]
        model.estimate = StateEstimate(t: t, probabilities: probs, source: .heuristic)

        guard model.activePresentation == nil, t - lastCardPlayhead > 15 else { return }
        if timeline.upcomingValley(after: t, within: 10) != nil, let card = provider.recallCard(at: t) {
            present(card, arm: .recallQuestion, at: t)
        } else if let i = valleyIndex, !recappedValleys.contains(i), t - valleys[i].start > 4,
                  let card = provider.recapCard(at: t) {
            recappedValleys.insert(i)
            present(card, arm: .recapCard, at: t)
        }
    }

    private func present(_ card: NudgeCard, arm: NudgeArm, at t: Double) {
        lastCardPlayhead = t
        model.activeNudge = arm
        model.activePresentation = .card(card)
    }
}
