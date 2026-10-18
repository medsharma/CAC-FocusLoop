import SwiftUI

/// Full-screen lecture player for Study Mode.
///
/// * Plays the bundled video (AVPlayer) or, when `lecture.videoFile` is nil/missing, a timer-driven
///   "no video" narration stage. Playback works on its own; every ~0.1 s the playhead is also sent
///   to the engine via `model.actions.setPlayhead`.
/// * Scrubber with predicted valley markers + predicted-engagement sparkline.
/// * Corner cortex (predicted brain response to this content, average brain) whose edge pulses
///   when a predicted valley is within 10 s or the playhead is inside one. Tap to expand.
/// * Renders `.card` presentations from `model.activePresentation`; pauses for recall questions.
public struct StudyModeView: View {
    let model: LiveSessionModel
    let lecture: LectureInfo
    let timeline: ContentTimeline
    var onClose: (() -> Void)?

    @State private var playback: StudyPlaybackController
    @State private var provider: StudyContentProvider
    @State private var engagementSamples: [Float]
    @State private var isVisible = false
    @State private var showCortexSheet = false
    @State private var showCortexInfo = false
    @State private var pausedForCard = false
    @State private var handledCardID: UUID?
    @State private var pausedForBackground = false
    @State private var didCloseAtEnd = false
    @Environment(\.scenePhase) private var scenePhase

    private let autoplay: Bool
    private let initialPlayhead: Double
    /// Scripted demo (SYNTHETIC): call `onClose` by itself when the lecture reaches its end.
    private let closeWhenFinished: Bool
    /// Scripted demo (SYNTHETIC): an unanswered card continues on its own after this many seconds.
    private let autoContinueCardsAfter: Double?

    public init(model: LiveSessionModel, lecture: LectureInfo, timeline: ContentTimeline,
                autoplay: Bool = false, initialPlayhead: Double = 0, closeWhenFinished: Bool = false,
                autoContinueCardsAfter: Double? = nil, onClose: (() -> Void)? = nil) {
        self.model = model
        self.lecture = lecture
        self.timeline = timeline
        self.onClose = onClose
        self.autoplay = autoplay
        self.initialPlayhead = initialPlayhead
        self.closeWhenFinished = closeWhenFinished
        self.autoContinueCardsAfter = autoContinueCardsAfter
        let controller = StudyPlaybackController(lecture: lecture)
        _playback = State(initialValue: controller)
        _provider = State(initialValue: StudyContentProvider(lecture: lecture))
        _engagementSamples = State(initialValue: StudyScrubberGeometry.samples(
            count: 120, duration: controller.duration) { timeline.engagement(at: $0) })
    }

    private var playhead: Double { playback.playhead }

    private var activeCard: NudgeCard? {
        guard case let .card(card)? = model.activePresentation, card.id != handledCardID else { return nil }
        return card
    }

    private var valleyWarning: Bool {
        timeline.upcomingValley(after: playhead, within: 10) != nil
            || StudyScrubberGeometry.valleyIndex(containing: playhead, in: timeline.valleys()) != nil
    }

    public var body: some View {
        ZStack {
            FLColor.bg.ignoresSafeArea()

            VStack(spacing: 0) {
                header
                    .padding(.horizontal, FLSpacing.gutter)
                    .padding(.top, FLSpacing.xs)
                    .padding(.bottom, FLSpacing.m)

                stage
                    .padding(.horizontal, FLSpacing.m)

                controls
                    .padding(.horizontal, FLSpacing.gutter)
                    .padding(.top, FLSpacing.l)

                Spacer(minLength: FLSpacing.m)

                lowerSection
                    .padding(.horizontal, FLSpacing.gutter)
                    .padding(.bottom, FLSpacing.xs)
            }

        }
        .overlay { cardLayer }
        .animation(FLMotion.standard, value: activeCard?.id)
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showCortexSheet) {
            StudyCortexSheet(timeline: timeline, playback: playback)
        }
        .onAppear {
            isVisible = true
            playback.onTick = { [model] t in model.actions.setPlayhead(t) }
            if initialPlayhead > 0, playback.playhead == 0 {
                playback.seek(to: initialPlayhead)
            } else {
                model.actions.setPlayhead(playback.playhead)
            }
            if autoplay { playback.play() }
            #if DEBUG
            if ProcessInfo.processInfo.arguments.contains("-studyOpenCortex") { showCortexSheet = true }
            #endif
        }
        .onDisappear {
            isVisible = false
            playback.pause()
            StudyAudioSession.restoreAfterLecture()
        }
        .onChange(of: activeCard?.id) { _, _ in handleCardChange() }
        .onChange(of: scenePhase) { _, phase in handleScenePhase(phase) }
        .onChange(of: playback.isAtEnd) { _, atEnd in
            // Scripted demo: the lecture is over, so end the session (saves it and opens its report).
            guard atEnd, closeWhenFinished, !didCloseAtEnd, let onClose else { return }
            didCloseAtEnd = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { onClose() }
        }
        .task(id: activeCard?.id) {
            // Scripted demo: an unanswered card continues by itself, so the demo runs hands-free.
            guard let after = autoContinueCardsAfter, let card = activeCard else { return }
            try? await Task.sleep(for: .seconds(after))
            guard !Task.isCancelled, activeCard?.id == card.id else { return }
            answer(card, choice: -1)
        }
    }

    /// The app went to the background: pause the lecture (the session engine pauses the session and its
    /// sensors); resume it on return if it was playing.
    private func handleScenePhase(_ phase: ScenePhase) {
        switch phase {
        case .background:
            if playback.isPlaying { playback.pause(); pausedForBackground = true }
        case .active:
            if pausedForBackground, activeCard?.kind != .recall { playback.play() }
            pausedForBackground = false
        default: break
        }
    }

    // MARK: Sections

    @ViewBuilder private var cardLayer: some View {
        if let card = activeCard {
            ZStack {
                Color.black.opacity(0.62)
                    .ignoresSafeArea()
                    .transition(.opacity)
                StudyModeCardOverlay(card: card,
                                     onAnswer: { answer(card, choice: $0) },
                                     onDismiss: { dismiss(card) },
                                     footnote: autoContinueCardsAfter.map {
                                         "Continues automatically in \(Int($0)) s"
                                     })
                    .id(card.id)
                    .padding(.horizontal, FLSpacing.gutter)
                    .transition(.scale(scale: 0.97).combined(with: .opacity))
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: FLSpacing.s) {
            if let onClose {
                FLIconButton("chevron.down", label: "Close Study Mode", identifier: "study.close", action: onClose)
            }
            VStack(alignment: .leading, spacing: 4) {
                FLEyebrow("Study Mode")
                Text(lecture.title)
                    .flText(.headline)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: FLSpacing.xs)
            StudyStateIndicator(state: model.state, hasEstimate: model.estimate != nil, risk: model.lapseRisk)
                .fixedSize()
                .layoutPriority(1)
        }
    }

    private var stage: some View {
        let shape = RoundedRectangle(cornerRadius: StudyModeStyle.stageRadius, style: .continuous)
        return ZStack {
            if let player = playback.player {
                StudyVideoSurface(player: player)
            } else {
                StudyNoVideoStage(lecture: lecture, recap: provider.currentRecap(at: playhead),
                                  progress: playback.progress, isPlaying: playback.isPlaying)
            }
        }
        .aspectRatio(playback.hasVideo ? 16 / 9 : 16 / 10, contentMode: .fit)
        .clipShape(shape)
        .overlay(shape.strokeBorder(FLColor.hairline, lineWidth: 0.5))
        .overlay(alignment: .top) { refocusToast.padding(.top, FLSpacing.s) }
        .contentShape(Rectangle())
        .onTapGesture { playback.toggle() }
        .accessibilityAction(named: playback.isPlaying ? "Pause" : "Play") { playback.toggle() }
    }

    /// "Back in X.X s" for a few seconds after the student refocuses (the same number the Focus screen shows).
    @ViewBuilder private var refocusToast: some View {
        let value = RefocusBanner.value(lastRefocusSeconds: model.lastRefocusSeconds,
                                        lastRefocusAt: model.lastRefocusAt, elapsed: model.elapsed)
        ZStack {
            if let v = value {
                HStack(spacing: FLSpacing.xs) {
                    Circle().fill(FLColor.focused).frame(width: 6, height: 6)
                    Text(UIFormat.backIn(v))
                        .font(FLFont.numeral(.subheadline, weight: .medium))
                        .foregroundStyle(FLColor.textPrimary)
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(FLColor.bgElevated.opacity(0.92), in: Capsule())
                .overlay(Capsule().strokeBorder(FLColor.hairlineStrong, lineWidth: 0.5))
                .transition(.scale(scale: 0.9).combined(with: .opacity))
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Refocused. \(UIFormat.backIn(v))")
            }
        }
        .animation(FLMotion.standard, value: value != nil)
        .allowsHitTesting(false)
    }

    private var controls: some View {
        VStack(spacing: FLSpacing.s) {
            HStack(alignment: .firstTextBaseline) {
                FLEyebrow("Predicted engagement")
                Spacer(minLength: FLSpacing.xs)
                StudyValleyStatusLine(timeline: timeline, playhead: playhead, highlight: valleyWarning)
            }

            StudyScrubber(duration: playback.duration, playhead: playhead, valleys: timeline.valleys(),
                          engagement: engagementSamples, onSeek: { playback.seek(to: $0) })

            HStack {
                Text(StudyScrubberGeometry.timeLabel(playhead))
                Spacer()
                HStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(StudyModeStyle.valley.opacity(StudyModeStyle.valleyBandOpacity + 0.15))
                        .frame(width: 10, height: 10)
                    Text("Predicted valley")
                }
                Spacer()
                Text("-" + StudyScrubberGeometry.timeLabel(playback.duration - playhead))
            }
            .font(FLFont.numeral(.caption))
            .foregroundStyle(FLColor.textTertiary)

            HStack(spacing: FLSpacing.xxl) {
                transportButton("gobackward.10", label: "Back 10 seconds") { playback.skip(by: -10) }
                Button { playback.toggle() } label: {
                    Image(systemName: playback.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 22, weight: .regular))
                        .foregroundStyle(FLColor.bg)
                        .contentTransition(.symbolEffect(.replace))
                        .frame(width: 64, height: 64)
                        .background(FLColor.textPrimary, in: Circle())
                }
                .buttonStyle(FLPressableStyle())
                .accessibilityLabel(playback.isPlaying ? "Pause" : "Play")
                transportButton("goforward.10", label: "Forward 10 seconds") { playback.skip(by: 10) }
            }
            .padding(.top, FLSpacing.s)
        }
    }

    private func transportButton(_ symbol: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 22, weight: .regular))
                .foregroundStyle(FLColor.textSecondary)
                .frame(width: 48, height: 48)
        }
        .buttonStyle(FLPressableStyle())
        .accessibilityLabel(label)
    }

    private var lowerSection: some View {
        VStack(alignment: .leading, spacing: FLSpacing.s) {
            HStack(alignment: .top, spacing: FLSpacing.l) {
                // Recap: plain text, no card.
                VStack(alignment: .leading, spacing: FLSpacing.xs) {
                    FLEyebrow(panelTitle)
                    Text(panelText)
                        .font(FLFont.callout)
                        .foregroundStyle(FLColor.textPrimary.opacity(0.88))
                        .lineSpacing(3)
                        .fixedSize(horizontal: false, vertical: true)
                        .animation(FLMotion.fade, value: panelText)
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)

                StudyCortexCorner(activations: timeline.activations(at: playhead), warning: valleyWarning,
                                  isMock: timeline.isMock, isLive: cortexIsLive, rotating: playback.isPlaying,
                                  onTap: { showCortexSheet = true })
                    .animation(.easeInOut(duration: 0.4), value: valleyWarning)
            }

            // One-line caption under the cortex, plus the single quiet info button.
            HStack(spacing: 0) {
                Spacer(minLength: 0)
                Text(StudyCortexCopy.short)
                    .font(.caption2)
                    .foregroundStyle(FLColor.textTertiary)
                    .lineLimit(1)
                    .accessibilityLabel(StudyCortexCopy.caption)
                StudyCortexInfoButton(isPreview: timeline.isMock, attribution: lecture.attribution,
                                      isPresented: $showCortexInfo)
            }
            .padding(.top, -FLSpacing.xxs)

            FLDivider()
            Text(lecture.attribution)
                .font(.caption2)
                .foregroundStyle(FLColor.textTertiary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }

    /// With video: the current recap. Without video the stage already shows it, so preview what's next.
    private var panelTitle: String { playback.hasVideo ? "In this part" : "Up next" }
    private var panelText: String {
        if playback.hasVideo { return provider.currentRecap(at: playhead)?.text ?? "Press play to begin." }
        let next = lecture.recaps.filter { $0.start > playhead }.min { $0.start < $1.start }
        return next?.text ?? "That's the end of this lecture."
    }

    /// The Metal cortex renders only on screen, uncovered, in the foreground, and while the phone is cool
    /// (thermal state serious or worse pauses it to a still placeholder).
    private var cortexIsLive: Bool {
        isVisible && !showCortexSheet && scenePhase == .active && !ThermalStatus.shared.cortexPaused
    }

    // MARK: Cards

    private func handleCardChange() {
        if let card = activeCard {
            if card.kind == .recall, playback.isPlaying {
                playback.pause()
                pausedForCard = true
            }
        } else {
            resumeIfPausedForCard()
        }
    }

    private func answer(_ card: NudgeCard, choice: Int) {
        handledCardID = card.id
        model.actions.answerRecall(choice)
        resumeIfPausedForCard()
    }

    private func dismiss(_ card: NudgeCard) {
        handledCardID = card.id
        model.actions.dismissNudge()
        resumeIfPausedForCard()
    }

    private func resumeIfPausedForCard() {
        guard pausedForCard else { return }
        pausedForCard = false
        playback.play()
    }
}

// MARK: - Previews

private struct StudyModePreviewHost: View {
    @State private var model = LiveSessionModel()
    @State private var driver: StudyModeDemoDriver?
    let lecture = StudyLectureLibrary.sampleLecture
    let timeline = StudyLectureLibrary.timeline(for: StudyLectureLibrary.sampleLecture)

    var body: some View {
        Group {
            if let timeline {
                StudyModeView(model: model, lecture: lecture, timeline: timeline, autoplay: true, onClose: {})
            } else {
                Text("sample_lecture.brain.json missing").foregroundStyle(.secondary)
            }
        }
        .onAppear {
            if driver == nil, let timeline {
                driver = StudyModeDemoDriver(model: model, lecture: lecture, timeline: timeline)
            }
        }
    }
}

#Preview("Study Mode (preview model, demo engine)") {
    StudyModePreviewHost()
}

#Preview("Recall card") {
    StudyModeCardOverlay(card: NudgeCard(kind: .recall, title: "Quick check",
                                         body: "Why do most leaves look green?",
                                         choices: ["They absorb green light", "They reflect green light"],
                                         correctIndex: 1),
                         onAnswer: { _ in }, onDismiss: {})
        .padding()
        .background(FLColor.bg)
}
