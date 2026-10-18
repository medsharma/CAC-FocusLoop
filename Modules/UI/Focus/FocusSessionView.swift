import SwiftUI

/// Immersive focus screen: orb, timer, "Back in X.X s", lapse-risk halo, nudge host.
public struct FocusSessionView: View {
    let model: LiveSessionModel
    let services: UIServices
    /// Called after the session is stopped, with the saved session id (nil if too short to save).
    var onFinish: (UUID?) -> Void
    var onOpenScience: () -> Void

    @State private var confirmEnd = false
    @State private var showBreathing = false
    @State private var refocusPulse = 0

    public init(model: LiveSessionModel, services: UIServices, onFinish: @escaping (UUID?) -> Void,
                onOpenScience: @escaping () -> Void) {
        self.model = model; self.services = services; self.onFinish = onFinish; self.onOpenScience = onOpenScience
    }

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// A session driven by the user's own recorded face video (compile-safe until `model.isRecorded` lands).
    private var faceVideo: AnyView? { model.isReplay ? nil : services.faceVideoView() }

    public var body: some View {
        ZStack {
            FLBackground(glow: model.state.color, center: UnitPoint(x: 0.5, y: 0.40), intensity: 0.08, radius: 0.42)
                .ignoresSafeArea()
                .animation(.easeInOut(duration: 1.2), value: model.state)

            VStack(spacing: 0) {
                topBar
                Spacer(minLength: FLSpacing.xs)
                orb
                stateBlock
                    .padding(.top, FLSpacing.xs)
                Spacer(minLength: FLSpacing.m)
                timerBlock
                Spacer(minLength: FLSpacing.m)
                riskLine
                controls
                    .padding(.top, FLSpacing.l)
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.bottom, FLSpacing.xs)

            SessionHintsOverlay(model: model, services: services, topInset: 56)

            NudgeHostLayer(model: model, services: services)

            if model.experiencePromptVisible {
                VStack {
                    Spacer()
                    services.experienceSampleView(model: model) ?? AnyView(PlaceholderExperienceSampleView(model: model))
                }
                .padding(.bottom, 120)
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.45, dampingFraction: 0.8), value: model.experiencePromptVisible)
        .statusBarHidden(false)
        .onChange(of: model.lastRefocusAt) { _, new in
            guard new != nil else { return }
            refocusPulse += 1
            FLHaptics.success()
        }
        .fullScreenCover(isPresented: $showBreathing) {
            services.breathingView(model: model, onClose: { showBreathing = false })
                ?? AnyView(BreathingScreen(model: model, onClose: { showBreathing = false }))
        }
        .confirmationDialog("End this session?", isPresented: $confirmEnd, titleVisibility: .visible) {
            Button("End and see report") { end() }
            Button("Keep studying", role: .cancel) {}
        }
    }

    // MARK: Pieces

    private var topBar: some View {
        HStack(spacing: FLSpacing.s) {
            FLIconButton("xmark", label: "End session", identifier: "focus.end") { confirmEnd = true }
            Spacer()
            FLBadgeRow(mock: model.studyMode && (model.timeline?.isMock ?? false), synthetic: model.isReplay,
                       demo: model.demoDirector, recorded: model.isRecorded, compact: true)
            Spacer()
            FLIconButton("waveform.path.ecg", label: "Open science view", identifier: "focus.science", action: onOpenScience)
        }
        .padding(.top, FLSpacing.xs)
    }

    private var orb: some View {
        GeometryReader { g in
            let side = min(g.size.width, g.size.height)
            ZStack {
                FocusOrb(probabilities: model.estimate?.probabilities ?? [1, 0, 0, 0],
                         lapseRisk: model.lapseRisk?.probability ?? 0,
                         riskAlert: model.lapseRisk?.isAlert ?? false,
                         paused: model.phase == .paused,
                         showsCore: faceVideo == nil)
                if let faceVideo {
                    faceWindow(faceVideo, diameter: side * 0.62)
                }
            }
            .frame(width: side, height: side)
            .position(x: g.size.width / 2, y: g.size.height / 2)
        }
        .frame(maxWidth: 360)
        .aspectRatio(1, contentMode: .fit)
        .padding(.horizontal, FLSpacing.l)
        .layoutPriority(1)
    }

    /// Where the face sits vertically in a typical portrait selfie video (fraction of its height).
    private static let faceCenterY: CGFloat = 0.30

    /// The video's frame inside a `diameter` square: full width for portrait video, shifted so the face is centred.
    static func faceAnchoredFrame(diameter d: CGFloat, aspect: CGFloat) -> CGRect {
        guard aspect > 0 else { return CGRect(x: 0, y: 0, width: d, height: d) }
        if aspect >= 1 {   // landscape: fill height, centre horizontally
            let w = d * aspect
            return CGRect(x: (d - w) / 2, y: 0, width: w, height: d)
        }
        let h = d / aspect
        let y = min(0, max(d - h, d / 2 - faceCenterY * h))
        return CGRect(x: 0, y: y, width: d, height: h)
    }

    private func faceAnchored(_ video: AnyView, diameter d: CGFloat) -> some View {
        let f = Self.faceAnchoredFrame(diameter: d, aspect: services.faceVideoAspectRatio() ?? 1)
        return Color.clear.overlay(alignment: .topLeading) {
            video.frame(width: f.width, height: f.height).offset(x: f.minX, y: f.minY)
        }
    }

    /// The recorded face video, circular-masked inside the orb's rings, with a faint live face mesh.
    private func faceWindow(_ video: AnyView, diameter: CGFloat) -> some View {
        ZStack {
            faceAnchored(video, diameter: diameter)
            RadialGradient(colors: [.clear, .clear, FLColor.bg.opacity(0.55)], center: .center,
                           startRadius: 0, endRadius: diameter / 2)
            TimelineView(.animation(minimumInterval: 1.0 / 30, paused: reduceMotion)) { _ in
                Canvas { gc, size in
                    guard let pts = services.faceMeshPoints(), !pts.isEmpty,
                          let aspect = services.faceVideoAspectRatio() else { return }
                    // Exactly on the face: same face-anchored geometry as the video underneath.
                    let frame = Self.faceAnchoredFrame(diameter: size.width, aspect: aspect)
                    for p in pts.map({ CGPoint(x: frame.minX + $0.x * frame.width, y: frame.minY + $0.y * frame.height) }) {
                        gc.fill(Path(ellipseIn: CGRect(x: p.x - 0.5, y: p.y - 0.5, width: 1, height: 1)),
                                with: .color(FLColor.accent.opacity(0.25)))
                    }
                }
            }
        }
        .frame(width: diameter, height: diameter)
        .clipShape(Circle())
        .overlay(Circle().strokeBorder(model.state.color.opacity(0.45), lineWidth: 0.75))
        .animation(.easeInOut(duration: 1.2), value: model.state)
        .accessibilityElement()
        .accessibilityLabel("Your recorded video, analysed on this device")
    }

    /// State label (headline + dot) and one supporting line, in fixed-height slots so strings swap, never overlap.
    private var stateBlock: some View {
        VStack(spacing: FLSpacing.xs) {
            ZStack {
                HStack(spacing: FLSpacing.xs) {
                    Circle().fill(model.state.color).frame(width: 6, height: 6)
                    Text(model.state.displayName).flText(.headline)
                }
                .id(model.state)
                .transition(.opacity)
            }
            .frame(height: 24)
            .clipped()
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Current state: \(model.state.displayName)")

            supportingLine
                .frame(height: 32)
        }
        .animation(.easeInOut(duration: 0.35), value: model.state)
    }

    /// "Back in X.X s" toast after a refocus; otherwise "Away X.X s" or the state caption.
    @ViewBuilder private var supportingLine: some View {
        let value = RefocusBanner.value(lastRefocusSeconds: model.lastRefocusSeconds,
                                        lastRefocusAt: model.lastRefocusAt, elapsed: model.elapsed)
        ZStack {
            if let v = value {
                HStack(spacing: FLSpacing.xs) {
                    Image(systemName: "checkmark")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(FLColor.accent)
                        .symbolEffect(.bounce, value: refocusPulse)
                    Text(UIFormat.backIn(v))
                        .font(.system(.subheadline, weight: .medium).monospacedDigit())
                        .foregroundStyle(FLColor.textPrimary)
                }
                .padding(.horizontal, FLSpacing.m)
                .frame(height: 32)
                .background(FLColor.accent.opacity(0.12), in: Capsule())
                .overlay(Capsule().strokeBorder(FLColor.accent.opacity(0.35), lineWidth: 0.5))
                .transition(.opacity.combined(with: .scale(scale: 0.92)))
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Refocused. \(UIFormat.backIn(v))")
                .accessibilityAddTraits(.updatesFrequently)
            } else if let away = model.lapseElapsed, model.state != .focused {
                Text("Away \(UIFormat.oneDecimal(away)) s")
                    .font(FLFont.callout.monospacedDigit())
                    .foregroundStyle(FLColor.textSecondary)
                    .transition(.opacity)
                    .accessibilityLabel("Away for \(UIFormat.oneDecimal(away)) seconds")
            } else {
                Text(model.faceDetected || model.phase != .running ? model.state.caption : "Looking for your face\u{2026}")
                    .flText(.callout)
                    .id(model.faceDetected || model.phase != .running ? model.state.caption : "looking")
                    .transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.3), value: value != nil)
        .animation(.easeOut(duration: 0.3), value: model.lapseElapsed != nil && model.state != .focused)
    }

    private var timerBlock: some View {
        VStack(spacing: FLSpacing.xs) {
            Text(UIFormat.timer(model.elapsed))
                .flText(.hero)
                .minimumScaleFactor(0.6)
                .lineLimit(1)
                .accessibilityLabel("Session time \(UIFormat.timer(model.elapsed))")
            Text(model.phase == .paused ? "Paused" : (model.studyMode ? "Study Mode" : "Focus session"))
                .flText(.eyebrow)
            if faceVideo != nil {
                Text("REC \u{00B7} your recording").flText(.caption)
                    .accessibilityLabel("Analysing your recording")
            }
        }
    }

    private var riskLine: some View {
        let p = Double(model.lapseRisk?.probability ?? 0)
        let alert = model.lapseRisk?.isAlert ?? false
        return VStack(alignment: .leading, spacing: FLSpacing.s) {
            HStack(alignment: .firstTextBaseline) {
                Text("Lapse risk \u{00B7} next 15 s").flText(.caption)
                if alert {
                    Text("High").font(FLFont.caption.weight(.medium)).foregroundStyle(FLColor.risk)
                }
                Spacer()
                Text(UIFormat.percent(p))
                    .font(FLFont.callout.monospacedDigit())
                    .foregroundStyle(alert ? FLColor.risk : FLColor.textPrimary)
            }
            FLLineMeter(value: p, color: alert ? FLColor.risk : FLColor.textSecondary, height: 3,
                        marker: model.lapseRisk.map { Double($0.threshold) })
                .animation(.easeOut(duration: 0.8), value: p)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Predicted lapse risk in the next 15 seconds: \(UIFormat.percent(p))\(alert ? ", above your threshold" : "")")
    }

    private var controls: some View {
        HStack(spacing: FLSpacing.s) {
            Button {
                FLHaptics.tap()
                model.phase == .paused ? model.actions.resume() : model.actions.pause()
            } label: {
                Label(model.phase == .paused ? "Resume" : "Pause",
                      systemImage: model.phase == .paused ? "play" : "pause")
            }
            .buttonStyle(FLSecondaryButtonStyle())
            Button {
                FLHaptics.tap(); showBreathing = true
            } label: {
                Label("Breathe", systemImage: "wind")
            }
            .buttonStyle(FLSecondaryButtonStyle())
        }
    }

    private func end() {
        Task {
            let id = await services.stopSession()
            onFinish(id)
        }
    }
}
