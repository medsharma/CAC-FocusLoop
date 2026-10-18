import SwiftUI

// Minimal fallbacks for slots other agents fill. The orchestrator swaps them via `UIServices`.

/// Fallback for the Calibration agent's work-zone flow.
public struct PlaceholderCalibrationView: View {
    var onFinish: () -> Void
    @State private var corner = 0
    private let targets: [UnitPoint] = [.center, .topLeading, .topTrailing, .bottomTrailing, .bottomLeading]
    public init(onFinish: @escaping () -> Void) { self.onFinish = onFinish }

    public var body: some View {
        VStack(spacing: FLSpacing.l) {
            VStack(spacing: FLSpacing.s) {
                Text("Your work zone").font(FLFont.display).foregroundStyle(FLColor.textPrimary)
                Text("Look at the center of your laptop screen, then each corner. FocusLoop learns where you look when you're working.")
                    .font(FLFont.body).foregroundStyle(FLColor.textSecondary).multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, FLSpacing.xl)
            ZStack {
                RoundedRectangle(cornerRadius: 18, style: .continuous).strokeBorder(FLColor.strokeStrong, lineWidth: 2)
                GeometryReader { g in
                    let t = targets[min(corner, targets.count - 1)]
                    Circle().fill(FLColor.accent).frame(width: 22, height: 22)
                        .shadow(color: FLColor.accent, radius: 10)
                        .position(x: 20 + (g.size.width - 40) * t.x, y: 20 + (g.size.height - 40) * t.y)
                        .animation(.spring(response: 0.5, dampingFraction: 0.75), value: corner)
                }
            }
            .aspectRatio(16 / 10, contentMode: .fit)
            .accessibilityLabel("Calibration target \(min(corner + 1, 5)) of 5")
            Text("Step \(min(corner + 1, 5)) of 5").font(FLFont.numeral(.callout)).foregroundStyle(FLColor.textSecondary)
            Spacer()
            Button(corner < targets.count - 1 ? "Next point" : "Finish calibration") {
                FLHaptics.tap()
                if corner < targets.count - 1 { corner += 1 } else { onFinish() }
            }
            .buttonStyle(FLPrimaryButtonStyle())
            Button("Skip for now", action: onFinish).font(FLFont.callout).foregroundStyle(FLColor.textSecondary)
        }
        .padding(.horizontal, FLSpacing.xl)
        .padding(.bottom, FLSpacing.l)
    }
}

/// Fallback for the Calibration agent's SART screen.
public struct PlaceholderSARTView: View {
    var onFinish: () -> Void
    public init(onFinish: @escaping () -> Void) { self.onFinish = onFinish }
    public var body: some View {
        VStack(spacing: FLSpacing.l) {
            Spacer()
            Text("3").font(.system(size: 120, weight: .bold, design: .rounded)).foregroundStyle(FLColor.textPrimary.opacity(0.9))
                .accessibilityHidden(true)
            Text("A 3-minute attention check").font(FLFont.display).foregroundStyle(FLColor.textPrimary)
                .multilineTextAlignment(.center)
            Text("Digits flash one at a time. Tap for every digit except 3. It teaches FocusLoop what your focus and your lapses look like.")
                .font(FLFont.body).foregroundStyle(FLColor.textSecondary).multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button("Start the check", action: onFinish).buttonStyle(FLPrimaryButtonStyle())
            Button("Skip for now", action: onFinish).font(FLFont.callout).foregroundStyle(FLColor.textSecondary)
        }
        .padding(.horizontal, FLSpacing.xl)
        .padding(.bottom, FLSpacing.l)
    }
}

/// Fallback Study Mode player (no video): the timer-driven lecture position, a corner cortex
/// that pulses before predicted valleys, and valley markers on the scrubber.
public struct PlaceholderStudyModeView: View {
    let lecture: LectureInfo
    let model: LiveSessionModel
    let services: UIServices
    var onFinish: (UUID?) -> Void
    var onOpenScience: () -> Void

    public init(lecture: LectureInfo, model: LiveSessionModel, services: UIServices,
                onFinish: @escaping (UUID?) -> Void, onOpenScience: @escaping () -> Void) {
        self.lecture = lecture; self.model = model; self.services = services
        self.onFinish = onFinish; self.onOpenScience = onOpenScience
    }

    public var body: some View {
        ZStack {
            FLBackground(tint: model.state.color).ignoresSafeArea()
            VStack(alignment: .leading, spacing: FLSpacing.m) {
                HStack {
                    FLIconButton("xmark", label: "End session") {
                        Task { onFinish(await services.stopSession()) }
                    }
                    Spacer()
                    FLBadgeRow(mock: model.timeline?.isMock ?? false, synthetic: model.isReplay, demo: model.demoDirector, compact: true)
                    Spacer()
                    FLIconButton("waveform.path.ecg.rectangle", label: "Open science view", action: onOpenScience)
                }
                ZStack(alignment: .topTrailing) {
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .fill(LinearGradient(colors: [Color(hex: 0x111A33), Color(hex: 0x0A0F1E)], startPoint: .top, endPoint: .bottom))
                        .overlay {
                            VStack(spacing: 6) {
                                Image(systemName: "play.rectangle.fill").font(.system(size: 40)).foregroundStyle(FLColor.textTertiary)
                                Text(lecture.title).font(FLFont.headline).foregroundStyle(FLColor.textPrimary)
                                Text("Lecture video plays here").font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                            }
                        }
                        .aspectRatio(16 / 9, contentMode: .fit)
                    CornerCortex(model: model).frame(width: 96, height: 96).padding(8)
                }
                Scrubber(model: model)
                HStack {
                    FLStateChip(model.state)
                    Spacer()
                    Text(UIFormat.timer(model.elapsed)).font(FLFont.numeral(.title3)).foregroundStyle(FLColor.textPrimary)
                }
                LanesCard(model: model)
                Text(lecture.attribution).font(.caption2).foregroundStyle(FLColor.textTertiary)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, FLSpacing.gutter)
            NudgeHostLayer(model: model, services: services)
        }
    }
}

/// Small cortex whose edge pulses when a predicted valley is coming up.
struct CornerCortex: View {
    let model: LiveSessionModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var body: some View {
        let upcoming = model.timeline?.upcomingValley(after: model.playhead, within: 10) != nil
        ZStack {
            Circle().fill(.black.opacity(0.4))
            BrainCortexView(activations: model.timeline?.activations(at: model.playhead) ?? [])
                .clipShape(Circle())
            TimelineView(.animation(minimumInterval: 1.0 / 30, paused: !upcoming || reduceMotion)) { ctx in
                let ph = 0.5 + 0.5 * sin(ctx.date.timeIntervalSinceReferenceDate * 4)
                Circle().strokeBorder(FLColor.distracted.opacity(upcoming ? 0.4 + 0.6 * ph : 0), lineWidth: 3)
            }
        }
        .accessibilityLabel(upcoming ? "Predicted low-engagement moment coming up" : "Predicted brain response to this content")
    }
}

struct Scrubber: View {
    let model: LiveSessionModel
    var body: some View {
        let d = max(1, model.timeline?.duration ?? 1)
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(FLColor.surfaceStrong).frame(height: 6)
                ForEach(Array((model.timeline?.valleys() ?? []).enumerated()), id: \.offset) { _, v in
                    Capsule().fill(FLColor.distracted.opacity(0.6))
                        .frame(width: max(4, g.size.width * (v.end - v.start) / d), height: 6)
                        .offset(x: g.size.width * v.start / d)
                }
                Capsule().fill(FLColor.accent).frame(width: g.size.width * model.playhead / d, height: 6)
                Circle().fill(.white).frame(width: 14, height: 14).offset(x: g.size.width * model.playhead / d - 7)
            }
            .frame(maxHeight: .infinity)
        }
        .frame(height: 20)
        .accessibilityElement()
        .accessibilityLabel("Lecture position \(UIFormat.timer(model.playhead)). Amber marks are predicted low-engagement moments.")
    }
}
