import SwiftUI

enum StudyCortexCopy {
    static let caption = "Predicted brain response to this content (average brain)"
    /// Appended while the brain timeline is a placeholder (never implies real model output).
    static let previewSuffix = " \u{00B7} preview model"

    static func caption(isPreview: Bool) -> String { caption + (isPreview ? previewSuffix : "") }
    /// One-line visible caption under the corner cortex.
    static let short = "Predicted brain response \u{00B7} average brain"
    /// The honesty note behind the info button (shown while the brain timeline is a placeholder).
    static let previewNote = "Brain predictions come from a placeholder preview model until the Axonal Labs model is connected. They show an average brain, not yours."
    /// Shown when real model output is connected.
    static let averageNote = "Brain predictions show the predicted response of an average brain to this content, not yours."
}

/// The single quiet info button near the cortex ("i" in a circle, 44pt hit area). Opens a short popover.
struct StudyCortexInfoButton: View {
    let isPreview: Bool
    let attribution: String
    @Binding var isPresented: Bool

    var body: some View {
        Button { isPresented = true } label: {
            Image(systemName: "info.circle")
                .font(.system(size: 15, weight: .regular))
                .foregroundStyle(FLColor.textSecondary)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(FLPressableStyle())
        .padding(.vertical, -14)
        .padding(.trailing, -12)
        .accessibilityLabel("About these brain predictions")
        .accessibilityIdentifier("study.cortexInfo")
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: FLSpacing.s) {
                Text(isPreview ? StudyCortexCopy.previewNote : StudyCortexCopy.averageNote)
                    .font(FLFont.callout)
                    .foregroundStyle(FLColor.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                if !attribution.isEmpty {
                    FLDivider()
                    Text(attribution)
                        .flText(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(FLSpacing.card)
            .frame(width: 300, alignment: .leading)
            .presentationCompactAdaptation(.popover)
            .presentationBackground(FLColor.bgElevated)
            .preferredColorScheme(.dark)
        }
    }
}

/// Small corner cortex. The Metal view is only created while `isLive` (on screen, not covered);
/// otherwise a static placeholder is shown so no GPU work happens. The edge pulses while
/// `warning` (a predicted valley is coming up, or the playhead is inside one).
struct StudyCortexCorner: View {
    let activations: [Float]
    let warning: Bool
    let isMock: Bool
    let isLive: Bool
    /// False while the lecture is paused (cortex holds still).
    var rotating: Bool = true
    var size: CGFloat = 128
    var onTap: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var shape: RoundedRectangle { RoundedRectangle(cornerRadius: StudyModeStyle.stageRadius, style: .continuous) }

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                shape.fill(FLColor.surface)
                if isLive {
                    BrainCortexView(activations: activations, autoRotate: rotating)
                        .padding(6)
                        .allowsHitTesting(false)
                } else {
                    Image(systemName: "brain")
                        .font(.system(size: 32, weight: .light))
                        .foregroundStyle(FLColor.textTertiary)
                }
                VStack {
                    HStack(alignment: .top) {
                        Spacer()
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .font(.system(size: 10, weight: .regular))
                            .foregroundStyle(FLColor.textSecondary)
                            .frame(width: 22, height: 22)
                            .background(FLColor.bg.opacity(0.6), in: Circle())
                    }
                    Spacer()
                }
                .padding(FLSpacing.xs)
            }
            .frame(width: size, height: size)
            .clipShape(shape)
            .overlay(shape.strokeBorder(FLColor.hairline, lineWidth: 0.5))
            .overlay(pulseStroke)
            .contentShape(shape)
            .onTapGesture(perform: onTap)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(StudyCortexCopy.caption(isPreview: isMock))
        .accessibilityHint(warning ? "A predicted low-engagement stretch is near. Double-tap to expand."
                                   : "Double-tap to expand.")
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("study.cortex")
    }

    /// A slow amber edge while a predicted valley is near (static under Reduce Motion).
    @ViewBuilder private var pulseStroke: some View {
        if warning {
            TimelineView(.animation(minimumInterval: 1 / 30, paused: reduceMotion)) { ctx in
                let phase = reduceMotion ? 0.6
                    : 0.5 + 0.5 * sin(ctx.date.timeIntervalSinceReferenceDate * 2 * .pi / 2.0)
                shape.strokeBorder(StudyModeStyle.valley.opacity(0.35 + 0.5 * phase), lineWidth: 1 + phase)
            }
            .allowsHitTesting(false)
            .transition(.opacity)
        }
    }
}

/// Seven network rows (BrainNetwork order, atlas colours): a small dot, the name and a 3pt line meter.
struct StudyNetworkBars: View {
    let activations: [Float]

    var body: some View {
        VStack(spacing: 0) {
            ForEach(Array(BrainNetwork.allCases.enumerated()), id: \.element) { i, n in
                let v = Double(n.rawValue < activations.count ? min(max(activations[n.rawValue], 0), 1) : 0)
                if i > 0 { FLDivider() }
                HStack(spacing: FLSpacing.s) {
                    Circle().fill(n.color).frame(width: 6, height: 6)
                    Text(n.displayName)
                        .font(FLFont.callout)
                        .foregroundStyle(FLColor.textPrimary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    FLLineMeter(value: v, color: n.color)
                        .frame(width: 112)
                        .animation(.easeOut(duration: 0.25), value: v)
                    Text("\(Int((v * 100).rounded()))")
                        .font(FLFont.numeral(.footnote))
                        .foregroundStyle(FLColor.textTertiary)
                        .frame(width: 28, alignment: .trailing)
                }
                .frame(minHeight: 40)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(n.displayName)
                .accessibilityValue("\(Int((v * 100).rounded())) percent predicted activity")
            }
        }
    }
}

/// Expanded cortex with network bars, driven live by the playback controller.
struct StudyCortexSheet: View {
    let timeline: ContentTimeline
    let playback: StudyPlaybackController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let t = playback.playhead
        let warning = timeline.upcomingValley(after: t, within: 10) != nil
            || StudyScrubberGeometry.valleyIndex(containing: t, in: timeline.valleys()) != nil
        ScrollView {
            VStack(alignment: .leading, spacing: FLSpacing.l) {
                HStack(alignment: .top, spacing: FLSpacing.s) {
                    VStack(alignment: .leading, spacing: FLSpacing.xs) {
                        FLEyebrow("Content")
                        Text("Content brain map").flText(.title)
                            .accessibilityAddTraits(.isHeader)
                        Text(StudyCortexCopy.caption)
                            .flText(.caption)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 0)
                    FLIconButton("xmark", label: "Close", identifier: "study.cortexSheet.close") { dismiss() }
                }

                ZStack {
                    RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous).fill(FLColor.surface)
                    RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous)
                        .strokeBorder(FLColor.hairline, lineWidth: 0.5)
                    if ThermalStatus.shared.cortexPaused {
                        Label("Paused while the phone cools down", systemImage: "thermometer.high")
                            .font(FLFont.callout)
                            .foregroundStyle(FLColor.textSecondary)
                    } else {
                        BrainCortexView(activations: timeline.activations(at: t), autoRotate: true)
                            .padding(FLSpacing.xs)
                    }
                }
                .frame(maxWidth: .infinity)
                .aspectRatio(1.05, contentMode: .fit)
                .background {
                    // The one glow on this sheet, behind the cortex only.
                    RadialGradient(colors: [FLColor.accent.opacity(0.07), .clear], center: .center,
                                   startRadius: 0, endRadius: 240)
                        .padding(-60)
                        .allowsHitTesting(false)
                }

                HStack {
                    FLEyebrow("Networks")
                    Spacer()
                    StudyValleyStatusLine(timeline: timeline, playhead: t, highlight: warning)
                }

                StudyNetworkBars(activations: timeline.activations(at: t))

                VStack(alignment: .leading, spacing: FLSpacing.xs) {
                    Text("Drag the brain to rotate. Values are model predictions for an average viewer, not a measurement of you.")
                }
                .flText(.caption)
                .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.top, FLSpacing.l)
            .padding(.bottom, FLSpacing.xl)
        }
        .scrollIndicators(.hidden)
        .presentationDetents([.large])
        .flSheet()
        .preferredColorScheme(.dark)
    }
}

/// "Predicted dip in 8 s" / "In a predicted dip" / "Engagement steady". Amber only while it matters.
struct StudyValleyStatusLine: View {
    let timeline: ContentTimeline
    let playhead: Double
    var highlight: Bool

    var body: some View {
        let inValley = StudyScrubberGeometry.valleyIndex(containing: playhead, in: timeline.valleys()) != nil
        let upcoming = timeline.upcomingValley(after: playhead, within: 10)
        let text: String = {
            if inValley { return "In a predicted dip" }
            if let u = upcoming { return "Predicted dip in \(max(1, Int((u.start - playhead).rounded(.up)))) s" }
            return "Engagement steady"
        }()
        HStack(spacing: 6) {
            Circle()
                .fill(highlight ? StudyModeStyle.valley : FLColor.textTertiary)
                .frame(width: 6, height: 6)
            Text(text)
                .font(.system(.footnote, weight: .medium))
                .monospacedDigit()
                .lineLimit(1)
        }
        .foregroundStyle(highlight ? StudyModeStyle.valley : FLColor.textSecondary)
        .animation(.easeInOut(duration: 0.3), value: highlight)
        .accessibilityElement(children: .combine)
    }
}
