import SwiftUI

/// Ambient edge glow. The pre-emptive variant is softer, cooler and slower. Never intercepts touches.
public struct GlowOverlayView: View {
    public var intensity: Float
    public var preemptive: Bool

    public init(intensity: Float = 0.7, preemptive: Bool = false) {
        self.intensity = intensity
        self.preemptive = preemptive
    }

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30, paused: reduceMotion)) { context in
            let period = preemptive ? 5.0 : 3.2
            let t = context.date.timeIntervalSinceReferenceDate
            // Reduce Motion: a steady glow instead of a breathing one.
            let breath = reduceMotion ? 0.6 : 0.5 + 0.5 * sin(t / period * 2 * .pi)
            let base = Double(min(max(intensity, 0), 1)) * (preemptive ? 0.55 : 0.85)
            let color = preemptive ? InterventionStyle.preemptive : InterventionStyle.calm
            ZStack {
                RoundedRectangle(cornerRadius: 56, style: .continuous)
                    .strokeBorder(color.opacity(base * (0.25 + 0.35 * breath)), lineWidth: preemptive ? 16 : 22)
                    .blur(radius: preemptive ? 28 : 24)
                RoundedRectangle(cornerRadius: 56, style: .continuous)
                    .strokeBorder(color.opacity(base * (0.15 + 0.25 * breath)), lineWidth: 1.5)
                    .blur(radius: 2)
            }
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// Bright, cool-light pulse for the drowsy state.
/// EXPERIMENTAL arm under test: the trial measures whether it helps this user. Never claim it works.
public struct LightPulseView: View {
    public var pulses: Int
    public var period: Double
    public var onFinish: (() -> Void)?

    @State private var startDate = Date()

    public init(pulses: Int = 3, period: Double = 2.0, onFinish: (() -> Void)? = nil) {
        self.pulses = pulses
        self.period = period
        self.onFinish = onFinish
    }

    public var body: some View {
        TimelineView(.animation) { context in
            let elapsed = context.date.timeIntervalSince(startDate)
            let total = Double(pulses) * period
            let phase = elapsed < total ? (elapsed.truncatingRemainder(dividingBy: period) / period) : 1
            let level = elapsed < total ? sin(phase * .pi) : 0
            ZStack(alignment: .bottom) {
                RadialGradient(colors: [InterventionStyle.coolLight.opacity(0.85 * level),
                                        InterventionStyle.coolEdge.opacity(0.25 * level)],
                               center: .center, startRadius: 20, endRadius: 600)
                    .ignoresSafeArea()
                Text("Experimental: cool-light pulse")
                    .font(FLFont.caption)
                    .foregroundStyle(FLColor.textSecondary)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                .background(FLColor.bgElevated, in: Capsule())
                .overlay(Capsule().strokeBorder(FLColor.hairlineStrong, lineWidth: 0.5))
                .opacity(elapsed < total ? 1 : 0)
                .padding(.bottom, FLSpacing.xxl)
            }
        }
        .allowsHitTesting(false)
        .environment(\.colorScheme, .dark)
        .task {
            try? await Task.sleep(nanoseconds: UInt64(max(0, Double(pulses) * period) * 1_000_000_000))
            if !Task.isCancelled { onFinish?() }
        }
    }
}

/// Convenience host: renders any `NudgePresentation` with the matching view.
/// Glow and light pulse are overlays (no touches); the others are full-screen calm sheets.
public struct NudgePresentationView: View {
    public var presentation: NudgePresentation
    public var heartRate: HeartRateSample?
    public var bpmHistory: [(t: Double, bpm: Float)]
    public var onDismiss: () -> Void
    public var onChoice: ((Int) -> Void)?

    public init(presentation: NudgePresentation, heartRate: HeartRateSample? = nil,
                bpmHistory: [(t: Double, bpm: Float)] = [], onDismiss: @escaping () -> Void,
                onChoice: ((Int) -> Void)? = nil) {
        self.presentation = presentation
        self.heartRate = heartRate
        self.bpmHistory = bpmHistory
        self.onDismiss = onDismiss
        self.onChoice = onChoice
    }

    public var body: some View {
        switch presentation {
        case let .glow(intensity, preemptive):
            GlowOverlayView(intensity: intensity, preemptive: preemptive)
        case .lightPulse:
            LightPulseView(onFinish: onDismiss)
        case let .card(card):
            NudgeCardView(card: card, onDismiss: onDismiss, onChoice: onChoice)
        case let .breathing(bpm):
            BreathingGuideView(targetBreathsPerMinute: bpm, heartRate: heartRate, bpmHistory: bpmHistory, onDone: onDismiss)
        case let .stretch(seconds):
            StretchCountdownView(seconds: seconds, onFinish: onDismiss)
        }
    }
}

#Preview("Glow") {
    ZStack {
        FLColor.bg.ignoresSafeArea()
        GlowOverlayView(intensity: 0.8)
    }
}

#Preview("Light pulse (experimental)") {
    ZStack {
        FLColor.bg.ignoresSafeArea()
        LightPulseView()
    }
}
