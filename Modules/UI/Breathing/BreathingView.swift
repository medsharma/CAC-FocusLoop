import Charts
import SwiftUI

/// Breathing phase helper: inhale for half the cycle, exhale for the other half.
public enum BreathPhase {
    /// Returns (0...1 expansion, isInhaling, seconds left in this phase).
    public static func at(_ t: Double, breathsPerMinute: Double) -> (Double, Bool, Double) {
        let period = 60 / max(1, breathsPerMinute)
        let x = (t.truncatingRemainder(dividingBy: period) + period).truncatingRemainder(dividingBy: period) / period
        let inhaling = x < 0.5
        let local = inhaling ? x / 0.5 : (x - 0.5) / 0.5
        let eased = 0.5 - 0.5 * cos(local * .pi)
        let left = (inhaling ? 0.5 - x : 1 - x) * period
        return (inhaling ? eased : 1 - eased, inhaling, left)
    }
}

/// Expanding / contracting breathing guide (~6 breaths/min).
public struct BreathingGuide: View {
    var breathsPerMinute: Double
    var compact: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var t0 = Date()

    public init(breathsPerMinute: Double = 6, compact: Bool = false) {
        self.breathsPerMinute = breathsPerMinute; self.compact = compact
    }

    public var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: false)) { ctx in
            let (e, inhaling, left) = BreathPhase.at(ctx.date.timeIntervalSince(t0), breathsPerMinute: breathsPerMinute)
            GeometryReader { geo in
                let side = min(geo.size.width, geo.size.height)
                // Reduce Motion: the ring steps between two sizes instead of gliding.
                let level = reduceMotion ? (inhaling ? 1.0 : 0.0) : e
                let s = 0.42 + 0.58 * level
                ZStack {
                    Circle().stroke(FLColor.hairline, lineWidth: 1)
                    Circle()
                        .stroke(FLColor.accent, lineWidth: compact ? 1.5 : 2)
                        .scaleEffect(s)
                        .opacity(0.55 + 0.45 * level)
                        .animation(reduceMotion ? .easeInOut(duration: 0.6) : nil, value: inhaling)
                    if !compact {
                        VStack(spacing: FLSpacing.xs) {
                            ZStack {
                                Text(inhaling ? "Breathe in" : "Breathe out")
                                    .flText(.title)
                                    .id(inhaling)
                                    .transition(.opacity)
                            }
                            .animation(FLMotion.fade, value: inhaling)
                            Text("\(Int(left.rounded(.up)))")
                                .font(FLFont.callout.monospacedDigit()).foregroundStyle(FLColor.textTertiary)
                        }
                    }
                }
                .frame(width: side, height: side)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .accessibilityElement()
            .accessibilityLabel(inhaling ? "Breathe in" : "Breathe out")
        }
    }
}

/// Fallback breathing biofeedback screen: guide + live BPM + trend line.
public struct BreathingScreen: View {
    let model: LiveSessionModel
    var onClose: () -> Void
    public init(model: LiveSessionModel, onClose: @escaping () -> Void) { self.model = model; self.onClose = onClose }

    public var body: some View {
        ZStack {
            FLColor.bg.ignoresSafeArea()
            VStack(spacing: 0) {
                HStack {
                    FLIconButton("xmark", label: "Close", action: onClose)
                    Spacer()
                    FLBadgeRow(synthetic: model.isReplay, demo: model.demoDirector)
                    Spacer()
                    Color.clear.frame(width: 44, height: 44)
                }
                .padding(.top, FLSpacing.xs)
                VStack(spacing: FLSpacing.xs) {
                    Text("Paced breathing").flText(.largeTitle)
                    Text("6 breaths a minute. Follow the circle.").flText(.callout)
                }
                .padding(.top, FLSpacing.l)
                Spacer(minLength: FLSpacing.l)
                BreathingGuide(breathsPerMinute: 6).frame(maxHeight: 320)
                Spacer(minLength: FLSpacing.l)
                HStack(alignment: .bottom, spacing: FLSpacing.m) {
                    FLMetric("Heart rate", value: UIFormat.bpm(model.heartRate?.bpm), unit: "BPM", color: FLColor.textPrimary)
                    HRTrendChart(points: Array(model.heartRateHistory.suffix(90)))
                        .frame(width: 140, height: 48)
                }
                .padding(.top, FLSpacing.m)
                .overlay(alignment: .top) { FLDivider() }
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.bottom, FLSpacing.m)
        }
    }
}

/// Tiny HR trend line (last ~90 s).
struct HRTrendChart: View {
    let points: [(t: Double, bpm: Float)]
    var body: some View {
        Chart {
            ForEach(points.indices, id: \.self) { i in
                LineMark(x: .value("t", points[i].t), y: .value("BPM", points[i].bpm))
                    .interpolationMethod(.catmullRom)
                    .lineStyle(FLChart.line)
                    .foregroundStyle(FLColor.heart)
            }
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartYScale(domain: .automatic(includesZero: false))
        .accessibilityLabel("Heart-rate trend")
    }
}
