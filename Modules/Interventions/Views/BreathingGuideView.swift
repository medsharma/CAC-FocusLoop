import Charts
import SwiftUI

/// Guided paced breathing (~6 breaths/min: inhale 4 s, exhale 6 s) with live heart-rate biofeedback:
/// current BPM, a trend line of recent BPM and the pulse waveform from `HeartRateSample.waveform`.
/// Heart rate is a camera-based (rPPG) estimate for wellness feedback only.
public struct BreathingGuideView: View {
    public var targetBreathsPerMinute: Double
    /// Latest heart-rate estimate (nil hides the BPM readout and waveform).
    public var heartRate: HeartRateSample?
    /// Recent (t, bpm) history for the trend line.
    public var bpmHistory: [(t: Double, bpm: Float)]
    /// Shows a "Done" button when set.
    public var onDone: (() -> Void)?

    @State private var startDate = Date()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public init(targetBreathsPerMinute: Double = 6, heartRate: HeartRateSample? = nil,
                bpmHistory: [(t: Double, bpm: Float)] = [], onDone: (() -> Void)? = nil) {
        self.targetBreathsPerMinute = targetBreathsPerMinute
        self.heartRate = heartRate
        self.bpmHistory = bpmHistory
        self.onDone = onDone
    }

    /// Breathing phase at `elapsed` seconds: (isInhale, progress 0...1 within the phase, seconds left, scale 0...1).
    public static func phase(elapsed: Double, breathsPerMinute: Double) -> (inhale: Bool, progress: Double, secondsLeft: Double, level: Double) {
        let cycle = 60 / max(breathsPerMinute, 1)
        let inhaleDur = cycle * 0.4
        let exhaleDur = cycle - inhaleDur
        let t = elapsed.truncatingRemainder(dividingBy: cycle)
        if t < inhaleDur {
            let p = t / inhaleDur
            return (true, p, inhaleDur - t, InterventionStyle.ease(p))
        }
        let p = (t - inhaleDur) / exhaleDur
        return (false, p, exhaleDur - (t - inhaleDur), 1 - InterventionStyle.ease(p))
    }

    public var body: some View {
        VStack(spacing: FLSpacing.xl) {
            TimelineView(.animation(minimumInterval: 1 / 30)) { context in
                let ph = Self.phase(elapsed: context.date.timeIntervalSince(startDate), breathsPerMinute: targetBreathsPerMinute)
                // Reduce Motion: the ring steps between its two sizes (cross-faded) instead of gliding.
                let level = reduceMotion ? (ph.inhale ? 1 : 0) : ph.level
                ZStack {
                    // Faint guide at full size, then the single accent ring that expands and contracts.
                    Circle().stroke(FLColor.hairline, lineWidth: 1)
                    Circle()
                        .stroke(InterventionStyle.calm, lineWidth: 2)
                        .scaleEffect(0.56 + 0.44 * level)
                        .opacity(0.55 + 0.45 * level)
                        .animation(reduceMotion ? .easeInOut(duration: 0.6) : nil, value: ph.inhale)
                    VStack(spacing: FLSpacing.xs) {
                        // Fixed slot: exactly one phase label is ever on screen.
                        ZStack {
                            Text(ph.inhale ? "Breathe in" : "Breathe out")
                                .flText(.title)
                                .lineLimit(1)
                                .id(ph.inhale)
                                .transition(.opacity)
                        }
                        .frame(width: 150, height: 30)
                        .animation(FLMotion.fade, value: ph.inhale)
                        Text("\(Int(ph.secondsLeft.rounded(.up)))")
                            .font(FLFont.numeral(.title3, weight: .light))
                            .foregroundStyle(FLColor.textTertiary)
                    }
                }
                .frame(width: 280, height: 280)
                .accessibilityElement(children: .combine)
                .accessibilityLabel(ph.inhale ? "Breathe in" : "Breathe out")
            }

            Text("About \(Int(targetBreathsPerMinute.rounded())) breaths a minute. In for 4, out for 6.")
                .flText(.callout)
                .multilineTextAlignment(.center)

            if heartRate != nil || !bpmHistory.isEmpty {
                biofeedback
            }

            if let onDone {
                Button("Done", action: onDone)
                    .buttonStyle(FLSecondaryButtonStyle(fullWidth: false))
            }
        }
        .padding(.horizontal, FLSpacing.gutter)
        // On black: opaque, so nothing underneath (e.g. Study Mode's video) bleeds through.
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(FLColor.bg.ignoresSafeArea())
        .environment(\.colorScheme, .dark)
    }

    /// Heart-rate biofeedback: a quiet metric row, a 1.5pt trend line and the pulse waveform. No card.
    private var biofeedback: some View {
        VStack(alignment: .leading, spacing: FLSpacing.s) {
            FLDivider()
            HStack(alignment: .firstTextBaseline, spacing: FLSpacing.xs) {
                Circle().fill(FLColor.heart).frame(width: 6, height: 6)
                    .alignmentGuide(.firstTextBaseline) { $0[.bottom] + 2 }
                if let hr = heartRate {
                    Text("\(Int(hr.bpm.rounded()))")
                        .flText(.metric)
                        .contentTransition(.numericText())
                    Text("BPM").font(FLFont.unit).foregroundStyle(FLColor.textTertiary)
                    Spacer()
                    QualityPill(quality: hr.quality, flagged: hr.motionFlagged)
                } else {
                    Text("--").flText(.metric)
                    Spacer()
                }
            }
            .accessibilityElement(children: .combine)
            if bpmHistory.count >= 2 || (heartRate?.waveform.count ?? 0) > 2 {
                HStack(spacing: FLSpacing.m) {
                    if bpmHistory.count >= 2 {
                        Chart(Array(bpmHistory.enumerated()), id: \.offset) { item in
                            LineMark(x: .value("Time", item.element.t), y: .value("BPM", item.element.bpm))
                                .interpolationMethod(.catmullRom)
                                .lineStyle(FLChart.line)
                                .foregroundStyle(FLColor.heart)
                        }
                        .chartYScale(domain: .automatic(includesZero: false))
                        .chartXAxis(.hidden)
                        .chartYAxis {
                            AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { _ in
                                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(FLColor.hairline)
                                AxisValueLabel().font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                            }
                        }
                        .frame(height: 56)
                    }
                    if let wave = heartRate?.waveform, wave.count > 2 {
                        PulseWaveformShape(samples: wave)
                            .stroke(FLColor.heart.opacity(0.85), style: FLChart.line)
                            .frame(height: 40)
                    }
                }
            }
            Text("Camera-based heart-rate estimate for wellness feedback. Not a medical measurement.")
                .flText(.caption)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: 440)
    }
}

/// Polyline of a -1...1 waveform.
struct PulseWaveformShape: Shape {
    var samples: [Float]

    func path(in rect: CGRect) -> Path {
        var p = Path()
        guard samples.count > 1 else { return p }
        let step = rect.width / CGFloat(samples.count - 1)
        for (i, s) in samples.enumerated() {
            let y = rect.midY - CGFloat(max(-1, min(1, s))) * rect.height / 2
            let pt = CGPoint(x: CGFloat(i) * step, y: y)
            if i == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
        }
        return p
    }
}

private struct QualityPill: View {
    var quality: Float
    var flagged: Bool

    var body: some View {
        let label = flagged ? "Hold still" : quality >= 0.6 ? "Good signal" : quality >= 0.3 ? "Fair signal" : "Weak signal"
        let color: Color = flagged ? FLColor.distracted : quality >= 0.6 ? FLColor.focused : quality >= 0.3 ? FLColor.distracted : FLColor.stressed
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label).font(.system(.footnote, weight: .medium)).foregroundStyle(FLColor.textSecondary)
        }
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(FLColor.surfaceHigh, in: Capsule())
        .overlay(Capsule().strokeBorder(FLColor.hairline, lineWidth: 0.5))
    }
}

#Preview("Breathing (SYNTHETIC HR)") {
    ZStack {
        FLColor.bg.ignoresSafeArea()
        BreathingGuideView(heartRate: BreathingGuidePreviewData.sample, bpmHistory: BreathingGuidePreviewData.history, onDone: {})
    }
}

/// SYNTHETIC preview data only.
private enum BreathingGuidePreviewData {
    static var sample: HeartRateSample {
        var wave: [Float] = []
        for i in 0..<90 { wave.append(Float(sin(Double(i) / 15 * Double.pi))) }
        return HeartRateSample(t: 60, bpm: 73, quality: 0.7, snrDB: 6, motionFlagged: false, baselineBPM: 68, waveform: wave)
    }

    static var history: [(t: Double, bpm: Float)] {
        var out: [(t: Double, bpm: Float)] = []
        for i in 0..<60 {
            let x = Double(i)
            let bpm: Double = 78 - x * 0.08 + sin(x / 4)
            out.append((t: x, bpm: Float(bpm)))
        }
        return out
    }
}
