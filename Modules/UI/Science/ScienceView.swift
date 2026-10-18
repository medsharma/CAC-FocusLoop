import Charts
import SwiftUI

/// The live "science" view: signals, heart rate, the predicted brain response to the content,
/// attention lanes, lapse-risk forecast and live bandit posteriors.
/// Quiet Dark editorial layout: eyebrow sections separated by hairlines; cards only where a visual needs a frame.
public struct ScienceView: View {
    let model: LiveSessionModel
    let services: UIServices
    /// When true, this view starts a preview session on appear (and stops it on disappear) if nothing is running.
    var ownsSession: Bool

    @State private var startedHere = false

    public init(model: LiveSessionModel, services: UIServices, ownsSession: Bool = true) {
        self.model = model; self.services = services; self.ownsSession = ownsSession
    }

    public var body: some View {
        ScrollViewReader { proxy in
        ScrollView {
            LazyVStack(alignment: .leading, spacing: FLSpacing.xxl) {
                ScienceHeader(model: model, services: services) { anchor in
                    withAnimation(FLMotion.standard) { proxy.scrollTo(anchor, anchor: .top) }
                }
                SignalsSection(model: model, services: services).id("signals")
                HeartSection(model: model).id("heart")
                StepBreak()
                StateProbabilitiesSection(model: model).id("state")
                StepBreak()
                BrainCard(model: model).id("brain")
                LanesCard(model: model, framed: false).id("lanes")
                LapseRiskSection(model: model).id("risk")
                StepBreak()
                InterveneSection(model: model).id("intervene")
                StepBreak()
                PosteriorsSection(model: model).id("bandit")
                FLDivider()
                Text("FocusLoop supports focus; it is not a medical device. Brain predictions reflect an average brain, not yours.")
                    .flText(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.top, FLSpacing.xs)
            .padding(.bottom, FLSpacing.xxl)
        }
        .modifier(FLLaunchScroll(proxy: proxy))
        }
        .flScreen("Science", inline: true)
        .onAppear {
            if ownsSession, model.phase != .running, model.phase != .paused {
                services.startSession(.science); startedHere = true
            }
        }
        .onDisappear {
            if startedHere { startedHere = false; Task { _ = await services.stopSession() } }
        }
    }
}

// MARK: - Narrative scaffolding

/// The five pipeline steps, in reading order. `anchor` is the scroll id of the step's first section.
private enum ScienceStep: Int, CaseIterable, Identifiable {
    case sense, understand, predict, intervene, learn
    var id: Int { rawValue }
    var name: String { ["Sense", "Understand", "Predict", "Intervene", "Learn"][rawValue] }
    var number: String { String(format: "%02d", rawValue + 1) }
    var eyebrow: String { "\(number)  \(name)" }
    var anchor: String { ["signals", "state", "brain", "intervene", "bandit"][rawValue] }
}

/// Hairline between pipeline steps.
private struct StepBreak: View {
    var body: some View { FLDivider() }
}

/// "What you're seeing": one or two plain-English sentences under a section title.
private struct Explainer: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text).flText(.callout)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// A section title with its step eyebrow, explainer and optional trailing accessory.
private struct StepHeader<Trailing: View>: View {
    let step: ScienceStep?
    let title: String
    let explainer: String
    let trailing: Trailing
    init(_ title: String, step: ScienceStep? = nil, explainer: String, @ViewBuilder trailing: () -> Trailing) {
        self.title = title; self.step = step; self.explainer = explainer; self.trailing = trailing()
    }
    var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.s) {
            if let step {
                FLSectionHeader(title, eyebrow: step.eyebrow, large: true) { trailing }
            } else {
                FLSectionHeader(title) { trailing }
            }
            Explainer(explainer)
        }
    }
}

private extension StepHeader where Trailing == EmptyView {
    init(_ title: String, step: ScienceStep? = nil, explainer: String) {
        self.init(title, step: step, explainer: explainer) { EmptyView() }
    }
}

// MARK: - Header

private struct ScienceHeader: View {
    let model: LiveSessionModel
    let services: UIServices
    var jump: (String) -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            HStack(spacing: FLSpacing.xs) {
                LiveDot(active: model.phase == .running)
                Text(model.phase == .running ? "Live" : "Idle").flText(.eyebrow)
                    .accessibilityHidden(true)
                Spacer(minLength: FLSpacing.xs)
            }
            VStack(alignment: .leading, spacing: FLSpacing.xs) {
                Text("Live pipeline").flText(.largeTitle)
                    .accessibilityAddTraits(.isHeader)
                Text("Watch FocusLoop sense your focus, understand it, predict a lapse, step in, and learn what works for you.")
                    .flText(.body)
                    .fixedSize(horizontal: false, vertical: true)
            }
            PipelineIndex(jump: jump)
        }
    }
}

/// Five numbered steps joined by a hairline; tap one to jump to it.
private struct PipelineIndex: View {
    var jump: (String) -> Void
    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            ForEach(ScienceStep.allCases) { step in
                if step != .sense { Spacer(minLength: FLSpacing.xs) }
                Button { FLHaptics.select(); jump(step.anchor) } label: {
                    VStack(alignment: .leading, spacing: 6) {
                        Circle().fill(FLColor.accent).frame(width: 5, height: 5)
                            .padding(.bottom, 4)
                        Text(step.number).font(FLFont.numeral(.caption2)).foregroundStyle(FLColor.textTertiary)
                        Text(step.name).font(FLFont.scaled(13, weight: .medium, relativeTo: .footnote))
                            .foregroundStyle(FLColor.textSecondary)
                            .lineLimit(1)
                    }
                    .frame(minHeight: 44, alignment: .leading)
                    .fixedSize()
                }
                .buttonStyle(FLPressableStyle())
                .accessibilityLabel("Step \(step.rawValue + 1): \(step.name)")
                .accessibilityHint("Scrolls to this step")
            }
        }
        .background(alignment: .topLeading) {
            Rectangle().fill(FLColor.hairlineStrong).frame(height: 0.5).offset(y: 2.25)
                .accessibilityHidden(true)
        }
    }
}

struct LiveDot: View {
    var active: Bool
    @State private var on = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var body: some View {
        Circle().fill(active ? FLColor.accent : FLColor.textTertiary)
            .frame(width: 6, height: 6)
            .overlay(Circle().stroke(FLColor.accent.opacity(active ? 0.5 : 0), lineWidth: 1)
                .scaleEffect(on ? 2.6 : 1).opacity(on ? 0 : 1))
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(.easeOut(duration: 1.6).repeatForever(autoreverses: false)) { on = true }
            }
            .accessibilityLabel(active ? "Live" : "Idle")
    }
}

// MARK: - Sense

private struct SignalsSection: View {
    let model: LiveSessionModel
    let services: UIServices
    var body: some View {
        let f = model.latestFrame
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            StepHeader("Face signals", step: .sense,
                       explainer: "The camera is read frame by frame and turned into numbers: blinks, eyelid closure, gaze and head pose. Each frame is discarded at once.")
            FaceMeshPanel(model: model, services: services).frame(height: 280)
                .overlay(alignment: .bottomTrailing) {
                    Text("0 bytes of video stored").font(FLFont.caption).foregroundStyle(FLColor.accent)
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(FLColor.bg.opacity(0.72), in: Capsule())
                        .padding(FLSpacing.s)
                }
            HStack(alignment: .top, spacing: FLSpacing.xs) {
                FLRingGauge(title: "Blink rate", value: GaugeScale.blinkRate(f?.blinkRate ?? 0),
                            valueText: "\(Int((f?.blinkRate ?? 0).rounded()))", unit: "/min", color: FLColor.accent)
                FLRingGauge(title: "PERCLOS", value: GaugeScale.perclos(f?.perclos ?? 0),
                            valueText: UIFormat.percent(Double(f?.perclos ?? 0)), unit: "closed",
                            color: FLColor.drowsy)
                FLRingGauge(title: "Gaze off zone", value: GaugeScale.gazeDeviation(f?.gazeDeviation ?? 0),
                            valueText: UIFormat.degrees(f?.gazeDeviation ?? 0), unit: "deviation",
                            color: FLColor.distracted)
                HeadPoseGauge(yaw: f?.headYaw ?? 0, pitch: f?.headPitch ?? 0, roll: f?.headRoll ?? 0)
            }
        }
    }
}

// MARK: - Heart

private struct HeartSection: View {
    let model: LiveSessionModel
    var body: some View {
        let hr = model.heartRate
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            StepHeader("Heart rate from the camera",
                       explainer: "Each heartbeat shifts your skin colour very slightly. FocusLoop reads that from the camera (rPPG) to get your pulse.") {
                SignalQuality(quality: hr?.quality ?? 0, motion: hr?.motionFlagged ?? false)
            }
            HStack(alignment: .center, spacing: FLSpacing.l) {
                VStack(alignment: .leading, spacing: FLSpacing.xxs) {
                    HStack(alignment: .firstTextBaseline, spacing: 4) {
                        Text(UIFormat.bpm(hr?.bpm)).flText(.metric)
                            .contentTransition(.numericText())
                            .animation(FLMotion.quick, value: hr?.bpm)
                        Text("BPM").font(FLFont.unit).foregroundStyle(FLColor.textTertiary)
                    }
                    Text(hr?.baselineBPM.map { "Resting \(UIFormat.bpm($0))" } ?? "Resting --").flText(.caption)
                }
                .fixedSize()
                .accessibilityElement(children: .combine)
                PulseWaveform(samples: hr?.waveform ?? [], color: FLColor.heart)
                    .frame(height: 56)
            }
        }
    }
}

struct SignalQuality: View {
    var quality: Float
    var motion: Bool
    var body: some View {
        let level = quality >= 0.7 ? 3 : quality >= 0.45 ? 2 : quality > 0.1 ? 1 : 0
        let label = motion ? "Motion" : ["No signal", "Poor", "Fair", "Good"][level]
        let color = motion ? FLColor.distracted : [FLColor.textTertiary, FLColor.stressed, FLColor.distracted, FLColor.focused][level]
        HStack(spacing: 6) {
            HStack(alignment: .bottom, spacing: 2) {
                ForEach(0..<3) { i in
                    Capsule().fill(i < level ? color : FLColor.hairlineStrong)
                        .frame(width: 2, height: CGFloat(5 + i * 3))
                }
            }
            Text(label).font(FLFont.caption).foregroundStyle(level == 0 && !motion ? FLColor.textTertiary : color)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Signal quality \(label)")
    }
}

/// Pulse waveform: a single 1.5pt line with a soft sweep that reads as "live".
struct PulseWaveform: View {
    var samples: [Float]
    var color: Color
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: reduceMotion)) { ctx in
            Canvas { gc, size in
                guard samples.count > 1 else {
                    var p = Path(); p.move(to: CGPoint(x: 0, y: size.height / 2)); p.addLine(to: CGPoint(x: size.width, y: size.height / 2))
                    gc.stroke(p, with: .color(FLColor.hairlineStrong), style: StrokeStyle(lineWidth: 1, dash: [2, 4]))
                    return
                }
                var p = Path()
                for (i, v) in samples.enumerated() {
                    let x = size.width * CGFloat(i) / CGFloat(samples.count - 1)
                    let y = size.height / 2 - CGFloat(max(-1, min(1, v))) * size.height * 0.42
                    i == 0 ? p.move(to: CGPoint(x: x, y: y)) : p.addLine(to: CGPoint(x: x, y: y))
                }
                gc.stroke(p, with: .color(color), style: FLChart.line)
                if !reduceMotion {
                    let x = CGFloat(ctx.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 2) / 2) * size.width
                    gc.fill(Path(CGRect(x: x - 40, y: 0, width: 40, height: size.height)),
                            with: .linearGradient(Gradient(colors: [FLColor.bg.opacity(0), FLColor.bg.opacity(0.7)]),
                                                  startPoint: CGPoint(x: x - 40, y: 0), endPoint: CGPoint(x: x, y: 0)))
                }
            }
        }
        .accessibilityLabel("Pulse waveform")
    }
}

// MARK: - Content (brain)

/// Framed cortex with one soft glow, the average-brain caption, and the 7 network bars.
struct BrainCard: View {
    let model: LiveSessionModel
    @State private var visible = true

    var body: some View {
        let acts = model.timeline?.activations(at: model.playhead) ?? Array(repeating: 0, count: 7)
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            StepHeader("Predicted brain response to this content", step: .predict,
                       explainer: "A model of an average brain predicts how strongly each moment of the lecture engages seven brain networks. Dips mark where content tends to lose people.")
            VStack(spacing: FLSpacing.s) {
                ZStack(alignment: .bottomTrailing) {
                    RadialGradient(colors: [FLColor.accent.opacity(0.09), FLColor.accent.opacity(0.03), .clear],
                                   center: .center, startRadius: 0, endRadius: 170)
                        .accessibilityHidden(true)
                    if visible && model.timeline != nil && !ThermalStatus.shared.cortexPaused {
                        BrainCortexView(activations: acts)
                            .accessibilityLabel("3D brain: predicted network response to this content, average brain")
                    } else if model.timeline == nil {
                        Text("No lecture content loaded.").flText(.caption)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                    Text(UIFormat.timer(model.playhead))
                        .font(FLFont.numeral(.footnote)).foregroundStyle(FLColor.textTertiary)
                        .padding(FLSpacing.m)
                        .accessibilityLabel("Lecture position \(UIFormat.timer(model.playhead))")
                }
                .frame(height: 300)
                .frame(maxWidth: .infinity)
                .overlay(alignment: .topTrailing) {
                    if model.timeline?.isMock ?? false { FLPreviewInfoButton().padding(FLSpacing.xxs) }
                }
                .background(FLColor.surface, in: RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous).strokeBorder(FLColor.hairline, lineWidth: 0.5))
                .clipShape(RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous))
                .onGeometryChange(for: Bool.self) { proxy in
                    guard let vis = proxy.bounds(of: .scrollView) else { return true }
                    return vis.intersects(CGRect(origin: .zero, size: proxy.size))
                } action: { visible = $0 }
                Text("Predicted brain response to this content (average brain)")
                    .flText(.caption)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            NetworkBars(activations: acts)
        }
    }
}

/// The 7 Yeo networks in `BrainNetwork` order as thin horizontal meters with short, readable labels.
/// VoiceOver reads each network's full name.
public struct NetworkBars: View {
    var activations: [Float]
    public init(activations: [Float]) { self.activations = activations }

    static func shortName(_ n: BrainNetwork) -> String {
        switch n {
        case .visual: "Visual"
        case .somatomotor: "Motor"
        case .dorsalAttention: "Dorsal attention"
        case .ventralAttention: "Salience"
        case .limbic: "Limbic"
        case .control: "Control"
        case .defaultMode: "Default mode"
        }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(BrainNetwork.allCases) { n in row(n) }
        }
    }

    private func value(_ n: BrainNetwork) -> Double {
        GaugeScale.clamp01(Double(n.rawValue < activations.count ? activations[n.rawValue] : 0))
    }

    private func row(_ n: BrainNetwork) -> some View {
        let v = value(n)
        return HStack(spacing: FLSpacing.s) {
            Text(Self.shortName(n)).font(FLFont.caption).foregroundStyle(FLColor.textSecondary)
                .lineLimit(1)
                .frame(width: 112, alignment: .leading)
            FLLineMeter(value: v, color: FLColor.accent, height: 3)
                .animation(.easeOut(duration: 0.35), value: v)
            Text(UIFormat.percent(v)).font(FLFont.numeral(.caption)).foregroundStyle(FLColor.textTertiary)
                .frame(width: 38, alignment: .trailing)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(n.displayName): \(UIFormat.percent(v)) of full predicted activity")
    }
}

// MARK: - Lanes

/// Two stacked lanes on one time axis: predicted content response (runs ahead of now) and your attention.
struct LanesCard: View {
    let model: LiveSessionModel
    /// Draw a card around the lanes (used when shown outside the Science view's sectioned layout).
    var framed = true

    var body: some View {
        let content = lanes
        if framed { content.flCard() } else { content }
    }

    private var lanes: some View {
        let now = model.elapsed
        let domain = (now - 60)...(now + 15)
        return VStack(alignment: .leading, spacing: FLSpacing.m) {
            FLSectionHeader("Content vs attention", eyebrow: framed ? "Two lanes" : nil)
            if !framed {
                Text("Top: how engaging the content is predicted to be, running 15 s ahead of now. Bottom: where your attention actually was.")
                    .flText(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
            VStack(alignment: .leading, spacing: FLSpacing.xs) {
                laneTitle("Predicted brain response to this content", trailing: "next 15 s")
                PredictedLane(model: model, domain: domain).frame(height: 64)
            }
            VStack(alignment: .leading, spacing: FLSpacing.xs) {
                laneTitle("Your attention", trailing: "last 60 s")
                AttentionLane(history: model.stateHistory, now: now, domain: domain).frame(height: 8)
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: FLSpacing.m) { valleyKey; stateKey }
                VStack(alignment: .leading, spacing: FLSpacing.xs) { valleyKey; stateKey }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Amber bands mark predicted valleys. Attention colors: focused mint, distracted amber, drowsy violet, stressed coral.")
        }
    }

    private var valleyKey: some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 1).fill(FLColor.distracted.opacity(0.35)).frame(width: 10, height: 10)
            Text("Predicted valley").font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
        }
        .fixedSize()
    }

    private var stateKey: some View {
        HStack(spacing: FLSpacing.s) {
            ForEach(AttentionState.allCases) { s in
                HStack(spacing: 4) {
                    Circle().fill(s.color).frame(width: 6, height: 6)
                    Text(s.displayName).font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                }
                .fixedSize()
            }
        }
    }

    private func laneTitle(_ s: String, trailing: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(s).font(FLFont.caption).foregroundStyle(FLColor.textSecondary)
            Spacer(minLength: FLSpacing.xs)
            Text(trailing).font(FLFont.caption).foregroundStyle(FLColor.textTertiary).accessibilityHidden(true)
        }
    }
}

private struct PredictedLane: View {
    let model: LiveSessionModel
    let domain: ClosedRange<Double>
    struct P: Identifiable { var t: Double; var e: Double; var valley: Bool; var id: Double { t } }

    var body: some View {
        let pts = samples()
        Chart {
            ForEach(pts) { p in
                if p.valley {
                    RectangleMark(xStart: .value("t", p.t - 0.5), xEnd: .value("t", p.t + 0.5), yStart: .value("e", 0), yEnd: .value("e", 1))
                        .foregroundStyle(FLColor.distracted.opacity(0.14))
                }
            }
            ForEach(pts) { p in
                AreaMark(x: .value("t", p.t), y: .value("Engagement", p.e))
                    .foregroundStyle(FLChart.area(FLColor.accentBlue))
                    .interpolationMethod(.catmullRom)
                LineMark(x: .value("t", p.t), y: .value("Engagement", p.e))
                    .foregroundStyle(FLColor.accentBlue)
                    .interpolationMethod(.catmullRom)
                    .lineStyle(FLChart.line)
            }
            RuleMark(x: .value("now", model.elapsed)).foregroundStyle(FLColor.textTertiary)
                .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 3]))
        }
        .chartXScale(domain: domain)
        .chartYScale(domain: 0...1)
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .clipped()
        .accessibilityLabel("Predicted engagement for the content, including the next 15 seconds")
    }

    private func samples() -> [P] {
        guard let tl = model.timeline, tl.duration > 0 else { return [] }
        let now = model.elapsed, ph = model.playhead
        let valleys = tl.valleys()
        return stride(from: domain.lowerBound.rounded(.down), through: domain.upperBound, by: 1).compactMap { t in
            guard t >= 0 else { return nil }
            var x = ph + (t - now)
            x = (x.truncatingRemainder(dividingBy: tl.duration) + tl.duration).truncatingRemainder(dividingBy: tl.duration)
            let inV = valleys.contains { x >= $0.start && x < $0.end }
            return P(t: t, e: Double(tl.engagement(at: x)), valley: inV)
        }
    }
}

struct AttentionLane: View {
    let history: [StateEstimate]
    let now: Double
    let domain: ClosedRange<Double>
    var body: some View {
        let recent = history.filter { $0.t >= domain.lowerBound - 1 }
        let segs = StateSegmenter.segments(recent.map { ($0.t, $0.state) }, end: now)
        Chart {
            RectangleMark(xStart: .value("start", domain.lowerBound), xEnd: .value("end", domain.upperBound),
                          yStart: .value("y", 0), yEnd: .value("y", 1))
                .foregroundStyle(FLColor.surfaceHigh)
            ForEach(Array(segs.enumerated()), id: \.offset) { _, s in
                RectangleMark(xStart: .value("start", s.start), xEnd: .value("end", s.end), yStart: .value("y", 0), yEnd: .value("y", 1))
                    .foregroundStyle(s.state.color)
            }
            RuleMark(x: .value("now", now)).foregroundStyle(FLColor.textTertiary)
                .lineStyle(StrokeStyle(lineWidth: 0.5))
        }
        .chartXScale(domain: domain)
        .chartYScale(domain: 0...1)
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .clipShape(Capsule())
        .accessibilityLabel("Your attention over the last minute: \(segs.last?.state.displayName ?? "no data") now")
    }
}

// MARK: - Understand

private struct StateProbabilitiesSection: View {
    let model: LiveSessionModel
    var body: some View {
        let probs = model.estimate?.probabilities ?? [1, 0, 0, 0]
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            StepHeader("Attention state", step: .understand,
                       explainer: "A small model on your phone turns the last few seconds of signals into the chance you are focused, distracted, drowsy or stressed.") {
                FLStateChip(model.state).fixedSize()
            }
            VStack(spacing: FLSpacing.m) {
                ForEach(AttentionState.allCases) { s in
                    let p = Double(s.index < probs.count ? probs[s.index] : 0)
                    HStack(spacing: FLSpacing.s) {
                        Circle().fill(s.color).frame(width: 6, height: 6)
                        Text(s.displayName).font(FLFont.callout).foregroundStyle(FLColor.textPrimary)
                            .frame(width: 92, alignment: .leading)
                        FLLineMeter(value: p, color: s.color, height: 3)
                            .animation(.easeOut(duration: 0.5), value: p)
                        Text(UIFormat.percent(p)).font(FLFont.numeral(.footnote)).foregroundStyle(FLColor.textSecondary)
                            .frame(width: 40, alignment: .trailing)
                    }
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("\(s.displayName) \(UIFormat.percent(p))")
                }
            }
            HStack {
                Text("Last minute").flText(.caption)
                Spacer()
                Text("Source: \(model.estimate?.source.rawValue.capitalized ?? "-")").flText(.caption)
            }
            StateProbabilityChart(history: Array(model.stateHistory.suffix(60))).frame(height: 72)
        }
    }
}

private struct StateProbabilityChart: View {
    let history: [StateEstimate]
    struct P: Identifiable { var t: Double; var s: AttentionState; var p: Double; var id: String { "\(t)-\(s.rawValue)" } }
    var body: some View {
        let pts = history.flatMap { e in AttentionState.allCases.map { P(t: e.t, s: $0, p: Double(e.probability(of: $0))) } }
        Chart(pts) { p in
            LineMark(x: .value("t", p.t), y: .value("p", p.p), series: .value("State", p.s.displayName))
                .foregroundStyle(by: .value("State", p.s.displayName))
                .interpolationMethod(.monotone)
                .lineStyle(FLChart.line)
        }
        .chartForegroundStyleScale(domain: AttentionState.allCases.map(\.displayName), range: AttentionState.allCases.map(\.color))
        .chartYScale(domain: 0...1)
        .chartLegend(.hidden)
        .chartXAxis(.hidden)
        .chartYAxis {
            AxisMarks(values: [0, 1]) { _ in AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(FLColor.hairline) }
        }
        .accessibilityLabel("State probabilities over the last minute")
    }
}

// MARK: - Predict

private struct LapseRiskSection: View {
    let model: LiveSessionModel
    var body: some View {
        let risk = model.lapseRisk
        let p = Double(risk?.probability ?? 0), th = Double(risk?.threshold ?? 0.6)
        let alert = risk?.isAlert ?? false
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            StepHeader("Lapse forecast",
                       explainer: "Your signals and the content's predicted dips are combined to forecast a lapse before it happens, so a nudge can arrive early.") {
                if risk?.usedContent ?? false { FLTag("Uses content", tint: FLColor.accentBlue) }
            }
            HStack(alignment: .lastTextBaseline) {
                VStack(alignment: .leading, spacing: FLSpacing.xxs) {
                    Text("P(lapse in next \(Int(risk?.horizon ?? 15)) s)").flText(.caption)
                    Text(UIFormat.percent(p)).flText(.metric)
                        .contentTransition(.numericText())
                        .animation(FLMotion.quick, value: p)
                }
                Spacer()
                Text(alert ? "Above your threshold" : "Below your threshold")
                    .font(FLFont.callout).foregroundStyle(alert ? FLColor.risk : FLColor.focused)
            }
            .accessibilityElement(children: .combine)
            FLLineMeter(value: p, color: FLColor.risk, height: 3, marker: th)
                .accessibilityElement()
                .accessibilityLabel("Lapse risk \(UIFormat.percent(p)), threshold \(UIFormat.percent(th))")
            RiskForecastChart(model: model).frame(height: 128).padding(.top, FLSpacing.m)
            AttentionLane(history: model.stateHistory, now: model.elapsed, domain: (model.elapsed - 60)...(model.elapsed + 15))
                .frame(height: 6)
            Text("Dashed line = forecast ahead of now").flText(.caption)
        }
    }
}

private struct RiskForecastChart: View {
    let model: LiveSessionModel
    struct P: Identifiable { var t: Double; var p: Double; var kind: String; var id: String { "\(kind)\(t)" } }
    var body: some View {
        let now = model.elapsed
        let hist = model.riskHistory.filter { $0.t >= now - 60 }.map { P(t: $0.t, p: Double($0.probability), kind: "Observed") }
        var fc = model.forecastLine.map { P(t: $0.t, p: Double($0.p), kind: "Forecast") }
        if let last = hist.last { fc.insert(P(t: last.t, p: last.p, kind: "Forecast"), at: 0) }
        let th = Double(model.lapseRisk?.threshold ?? 0.6)
        return Chart {
            RuleMark(y: .value("Threshold", th)).foregroundStyle(FLColor.hairlineStrong)
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                .annotation(position: .top, alignment: .leading, spacing: 2) {
                    Text("threshold").font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                }
            ForEach(hist) { p in
                AreaMark(x: .value("t", p.t), y: .value("risk", p.p))
                    .foregroundStyle(FLChart.area(FLColor.risk))
                    .interpolationMethod(.monotone)
                LineMark(x: .value("t", p.t), y: .value("risk", p.p), series: .value("k", "Observed"))
                    .foregroundStyle(FLColor.risk)
                    .interpolationMethod(.monotone)
                    .lineStyle(FLChart.line)
            }
            ForEach(fc) { p in
                LineMark(x: .value("t", p.t), y: .value("risk", p.p), series: .value("k", "Forecast"))
                    .foregroundStyle(FLColor.risk.opacity(0.8))
                    .interpolationMethod(.monotone)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: [4, 4]))
            }
            RuleMark(x: .value("now", now)).foregroundStyle(FLColor.textTertiary)
                .lineStyle(StrokeStyle(lineWidth: 0.5))
                .annotation(position: .top, spacing: 2) { Text("now").font(FLFont.caption).foregroundStyle(FLColor.textSecondary) }
        }
        .chartXScale(domain: (now - 60)...(now + 15))
        .chartYScale(domain: 0...1)
        .chartXAxis(.hidden)
        .chartYAxis {
            AxisMarks(position: .trailing, values: [0, 0.5, 1]) { v in
                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(FLColor.hairline)
                AxisValueLabel { if let d = v.as(Double.self) { Text(UIFormat.percent(d)).font(FLFont.caption).foregroundStyle(FLColor.textTertiary) } }
            }
        }
        .accessibilityLabel("Lapse risk over the last minute with a 15 second forecast")
    }
}

// MARK: - Intervene

/// The most recent nudge decision and the randomization behind it.
private struct InterveneSection: View {
    let model: LiveSessionModel
    var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            StepHeader("Nudges", step: .intervene,
                       explainer: "When you drift, or a lapse is forecast, FocusLoop picks a gentle nudge at random from the ones that fit, including sometimes no nudge at all, so it can measure what really helps.")
            if let d = model.lastDecision {
                DecisionSummary(decision: d, demo: model.demoDirector && d.randomizationProbability >= 1)
                OddsList(decision: d)
            } else {
                Text("No nudge yet this session.").flText(.callout)
            }
        }
    }
}

private struct DecisionSummary: View {
    let decision: NudgeDecision
    let demo: Bool
    var body: some View {
        let kind = decision.policy == .preemptive ? "Before a predicted lapse" : "Reactive"
        let detail = demo ? kind : "\(kind) \u{00B7} p = \(String(format: "%.2f", decision.randomizationProbability))"
        VStack(alignment: .leading, spacing: FLSpacing.xxs) {
            Text(demo ? "Last nudge" : "Last pick").flText(.caption)
            Text(decision.arm.displayName).flText(.title)
            Text(detail).flText(.callout)
        }
        .accessibilityElement(children: .combine)
    }
}

/// Selection probability of every eligible arm at the last decision; the picked arm in the accent.
private struct OddsList: View {
    let decision: NudgeDecision
    private struct Odd: Identifiable { let arm: NudgeArm; let p: Double; var id: String { arm.rawValue } }
    private var odds: [Odd] {
        decision.armProbabilities.compactMap { k, v in NudgeArm(rawValue: k).map { Odd(arm: $0, p: v) } }
            .sorted { $0.p == $1.p ? $0.arm.displayName < $1.arm.displayName : $0.p > $1.p }
    }
    var body: some View {
        let list = odds
        if list.count > 1 {
            VStack(alignment: .leading, spacing: FLSpacing.s) {
                FLEyebrow("The odds it rolled")
                ForEach(list) { o in row(o, picked: o.arm == decision.arm) }
            }
        }
    }
    private func row(_ o: Odd, picked: Bool) -> some View {
        HStack(spacing: FLSpacing.s) {
            Text(o.arm.displayName).font(FLFont.caption)
                .foregroundStyle(picked ? FLColor.textPrimary : FLColor.textSecondary)
                .lineLimit(2)
                .frame(width: 124, alignment: .leading)
            FLLineMeter(value: o.p, color: picked ? FLColor.accent : FLColor.textTertiary, height: 2)
            Text(String(format: "%.2f", o.p)).font(FLFont.numeral(.caption))
                .foregroundStyle(picked ? FLColor.accent : FLColor.textTertiary)
                .frame(width: 40, alignment: .trailing)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(o.arm.displayName): probability \(String(format: "%.2f", o.p))\(picked ? ", picked" : "")")
    }
}

// MARK: - Learn

private struct PosteriorsSection: View {
    let model: LiveSessionModel
    var body: some View {
        let groups: [(String, [ArmPosterior])] = groupsFor(model.posteriors)
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            StepHeader("What the trial is learning", step: .learn,
                       explainer: "Every nudge is a tiny randomized experiment. These beliefs update after each one, so FocusLoop picks what brings you back more often.")
            Text("Each bar is the posterior chance a nudge brings you back (Beta mean). n = times tried.")
                .flText(.caption)
                .fixedSize(horizontal: false, vertical: true)
            if groups.isEmpty {
                Text("No nudges yet.").flText(.callout)
            }
            ForEach(groups, id: \.0) { title, arms in
                VStack(alignment: .leading, spacing: FLSpacing.s) {
                    FLEyebrow(title)
                    ForEach(arms) { a in PosteriorRow(p: a, highlighted: isLast(a)) }
                }
            }
        }
    }

    private func isLast(_ a: ArmPosterior) -> Bool {
        guard let d = model.lastDecision, d.arm == a.arm, d.policy == a.policy else { return false }
        return d.policy == .preemptive || d.context.state == a.state
    }

    private func groupsFor(_ ps: [ArmPosterior]) -> [(String, [ArmPosterior])] {
        var out: [(String, [ArmPosterior])] = []
        for s in AttentionState.allCases where s != .focused {
            let g = ps.filter { $0.policy == .reactive && $0.state == s }
            if !g.isEmpty { out.append(("When \(s.displayName.lowercased())", g)) }
        }
        let pre = ps.filter { $0.policy == .preemptive }
        if !pre.isEmpty { out.append(("Before a predicted lapse", pre)) }
        return out
    }
}

private struct PosteriorRow: View {
    let p: ArmPosterior
    var highlighted: Bool
    var body: some View {
        let sd = (p.alpha * p.beta / ((p.alpha + p.beta) * (p.alpha + p.beta) * (p.alpha + p.beta + 1))).squareRoot()
        let tint = p.arm == .noNudge ? FLColor.textTertiary : FLColor.accent
        HStack(spacing: FLSpacing.s) {
            Text(p.arm.displayName).font(FLFont.caption)
                .foregroundStyle(highlighted ? FLColor.textPrimary : FLColor.textSecondary)
                .lineLimit(2)
                .frame(width: 124, alignment: .leading)
            GeometryReader { g in
                let w = g.size.width
                ZStack(alignment: .leading) {
                    Capsule().fill(FLColor.hairlineStrong).frame(height: 2)
                    Capsule().fill(tint.opacity(0.22))
                        .frame(width: max(2, w * 2 * sd), height: 6)
                        .offset(x: w * max(0, p.mean - sd))
                    Capsule().fill(tint).frame(width: max(2, w * p.mean), height: 2)
                    Circle().fill(tint).frame(width: 6, height: 6).offset(x: w * p.mean - 3)
                }
                .frame(maxHeight: .infinity)
            }
            .frame(height: 16)
            .animation(.easeOut(duration: 0.5), value: p.mean)
            Text("n=\(p.pulls)").font(FLFont.numeral(.caption)).foregroundStyle(highlighted ? FLColor.accent : FLColor.textTertiary)
                .frame(width: 40, alignment: .trailing)
        }
        .frame(minHeight: 24)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(p.arm.displayName): posterior mean \(UIFormat.percent(p.mean)), tried \(p.pulls) times\(highlighted ? ", last pick" : "")")
    }
}
