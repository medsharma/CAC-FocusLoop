import Charts
import SwiftUI
import UIKit

/// Renders a `SessionReport` (built by Persistence; `PreviewReportBuilder` before integration).
public struct SessionReportView: View {
    let report: SessionReport
    let services: UIServices?
    /// True while the on-device coach is still writing this report's insights.
    var insightsPending = false

    @State private var exportURL: URL?
    @State private var imageURL: URL?
    @State private var hrURL: URL?
    @State private var exporting = false
    @State private var exportError: String?
    @State private var sharing: ShareItem?

    public init(report: SessionReport, services: UIServices?, insightsPending: Bool = false) {
        self.report = report; self.services = services; self.insightsPending = insightsPending
    }

    public var body: some View {
        ScrollViewReader { proxy in
        ScrollView {
            VStack(alignment: .leading, spacing: FLSpacing.xxl) {
                ReportHeader(report: report)
                FocusTimelineSection(report: report)
                FLDivider()
                RefocusSection(report: report).id("refocus")
                if !report.heartRate.isEmpty {
                    FLDivider()
                    HeartRateSection(report: report)
                }
                if !report.predictedDrift.isEmpty {
                    FLDivider()
                    DriftOverlaySection(report: report).id("drift")
                }
                FLDivider()
                PersonalTrialCard(effects: report.effects, isSynthetic: report.isReplay).id("trial")
                FLDivider()
                InsightsCard(insights: report.insights, isSynthetic: report.isReplay, loading: insightsPending)
                VStack(alignment: .leading, spacing: FLSpacing.l) {
                    exportButton.id("export")
                    VStack(spacing: FLSpacing.xs) {
                        Text("FocusLoop supports focus; it is not a medical device.")
                            .flText(.caption)
                            .frame(maxWidth: .infinity)
                        FLDataFootnote(mock: report.contentIsMock && report.lectureID != nil, synthetic: report.isReplay,
                                       demo: services?.sessionUsedDemoNudges(report.sessionID) ?? false)
                    }
                }
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.top, FLSpacing.xs)
            .padding(.bottom, FLSpacing.xxl)
        }
        .modifier(FLLaunchScroll(proxy: proxy))
        }
        .flScreen("Session report", inline: true)
        .alert("Export failed", isPresented: Binding(get: { exportError != nil }, set: { if !$0 { exportError = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(exportError ?? "") }
        .sheet(item: $sharing) { item in
            ActivityShareSheet(url: item.url).ignoresSafeArea().presentationDetents([.medium, .large])
        }
    }

    @ViewBuilder private var exportButton: some View {
        if let services {
            VStack(spacing: FLSpacing.xs) {
                if let url = exportURL {
                    ShareLink(item: url) {
                        Label("Share PDF", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(FLPrimaryButtonStyle())
                    .accessibilityIdentifier("report.sharePDF")
                } else {
                    Button {
                        exporting = true
                        Task {
                            do {
                                let url = try await services.exportReport(report)
                                exportURL = url
                                sharing = ShareItem(url: url)    // one tap: the share sheet opens straight away
                            } catch { exportError = error.localizedDescription }
                            exporting = false
                        }
                    } label: {
                        HStack(spacing: FLSpacing.xs) {
                            if exporting { ProgressView().tint(FLColor.onAccent) }
                            Label("Export PDF", systemImage: "doc.richtext")
                        }
                    }
                    .buttonStyle(FLPrimaryButtonStyle())
                    .disabled(exporting)
                    .accessibilityIdentifier("report.export")
                }
                HStack(spacing: FLSpacing.l) {
                    if let url = imageURL {
                        ShareLink(item: url) { Label("Share image", systemImage: "square.and.arrow.up") }
                    } else {
                        Button {
                            Task {
                                do { imageURL = try await services.exportReportImage(report) } catch { exportError = error.localizedDescription }
                            }
                        } label: { Label("Export image", systemImage: "photo") }
                    }
                    if !report.heartRate.isEmpty {
                        if let url = hrURL {
                            ShareLink(item: url) { Label("Share heart rate", systemImage: "square.and.arrow.up") }
                        } else {
                            Button {
                                do { hrURL = try services.exportHeartRateCSV(report) } catch { exportError = error.localizedDescription }
                            } label: { Label("Heart rate (CSV)", systemImage: "heart.text.square") }
                        }
                    }
                }
                .buttonStyle(FLTextButtonStyle(color: FLColor.textSecondary))
                .labelStyle(FLButtonLabelStyle())
                .font(FLFont.callout)
                .frame(maxWidth: .infinity)
            }
        }
    }
}

// MARK: - Header

private struct ReportHeader: View {
    let report: SessionReport
    var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            VStack(alignment: .leading, spacing: FLSpacing.s) {
                Text(report.startedAt.formatted(date: .abbreviated, time: .shortened)).flText(.eyebrow)
                Text(report.lectureTitle ?? "Focus session").flText(.largeTitle)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.isHeader)
            }
            FLDivider()
            FLMetricStrip([
                FLMetric("Focused", value: UIFormat.percent(report.focusedFraction), color: FLColor.focused),
                FLMetric("Duration", value: UIFormat.timer(report.duration)),
                FLMetric("Back in", value: report.medianRefocusSeconds.map(UIFormat.oneDecimal) ?? "--", unit: "s"),
            ])
            FLDivider()
        }
    }
}

/// Section title + eyebrow + optional trailing accessory, with the standard gap to its content.
private struct ReportSection<Trailing: View, Content: View>: View {
    let title: String
    let eyebrow: String
    let trailing: Trailing
    let content: Content
    init(_ title: String, eyebrow: String, @ViewBuilder trailing: () -> Trailing, @ViewBuilder content: () -> Content) {
        self.title = title; self.eyebrow = eyebrow; self.trailing = trailing(); self.content = content()
    }
    var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader(title, eyebrow: eyebrow, large: true) { trailing }
            content
        }
    }
}

private extension ReportSection where Trailing == EmptyView {
    init(_ title: String, eyebrow: String, @ViewBuilder content: () -> Content) {
        self.init(title, eyebrow: eyebrow, trailing: { EmptyView() }, content: content)
    }
}

/// Session-time x axis: caption labels in `textTertiary`, no grid.
private func timerAxis() -> some AxisContent {
    AxisMarks(values: .automatic(desiredCount: 4)) { v in
        AxisValueLabel { if let d = v.as(Double.self) { Text(UIFormat.timer(d)).font(FLFont.caption).foregroundStyle(FLColor.textTertiary) } }
    }
}

// MARK: - Focus timeline

private struct FocusTimelineSection: View {
    let report: SessionReport
    var body: some View {
        let fr = StateSegmenter.fractions(report.segments)
        ReportSection("Focus timeline", eyebrow: "Your session") {
            VStack(alignment: .leading, spacing: FLSpacing.m) {
                Chart {
                    ForEach(Array(report.segments.enumerated()), id: \.offset) { _, s in
                        RectangleMark(xStart: .value("Start", s.start), xEnd: .value("End", s.end),
                                      yStart: .value("y", 0), yEnd: .value("y", 1))
                            .foregroundStyle(s.state.color)
                    }
                }
                .chartYAxis(.hidden)
                .chartXAxis { timerAxis() }
                .chartXScale(domain: 0...max(1, report.duration))
                .chartPlotStyle { $0.clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous)) }
                .frame(height: 44)
                .accessibilityLabel("Focus timeline. Drift segments are colored by state.")
                Grid(alignment: .leading, horizontalSpacing: FLSpacing.l, verticalSpacing: FLSpacing.s) {
                    GridRow { stateKey(.focused, fr); stateKey(.distracted, fr) }
                    GridRow { stateKey(.drowsy, fr); stateKey(.stressed, fr) }
                }
            }
        }
    }

    private func stateKey(_ s: AttentionState, _ fr: [Double]) -> some View {
        HStack(spacing: FLSpacing.xs) {
            Circle().fill(s.color).frame(width: 6, height: 6)
            Text(s.displayName).font(FLFont.callout).foregroundStyle(FLColor.textSecondary)
            Text(UIFormat.percent(s.index < fr.count ? fr[s.index] : 0)).font(FLFont.numeral(.subheadline))
                .foregroundStyle(FLColor.textPrimary)
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Refocus

private struct RefocusSection: View {
    let report: SessionReport
    var body: some View {
        ReportSection("Refocus times", eyebrow: "How fast you came back") {
            Text("\(report.nudgeCount) nudges, \(report.preemptiveCount) pre-emptive").flText(.caption)
        } content: {
            if report.refocusTimes.isEmpty {
                Text("No lapses this session.").flText(.callout)
            } else {
                Chart {
                    ForEach(Array(report.refocusTimes.enumerated()), id: \.offset) { i, s in
                        BarMark(x: .value("Lapse", "\(i + 1)"), y: .value("Seconds", s), width: .fixed(14))
                            .foregroundStyle(FLColor.accent)
                            .cornerRadius(FLChart.barRadius)
                            .annotation(position: .top, spacing: 4) {
                                Text(UIFormat.oneDecimal(s)).font(FLFont.numeral(.caption)).foregroundStyle(FLColor.textSecondary)
                            }
                    }
                    if let m = report.medianRefocusSeconds {
                        RuleMark(y: .value("Median", m)).foregroundStyle(FLColor.hairlineStrong)
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                            .annotation(position: .top, alignment: .leading, spacing: 2) {
                                Text("median").font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                            }
                    }
                }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .trailing, values: .automatic(desiredCount: 3)) {
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(FLColor.hairline)
                        AxisValueLabel().font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                    }
                }
                .frame(height: 140)
                .accessibilityLabel("Refocus times in seconds, median \(report.medianRefocusSeconds.map(UIFormat.oneDecimal) ?? "n/a")")
            }
        }
    }
}

private struct HeartRateSection: View {
    let report: SessionReport
    var body: some View {
        ReportSection("Heart rate", eyebrow: "rPPG over the session") {
            if let r = report.restingHeartRate { Text("Resting \(Int(r)) BPM").flText(.caption) }
        } content: {
            Chart {
                ForEach(Array(report.heartRate.enumerated()), id: \.offset) { _, p in
                    LineMark(x: .value("t", p.t), y: .value("BPM", p.value))
                        .foregroundStyle(FLColor.heart).interpolationMethod(.catmullRom)
                        .lineStyle(FLChart.line)
                }
                if let r = report.restingHeartRate {
                    RuleMark(y: .value("Resting", r)).foregroundStyle(FLColor.hairlineStrong)
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                        .annotation(position: .bottom, alignment: .leading, spacing: 2) {
                            Text("resting").font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                        }
                }
            }
            .chartYScale(domain: .automatic(includesZero: false))
            .chartXAxis { timerAxis() }
            .chartYAxis {
                AxisMarks(position: .trailing, values: .automatic(desiredCount: 3)) {
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(FLColor.hairline)
                    AxisValueLabel().font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                }
            }
            .frame(height: 140)
            .accessibilityLabel("Heart rate over the session")
        }
    }
}

// MARK: - Predicted vs actual

private struct DriftOverlaySection: View {
    let report: SessionReport
    /// The cortex Metal view only exists while this section is on screen (spec 2: pause it off screen).
    @State private var cortexVisible = true
    struct P: Identifiable { var t: Double; var v: Double; var k: String; var id: String { "\(k)\(t)" } }
    static let predicted = "Predicted to lose you", actual = "Where you drifted"
    var body: some View {
        let pts = report.predictedDrift.map { P(t: $0.t, v: $0.value, k: Self.predicted) }
            + report.actualDrift.map { P(t: $0.t, v: $0.value, k: Self.actual) }
        ReportSection("Where this lecture was predicted to lose you vs where you actually drifted", eyebrow: "Content vs you") {
            VStack(alignment: .leading, spacing: FLSpacing.m) {
                HStack(spacing: FLSpacing.l) {
                    key(Self.predicted, color: FLColor.accentBlue, dashed: true)
                    key(Self.actual, color: FLColor.distracted, dashed: false)
                    Spacer(minLength: 0)
                }
                Chart(pts) { p in
                    LineMark(x: .value("Lecture time", p.t), y: .value("Scaled", p.v), series: .value("Series", p.k))
                        .foregroundStyle(by: .value("Series", p.k))
                        .interpolationMethod(.catmullRom)
                        .lineStyle(StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: p.k == Self.predicted ? [4, 4] : []))
                    if let b = report.biggestDriftPlayhead {
                        RuleMark(x: .value("Biggest drift", b)).foregroundStyle(FLColor.hairlineStrong)
                            .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                }
                .chartForegroundStyleScale([Self.predicted: FLColor.accentBlue, Self.actual: FLColor.distracted])
                .chartLegend(.hidden)
                .chartYAxis(.hidden)
                .chartXAxis { timerAxis() }
                .frame(height: 160)
                .accessibilityLabel("Predicted drift versus actual drift across the lecture, \(UIFormat.correlation(report.driftCorrelation))")
                Text("Agreement \(UIFormat.correlation(report.driftCorrelation))").font(FLFont.numeral(.footnote))
                    .foregroundStyle(FLColor.textSecondary)

                if let acts = report.biggestDriftActivations {
                    VStack(alignment: .leading, spacing: FLSpacing.l) {
                        HStack(alignment: .center, spacing: FLSpacing.m) {
                            ZStack {
                                RadialGradient(colors: [FLColor.accent.opacity(0.09), .clear], center: .center, startRadius: 0, endRadius: 90)
                                    .accessibilityHidden(true)
                                if cortexVisible && !ThermalStatus.shared.cortexPaused {
                                    BrainCortexView(activations: acts, autoRotate: false)
                                }
                            }
                            .frame(width: 150, height: 140)
                            .background(FLColor.surface, in: RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous).strokeBorder(FLColor.hairline, lineWidth: 0.5))
                            .clipShape(RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous))
                            .onGeometryChange(for: Bool.self) { proxy in
                                guard let vis = proxy.bounds(of: .scrollView) else { return true }
                                return vis.intersects(CGRect(origin: .zero, size: proxy.size))
                            } action: { cortexVisible = $0 }
                            .accessibilityElement()
                            .accessibilityLabel("Brain snapshot at the biggest drift moment, predicted response, average brain")
                            .overlay(alignment: .topTrailing) {
                                if report.contentIsMock { FLPreviewInfoButton() }
                            }
                            VStack(alignment: .leading, spacing: FLSpacing.xxs) {
                                Text("Biggest drift").flText(.eyebrow)
                                Text(UIFormat.timer(report.biggestDriftPlayhead ?? 0)).flText(.metric)
                                Text("into the lecture").flText(.caption)
                            }
                            .accessibilityElement(children: .combine)
                        }
                        Text("Predicted brain response to this content (average brain)").flText(.caption)
                        NetworkBars(activations: acts)
                    }
                    .padding(.top, FLSpacing.s)
                }
            }
        }
    }

    private func key(_ s: String, color: Color, dashed: Bool) -> some View {
        HStack(spacing: 6) {
            Path { p in p.move(to: CGPoint(x: 0, y: 1)); p.addLine(to: CGPoint(x: 16, y: 1)) }
                .stroke(color, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: dashed ? [3, 3] : []))
                .frame(width: 16, height: 2)
            Text(s).font(FLFont.caption).foregroundStyle(FLColor.textSecondary)
        }
        .fixedSize()
    }
}

// MARK: - Personal trial

/// "Your Personal Trial": a forest plot of each arm's effect vs no nudge (dot = estimate, 1.5pt line = interval),
/// with a shared zero line and n per arm.
public struct PersonalTrialCard: View {
    let effects: [ArmEffect]
    let isSynthetic: Bool
    public init(effects: [ArmEffect], isSynthetic: Bool) { self.effects = effects; self.isSynthetic = isSynthetic }

    public var body: some View {
        let shown = Array(effects.prefix(8))
        let domain = EffectAxis.domain(shown)
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader("Your Personal Trial", eyebrow: "What brings you back", large: true)
            Text("Each nudge vs no nudge, randomized during your sessions. Dot = estimated effect on refocusing; line = \(Int((shown.first?.level ?? 0.9) * 100))% interval.")
                .flText(.callout)
                .fixedSize(horizontal: false, vertical: true)
            if shown.isEmpty {
                Text("No randomized nudges yet. Your trial starts with your next session.")
                    .flText(.callout)
            } else {
                VStack(spacing: 0) {
                    ForEach(shown) { e in EffectRow(effect: e, domain: domain, color: color(e), label: label(e)) }
                    HStack(spacing: EffectRow.gap) {
                        Color.clear.frame(width: EffectRow.labelWidth, height: 1)
                        GeometryReader { g in
                            let zeroX = g.size.width * CGFloat(-domain.lowerBound / (domain.upperBound - domain.lowerBound))
                            ZStack(alignment: .topLeading) {
                                Text(UIFormat.signedPoints(domain.lowerBound)).fixedSize()
                                Text("0").fixedSize().position(x: zeroX, y: 8)
                                Text(UIFormat.signedPoints(domain.upperBound)).fixedSize()
                                    .frame(maxWidth: .infinity, alignment: .trailing)
                            }
                        }
                        .frame(height: 16)
                        Color.clear.frame(width: EffectRow.nWidth, height: 1)
                    }
                    .font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                    .padding(.top, FLSpacing.xs)
                    HStack(spacing: EffectRow.gap) {
                        Color.clear.frame(width: EffectRow.labelWidth, height: 1)
                        HStack {
                            Text("\u{2190} worse")
                            Spacer()
                            Text("better \u{2192}")
                        }
                        Color.clear.frame(width: EffectRow.nWidth, height: 1)
                    }
                    .font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                    .padding(.top, 2)
                    .accessibilityHidden(true)
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(shown.map { "\($0.arm.displayName): \(UIFormat.signedPoints($0.effect)), \(EffectAxis.verdict($0)), n \($0.n)" }.joined(separator: ". "))
            }
            Text("Intervals narrow as you study more.").flText(.caption)
        }
    }

    private func label(_ e: ArmEffect) -> String {
        e.policy == .preemptive ? "\(e.arm.displayName) (pre)" : e.arm.displayName
    }
    private func color(_ e: ArmEffect) -> Color {
        if e.arm.isExperimental || e.n < 3 { return e.arm.isExperimental ? FLColor.drowsy : FLColor.textSecondary }
        return e.lower > 0 ? FLColor.focused : (e.upper < 0 ? FLColor.stressed : FLColor.accentBlue)
    }
}

/// One forest-plot row: label + verdict, interval line and estimate dot on a shared axis, n on the right.
/// Rows sit flush so the zero line reads as one continuous rule.
private struct EffectRow: View {
    static let labelWidth: CGFloat = 128
    static let nWidth: CGFloat = 36
    static let gap: CGFloat = FLSpacing.s
    let effect: ArmEffect
    let domain: ClosedRange<Double>
    let color: Color
    let label: String
    var body: some View {
        HStack(spacing: Self.gap) {
            VStack(alignment: .leading, spacing: 2) {
                Text(label).font(FLFont.caption).foregroundStyle(FLColor.textPrimary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Text(EffectAxis.verdict(effect)).font(FLFont.scaled(12, weight: .regular, relativeTo: .caption1))
                    .foregroundStyle(color == FLColor.textSecondary ? FLColor.textTertiary : color)
            }
            .frame(width: Self.labelWidth, alignment: .leading)
            .padding(.vertical, FLSpacing.s)
            GeometryReader { g in
                let w = g.size.width, span = domain.upperBound - domain.lowerBound
                let x: (Double) -> CGFloat = { CGFloat((min(domain.upperBound, max(domain.lowerBound, $0)) - domain.lowerBound) / span) * w }
                ZStack(alignment: .leading) {
                    Rectangle().fill(FLColor.hairlineStrong).frame(width: 1).frame(maxHeight: .infinity)
                        .offset(x: x(0) - 0.5)
                    Capsule().fill(color)
                        .frame(width: max(1.5, x(effect.upper) - x(effect.lower)), height: 1.5)
                        .offset(x: x(effect.lower))
                    Circle().fill(color).frame(width: 8, height: 8)
                        .overlay(Circle().stroke(FLColor.bg, lineWidth: 1.5))
                        .offset(x: x(effect.effect) - 4)
                }
                .frame(maxHeight: .infinity)
            }
            Text("n=\(effect.n)").font(FLFont.numeral(.caption)).foregroundStyle(FLColor.textTertiary)
                .frame(width: Self.nWidth, alignment: .trailing)
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - Insights

public struct InsightsCard: View {
    let insights: [Insight]
    let isSynthetic: Bool
    let loading: Bool
    public init(insights: [Insight], isSynthetic: Bool, loading: Bool = false) {
        self.insights = insights; self.isSynthetic = isSynthetic; self.loading = loading
    }
    public var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader("Coach insights", eyebrow: "On-device", large: true)
            if insights.isEmpty && loading {
                HStack(spacing: FLSpacing.s) {
                    ProgressView().controlSize(.small).tint(FLColor.textSecondary)
                    Text("Writing insights on your phone\u{2026}").flText(.callout)
                }
            } else if insights.isEmpty {
                Text("Insights appear after a few sessions.").flText(.callout)
            }
            if !insights.isEmpty {
                VStack(alignment: .leading, spacing: FLSpacing.m) {
                    ForEach(insights) { i in
                        HStack(alignment: .firstTextBaseline, spacing: FLSpacing.s) {
                            Image(systemName: i.source == .foundationModels ? "sparkles" : "text.quote")
                                .font(.system(size: 15, weight: .regular))
                                .symbolRenderingMode(.hierarchical)
                                .foregroundStyle(FLColor.accent)
                                .frame(width: 20)
                                .accessibilityHidden(true)
                            Text(i.text).font(FLFont.body).foregroundStyle(FLColor.textPrimary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        }
    }
}

/// Static page rendered into the PDF / PNG export (`ImageRenderer` cannot capture the live Metal cortex,
/// so the caller passes an offscreen `CortexSnapshotter` still). Print-friendly: black on white, hairlines,
/// outlined honesty labels.
struct ReportExportPage: View {
    let report: SessionReport
    var cortexImage: UIImage? = nil

    static let cortexCaption = "Predicted brain response to this content (average brain)"

    private static let ink = Color(white: 0.07)
    private static let muted = Color(white: 0.36)
    private static let rule = Color(white: 0.85)
    private static let steel = Color(hex: 0x3E5C8A)
    private static let amber = Color(hex: 0xB86E00)

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            // Masthead
            HStack(alignment: .firstTextBaseline) {
                Text("FOCUSLOOP SESSION REPORT").font(.system(size: 9, weight: .semibold)).tracking(1.4).foregroundStyle(Self.muted)
                Spacer()
                Text(report.startedAt.formatted(date: .long, time: .shortened)).font(.system(size: 9)).foregroundStyle(Self.muted)
            }
            VStack(alignment: .leading, spacing: 8) {
                Text(report.lectureTitle ?? "Focus session").font(.system(size: 26, weight: .semibold)).tracking(-0.4)
            }
            rule
            HStack(alignment: .top, spacing: 0) {
                metric("Focused", UIFormat.percent(report.focusedFraction), unit: nil)
                divider
                metric("Duration", UIFormat.timer(report.duration), unit: nil)
                divider
                metric("Median refocus", report.medianRefocusSeconds.map(UIFormat.oneDecimal) ?? "n/a", unit: "s")
            }
            .fixedSize(horizontal: false, vertical: true)
            rule

            if !report.predictedDrift.isEmpty {
                section("Content vs you", "Where this lecture was predicted to lose you vs where you actually drifted")
                Chart {
                    ForEach(Array(report.predictedDrift.enumerated()), id: \.offset) { _, p in
                        LineMark(x: .value("Lecture time", p.t), y: .value("Value", p.value), series: .value("Series", "Predicted"))
                            .foregroundStyle(Self.steel)
                            .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                    }
                    ForEach(Array(report.actualDrift.enumerated()), id: \.offset) { _, p in
                        LineMark(x: .value("Lecture time", p.t), y: .value("Value", p.value), series: .value("Series", "Actual"))
                            .foregroundStyle(Self.amber)
                            .lineStyle(StrokeStyle(lineWidth: 1.5))
                    }
                }
                .chartYAxis(.hidden)
                .chartXAxis {
                    AxisMarks(values: .automatic(desiredCount: 5)) { v in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(Self.rule)
                        AxisValueLabel { if let d = v.as(Double.self) { Text(UIFormat.timer(d)).font(.system(size: 8)).foregroundStyle(Self.muted) } }
                    }
                }
                .frame(height: 120)
                Text("Dashed blue: predicted drift (brain model). Orange: your drift. \(UIFormat.correlation(report.driftCorrelation))")
                    .font(.system(size: 9)).foregroundStyle(Self.muted)
            }
            if let cortexImage {
                HStack(alignment: .center, spacing: 16) {
                    Image(uiImage: cortexImage).resizable().scaledToFit()
                        .frame(width: 180, height: 135)
                        .background(Color(white: 0.08), in: RoundedRectangle(cornerRadius: 12))
                    VStack(alignment: .leading, spacing: 4) {
                        Text("BIGGEST DRIFT").font(.system(size: 8, weight: .semibold)).tracking(1.2).foregroundStyle(Self.muted)
                        Text(UIFormat.timer(report.biggestDriftPlayhead ?? 0)).font(.system(size: 22, weight: .light).monospacedDigit())
                        Text(Self.cortexCaption).font(.system(size: 9)).foregroundStyle(Self.muted)
                    }
                }
            }

            section("What brings you back", "Your Personal Trial")
            if report.effects.isEmpty {
                Text("No randomized nudges yet.").font(.system(size: 10)).foregroundStyle(Self.muted)
            } else {
                VStack(spacing: 0) {
                    ForEach(report.effects) { e in
                        HStack {
                            Text(e.policy == .preemptive ? "\(e.arm.displayName) (pre)" : e.arm.displayName)
                            Spacer()
                            Text("\(UIFormat.signedPoints(e.effect))  [\(UIFormat.signedPoints(e.lower)), \(UIFormat.signedPoints(e.upper))]").monospacedDigit()
                            Text("n=\(e.n)").monospacedDigit().foregroundStyle(Self.muted).frame(width: 40, alignment: .trailing)
                        }
                        .font(.system(size: 10))
                        .padding(.vertical, 5)
                        .overlay(alignment: .bottom) { Rectangle().fill(Self.rule).frame(height: 0.5) }
                    }
                }
            }
            if !report.insights.isEmpty {
                section("On-device", "Insights")
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(report.insights) { Text($0.text).font(.system(size: 10)) }
                }
            }
            rule
            VStack(alignment: .leading, spacing: 3) {
                Text("FocusLoop supports focus. It is not a medical device. Brain predictions reflect an average brain, not you.")
                    .font(.system(size: 8)).foregroundStyle(Self.muted)
                if let note = FLDataFootnote.text(mock: report.contentIsMock, synthetic: report.isReplay, demo: false) {
                    Text(note).font(.system(size: 8)).foregroundStyle(Self.muted)
                }
            }
        }
        .padding(40)
        .foregroundStyle(Self.ink)
        .background(.white)
    }

    private var rule: some View { Rectangle().fill(Self.rule).frame(height: 0.5) }
    private var divider: some View { Rectangle().fill(Self.rule).frame(width: 0.5).padding(.horizontal, 16) }

    private func metric(_ title: String, _ value: String, unit: String?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title.uppercased()).font(.system(size: 8, weight: .semibold)).tracking(1.2).foregroundStyle(Self.muted)
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value).font(.system(size: 26, weight: .light).monospacedDigit()).tracking(-0.5)
                if let unit { Text(unit).font(.system(size: 11)).foregroundStyle(Self.muted) }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func section(_ eyebrow: String, _ title: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(eyebrow.uppercased()).font(.system(size: 8, weight: .semibold)).tracking(1.2).foregroundStyle(Self.muted)
            Text(title).font(.system(size: 13, weight: .semibold))
        }
        .padding(.top, 4)
    }
}

/// A file to hand to the system share sheet.
struct ShareItem: Identifiable {
    let url: URL
    var id: URL { url }
}

/// `UIActivityViewController` wrapper, so an export can open the share sheet without a second tap.
struct ActivityShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
