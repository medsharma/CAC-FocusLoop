import Charts
import SwiftUI

// Sustained attention (SART) on the Attention Profile: a compact section, and a detail screen with the latest
// run, trends across runs, a plain-English explainer, optional verified reference ranges and "Run SART again".

/// Display helpers shared by the section, the detail screen and tests.
public enum SARTFormat {
    public static func percent(_ f: Double?) -> String { f.map { "\(Int(($0 * 100).rounded()))" } ?? "--" }
    public static func ms(_ v: Double?) -> String { v.map { "\(Int($0.rounded()))" } ?? "--" }
    public static func cv(_ v: Double?) -> String { v.map { String(format: "%.2f", $0) } ?? "--" }
}

/// Compact section for the Attention Profile.
struct SustainedAttentionSection: View {
    let runs: [SARTRunRecord]
    let onRun: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader(SARTCopy.title, eyebrow: "Attention check", large: true) {
                if let latest = runs.last, latest.origin != .device { SARTOriginTag(origin: latest.origin) }
            }
            if let latest = runs.last {
                VStack(alignment: .leading, spacing: FLSpacing.m) {
                    SARTLatestStrip(metrics: latest.metrics)
                    Text("Latest of \(runs.count) \(runs.count == 1 ? "run" : "runs") \u{00B7} \(latest.date.formatted(date: .abbreviated, time: .shortened))")
                        .flText(.caption)
                }
                NavigationLink(value: FLRoute.sustainedAttention) {
                    FLRow("Trends and what they mean", symbol: "chart.line.uptrend.xyaxis")
                }
                .buttonStyle(FLPressableStyle())
                .flRowGroup()
                .accessibilityIdentifier("history.sart.details")
            } else {
                Text(SARTCopy.emptyBody).flText(.callout).fixedSize(horizontal: false, vertical: true)
                Button { FLHaptics.tap(); onRun() } label: { Label("Run the attention check", systemImage: "hand.tap") }
                    .buttonStyle(FLSecondaryButtonStyle())
                    .accessibilityIdentifier("history.sart.run")
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("history.sart")
    }
}

/// Taps on 3 · Variability · Response time.
struct SARTLatestStrip: View {
    let metrics: SARTMetrics
    var body: some View {
        FLMetricStrip([
            FLMetric("Taps on 3", value: SARTFormat.percent(metrics.commissionRate), unit: metrics.commissionRate == nil ? nil : "%"),
            FLMetric("Variability", value: SARTFormat.cv(metrics.rtCV), unit: metrics.rtCV == nil ? nil : "CV"),
            FLMetric("Response", value: SARTFormat.ms(metrics.meanGoRTms), unit: metrics.meanGoRTms == nil ? nil : "ms"),
        ])
    }
}

/// Quiet tag for runs that did not come from the phone's camera-equipped session (Simulator / SAMPLE).
struct SARTOriginTag: View {
    let origin: SARTRunOrigin
    var body: some View {
        switch origin {
        case .sample: FLTag("SAMPLE", tint: FLColor.demo, accessibilityText: "Sample run: generated for screenshots, not a person's taps")
        case .simulator: FLTag("SIMULATOR", tint: FLColor.textSecondary, accessibilityText: "Run in the Simulator: real taps, no camera")
        case .device: EmptyView()
        }
    }
}

/// Detail screen: latest run, trends, explainer, reference ranges, past runs, Run SART again.
public struct SustainedAttentionDetailView: View {
    let services: UIServices
    @State private var runs: [SARTRunRecord] = []
    @State private var loaded = false
    @State private var running = false

    public init(services: UIServices) { self.services = services }

    public var body: some View {
        ScrollViewReader { proxy in
        ScrollView {
            VStack(alignment: .leading, spacing: FLSpacing.xxl) {
                if loaded && runs.isEmpty {
                    empty
                } else if let latest = runs.last {
                    latestBlock(latest)
                    trends.id("trend")
                    FLDivider()
                    explainer.id("explainer")
                    FLDivider()
                    pastRuns.id("runs")
                    runButton
                }
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.top, FLSpacing.xs)
            .padding(.bottom, FLSpacing.xxl)
        }
        .modifier(FLLaunchScroll(proxy: proxy))
        }
        .flScreen(SARTCopy.title)
        .onAppear(perform: load)
        .fullScreenCover(isPresented: $running, onDismiss: load) {
            let done = { running = false }
            ZStack(alignment: .topTrailing) {
                services.sartView(onFinish: done) ?? AnyView(PlaceholderSARTView(onFinish: done))
                FLIconButton("xmark", label: "Close", action: done)
                    .padding(.trailing, FLSpacing.m).padding(.top, FLSpacing.xs)
            }
        }
        .accessibilityIdentifier("sart.detail")
    }

    private func load() {
        runs = services.sartHistory()
        loaded = true
    }

    // MARK: Sections

    private var empty: some View {
        VStack(alignment: .leading, spacing: FLSpacing.m) {
            Image(systemName: "hand.tap")
                .font(.system(size: 28, weight: .regular))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(FLColor.textSecondary)
                .accessibilityHidden(true)
            Text(SARTCopy.emptyTitle).flText(.title).accessibilityIdentifier("sart.empty")
            Text(SARTCopy.emptyBody).flText(.body).fixedSize(horizontal: false, vertical: true)
            Text(SARTCopy.intro).flText(.callout).fixedSize(horizontal: false, vertical: true)
                .padding(.bottom, FLSpacing.m)
            runButton
        }
        .padding(.top, FLSpacing.xl)
    }

    private func latestBlock(_ r: SARTRunRecord) -> some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            HStack(alignment: .firstTextBaseline) {
                FLEyebrow("Latest run \u{00B7} \(r.date.formatted(date: .abbreviated, time: .shortened))")
                Spacer(minLength: 8)
                SARTOriginTag(origin: r.origin)
            }
            Text(SARTCopy.intro).flText(.callout).fixedSize(horizontal: false, vertical: true)
            FLDivider()
            SARTLatestStrip(metrics: r.metrics)
            FLDivider()
            FLMetricStrip([
                FLMetric("Missed", value: SARTFormat.percent(r.metrics.omissionRate), unit: r.metrics.omissionRate == nil ? nil : "%"),
                FLMetric("Spread", value: SARTFormat.ms(r.metrics.rtSDms), unit: r.metrics.rtSDms == nil ? nil : "ms SD"),
                FLMetric("Too fast", value: "\(r.metrics.anticipationCount)", unit: "taps"),
            ])
            FLDivider()
            Text("\(r.metrics.commissionCount) of \(r.metrics.noGoCount) threes tapped \u{00B7} \(r.metrics.omissionCount) of \(r.metrics.goCount) digits missed \u{00B7} \(r.metrics.trialCount) trials")
                .flText(.caption)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("sart.latestCounts")
        }
    }

    @ViewBuilder private var trends: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader("Across your runs", eyebrow: "Trend", large: true)
            if runs.count < 2 {
                Text("Your trend appears after your second run.").flText(.callout)
            } else {
                SARTTrendChart(title: "Taps on 3", runs: runs, value: { $0.commissionRate }, format: { "\(Int(($0 * 100).rounded()))%" },
                               band: SARTReferenceData.range(.commissionRate), color: FLColor.accent)
                    .accessibilityIdentifier("sart.chart.commission")
                SARTTrendChart(title: "Variability (CV)", runs: runs, value: { $0.rtCV }, format: { String(format: "%.2f", $0) },
                               band: SARTReferenceData.range(.rtCV), color: FLColor.accentBlue)
                    .accessibilityIdentifier("sart.chart.cv")
            }
            referenceNote
        }
    }

    @ViewBuilder private var referenceNote: some View {
        let trials = runs.last?.metrics.trialCount ?? SARTConfig().trialCount
        if SARTReferenceData.ranges.isEmpty {
            Text(SARTReferenceData.noVerifiedSourceNote).flText(.caption)
                .accessibilityIdentifier("sart.reference")
        } else {
            VStack(alignment: .leading, spacing: FLSpacing.xs) {
                ForEach(SARTReferenceData.ranges) { r in
                    (Text(r.metric.label + ". ").foregroundStyle(FLColor.textSecondary)
                        + Text(SARTReferenceData.sentence(r, ourTrials: trials))).flText(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Text("Shaded bands show these ranges. Compare with your own runs first.").flText(.caption)
            }
            .accessibilityIdentifier("sart.reference")
        }
    }

    private var explainer: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader("What this means", eyebrow: "Plain English", large: true)
            VStack(alignment: .leading, spacing: FLSpacing.m) {
                ForEach(SARTCopy.explainers, id: \.title) { e in
                    VStack(alignment: .leading, spacing: FLSpacing.xxs) {
                        Text(e.title).flText(.headline)
                        Text(e.text).flText(.callout).fixedSize(horizontal: false, vertical: true)
                    }
                    .accessibilityElement(children: .combine)
                }
            }
            Text(SARTCopy.personalNote).flText(.caption).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var pastRuns: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader("Runs", eyebrow: "History", large: true)
            VStack(spacing: 0) {
                ForEach(Array(runs.reversed().enumerated()), id: \.element.id) { i, r in
                    if i > 0 { FLDivider(inset: 16) }
                    HStack(spacing: FLSpacing.s) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(r.date.formatted(date: .abbreviated, time: .shortened)).font(FLFont.body)
                                .foregroundStyle(FLColor.textPrimary)
                            Text("Taps on 3 \(SARTFormat.percent(r.metrics.commissionRate))% \u{00B7} CV \(SARTFormat.cv(r.metrics.rtCV)) \u{00B7} \(SARTFormat.ms(r.metrics.meanGoRTms)) ms")
                                .flText(.caption).lineLimit(1).minimumScaleFactor(0.8)
                        }
                        Spacer(minLength: 4)
                        SARTOriginTag(origin: r.origin)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .frame(minHeight: 60)
                    .accessibilityElement(children: .combine)
                    .accessibilityIdentifier("sart.run")
                }
            }
            .flRowGroup()
        }
    }

    private var runButton: some View {
        Button { FLHaptics.tap(); running = true } label: { Label("Run SART again", systemImage: "hand.tap") }
            .buttonStyle(FLPrimaryButtonStyle())
            .accessibilityIdentifier("sart.runAgain")
    }
}

/// One metric across runs: 1.5pt line, points when ≤ 8 runs, optional verified reference band, direct label.
struct SARTTrendChart: View {
    let title: String
    let runs: [SARTRunRecord]
    let value: (SARTMetrics) -> Double?
    let format: (Double) -> String
    let band: SARTReferenceRange?
    let color: Color

    private struct P: Identifiable { var date: Date; var v: Double; var id: Date { date } }

    /// Label every run when there are few; otherwise the first, middle and latest.
    static func labelDates(_ dates: [Date]) -> [Date] {
        guard dates.count > 5 else { return dates }
        return [dates[0], dates[dates.count / 2], dates[dates.count - 1]]
    }

    /// Runs span with a margin on both sides so the first and last points and labels are not clipped.
    static func xDomain(_ dates: [Date]) -> ClosedRange<Date> {
        let a = dates.min() ?? Date(), b = dates.max() ?? a
        let pad = max(3600 * 6, b.timeIntervalSince(a) * 0.08)
        return a.addingTimeInterval(-pad)...b.addingTimeInterval(pad)
    }

    var body: some View {
        let pts = runs.compactMap { r in value(r.metrics).map { P(date: r.date, v: $0) } }
        VStack(alignment: .leading, spacing: FLSpacing.s) {
            HStack(alignment: .firstTextBaseline) {
                FLEyebrow(title)
                Spacer()
                if let last = pts.last { Text(format(last.v)).font(FLFont.numeral(.subheadline)).foregroundStyle(FLColor.textPrimary) }
            }
            Chart {
                if let band, !pts.isEmpty {
                    let d = Self.xDomain(pts.map(\.date))
                    RectangleMark(xStart: .value("From", d.lowerBound), xEnd: .value("To", d.upperBound),
                                  yStart: .value("Low", band.low), yEnd: .value("High", band.high))
                        .foregroundStyle(FLColor.textPrimary.opacity(0.06))
                }
                ForEach(pts) { p in
                    if band == nil {
                        AreaMark(x: .value("Date", p.date), y: .value(title, p.v))
                            .foregroundStyle(FLChart.area(color)).interpolationMethod(.monotone)
                    }
                    LineMark(x: .value("Date", p.date), y: .value(title, p.v))
                        .foregroundStyle(color).lineStyle(FLChart.line).interpolationMethod(.monotone)
                    if pts.count <= 8 {
                        PointMark(x: .value("Date", p.date), y: .value(title, p.v)).foregroundStyle(color).symbolSize(20)
                    }
                }
            }
            .chartYScale(domain: .automatic(includesZero: true))
            .chartXScale(domain: Self.xDomain(pts.map(\.date)))
            .chartYAxis {
                AxisMarks(position: .trailing, values: .automatic(desiredCount: 3)) { v in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(FLColor.hairline)
                    AxisValueLabel { if let d = v.as(Double.self) { Text(format(d)).font(FLFont.caption).foregroundStyle(FLColor.textTertiary) } }
                }
            }
            .chartXAxis {
                AxisMarks(values: Self.labelDates(pts.map(\.date))) {
                    AxisValueLabel(format: .dateTime.month(.abbreviated).day(), centered: true)
                        .font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                }
            }
            .frame(height: 130)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("\(title) across \(pts.count) runs: " + pts.map { format($0.v) }.joined(separator: ", "))
        }
    }
}
