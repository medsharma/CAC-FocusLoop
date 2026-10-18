import Charts
import SwiftUI

/// History and Attention Profile: trends across sessions + session list linking to reports.
public struct HistoryView: View {
    let services: UIServices
    var openReport: (UUID) -> Void

    @State private var summary: AggregateSummary?
    @State private var series = ProfileSeries.empty
    @State private var sessions: [UISessionSummary] = []
    @State private var insights: [Insight] = []
    @State private var loaded = false
    @State private var sartRuns: [SARTRunRecord] = []
    @State private var runningSART = false
    @State private var showDoctorReport = false

    public init(services: UIServices, openReport: @escaping (UUID) -> Void) {
        self.services = services; self.openReport = openReport
    }

    public var body: some View {
        ScrollViewReader { proxy in
        ScrollView {
            VStack(alignment: .leading, spacing: FLSpacing.xxl) {
                if loaded && sessions.isEmpty {
                    emptyState
                    FLDivider()
                    SustainedAttentionSection(runs: sartRuns) { runningSART = true }.id("sart")
                    FLDivider()
                    doctorRow
                } else {
                    if let s = summary { ProfileSummarySection(summary: s) }
                    if series.points.count > 1 { FocusTrendSection(series: series); FLDivider() }
                    if series.focusByHour.contains(where: { $0 != nil }) { BestHoursSection(series: series); FLDivider() }
                    SustainedAttentionSection(runs: sartRuns) { runningSART = true }.id("sart")
                    FLDivider()
                    if let s = summary, !s.armEffects.isEmpty {
                        PersonalTrialCard(effects: s.armEffects, isSynthetic: s.isReplay).id("trial")
                        FLDivider()
                    }
                    InsightsCard(insights: insights, isSynthetic: summary?.isReplay ?? false, loading: !loaded)
                    FLDivider()
                    doctorRow.id("doctor")
                    FLDivider()
                    sessionList.id("sessions")
                }
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.top, FLSpacing.xs)
            .padding(.bottom, FLSpacing.xxl)
        }
        .modifier(FLLaunchScroll(proxy: proxy))
        }
        .flScreen("Attention Profile")
        .task(id: services.listSessions().count) { await load() }
        .refreshable { await load() }
        .onAppear { sartRuns = services.sartHistory() }
        .fullScreenCover(isPresented: $runningSART, onDismiss: { sartRuns = services.sartHistory() }) {
            let done = { runningSART = false }
            ZStack(alignment: .topTrailing) {
                services.sartView(onFinish: done) ?? AnyView(PlaceholderSARTView(onFinish: done))
                FLIconButton("xmark", label: "Close", action: done)
                    .padding(.trailing, FLSpacing.m).padding(.top, FLSpacing.xs)
            }
        }
        .sheet(isPresented: $showDoctorReport) { DoctorReportSheet(services: services) }
        .task {
            // `-doctorReport` (screenshots): open the clinician-report sheet on launch.
            if ProcessInfo.processInfo.arguments.contains("-doctorReport") {
                try? await Task.sleep(for: .milliseconds(700))
                showDoctorReport = true
            }
        }
    }

    /// "Share with your doctor": a quiet row group that opens the clinician-report sheet.
    private var doctorRow: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader("Share with your doctor", eyebrow: "For a clinician", large: true)
            Text("A printable summary of how your sustained attention behaves, made on this phone. You decide who sees it.")
                .flText(.callout).fixedSize(horizontal: false, vertical: true)
            Button { FLHaptics.tap(); showDoctorReport = true } label: {
                FLRow("Create a PDF summary", subtitle: "Real sessions only, unless you include demo data", symbol: "doc.text")
            }
            .buttonStyle(FLPressableStyle())
            .flRowGroup()
            .accessibilityIdentifier("history.doctorReport")
        }
    }

    private func load() async {
        sartRuns = services.sartHistory()
        sessions = services.listSessions()
        series = services.profileSeries()
        summary = await services.aggregateSummary()
        insights = await services.profileInsights()
        loaded = true
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: FLSpacing.m) {
            Image(systemName: "chart.xyaxis.line")
                .font(.system(size: 28, weight: .regular))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(FLColor.textSecondary)
                .accessibilityHidden(true)
            Text("Your profile builds as you study").accessibilityIdentifier("history.empty").flText(.title)
                .fixedSize(horizontal: false, vertical: true)
            Text("After a few sessions you'll see when your focus tends to dip, how fast you come back, and which nudges help you.")
                .flText(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, FLSpacing.xl)
    }

    private var sessionList: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader("Sessions", eyebrow: "History", large: true)
            VStack(spacing: 0) {
                ForEach(Array(sessions.enumerated()), id: \.element.id) { i, s in
                    if i > 0 { FLDivider(inset: 16 + 40 + 12) }
                    Button { FLHaptics.tap(); openReport(s.id) } label: { SessionRow(s: s) }
                        .buttonStyle(FLPressableStyle())
                        .accessibilityIdentifier("history.session")
                }
            }
            .flRowGroup()
        }
    }
}

private struct SessionRow: View {
    let s: UISessionSummary
    var body: some View {
        HStack(spacing: FLSpacing.s) {
            ZStack {
                Circle().stroke(FLColor.hairlineStrong, lineWidth: 2)
                Circle().trim(from: 0, to: s.focusedFraction).stroke(FLColor.focused, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                Text("\(Int((s.focusedFraction * 100).rounded()))").font(FLFont.numeral(.caption))
                    .foregroundStyle(FLColor.textPrimary)
            }
            .frame(width: 40, height: 40)
            VStack(alignment: .leading, spacing: 2) {
                Text(s.lectureTitle ?? "Focus session").font(FLFont.body).foregroundStyle(FLColor.textPrimary).lineLimit(1)
                Text("\(s.startedAt.formatted(date: .abbreviated, time: .shortened)) \u{00B7} \(UIFormat.timer(s.duration))")
                    .flText(.caption).lineLimit(1)
            }
            Spacer(minLength: 4)
            Image(systemName: "chevron.right")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(FLColor.textTertiary)
                .accessibilityHidden(true)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(minHeight: 64)
        .contentShape(Rectangle())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(s.lectureTitle ?? "Focus session"), \(s.startedAt.formatted(date: .abbreviated, time: .shortened)), \(UIFormat.timer(s.duration)), \(UIFormat.percent(s.focusedFraction)) focused")
        .accessibilityHint("Opens the session report")
        .accessibilityAddTraits(.isButton)
    }
}

private struct ProfileSummarySection: View {
    let summary: AggregateSummary
    var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            VStack(alignment: .leading, spacing: FLSpacing.xs) {
                FLEyebrow("This week")
                Text("\(summary.sessionCount) sessions \u{00B7} \(UIFormat.minutes(summary.totalMinutes))").flText(.body)
            }
            FLDivider()
            FLMetricStrip([
                FLMetric("Focused", value: UIFormat.percent(summary.focusedFraction), color: FLColor.focused),
                FLMetric("Back in", value: summary.medianRefocusSeconds.map(UIFormat.oneDecimal) ?? "--", unit: "s"),
                FLMetric("Resting HR", value: summary.restingHeartRate.map { "\(Int($0))" } ?? "--", unit: "BPM"),
            ])
            FLDivider()
            VStack(alignment: .leading, spacing: FLSpacing.m) {
                FLEyebrow("Time in each state")
                GeometryReader { g in
                    HStack(spacing: 2) {
                        ForEach(AttentionState.allCases) { s in
                            Rectangle().fill(s.color).frame(width: max(0, g.size.width * fraction(s) - 2))
                        }
                    }
                    .clipShape(Capsule())
                }
                .frame(height: 6)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(AttentionState.allCases.map { "\($0.displayName) \(UIFormat.percent(fraction($0)))" }.joined(separator: ", "))
                Grid(alignment: .leading, horizontalSpacing: FLSpacing.l, verticalSpacing: FLSpacing.s) {
                    GridRow { key(.focused); key(.distracted) }
                    GridRow { key(.drowsy); key(.stressed) }
                }
            }
        }
    }

    private func fraction(_ s: AttentionState) -> Double {
        s.index < summary.stateFractions.count ? summary.stateFractions[s.index] : 0
    }

    private func key(_ s: AttentionState) -> some View {
        HStack(spacing: FLSpacing.xs) {
            Circle().fill(s.color).frame(width: 6, height: 6)
            Text(s.displayName).font(FLFont.callout).foregroundStyle(FLColor.textSecondary)
            Text(UIFormat.percent(fraction(s))).font(FLFont.numeral(.subheadline)).foregroundStyle(FLColor.textPrimary)
        }
        .accessibilityElement(children: .combine)
    }
}

private struct FocusTrendSection: View {
    let series: ProfileSeries
    var body: some View {
        // One point per day (minute-weighted), so several sessions on one day don't stack into a spike.
        let daily = Self.dailyPoints(series.points)
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader("Focus across sessions", eyebrow: "Trend", large: true)
            Chart(daily) { p in
                AreaMark(x: .value("Date", p.date), y: .value("Focused", p.focusedFraction))
                    .foregroundStyle(FLChart.area(FLColor.focused))
                    .interpolationMethod(.monotone)
                LineMark(x: .value("Date", p.date), y: .value("Focused", p.focusedFraction))
                    .foregroundStyle(FLColor.focused).interpolationMethod(.monotone)
                    .lineStyle(FLChart.line)
                if daily.count <= 8 {
                    PointMark(x: .value("Date", p.date), y: .value("Focused", p.focusedFraction))
                        .foregroundStyle(FLColor.focused).symbolSize(20)
                }
            }
            .chartYScale(domain: 0...1)
            .chartYAxis {
                AxisMarks(position: .trailing, values: [0, 0.5, 1]) { v in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(FLColor.hairline)
                    AxisValueLabel { if let d = v.as(Double.self) { Text(UIFormat.percent(d)).font(FLFont.caption).foregroundStyle(FLColor.textTertiary) } }
                }
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .day)) {
                    AxisValueLabel(format: .dateTime.weekday(.narrow)).font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                }
            }
            .frame(height: 150)
            .accessibilityLabel("Focused fraction across sessions")
            if series.points.contains(where: { $0.medianRefocusSeconds != nil }) {
                VStack(alignment: .leading, spacing: FLSpacing.m) {
                    FLEyebrow("Median time to refocus")
                    Chart(daily.filter { $0.medianRefocusSeconds != nil }) { p in
                        BarMark(x: .value("Date", p.date, unit: .day), y: .value("Seconds", p.medianRefocusSeconds ?? 0), width: .fixed(10))
                            .foregroundStyle(FLColor.accent).cornerRadius(FLChart.barRadius)
                            .annotation(position: .top, spacing: 4) {
                                Text(UIFormat.oneDecimal(p.medianRefocusSeconds ?? 0)).font(FLFont.numeral(.caption)).foregroundStyle(FLColor.textTertiary)
                            }
                    }
                    .chartXAxis {
                        AxisMarks(values: .stride(by: .day)) {
                            AxisValueLabel(format: .dateTime.weekday(.narrow)).font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                        }
                    }
                    .chartYAxis(.hidden)
                    .frame(height: 100)
                    .accessibilityLabel("Median refocus seconds per session")
                }
                .padding(.top, FLSpacing.s)
            }
        }
    }

    /// Aggregates sessions to one point per calendar day, plotted at midday.
    static func dailyPoints(_ points: [ProfileSeries.Point], calendar: Calendar = .current) -> [ProfileSeries.Point] {
        let byDay = Dictionary(grouping: points) { calendar.startOfDay(for: $0.date) }
        return byDay.keys.sorted().map { day in
            let ps = byDay[day] ?? []
            let minutes = ps.reduce(0) { $0 + $1.minutes }
            let focused = minutes > 0 ? ps.reduce(0) { $0 + $1.focusedFraction * $1.minutes } / minutes
                                      : ps.map(\.focusedFraction).reduce(0, +) / Double(max(1, ps.count))
            let refocus = ps.compactMap(\.medianRefocusSeconds).sorted()
            let hr = ps.compactMap(\.restingHeartRate)
            return ProfileSeries.Point(date: calendar.date(byAdding: .hour, value: 12, to: day) ?? day,
                                       focusedFraction: focused,
                                       medianRefocusSeconds: refocus.isEmpty ? nil : refocus[refocus.count / 2],
                                       restingHeartRate: hr.isEmpty ? nil : hr.reduce(0, +) / Double(hr.count),
                                       minutes: minutes)
        }
    }
}

private struct BestHoursSection: View {
    let series: ProfileSeries
    var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.l) {
            FLSectionHeader("When you focus best", eyebrow: "Time of day", large: true)
            VStack(spacing: FLSpacing.xs) {
                HStack(alignment: .bottom, spacing: 3) {
                    ForEach(0..<24, id: \.self) { h in
                        let v = h < series.focusByHour.count ? series.focusByHour[h] : nil
                        RoundedRectangle(cornerRadius: FLChart.barRadius / 2)
                            .fill(v == nil ? FLColor.hairlineStrong : FLColor.focused.opacity(0.3 + 0.7 * (v ?? 0)))
                            .frame(height: v.map { 6 + 74 * $0 } ?? 2)
                            .frame(maxWidth: .infinity)
                    }
                }
                .frame(height: 80, alignment: .bottom)
                HStack(spacing: 0) {
                    ForEach([0, 6, 12, 18], id: \.self) { h in
                        Text(UIFormat.hour(h).replacingOccurrences(of: " ", with: "").lowercased())
                            .font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Focus by hour of day")
        }
    }
}
