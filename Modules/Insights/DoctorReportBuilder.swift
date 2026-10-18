import Foundation

// "Share with your doctor": a print-friendly summary of how the student's sustained attention behaves, as
// information to bring to a clinician. This file is pure (no UI): `DoctorReportBuilder.build` turns stored
// records into `DoctorReportContent` (text, tables and chart series), which `DoctorReportExporter` lays out
// as a US Letter PDF. Tests scan `allText` for wording the report must never use.
//
// Data honesty: by default only real sessions are included (live camera; not the synthetic Replay, not a
// RECORDED demo video, not a Demo Director session). With `includeDemo`, demo sessions are added and every
// section they feed is marked "demo data". SART runs are all included (their taps are real), except DEBUG
// SAMPLE runs, which appear only with `includeDemo` and are marked as demo data.

public struct DoctorReportContent: Equatable, Sendable {
    /// Required on page 1 and in every page footer.
    public static let statement = "This is not a diagnosis. FocusLoop is a wellness tool that measures attention signals with a phone camera; it cannot diagnose any condition. Measurements can be affected by lighting, motion and device placement."
    public static let title = "FocusLoop attention summary"
    public static let demoMark = "demo data"

    public struct KeyValue: Equatable, Sendable {
        public var key: String
        public var value: String
        public init(_ key: String, _ value: String) { self.key = key; self.value = value }
    }

    public struct Table: Equatable, Sendable {
        public var headers: [String]
        public var rows: [[String]]
        /// Relative column widths (same count as headers).
        public var weights: [Double]
        public init(headers: [String], rows: [[String]], weights: [Double]? = nil) {
            self.headers = headers; self.rows = rows
            self.weights = weights ?? Array(repeating: 1, count: headers.count)
        }
    }

    public struct ChartSeries: Equatable, Sendable {
        public struct Point: Equatable, Sendable {
            public var date: Date
            public var value: Double
            public init(date: Date, value: Double) { self.date = date; self.value = value }
        }
        public var title: String
        /// Axis value format.
        public enum Format: String, Sendable { case percent, bpm, ratio }
        public var format: Format
        public var points: [Point]
        /// Optional verified reference band (same units as the points).
        public var band: ClosedRange<Double>?
        public init(title: String, format: Format, points: [Point], band: ClosedRange<Double>? = nil) {
            self.title = title; self.format = format; self.points = points; self.band = band
        }
    }

    public enum Block: Equatable, Sendable {
        case paragraph(String)
        case keyValues([KeyValue])
        case table(Table)
        case bullets([String])
        case chart(ChartSeries)
        /// Small muted text (caveats, sources).
        case note(String)
    }

    public struct Section: Equatable, Sendable, Identifiable {
        public var id: String
        public var number: Int
        public var title: String
        /// True if any number in this section comes from demo data (rendered with a "demo data" mark).
        public var isDemoData: Bool
        public var blocks: [Block]
    }

    public var dateRange: String
    public var generated: String
    /// One line under the statement on page 1 describing what data is included.
    public var dataScope: String
    public var includesDemo: Bool
    public var realSessionCount: Int
    public var demoSessionCount: Int
    public var sections: [Section]

    public func section(_ id: String) -> Section? { sections.first { $0.id == id } }

    /// Every word of the report, for tests and accessibility.
    public var allText: String {
        var parts = [Self.title, dateRange, generated, Self.statement, dataScope]
        for s in sections {
            parts.append(s.title)
            if s.isDemoData { parts.append(Self.demoMark) }
            for b in s.blocks {
                switch b {
                case let .paragraph(t), let .note(t): parts.append(t)
                case let .keyValues(kv): parts += kv.flatMap { [$0.key, $0.value] }
                case let .table(t): parts += t.headers + t.rows.flatMap { $0 }
                case let .bullets(b): parts += b
                case let .chart(c): parts.append(c.title)
                }
            }
        }
        return parts.joined(separator: "\n")
    }
}

public enum DoctorReportBuilder {

    public struct Input: Sendable {
        public var sessions: [SessionRecord]
        /// Effects from the real Personal Trial (only real sessions feed it).
        public var realArmEffects: [ArmEffect]
        /// Effects from the demo/SYNTHETIC trial (shown only with `includeDemo`, marked demo data).
        public var demoArmEffects: [ArmEffect]
        public var sartRuns: [SARTRunRecord]
        public var includeDemo: Bool
        public var now: Date
        public var calendar: Calendar
        public init(sessions: [SessionRecord], realArmEffects: [ArmEffect] = [], demoArmEffects: [ArmEffect] = [],
                    sartRuns: [SARTRunRecord] = [], includeDemo: Bool = false, now: Date = Date(),
                    calendar: Calendar = .current) {
            self.sessions = sessions; self.realArmEffects = realArmEffects; self.demoArmEffects = demoArmEffects
            self.sartRuns = sartRuns; self.includeDemo = includeDemo; self.now = now; self.calendar = calendar
        }
    }

    /// `SessionRecord.sourceKind` of a session run with Demo Director on.
    public static let demoDirectorSourceKind = "demoDirector"

    /// Replay (synthetic), RECORDED video and Demo Director sessions are demo sessions. Older Demo Director
    /// records without the source kind are recognised by their scripted (deterministic) nudges.
    public static func isDemoSession(_ r: SessionRecord) -> Bool {
        if r.isReplay || r.isRecorded || r.sourceKind == demoDirectorSourceKind { return true }
        return r.events.contains { e in
            if case let .nudge(d) = e { return d.randomizationProbability >= 1 && d.arm != .noNudge
                && d.armProbabilities.count == 1 }
            return false
        }
    }

    public static func build(_ input: Input) -> DoctorReportContent {
        let real = input.sessions.filter { !isDemoSession($0) }.sorted { $0.startedAt < $1.startedAt }
        let demo = input.sessions.filter(isDemoSession).sorted { $0.startedAt < $1.startedAt }
        let included = (input.includeDemo ? real + demo : real).sorted { $0.startedAt < $1.startedAt }
        let sessionsAreDemo = input.includeDemo && !demo.isEmpty
        let runs = input.sartRuns.filter { input.includeDemo || $0.origin.isRealTaps }.sorted { $0.date < $1.date }
        let demoIDs = Set(demo.map(\.id))

        // Date range over everything included.
        let dates = included.map(\.startedAt) + runs.map(\.date)
        let range: String
        if let lo = dates.min(), let hi = dates.max() {
            range = lo.formatted(date: .long, time: .omitted) + " \u{2013} " + hi.formatted(date: .long, time: .omitted)
        } else {
            range = "No data recorded yet"
        }

        var scope = "Includes \(count(real.count, "real session")) measured live with the phone camera"
        if input.includeDemo, !demo.isEmpty {
            scope += ", plus \(count(demo.count, "demo session")) (marked \u{201C}demo data\u{201D})"
        } else if !demo.isEmpty {
            scope += "; \(count(demo.count, "demo session")) left out"
        }
        scope += ", and \(count(runs.count, "attention-check (SART) run")).".replacingOccurrences(of: "..", with: ".")

        let sections = [
            overview(included, real: real, demoCount: demo.count, isDemo: sessionsAreDemo, demoIDs: demoIDs,
                     includeDemo: input.includeDemo, calendar: input.calendar),
            sart(runs),
            heartRate(included, isDemo: sessionsAreDemo, demoIDs: demoIDs),
            whatHelped(real: real, input: input, demoCount: demo.count),
            selfReports(included, demoIDs: demoIDs),
            clinicianNotes(),
        ].enumerated().map { i, s in var s = s; s.number = i + 1; return s }

        return DoctorReportContent(
            dateRange: range,
            generated: "Generated on this phone on \(input.now.formatted(date: .long, time: .shortened))",
            dataScope: scope, includesDemo: input.includeDemo, realSessionCount: real.count,
            demoSessionCount: demo.count, sections: sections)
    }

    // MARK: - Sections

    static func overview(_ s: [SessionRecord], real: [SessionRecord], demoCount: Int, isDemo: Bool, demoIDs: Set<UUID>,
                         includeDemo: Bool, calendar: Calendar) -> DoctorReportContent.Section {
        var blocks: [DoctorReportContent.Block] = []
        if s.isEmpty {
            var t = "No real sessions have been recorded yet. This summary only includes sessions measured live with the phone camera"
            t += demoCount > 0 && !includeDemo ? "; \(count(demoCount, "demo session")) \(demoCount == 1 ? "was" : "were") left out." : "."
            blocks.append(.paragraph(t))
        } else {
            let sum = AttentionProfileBuilder.summary(sessions: s, calendar: calendar)
            let minutes = sum.totalMinutes
            func stateLine(_ st: AttentionState) -> String {
                let f = st.index < sum.stateFractions.count ? sum.stateFractions[st.index] : 0
                return "\(fmtMinutes(minutes * f)) (\(pct(f)))"
            }
            blocks.append(.paragraph("Each session is a study period with the phone propped up facing the student. FocusLoop estimates, second by second, whether attention looks focused, distracted, drowsy or stressed, from camera and motion signals."))
            blocks.append(.keyValues([
                .init("Sessions", "\(s.count)" + (isDemo ? " (\(s.filter { demoIDs.contains($0.id) }.count) demo)" : "")),
                .init("Total time", fmtMinutes(minutes)),
                .init("Estimated focused", pct(sum.focusedFraction) + " of the time"),
                .init("Focused", stateLine(.focused)),
                .init("Distracted", stateLine(.distracted)),
                .init("Drowsy", stateLine(.drowsy)),
                .init("Stressed", stateLine(.stressed)),
                .init("Typical minute focus first drops", sum.typicalDropMinute.map { "Minute \(Int($0) + 1) of a session" } ?? "Not enough data yet"),
                .init("Median time to refocus", sum.medianRefocusSeconds.map { String(format: "%.1f s", $0) } ?? "Not enough data yet"),
                .init("Best hours (most focused)", sum.bestHours.isEmpty ? "Not enough data yet" : sum.bestHours.map(hourName).joined(separator: ", ")),
            ]))
            let recent = Array(s.suffix(30))
            var headers = ["Date", "Minutes", "Focused", "Refocus (median)"]
            if isDemo { headers.append("Data") }
            let rows: [[String]] = recent.reversed().map { r in
                let one = AttentionProfileBuilder.summary(sessions: [r], calendar: calendar)
                var row = [r.startedAt.formatted(date: .abbreviated, time: .shortened),
                           String(format: "%.0f", AttentionProfileBuilder.sessionSeconds(r) / 60),
                           pct(one.focusedFraction),
                           one.medianRefocusSeconds.map { String(format: "%.1f s", $0) } ?? "\u{2013}"]
                if isDemo { row.append(demoIDs.contains(r.id) ? "Demo" : "Real") }
                return row
            }
            blocks.append(.table(.init(headers: headers, rows: rows, weights: isDemo ? [2.2, 1, 1, 1.3, 0.8] : [2.2, 1, 1, 1.3])))
            if s.count > recent.count { blocks.append(.note("Showing the \(recent.count) most recent of \(s.count) sessions; the totals above include all of them.")) }
            blocks.append(.note("States are estimates from camera and motion signals, not observations by a person. Time of day uses each session's start time."))
        }
        return .init(id: "overview", number: 0, title: "Overview", isDemoData: isDemo, blocks: blocks)
    }

    static func sart(_ runs: [SARTRunRecord]) -> DoctorReportContent.Section {
        var blocks: [DoctorReportContent.Block] = []
        let hasSamples = runs.contains { $0.origin == .sample }
        blocks.append(.paragraph("The Sustained Attention to Response Task (SART; Robertson et al., 1997) is a standard research task. Digits 1 to 9 flash for 250 ms, each followed by a 900 ms mask; the student taps for every digit except 3. Taps on 3 show moments when a habitual response slipped through; response-time variability shows how steady attention stayed. FocusLoop\u{2019}s version runs on the phone touchscreen and is shorter than the original."))
        if runs.isEmpty {
            blocks.append(.paragraph("No attention-check runs yet."))
        } else {
            let rows: [[String]] = runs.reversed().map { r in
                let m = r.metrics
                return [r.date.formatted(date: .abbreviated, time: .shortened),
                        "\(m.trialCount)",
                        m.commissionRate.map { "\(pct($0)) (\(m.commissionCount)/\(m.noGoCount))" } ?? "\u{2013}",
                        m.omissionRate.map { "\(pct($0)) (\(m.omissionCount))" } ?? "\u{2013}",
                        m.meanGoRTms.map { "\(Int($0.rounded())) ms" } ?? "\u{2013}",
                        m.rtSDms.map { "\(Int($0.rounded())) ms" } ?? "\u{2013}",
                        m.rtCV.map { String(format: "%.2f", $0) } ?? "\u{2013}",
                        "\(m.anticipationCount)",
                        r.origin == .sample ? "SAMPLE (demo data)" : r.origin.label]
            }
            blocks.append(.table(.init(headers: ["Date", "Trials", "Taps on 3", "Missed", "Mean RT", "RT SD", "RT CV", "<100 ms", "Run on"],
                                       rows: rows, weights: [2.0, 0.8, 1.4, 1.1, 1.1, 1.0, 0.9, 0.9, 1.5])))
            blocks.append(.note("Taps on 3 = commission errors (no-go digits tapped). Missed = omissions (go digits not tapped). RT = response time on go digits, excluding taps under 100 ms (counted in the <100 ms column) and trials with disrupted display timing. CV = SD divided by mean. \u{201C}Simulator\u{201D} runs were done on a computer\u{2019}s iPhone simulator: the taps are real, but there was no camera."))
            if runs.count >= 2 {
                var trend: [String] = []
                if let a = runs.first?.metrics.commissionRate, let b = runs.last?.metrics.commissionRate {
                    trend.append("Taps on 3 went from \(pct(a)) in the first run to \(pct(b)) in the latest.")
                }
                if let a = runs.first?.metrics.rtCV, let b = runs.last?.metrics.rtCV {
                    trend.append(String(format: "Response-time variability (CV) went from %.2f to %.2f.", a, b))
                }
                if !trend.isEmpty { blocks.append(.paragraph(trend.joined(separator: " ") + " Changes between a few runs can reflect practice, sleep or time of day.")) }
                let c = runs.compactMap { r in r.metrics.commissionRate.map { DoctorReportContent.ChartSeries.Point(date: r.date, value: $0) } }
                let v = runs.compactMap { r in r.metrics.rtCV.map { DoctorReportContent.ChartSeries.Point(date: r.date, value: $0) } }
                let cRef = SARTReferenceData.range(.commissionRate), vRef = SARTReferenceData.range(.rtCV)
                if c.count >= 2 { blocks.append(.chart(.init(title: "Taps on 3 (commission error rate) by run", format: .percent, points: c,
                                                             band: cRef.map { $0.low...$0.high }))) }
                if v.count >= 2 { blocks.append(.chart(.init(title: "Response-time variability (CV) by run", format: .ratio, points: v,
                                                             band: vRef.map { $0.low...$0.high }))) }
            }
            let trials = runs.last?.metrics.trialCount ?? SARTConfig().trialCount
            if SARTReferenceData.ranges.isEmpty {
                blocks.append(.note(SARTReferenceData.noVerifiedSourceNote))
            } else {
                blocks.append(.paragraph("Published reference ranges (shaded on the charts):"))
                blocks.append(.bullets(SARTReferenceData.ranges.map { "\($0.metric.label): " + SARTReferenceData.sentence($0, ourTrials: trials) + " " + $0.derivation + "." }))
            }
        }
        return .init(id: "sart", number: 0, title: "Sustained attention (SART)", isDemoData: hasSamples, blocks: blocks)
    }

    static func heartRate(_ s: [SessionRecord], isDemo: Bool, demoIDs: Set<UUID>) -> DoctorReportContent.Section {
        var blocks: [DoctorReportContent.Block] = []
        let per = s.compactMap { r -> (SessionRecord, Double)? in
            AttentionProfileBuilder.restingHeartRate(r.ticks).map { (r, $0) }
        }
        let demoUsed = isDemo && per.contains { demoIDs.contains($0.0.id) }
        if per.isEmpty {
            blocks.append(.paragraph(s.isEmpty ? "No sessions to summarise." : "No camera heart-rate readings of good enough quality yet."))
        } else {
            let values = per.map(\.1)
            var kv: [DoctorReportContent.KeyValue] = [
                .init("Sessions with a reading", "\(per.count)"),
                .init("Median resting heart rate", "\(Int((AttentionProfileBuilder.median(values) ?? 0).rounded())) BPM"),
                .init("Range across sessions", "\(Int((values.min() ?? 0).rounded()))\u{2013}\(Int((values.max() ?? 0).rounded())) BPM"),
            ]
            if let slope = AttentionProfileBuilder.heartRateTrendPerSession(s) {
                let r = (slope * 10).rounded() / 10
                kv.append(.init("Trend", r == 0 ? "No change (0.0 BPM per session)"
                                : String(format: "%@%.1f BPM per session", r > 0 ? "+" : "\u{2212}", abs(r))))
            }
            blocks.append(.keyValues(kv))
            if per.count >= 2 {
                blocks.append(.chart(.init(title: "Resting heart rate by session (camera estimate)", format: .bpm,
                                           points: per.map { .init(date: $0.0.startedAt, value: $0.1) })))
            }
        }
        blocks.append(.note("Resting heart rate here is the median camera reading while attention looked focused, using good-quality readings only. Camera heart rate (remote photoplethysmography) is an estimate: it is less accurate for some skin tones and in poor or changing lighting, and with movement. It is not a medical measurement."))
        return .init(id: "heartRate", number: 0, title: "Heart rate (camera)", isDemoData: demoUsed, blocks: blocks)
    }

    static func whatHelped(real: [SessionRecord], input: Input, demoCount: Int) -> DoctorReportContent.Section {
        var blocks: [DoctorReportContent.Block] = []
        blocks.append(.paragraph("When FocusLoop noticed attention drifting, it chose at random between a gentle nudge (such as a soft glow, a short stretch or paced breathing) and no nudge at all. Comparing the two shows what tends to bring this student back. Each estimate is the difference in how often the student was focused again within 20 seconds, compared with no nudge, with a 90% interval."))
        let realRows = effectRows(input.realArmEffects)
        if real.isEmpty {
            blocks.append(.paragraph("No real sessions yet, so there are no results."))
        } else if realRows.isEmpty {
            blocks.append(.paragraph("No nudges have been compared yet."))
        } else {
            blocks.append(.table(.init(headers: ["Nudge", "Estimated change", "90% interval", "Tries", "Reading"], rows: realRows,
                                       weights: [2.2, 1.3, 1.6, 0.7, 1.6])))
        }
        var isDemo = false
        if input.includeDemo, demoCount > 0 {
            let demoRows = effectRows(input.demoArmEffects)
            if !demoRows.isEmpty {
                isDemo = true
                blocks.append(.paragraph("From demo sessions (demo data, kept separate from the student\u{2019}s own results):"))
                blocks.append(.table(.init(headers: ["Nudge", "Estimated change", "90% interval", "Tries", "Reading"], rows: demoRows,
                                           weights: [2.2, 1.3, 1.6, 0.7, 1.6])))
            }
        }
        blocks.append(.note("\u{201C}Still learning\u{201D} means the interval still includes zero: there is not yet enough evidence either way. Points are percentage points."))
        return .init(id: "whatHelped", number: 0, title: "What helped", isDemoData: isDemo, blocks: blocks)
    }

    static func effectRows(_ effects: [ArmEffect]) -> [[String]] {
        effects.filter { $0.arm != .noNudge }.map { e in
            let name = e.policy == .preemptive ? "\(e.arm.displayName) (before a dip)" : e.arm.displayName
            return [name, points(e.effect), "\(points(e.lower)) to \(points(e.upper))", "\(e.n)", reading(e)]
        }
    }

    /// Plain-language reading of an effect interval.
    public static func reading(_ e: ArmEffect) -> String {
        if e.arm.isExperimental { return "Experimental" }
        if e.n < 3 { return "Too few tries yet" }
        if e.lower > 0 { return "Seems to help" }
        if e.upper < 0 { return "Seems not to help" }
        return "Still learning"
    }

    static func selfReports(_ s: [SessionRecord], demoIDs: Set<UUID>) -> DoctorReportContent.Section {
        var focused = 0, tired = 0, distracted = 0, stressed = 0, unspecified = 0
        var demoUsed = false
        for r in s {
            for e in r.events {
                guard case let .experienceSample(_, f, reason) = e else { continue }
                if demoIDs.contains(r.id) { demoUsed = true }
                if f { focused += 1; continue }
                switch reason {
                case .tired?: tired += 1
                case .distracted?: distracted += 1
                case .stressed?: stressed += 1
                case nil: unspecified += 1
                }
            }
        }
        let total = focused + tired + distracted + stressed + unspecified
        var blocks: [DoctorReportContent.Block] = [
            .paragraph("During some sessions FocusLoop asked \u{201C}Were you focused just now?\u{201D} These are the student\u{2019}s own answers."),
        ]
        if total == 0 {
            blocks.append(.paragraph("No check-in answers yet."))
        } else {
            func share(_ n: Int) -> String { "\(n) (\(pct(Double(n) / Double(total))))" }
            blocks.append(.keyValues([
                .init("Answers", "\(total)"),
                .init("Focused", share(focused)),
                .init("Not focused: tired", share(tired)),
                .init("Not focused: distracted", share(distracted)),
                .init("Not focused: stressed", share(stressed)),
                .init("Not focused: no reason given", share(unspecified)),
            ]))
        }
        return .init(id: "selfReports", number: 0, title: "Self-reports", isDemoData: demoUsed, blocks: blocks)
    }

    static func clinicianNotes() -> DoctorReportContent.Section {
        .init(id: "notes", number: 0, title: "Notes for the clinician", isDemoData: false, blocks: [
            .paragraph("The student may bring this summary to discuss attention. It describes how their sustained attention behaves during study sessions; it is information to share, not a conclusion."),
            .bullets([
                "How it is measured: the phone\u{2019}s front camera tracks face position, gaze direction, blinks and head movement; phone motion shows when the phone is picked up. A model turns these signals into a focused / distracted / drowsy / stressed estimate each second. Everything is processed on the phone and no video is stored.",
                "Heart rate comes from tiny colour changes in the skin seen by the camera (remote photoplethysmography). It is an estimate, less accurate for some skin tones and lighting conditions.",
                "The attention-state model has not been validated against clinical measures. Its estimates can be wrong, especially in poor lighting, with movement, or if the phone is placed differently from calibration.",
                "The SART follows Robertson et al. (1997) timing but runs on a phone touchscreen with 160 trials (about 3 minutes) by default, rather than 225 trials on a computer.",
                "Nudge results compare the student only with themself, over a small number of sessions; intervals are wide early on.",
                "Self-reports are brief in-the-moment answers and may be affected by being asked.",
            ]),
        ])
    }

    // MARK: - Formatting

    static func pct(_ f: Double) -> String { "\(Int((f * 100).rounded()))%" }

    static func points(_ f: Double) -> String {
        let v = Int((f * 100).rounded())
        return v > 0 ? "+\(v) pts" : (v < 0 ? "\u{2212}\(-v) pts" : "0 pts")
    }

    static func fmtMinutes(_ m: Double) -> String {
        let v = Int(m.rounded())
        return v >= 60 ? String(format: "%d h %02d min", v / 60, v % 60) : "\(v) min"
    }

    static func hourName(_ h: Int) -> String {
        let hh = ((h % 24) + 24) % 24
        return "\(hh % 12 == 0 ? 12 : hh % 12) \(hh < 12 ? "AM" : "PM")"
    }

    static func count(_ n: Int, _ noun: String) -> String { "\(n) \(noun)\(n == 1 ? "" : "s")" }
}

/// Words and phrases the clinician report and the sustained-attention screens must never use (the required
/// not-a-diagnosis statement is the only exception and is removed before scanning).
public enum DoctorReportWording {
    public static let banned = ["adhd", "diagnos", "disorder", "symptom", "treat", "patient", "clinically proven",
                                "normal", "abnormal"]

    /// Banned terms found in `text` (case-insensitive, matched at word starts), after removing the statement.
    public static func violations(in text: String) -> [String] {
        let cleaned = text.replacingOccurrences(of: DoctorReportContent.statement, with: " ").lowercased()
        return banned.filter { term in
            let pattern = "\\b" + NSRegularExpression.escapedPattern(for: term)
            return cleaned.range(of: pattern, options: .regularExpression) != nil
        }
    }
}
