import Foundation

/// Builds a `SessionReport` from a `SessionRecord`. PREVIEW-ONLY stand-in for the Persistence agent's
/// report builder so the report screen renders before integration. Numbers come from SYNTHETIC replay.
public enum PreviewReportBuilder {
    public static func build(record r: SessionRecord, timeline: ContentTimeline?, lectureTitle: String?,
                             effects: [ArmEffect], nudgeCount: Int, preemptiveCount: Int) -> SessionReport {
        let ticks = r.ticks
        let segs = StateSegmenter.segments(ticks.map { ($0.t, $0.state) }, end: r.duration)
        let fractions = StateSegmenter.fractions(segs)
        var refocus: [Double] = []
        for e in r.events { if case let .refocused(_, s) = e { refocus.append(s) } }
        let hr = ticks.compactMap { t in t.heartRate.map { SessionReport.Point(t: t.t, value: Double($0)) } }

        // Predicted vs actual drift on the lecture playhead axis (first pass through the lecture only).
        var pass: [SessionTick] = []
        for t in ticks {
            guard let ph = t.playhead else { break }
            if let last = pass.last?.playhead, ph < last { break }
            pass.append(t)
        }
        var predicted: [SessionReport.Point] = [], actual: [SessionReport.Point] = []
        var r2: Double?, biggest: Double?, acts: [Float]?
        if let tl = timeline, r.lectureID != nil, pass.count >= 10 {
            let ph = pass.map { $0.playhead ?? 0 }
            let pred = UIStats.minMax(ph.map { Double(tl.driftRisk(at: $0)) })
            let raw = pass.map { 1 - Double($0.probabilities.first ?? 1) }
            let act = UIStats.minMax(UIStats.ema(raw, alpha: 0.3))
            predicted = zip(ph, pred).map { .init(t: $0, value: $1) }
            actual = zip(ph, act).map { .init(t: $0, value: $1) }
            r2 = UIStats.pearson(pred, act)
            if let i = raw.indices.max(by: { raw[$0] < raw[$1] }) {
                biggest = ph[i]
                acts = tl.activations(at: ph[i])
            }
        }
        let sortedHR = hr.map(\.value).sorted()
        let resting = sortedHR.isEmpty ? nil : sortedHR[sortedHR.count / 10]   // 10th percentile as a resting proxy
        let insights = templateInsights(focused: fractions.first ?? 0, medianRefocus: UIStats.median(refocus),
                                        correlation: r2, nudges: nudgeCount, isReplay: r.isReplay)
        return SessionReport(sessionID: r.id, startedAt: r.startedAt, duration: r.duration, lectureID: r.lectureID,
                             lectureTitle: lectureTitle, contentIsMock: r.contentIsMock, isReplay: r.isReplay,
                             segments: segs, focusedFraction: fractions.first ?? 0, refocusTimes: refocus,
                             medianRefocusSeconds: UIStats.median(refocus), heartRate: hr, restingHeartRate: resting,
                             predictedDrift: predicted, actualDrift: actual, driftCorrelation: r2,
                             biggestDriftPlayhead: biggest, biggestDriftActivations: acts, nudgeCount: nudgeCount,
                             preemptiveCount: preemptiveCount, effects: effects, insights: insights)
    }

    /// Template coach text from aggregated numbers (stand-in for the Insights agent).
    static func templateInsights(focused: Double, medianRefocus: Double?, correlation: Double?, nudges: Int,
                                 isReplay: Bool) -> [Insight] {
        let prefix = isReplay ? "In this SYNTHETIC replay session, " : ""
        var out = [Insight(text: "\(prefix)you stayed focused for \(UIFormat.percent(focused)) of the time.", source: .template)]
        if let m = medianRefocus {
            out.append(Insight(text: "When attention slipped, you were typically back in \(UIFormat.oneDecimal(m)) seconds.",
                               source: .template))
        }
        if let c = correlation {
            let strength = abs(c) >= 0.5 ? "closely" : abs(c) >= 0.25 ? "partly" : "only loosely"
            out.append(Insight(text: "Your drift \(strength) followed the lecture's predicted low-engagement moments (\(UIFormat.correlation(c))).",
                               source: .template))
        }
        if nudges > 0 {
            out.append(Insight(text: "FocusLoop tried \(nudges) nudge\(nudges == 1 ? "" : "s") and is learning which ones help you most.",
                               source: .template))
        }
        return out
    }
}
