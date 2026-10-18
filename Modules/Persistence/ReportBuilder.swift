import Foundation

/// Pure `SessionRecord` -> `SessionReport` transformation (no I/O, deterministic).
///
/// - Segments: 1 Hz ticks are merged into runs of equal state; each tick covers [t, next t)
///   (the last one up to `duration`, at most 1 s). Runs shorter than `minRunSeconds` are absorbed
///   into the previous run (or the next one if first), then equal neighbours are re-merged.
/// - `focusedFraction`: time-weighted share of raw ticks in `focused`.
/// - Refocus times: `.refocused(afterSeconds:)` events in session order; median of those.
/// - Heart rate: ticks with a BPM and `heartRateQuality >= minHeartRateQuality`. Resting HR is the
///   caller's personal baseline if given, else the 20th percentile of those points (needs >= 10).
/// - Drift overlay (only ticks with a lecture playhead): predicted = `timeline.driftRisk(at: playhead)`
///   (falling back to `tick.predictedDriftRisk`); actual = 1 - P(focused) smoothed by a centred 5 s
///   moving average in session time. Each curve is min-max scaled to 0...1 independently, both on the
///   playhead axis, in session order. Pearson r is computed on the pairs (nil if < 20 or zero variance).
/// - Biggest drift: max of the smoothed actual drift -> its playhead and `timeline.activations(at:)`.
/// - Nudges: `.nudge` decisions excluding the `noNudge` control arm (nothing was shown for those);
///   `preemptiveCount` is the pre-emptive subset.
public enum ReportBuilder {
    public static let minRunSeconds = 2.0
    public static let smoothingWindowSeconds = 5.0
    public static let minCorrelationPoints = 20
    public static let minHeartRateQuality: Float = 0.5
    public static let minRestingPoints = 10

    public static func build(record: SessionRecord, timeline: ContentTimeline?, lectureTitle: String?,
                             effects: [ArmEffect], restingHeartRate personalBaseline: Double? = nil) -> SessionReport {
        let ticks = record.ticks.sorted { $0.t < $1.t }
        let spans = tickSpans(ticks, duration: record.duration)

        // Focus timeline
        let segments = segmentsFrom(ticks: ticks, spans: spans)
        let total = spans.reduce(0) { $0 + ($1.end - $1.start) }
        let focused = zip(ticks, spans).reduce(0.0) { $0 + ($1.0.state == .focused ? $1.1.end - $1.1.start : 0) }
        let focusedFraction = total > 0 ? focused / total : 0

        // Refocus
        let refocus = record.events
            .compactMap { e -> (Double, Double)? in
                if case let .refocused(t, after) = e { return (t, after) } else { return nil }
            }
            .sorted { $0.0 < $1.0 }
            .map(\.1)

        // Heart rate
        let hr = ticks.compactMap { tick -> SessionReport.Point? in
            guard let bpm = tick.heartRate, bpm.isFinite, bpm > 0,
                  let q = tick.heartRateQuality, q >= minHeartRateQuality else { return nil }
            return SessionReport.Point(t: tick.t, value: Double(bpm))
        }
        let resting = personalBaseline ?? (hr.count >= minRestingPoints ? percentile(hr.map(\.value), 0.2) : nil)

        // Predicted vs actual drift on the playhead axis
        let actualSmoothed = smoothedNotFocused(ticks)
        var playheads: [Double] = [], predicted: [Double] = [], actual: [Double] = []
        for (i, tick) in ticks.enumerated() {
            guard let p = tick.playhead, p.isFinite else { continue }
            let pred: Double? = timeline.map { Double($0.driftRisk(at: p)) } ?? tick.predictedDriftRisk.map(Double.init)
            guard let pr = pred, pr.isFinite else { continue }
            playheads.append(p); predicted.append(pr); actual.append(actualSmoothed[i])
        }
        let predScaled = minMaxScale(predicted), actScaled = minMaxScale(actual)
        let predictedDrift = zip(playheads, predScaled).map { SessionReport.Point(t: $0, value: $1) }
        let actualDrift = zip(playheads, actScaled).map { SessionReport.Point(t: $0, value: $1) }
        let r = pearson(predicted, actual)

        var biggestPlayhead: Double?
        var biggestActivations: [Float]?
        if let iMax = actual.indices.max(by: { actual[$0] < actual[$1] }) {
            biggestPlayhead = playheads[iMax]
            biggestActivations = timeline?.activations(at: playheads[iMax])
        }

        // Nudges
        let decisions = record.events.compactMap { e -> NudgeDecision? in
            if case let .nudge(d) = e, d.arm != .noNudge { return d } else { return nil }
        }

        return SessionReport(
            sessionID: record.id, startedAt: record.startedAt, duration: record.duration,
            lectureID: record.lectureID, lectureTitle: lectureTitle, contentIsMock: record.contentIsMock,
            isReplay: record.isReplay, segments: segments, focusedFraction: focusedFraction,
            refocusTimes: refocus, medianRefocusSeconds: refocus.isEmpty ? nil : percentile(refocus, 0.5),
            heartRate: hr, restingHeartRate: resting,
            predictedDrift: predictedDrift, actualDrift: actualDrift, driftCorrelation: r,
            biggestDriftPlayhead: biggestPlayhead, biggestDriftActivations: biggestActivations,
            nudgeCount: decisions.count, preemptiveCount: decisions.filter { $0.policy == .preemptive }.count,
            effects: effects, insights: record.insights)
    }

    // MARK: - Pieces (internal for tests)

    static func tickSpans(_ ticks: [SessionTick], duration: Double) -> [(start: Double, end: Double)] {
        ticks.indices.map { i in
            let t = ticks[i].t
            let end = i + 1 < ticks.count ? ticks[i + 1].t : max(t, min(duration, t + 1))
            return (t, max(t, end))
        }
    }

    static func segmentsFrom(ticks: [SessionTick], spans: [(start: Double, end: Double)]) -> [SessionReport.Segment] {
        var runs: [SessionReport.Segment] = []
        for (tick, span) in zip(ticks, spans) {
            if let last = runs.last, last.state == tick.state {
                runs[runs.count - 1].end = span.end
            } else {
                runs.append(.init(start: span.start, end: span.end, state: tick.state))
            }
        }
        guard runs.count > 1 else { return runs }

        // Absorb short runs.
        var out: [SessionReport.Segment] = []
        var carryStart: Double?   // a leading short run waiting to join the next run
        for run in runs {
            var run = run
            if let s = carryStart { run.start = s; carryStart = nil }
            if run.end - run.start < minRunSeconds {
                if !out.isEmpty { out[out.count - 1].end = run.end } else { carryStart = run.start }
                continue
            }
            if let last = out.last, last.state == run.state {
                out[out.count - 1].end = run.end
            } else {
                out.append(run)
            }
        }
        if let s = carryStart {   // everything was short: keep it as one run of the first state
            out.append(.init(start: s, end: runs.last!.end, state: runs.first!.state))
        }
        return out
    }

    /// 1 - P(focused), centred moving average over `smoothingWindowSeconds` of session time.
    static func smoothedNotFocused(_ ticks: [SessionTick]) -> [Double] {
        let raw = ticks.map { 1 - Double($0.probabilities.indices.contains(AttentionState.focused.index)
                                          ? $0.probabilities[AttentionState.focused.index]
                                          : ($0.state == .focused ? 1 : 0)) }
        let half = smoothingWindowSeconds / 2
        var out = [Double](repeating: 0, count: ticks.count)
        var lo = 0, hi = 0, sum = 0.0
        for i in ticks.indices {
            while hi < ticks.count && ticks[hi].t <= ticks[i].t + half { sum += raw[hi]; hi += 1 }
            while lo < hi && ticks[lo].t < ticks[i].t - half { sum -= raw[lo]; lo += 1 }
            out[i] = sum / Double(max(1, hi - lo))
        }
        return out
    }

    static func minMaxScale(_ v: [Double]) -> [Double] {
        guard let lo = v.min(), let hi = v.max() else { return [] }
        let range = hi - lo
        return range > 1e-12 ? v.map { ($0 - lo) / range } : v.map { _ in 0 }
    }

    static func pearson(_ x: [Double], _ y: [Double]) -> Double? {
        let n = min(x.count, y.count)
        guard n >= minCorrelationPoints else { return nil }
        let mx = x.prefix(n).reduce(0, +) / Double(n), my = y.prefix(n).reduce(0, +) / Double(n)
        var sxy = 0.0, sxx = 0.0, syy = 0.0
        for i in 0..<n {
            let dx = x[i] - mx, dy = y[i] - my
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy
        }
        guard sxx > 1e-12, syy > 1e-12 else { return nil }
        return max(-1, min(1, sxy / (sxx * syy).squareRoot()))
    }

    /// Linear-interpolated percentile, p in 0...1.
    static func percentile(_ values: [Double], _ p: Double) -> Double? {
        let s = values.sorted()
        guard !s.isEmpty else { return nil }
        let pos = p * Double(s.count - 1)
        let lo = Int(pos.rounded(.down)), hi = Int(pos.rounded(.up))
        return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
    }
}
