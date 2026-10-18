import Foundation

/// Derives labelled 10 s feature windows from a SART run (spec §1.5).
///
/// * Not focused (`isLapse = true`, label `.distracted`):
///   - the window preceding each commission error: [onset - window, onset);
///   - windows ending at a trial whose rolling RT coefficient of variation (over the last
///     `cvWindowTrials` trials, 4-8) is above the user's own `highCVPercentile` (80th) in this run.
///   Overlapping not-focused intervals are merged so no frame is exported twice.
/// * Focused (`isLapse = false`): the SART span is tiled into consecutive windows; a tile is focused if it
///   contains no commission/omission error, does not overlap a not-focused interval, and every rolling CV in it
///   is at or below the user's `stableCVPercentile` (median).
///
/// Trial times and frame times must share the session clock (`SARTEngine.clockOrigin`). Windows are clipped
/// to the SART span and dropped if shorter than half a window or with less than `minFrameCoverage` of the
/// expected 10 Hz frames.
public struct SARTLabeler: Sendable {
    public var windowSeconds: Double = FeatureFrame.stateWindowSeconds
    /// Rolling window for the RT CV, in trials (clamped to 4...8).
    public var cvWindowTrials = 6
    /// Minimum go-trial RTs inside a rolling window for its CV to count.
    public var minRTsForCV = 3
    public var highCVPercentile = 0.8
    public var stableCVPercentile = 0.5
    public var minFrameCoverage = 0.5

    /// CVs within this of a percentile count as equal (sums of the same RTs in a different order differ by ~1e-17).
    static let tieTolerance = 1e-9

    public init() {}

    /// Rolling RT CV ending at each trial (nil where too few go RTs). Same indexing as `trials`.
    public func rollingCV(_ trials: [SARTTrial]) -> [Double?] {
        let k = min(8, max(4, cvWindowTrials))
        return trials.indices.map { i in
            let lo = max(0, i - k + 1)
            let rts = trials[lo...i].filter { !$0.isNoGo }.compactMap(\.rt)
            guard rts.count >= minRTsForCV else { return nil }
            return CalibrationStats.coefficientOfVariation(rts)
        }
    }

    public func label(trials input: [SARTTrial], frames: [FeatureFrame], sessionID: UUID) -> [LabeledWindow] {
        let trials = input.filter(\.isComplete).sorted { $0.onset < $1.onset }
        guard let spanStart = trials.first?.onset, let spanEnd = trials.last?.end else { return [] }
        let cvs = rollingCV(trials)
        let valid = cvs.compactMap { $0 }
        let high = CalibrationStats.percentile(valid, highCVPercentile)
        let stable = CalibrationStats.percentile(valid, stableCVPercentile)

        // Not-focused intervals.
        var lapse: [(Double, Double)] = []
        for (i, tr) in trials.enumerated() {
            if tr.commission { lapse.append((tr.onset - windowSeconds, tr.onset)) }
            if let cv = cvs[i], let high, cv > high + Self.tieTolerance, let end = tr.end {
                lapse.append((end - windowSeconds, end))
            }
        }
        let merged = Self.merge(lapse.compactMap { clip($0, spanStart, spanEnd) })

        // Focused tiles.
        var focused: [(Double, Double)] = []
        var s = spanStart
        while s + windowSeconds <= spanEnd + 1e-9 {
            let e = s + windowSeconds
            defer { s = e }
            if merged.contains(where: { $0.0 < e && $0.1 > s }) { continue }
            let inside = trials.indices.filter { trials[$0].onset >= s && trials[$0].onset < e }
            if inside.isEmpty || inside.contains(where: { trials[$0].isError }) { continue }
            let insideCV = inside.compactMap { cvs[$0] }
            guard let stable, !insideCV.isEmpty, insideCV.allSatisfy({ $0 <= stable + Self.tieTolerance }) else { continue }
            focused.append((s, e))
        }

        let expectedPerSecond = FeatureFrame.rateHz
        func window(_ iv: (Double, Double), lapse: Bool) -> LabeledWindow? {
            let f = calibrationFrames(frames, from: iv.0, to: iv.1)
            guard Double(f.count) >= minFrameCoverage * (iv.1 - iv.0) * expectedPerSecond, !f.isEmpty else { return nil }
            return LabeledWindow(sessionID: sessionID, start: iv.0, end: iv.1, label: lapse ? .distracted : .focused,
                                 isLapse: lapse, source: .sart, frames: f)
        }
        let out = merged.compactMap { window($0, lapse: true) } + focused.compactMap { window($0, lapse: false) }
        return out.sorted { $0.start < $1.start }
    }

    private func clip(_ iv: (Double, Double), _ lo: Double, _ hi: Double) -> (Double, Double)? {
        let a = max(iv.0, lo), b = min(iv.1, hi)
        return b - a >= windowSeconds / 2 ? (a, b) : nil
    }

    static func merge(_ ivs: [(Double, Double)]) -> [(Double, Double)] {
        var out: [(Double, Double)] = []
        for iv in ivs.sorted(by: { $0.0 < $1.0 }) {
            if let last = out.last, iv.0 <= last.1 {
                out[out.count - 1].1 = max(last.1, iv.1)
            } else {
                out.append(iv)
            }
        }
        return out
    }
}
