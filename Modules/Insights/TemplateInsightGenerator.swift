import Foundation

/// Deterministic coach: 2-4 insights (1-2 sentences each) built from `AggregateSummary` only.
/// Phrasing varies with a stable seed derived from the numbers, so the same summary always gives
/// the same text. Wellness framing; effect claims only when the interval excludes 0.
public final class TemplateInsightGenerator: InsightGenerator {

    public init() {}

    public var usesFoundationModels: Bool { false }

    public func insights(for summary: AggregateSummary) async -> [Insight] {
        texts(for: summary).map { Insight(text: $0, source: .template) }
    }

    /// Synchronous core (also the fallback used by the Foundation Models generator).
    public func texts(for s: AggregateSummary) -> [String] {
        guard s.sessionCount > 0, s.totalMinutes > 0 || s.stateFractions.contains(where: { $0 > 0 }) else {
            return [
                "No sessions recorded yet. Start a study session and FocusLoop will learn when your focus tends to drift.",
                "Insights use only summary numbers that stay on this device.",
            ]
        }
        let seed = Self.seed(for: s)
        let replay = s.isReplay

        var primary: [String] = []
        primary.append(focusInsight(s, seed))
        if let n = nudgeInsight(s, seed) { primary.append(n) }
        if let d = driftInsight(s, seed) { primary.append(d) }

        var secondary: [String] = []
        if let b = bestHoursInsight(s, seed) { secondary.append(b) }
        if let h = heartRateInsight(s, seed) { secondary.append(h) }
        if let r = refocusInsight(s, seed) { secondary.append(r) }
        if secondary.count > 1 {
            // Rotate which secondary topic leads, for variety across summaries.
            let k = Int(seed % UInt64(secondary.count))
            secondary = Array(secondary[k...] + secondary[..<k])
        }

        var out = Array((primary + secondary).prefix(4)).map {
            InsightFacts.qualified($0, mock: false, replay: replay)
        }
        if out.count < 2 {
            out.append("Keep running sessions so FocusLoop can learn what supports your focus.")
        }
        return out
    }

    // MARK: - Topics

    private func focusInsight(_ s: AggregateSummary, _ seed: UInt64) -> String {
        let pct = InsightsFormat.percent(s.focusedFraction)
        let mins = InsightsFormat.minutes(s.totalMinutes)
        let sess = InsightsFormat.sessions(s.sessionCount)
        guard let drop = s.typicalDropMinute else {
            return pick(seed, 0, [
                "You were focused for \(pct) of \(mins) tracked minutes across \(sess). No usual drop point stands out yet.",
                "Across \(sess) you stayed focused \(pct) of the time, with no single minute where focus usually slipped.",
            ])
        }
        let m = InsightsFormat.minutes(drop)
        return pick(seed, 0, [
            "Your focus most often starts to slip around minute \(m) of a session. Across \(sess) you were focused \(pct) of the time.",
            "Across \(sess), focus tended to fade around minute \(m). A short reset just before then may help you stay with it.",
            "You stayed focused for \(pct) of \(mins) tracked minutes, and longer lapses usually began around minute \(m).",
        ])
    }

    private func nudgeInsight(_ s: AggregateSummary, _ seed: UInt64) -> String? {
        let effects = InsightFacts.reportableEffects(s)
        guard !effects.isEmpty else { return nil }
        func interval(_ e: ArmEffect) -> String {
            "\(Int((e.level * 100).rounded()))% interval \(InsightsFormat.points(e.lower)) to \(InsightsFormat.points(e.upper)) points"
        }
        func outcome(_ e: ArmEffect) -> String {
            e.policy == .preemptive ? "avoid a lapse" : "refocus within 20 s"
        }
        if let best = effects.first(where: { InsightArmVerdict.of($0) == .helps }) {
            let pts = InsightsFormat.points(best.effect)
            return pick(seed, 1, [
                "\(best.arm.displayName) has helped you so far: you were about \(pts) points more likely to \(outcome(best)) than with no nudge (\(interval(best))).",
                "Your personal trial shows \(best.arm.displayName) working for you, at \(pts) points vs no nudge (\(interval(best))). FocusLoop keeps testing it as it learns what works for you.",
            ])
        }
        if let worst = effects.last(where: { InsightArmVerdict.of($0) == .notHelping }) {
            return "\(worst.arm.displayName) has not helped so far (\(interval(worst)) vs no nudge), so the trial will likely pick it less often."
        }
        let n = effects.reduce(0) { $0 + max(0, $1.n) }
        return pick(seed, 1, [
            "FocusLoop is still learning which nudges work for you: every estimate so far has an interval that includes zero.",
            "No nudge has a clear effect yet after \(n) tries, so FocusLoop is still learning what works for you.",
        ])
    }

    private func driftInsight(_ s: AggregateSummary, _ seed: UInt64) -> String? {
        guard let c = s.contentDriftCorrelation, c.isFinite else { return nil }
        let r = InsightsFormat.r(c)
        let what = s.sessionCount == 1
            ? "the predicted brain response to this lecture (average brain)"
            : "the predicted brain response to your lectures (average brain)"
        let text: String
        if c >= 0.3 {
            text = pick(seed, 2, [
                "Your drift lined up with the content's predicted valleys (r = \(r)): when \(what) dipped, your attention tended to dip too.",
                "When \(what) dipped, you tended to drift as well (r = \(r)). Active-recall prompts before those valleys may help.",
            ])
        } else if c >= 0.1 {
            text = "Your drift lined up only loosely with \(what) (r = \(r))."
        } else if c > -0.1 {
            text = "Your drift did not track \(what) (r = \(r)), so the predicted valleys did not explain when you drifted."
        } else {
            text = "Your drift tended to run opposite to \(what) (r = \(r)), drifting more when the content was predicted to be engaging."
        }
        return InsightFacts.qualified(text, mock: s.contentIsMock, replay: false)
    }

    private func bestHoursInsight(_ s: AggregateSummary, _ seed: UInt64) -> String? {
        guard !s.bestHours.isEmpty else { return nil }
        let hours = s.bestHours.map(InsightsFormat.hour)
        if hours.count == 1 {
            return pick(seed, 3, [
                "Your strongest focus came in sessions starting around \(hours[0]). Scheduling demanding work then may help.",
                "Sessions that started around \(hours[0]) were your most focused so far.",
            ])
        }
        return pick(seed, 3, [
            "Your best focus hours so far start around \(InsightsFormat.list(hours)), with \(hours[0]) on top.",
            "You focused best in sessions starting around \(hours[0]), followed by \(InsightsFormat.list(Array(hours.dropFirst()))).",
        ])
    }

    private func heartRateInsight(_ s: AggregateSummary, _ seed: UInt64) -> String? {
        guard let hr = s.restingHeartRate, hr.isFinite else { return nil }
        let bpm = InsightsFormat.bpm(hr)
        let base = pick(seed, 4, [
            "Your heart rate while focused settled around \(bpm)",
            "While you were focused, your heart rate sat around \(bpm)",
        ])
        guard let t = s.heartRateTrendPerSession, t.isFinite else {
            return base + ". It is a general wellness reading from the camera, not a medical measurement."
        }
        if abs(t) < 0.5 { return base + ", and it has stayed steady across sessions." }
        let amount = String(format: "%.1f", locale: InsightsFormat.posix, abs(t))
        return base + ", and it has been trending \(t < 0 ? "down" : "up") by about \(amount) BPM per session."
    }

    private func refocusInsight(_ s: AggregateSummary, _ seed: UInt64) -> String? {
        guard let r = s.medianRefocusSeconds, r.isFinite else { return nil }
        let sec = InsightsFormat.seconds(r)
        return pick(seed, 5, [
            "When your focus slipped, you typically got back on track in about \(sec).",
            "Your typical time to get back to focus was about \(sec).",
        ])
    }

    // MARK: - Deterministic variety

    private func pick(_ seed: UInt64, _ salt: UInt64, _ options: [String]) -> String {
        options[Int((seed &+ salt &* 0x9E37_79B9) % UInt64(options.count))]
    }

    /// Stable FNV-1a hash of the rounded headline numbers (Swift's `Hasher` is randomised per launch).
    static func seed(for s: AggregateSummary) -> UInt64 {
        let key = [
            "\(s.sessionCount)", "\(Int(s.totalMinutes.rounded()))", "\(Int((s.focusedFraction * 1000).rounded()))",
            "\(s.typicalDropMinute.map { Int($0) } ?? -1)", "\(s.bestHours)",
            "\(s.restingHeartRate.map { Int($0.rounded()) } ?? -1)",
        ].joined(separator: "|")
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in key.utf8 {
            h ^= UInt64(b)
            h = h &* 0x0000_0100_0000_01B3
        }
        return h
    }
}
