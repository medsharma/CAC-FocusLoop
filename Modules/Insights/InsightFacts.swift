import Foundation

/// Shared number formatting and the compact fact list used both by the template coach and as the
/// ONLY content of the Foundation Models prompt.
enum InsightsFormat {
    static let posix = Locale(identifier: "en_US_POSIX")

    static func percent(_ x: Double) -> String { "\(Int((x * 100).rounded()))%" }

    static func minutes(_ x: Double) -> String { "\(Int(x.rounded()))" }

    static func seconds(_ x: Double) -> String {
        x >= 10 ? "\(Int(x.rounded())) s" : String(format: "%.1f s", locale: posix, x)
    }

    static func bpm(_ x: Double) -> String { "\(Int(x.rounded())) BPM" }

    static func hour(_ h: Int) -> String {
        let hh = ((h % 24) + 24) % 24
        let twelve = hh % 12 == 0 ? 12 : hh % 12
        return "\(twelve) \(hh < 12 ? "AM" : "PM")"
    }

    static func r(_ x: Double) -> String { String(format: "%.2f", locale: posix, x) }

    static func signed(_ x: Double, decimals: Int = 0) -> String {
        let s = String(format: "%.\(decimals)f", locale: posix, abs(x))
        if s.allSatisfy({ $0 == "0" || $0 == "." }) { return s }
        return (x < 0 ? "-" : "+") + s
    }

    /// Effect in percentage points, e.g. "+18".
    static func points(_ x: Double) -> String { signed(x * 100) }

    static func list(_ items: [String]) -> String {
        switch items.count {
        case 0: return ""
        case 1: return items[0]
        case 2: return "\(items[0]) and \(items[1])"
        default: return items.dropLast().joined(separator: ", ") + ", and " + items.last!
        }
    }

    static func sessions(_ n: Int) -> String { n == 1 ? "1 session" : "\(n) sessions" }
}

/// What the trial says about one arm, decided from its interval (never from the point estimate alone).
enum InsightArmVerdict: String {
    /// Interval entirely above 0 (and not the experimental arm).
    case helps
    /// Interval entirely below 0.
    case notHelping = "not helping"
    /// Interval includes 0, or experimental arm: no claim.
    case stillLearning = "still learning"

    static func of(_ e: ArmEffect) -> InsightArmVerdict {
        if e.arm.isExperimental { return .stillLearning }    // spec: never claim the light pulse works
        if e.lower > 0 { return .helps }
        if e.upper < 0 { return .notHelping }
        return .stillLearning
    }
}

/// Serialises an `AggregateSummary` into a compact, numbers-only fact list.
public enum InsightFacts {

    public static let mockQualifier = "(based on MOCK content predictions)"
    public static let replayQualifier = "(SYNTHETIC replay data)"

    /// Appends the MOCK / SYNTHETIC qualifier(s) to `text` when they apply and are not already present.
    /// Inline "(SYNTHETIC replay data)" / "(based on MOCK…)" qualifiers in insight sentences. Off by default:
    /// the report shows one data footnote instead (product decision). Tests turn it on to check the labelling path.
    nonisolated(unsafe) public static var inlineDataLabels = false

    static func qualified(_ text: String, mock: Bool, replay: Bool) -> String {
        guard inlineDataLabels else { return text }
        var parts: [String] = []
        if mock, !text.contains("MOCK") { parts.append("based on MOCK content predictions") }
        if replay, !text.contains("SYNTHETIC") { parts.append("SYNTHETIC replay data") }
        guard !parts.isEmpty else { return text }
        return text + " (" + parts.joined(separator: "; ") + ")"
    }

    /// Nudge effects worth talking about (control arm excluded), largest effect first.
    static func reportableEffects(_ s: AggregateSummary) -> [ArmEffect] {
        s.armEffects.filter { $0.arm != .noNudge && $0.effect.isFinite && $0.lower.isFinite && $0.upper.isFinite }
            .sorted { $0.effect != $1.effect ? $0.effect > $1.effect : $0.arm.rawValue < $1.arm.rawValue }
    }

    /// One "- key: value" line per available fact, plus every number the text contains
    /// (for `InsightGuardrails.unsupportedNumbers`).
    public static func factList(for s: AggregateSummary) -> (text: String, numbers: [Double]) {
        var lines: [String] = []
        lines.append("sessions: \(s.sessionCount)")
        lines.append("minutes tracked: \(InsightsFormat.minutes(s.totalMinutes))")
        lines.append("time focused: \(InsightsFormat.percent(s.focusedFraction))")
        let names = AttentionState.allCases.map(\.rawValue)
        if s.stateFractions.count == names.count {
            let other = (1..<names.count).map { "\(names[$0]) \(InsightsFormat.percent(s.stateFractions[$0]))" }
            lines.append("time in other states: " + other.joined(separator: ", "))
        }
        if let m = s.typicalDropMinute, m.rounded() >= 1 {   // "minute 0" reads as nonsense
            lines.append("focus most often starts to drop around session minute: \(InsightsFormat.minutes(m))")
        }
        if let r = s.medianRefocusSeconds {
            lines.append("typical time to get back to focus: \(InsightsFormat.seconds(r))")
        }
        if !s.bestHours.isEmpty {
            lines.append("best focus when sessions start around: " + s.bestHours.map(InsightsFormat.hour).joined(separator: ", "))
        }
        if let hr = s.restingHeartRate {
            lines.append("typical heart rate while focused: \(InsightsFormat.bpm(hr))")
        }
        if let t = s.heartRateTrendPerSession {
            let dir = abs(t) < 0.5 ? "steady" : (t < 0 ? "trending down" : "trending up")
            lines.append("heart rate change per session: \(InsightsFormat.signed(t, decimals: 1)) BPM (\(dir))")
        }
        if let c = s.contentDriftCorrelation {
            lines.append("correlation between the predicted brain response to the lecture (average brain) "
                         + "drift risk and the student's actual drift: r = \(InsightsFormat.r(c))"
                         + (s.contentIsMock ? " [content predictions are MOCK]" : ""))
        }
        for e in reportableEffects(s) {
            let kind = e.policy == .preemptive ? "lapse avoided" : "refocused within 20 s"
            lines.append("nudge \"\(e.arm.displayName)\": \(InsightsFormat.points(e.effect)) points (\(kind)) vs no nudge, "
                         + "\(Int((e.level * 100).rounded()))% interval \(InsightsFormat.points(e.lower)) to "
                         + "\(InsightsFormat.points(e.upper)), n = \(e.n), verdict: \(InsightArmVerdict.of(e).rawValue)")
        }
        if inlineDataLabels, s.isReplay { lines.append("data label: SYNTHETIC replay data (not a real student)") }
        if inlineDataLabels, s.contentIsMock { lines.append("content label: MOCK content predictions") }

        let text = lines.map { "- " + $0 }.joined(separator: "\n")
        var nums = InsightGuardrails.numbers(in: text)
        // Equivalent spellings the model may legitimately use.
        nums += s.bestHours.map(Double.init)
        nums += s.stateFractions
        nums.append(s.focusedFraction)
        return (text, nums)
    }
}
