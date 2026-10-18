import Foundation

/// A published range for one SART metric, shown only when it has been verified against the primary source.
/// Never add a range that was not read in the source itself.
public struct SARTReferenceRange: Equatable, Sendable, Identifiable {
    public enum Metric: String, Sendable {
        case commissionRate, omissionRate, meanRT, rtCV
        public var label: String {
            switch self {
            case .commissionRate: "Taps on 3"
            case .omissionRate: "Missed digits"
            case .meanRT: "Response time"
            case .rtCV: "Variability (CV)"
            }
        }
    }

    public var metric: Metric
    /// Same units as `SARTMetrics` (fractions for rates, ms for RT, unitless for CV).
    public var low: Double
    public var high: Double
    /// Who was measured, e.g. "adults".
    public var population: String
    /// In-app short citation, e.g. "Robertson et al., 1997".
    public var citation: String
    /// How the range was derived (e.g. "mean ± 1 SD of the control group, n = 30").
    public var derivation: String
    /// Trials in the published task (our default task is 160).
    public var publishedTrials: Int
    public var id: String { metric.rawValue + citation }

    public init(metric: Metric, low: Double, high: Double, population: String, citation: String, derivation: String,
                publishedTrials: Int) {
        self.metric = metric; self.low = low; self.high = high; self.population = population
        self.citation = citation; self.derivation = derivation; self.publishedTrials = publishedTrials
    }
}

public enum SARTReferenceData {
    /// Verified ranges (checked against the primary sources). Empty means none could be verified.
    ///
    /// Both sources describe one online sample (N = 504, mean age 32, SD 11) that ran the standard 225-trial SART
    /// (250 ms digit, 900 ms mask, no-go 3) on home computers. Ranges are mean ± 1 SD as published.
    /// * Cheyne, Solman, Carriere & Smilek (2009), Cognition 111:98-113, Table 1: NOGO errors 10.89 (SD 5.83) of 25
    ///   -> 5.06-16.72 of 25 = 20.2%-66.9%; RT CV 0.28 (SD 0.12) -> 0.16-0.40 (CV over RTs > 200 ms).
    /// * Cheyne, Carriere & Smilek (2006), Consciousness and Cognition 15:578-592, Table 4: SART RTs 311.83 ms
    ///   (SD 73.03) -> 238.8-384.9 ms.
    public static let ranges: [SARTReferenceRange] = [
        SARTReferenceRange(metric: .commissionRate, low: 5.06 / 25, high: 16.72 / 25, population: "adults",
                           citation: "Cheyne et al., 2009",
                           derivation: "Mean \u{00B1} 1 SD of 504 adults who did the task online (10.89 \u{00B1} 5.83 of 25 threes tapped)",
                           publishedTrials: 225),
        SARTReferenceRange(metric: .rtCV, low: 0.16, high: 0.40, population: "adults",
                           citation: "Cheyne et al., 2009",
                           derivation: "Mean \u{00B1} 1 SD of the same 504 adults (CV 0.28 \u{00B1} 0.12)",
                           publishedTrials: 225),
        SARTReferenceRange(metric: .meanRT, low: 238.80, high: 384.86, population: "adults",
                           citation: "Cheyne et al., 2006",
                           derivation: "Mean \u{00B1} 1 SD of the same online sample (312 \u{00B1} 73 ms)",
                           publishedTrials: 225),
    ]

    public static let noVerifiedSourceNote = "Reference ranges not shown: no verified source."

    public static func range(_ metric: SARTReferenceRange.Metric) -> SARTReferenceRange? {
        ranges.first { $0.metric == metric }
    }

    /// "Typical range in published studies of adults: 4–12% (Source, year)". Rates as percentages.
    public static func sentence(_ r: SARTReferenceRange, ourTrials: Int) -> String {
        let value: (Double) -> String
        switch r.metric {
        case .commissionRate, .omissionRate: value = { "\(Int(($0 * 100).rounded()))" }
        case .meanRT: value = { "\(Int($0.rounded()))" }
        case .rtCV: value = { String(format: "%.2f", $0) }
        }
        let unit: String
        switch r.metric {
        case .commissionRate, .omissionRate: unit = "%"
        case .meanRT: unit = " ms"
        case .rtCV: unit = ""
        }
        var s = "Typical range in published studies of \(r.population): \(value(r.low))\u{2013}\(value(r.high))\(unit) (\(r.citation))."
        if ourTrials != r.publishedTrials {
            s += " That task had \(r.publishedTrials) trials; yours had \(ourTrials), so compare loosely."
        }
        return s
    }
}
