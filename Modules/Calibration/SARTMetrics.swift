import Foundation

// Sustained-attention scores for one SART run, and the persisted record of a run.
//
// Definitions (all computed on completed trials only):
// * Commission error rate = no-go digits (3) tapped / no-go digits shown.
// * Omission rate         = go digits with no tap / go digits shown.
// * Anticipation          = a go-trial tap faster than 100 ms after digit onset. At that speed the tap cannot be a
//                           response to the digit (it is a late tap for the previous digit or a guess), so
//                           anticipations are counted separately and left out of every reaction-time statistic.
// * Go RT statistics      = mean, sample SD (n - 1) and coefficient of variation (SD / mean) of go-trial reaction
//                           times in ms, excluding anticipations and trials whose display timing was disrupted
//                           (`SARTTrial.timingValid == false`, e.g. the app was backgrounded).
// * Post-error slowing    = mean RT of the first go trial after each commission error minus the mean RT of go
//                           trials that follow a correct go trial. Nil without a usable post-error RT.

public struct SARTMetrics: Codable, Equatable, Sendable {
    /// Taps faster than this (ms after onset) are anticipations.
    public static let anticipationThresholdMs: Double = 100

    public var trialCount: Int
    public var goCount: Int
    public var noGoCount: Int
    public var commissionCount: Int
    public var omissionCount: Int
    /// Fraction 0...1 (nil without no-go trials).
    public var commissionRate: Double?
    /// Fraction 0...1 (nil without go trials).
    public var omissionRate: Double?
    public var anticipationCount: Int
    /// Go RTs that entered the RT statistics.
    public var rtCount: Int
    public var meanGoRTms: Double?
    public var rtSDms: Double?
    /// SD / mean (unitless).
    public var rtCV: Double?
    public var postErrorSlowingMs: Double?

    public init(trialCount: Int, goCount: Int, noGoCount: Int, commissionCount: Int, omissionCount: Int,
                commissionRate: Double?, omissionRate: Double?, anticipationCount: Int, rtCount: Int,
                meanGoRTms: Double?, rtSDms: Double?, rtCV: Double?, postErrorSlowingMs: Double?) {
        self.trialCount = trialCount; self.goCount = goCount; self.noGoCount = noGoCount
        self.commissionCount = commissionCount; self.omissionCount = omissionCount
        self.commissionRate = commissionRate; self.omissionRate = omissionRate
        self.anticipationCount = anticipationCount; self.rtCount = rtCount
        self.meanGoRTms = meanGoRTms; self.rtSDms = rtSDms; self.rtCV = rtCV
        self.postErrorSlowingMs = postErrorSlowingMs
    }

    /// True if a go trial's tap counts as an anticipation.
    public static func isAnticipation(_ t: SARTTrial) -> Bool {
        guard !t.isNoGo, let rt = t.rtMs else { return false }
        return rt < anticipationThresholdMs
    }

    /// The go-trial RT (ms) that enters the RT statistics, or nil.
    public static func usableRT(_ t: SARTTrial) -> Double? {
        guard !t.isNoGo, t.timingValid, let rt = t.rtMs, rt >= anticipationThresholdMs else { return nil }
        return rt
    }

    public static func compute(_ input: [SARTTrial]) -> SARTMetrics {
        let trials = input.filter(\.isComplete).sorted { $0.index < $1.index }
        let go = trials.filter { !$0.isNoGo }
        let noGo = trials.filter(\.isNoGo)
        let commissions = noGo.filter(\.tapped).count
        let omissions = go.filter { !$0.tapped }.count
        let rts = go.compactMap(usableRT)
        let mean = CalibrationStats.mean(rts)
        let sd = CalibrationStats.standardDeviation(rts)
        let cv = CalibrationStats.coefficientOfVariation(rts)

        // Post-error slowing.
        var postError: [Double] = []
        var postCorrect: [Double] = []
        for i in trials.indices.dropFirst() {
            let prev = trials[i - 1], cur = trials[i]
            guard let rt = usableRT(cur) else { continue }
            if prev.commission { postError.append(rt) }
            else if !prev.isNoGo, prev.tapped, !isAnticipation(prev) { postCorrect.append(rt) }
        }
        let pes: Double?
        if let e = CalibrationStats.mean(postError), let c = CalibrationStats.mean(postCorrect) { pes = e - c } else { pes = nil }

        return SARTMetrics(trialCount: trials.count, goCount: go.count, noGoCount: noGo.count,
                           commissionCount: commissions, omissionCount: omissions,
                           commissionRate: noGo.isEmpty ? nil : Double(commissions) / Double(noGo.count),
                           omissionRate: go.isEmpty ? nil : Double(omissions) / Double(go.count),
                           anticipationCount: go.filter(isAnticipation).count, rtCount: rts.count,
                           meanGoRTms: mean, rtSDms: sd, rtCV: cv, postErrorSlowingMs: pes)
    }
}

/// Where a SART run came from. The taps are real in every case except `.sample`.
public enum SARTRunOrigin: String, Codable, Sendable, CaseIterable {
    /// A phone with the face-tracking camera available (features were recorded alongside the taps).
    case device
    /// The iOS Simulator (taps are real, there is no camera).
    case simulator
    /// DEBUG builds only: generated SAMPLE runs for screenshots. Never real, never in reports by default.
    case sample

    public var label: String {
        switch self {
        case .device: "Phone"
        case .simulator: "Simulator"
        case .sample: "SAMPLE"
        }
    }

    /// Real taps by a person (device or Simulator).
    public var isRealTaps: Bool { self != .sample }

    /// The origin of a run done in this build/environment right now.
    public static var current: SARTRunOrigin {
        #if targetEnvironment(simulator)
        return .simulator
        #else
        return .device
        #endif
    }
}

/// One completed SART run, persisted for the sustained-attention profile. Numbers only.
public struct SARTRunRecord: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var date: Date
    public var origin: SARTRunOrigin
    /// True when the camera recorded features during the run (labelled windows were made from it).
    public var sensorsRecorded: Bool
    public var config: SARTConfig
    public var trials: [SARTTrial]
    public var metrics: SARTMetrics

    public init(id: UUID = UUID(), date: Date = Date(), origin: SARTRunOrigin = .current, sensorsRecorded: Bool = false,
                config: SARTConfig, trials: [SARTTrial]) {
        self.id = id; self.date = date; self.origin = origin; self.sensorsRecorded = sensorsRecorded
        self.config = config; self.trials = trials
        self.metrics = SARTMetrics.compute(trials)
    }

    public init(result: SARTResult, date: Date = Date(), origin: SARTRunOrigin = .current, sensorsRecorded: Bool = false) {
        self.init(date: date, origin: origin, sensorsRecorded: sensorsRecorded, config: result.config, trials: result.trials)
    }

    /// True when the task used Robertson et al.'s (1997) timing: 250 ms digit, 900 ms mask, no-go digit 3.
    public var usesStandardTiming: Bool {
        abs(config.stimulusDuration - 0.25) < 1e-6 && abs(config.maskDuration - 0.9) < 1e-6 && config.noGoDigit == 3
    }
}

/// Trend across runs (oldest first) for one metric: the change from the first to the latest run.
public enum SARTTrend {
    /// Runs sorted oldest first, real-tap runs only unless `includeSamples`.
    public static func ordered(_ runs: [SARTRunRecord], includeSamples: Bool = true) -> [SARTRunRecord] {
        runs.filter { includeSamples || $0.origin.isRealTaps }.sorted { $0.date < $1.date }
    }

    /// Latest minus first value of `metric` over runs where it exists (nil with fewer than 2 values).
    public static func change(_ runs: [SARTRunRecord], _ metric: (SARTMetrics) -> Double?) -> Double? {
        let v = ordered(runs).compactMap { metric($0.metrics) }
        guard v.count >= 2, let first = v.first, let last = v.last else { return nil }
        return last - first
    }
}

/// Plain-language copy for the sustained-attention screens and the clinician report (kept here so tests can
/// scan every sentence for wording the app must not use).
public enum SARTCopy {
    public static let title = "Sustained attention"
    public static let intro = "A three-minute tap task (the SART) that shows how steady your attention stays when a task is repetitive. Digits flash one at a time; you tap for every digit except 3."

    public static let explainers: [(title: String, text: String)] = [
        ("Taps on 3", "How often you tapped when you meant to hold back. Lower means you caught yourself more often, a sign of steadier attention on a repetitive task."),
        ("Missed digits", "How often you didn't tap for a digit at all. Lower is steadier."),
        ("Response time", "Your average time to tap, in milliseconds. Faster is not always better: rushing tends to go with more taps on 3."),
        ("Variability", "How much your response time wobbles from tap to tap (coefficient of variation = spread divided by average). Lower variability means steadier attention."),
        ("Too-fast taps", "Taps under 100 ms, too fast to be a response to the digit. They are left out of the response-time numbers."),
    ]

    public static let personalNote = "Your numbers are most useful compared with your own earlier runs. Sleep, time of day, practice and how you hold the phone all change them."

    public static let emptyTitle = "No attention check yet"
    public static let emptyBody = "Run the three-minute tap task to see how steady your attention is. Each run is saved here so you can watch your own trend."
}
