import Foundation

// Sustained Attention to Response Task (Robertson, Manly, Andrade, Baddeley & Yiend, 1997).
// Digits 1-9 shown for 250 ms, each followed by a 900 ms mask (ring with a diagonal cross);
// respond to every digit except the no-go digit 3. Digit font size is randomised across 5 sizes.
//
// Default length: 160 trials x 1.15 s = 184 s (about 3.07 min), to match the spec's "about 3 minutes".
// (Robertson's original block is 225 trials = 4.3 min; set `trialCount = 225` to reproduce it.)
//
// The engine is pure: it is driven by `tick(frameTime:)` once per display frame and `tap(at:)` per touch,
// both on the CACurrentMediaTime clock (CADisplayLink.targetTimestamp / UITouch.timestamp). Tests inject
// these times directly. All recorded times are on the *session clock*: `mediaTime - clockOrigin`.

public struct SARTConfig: Codable, Equatable, Sendable {
    public var trialCount = 160
    public var stimulusDuration: Double = 0.25
    public var maskDuration: Double = 0.9
    public var noGoDigit = 3
    /// Five digit sizes in points (Robertson et al. used 48, 72, 94, 100 and 120 pt).
    public var fontSizes: [Double] = [48, 72, 94, 100, 120]
    /// Practice blocks are scored for feedback but never used for labels.
    public var isPractice = false
    /// Fixed seed for a reproducible sequence; nil = random.
    public var seed: UInt64?
    /// A phase switch fires when it is due within this tolerance (absorbs display-timestamp jitter
    /// so we don't overshoot by a whole frame). Must stay below half a frame at 120 Hz.
    public var frameTolerance: Double = 0.004
    /// A display gap longer than this during a trial (app backgrounded, hitch) marks the trial's timing invalid.
    public var maxFrameGap: Double = 0.1

    public init() {}

    public static var practice: SARTConfig {
        var c = SARTConfig()
        c.trialCount = 18
        c.isPractice = true
        return c
    }

    /// Stimulus-onset asynchrony (stimulus + mask) in seconds.
    public var trialDuration: Double { stimulusDuration + maskDuration }
    public var estimatedDuration: Double { Double(trialCount) * trialDuration }
}

public struct SARTTrial: Codable, Equatable, Sendable, Identifiable {
    public var index: Int
    public var digit: Int
    public var fontSize: Double
    public var isNoGo: Bool
    /// Session time of the display frame on which the digit appeared.
    public var onset: Double
    /// Session time of the display frame on which the mask replaced the digit.
    public var maskOnset: Double?
    /// Session time of the first touch in this trial's response window (onset ..< next onset).
    public var tapTime: Double?
    /// Session time this trial's response window closed (next onset or end of task).
    public var end: Double?
    /// False if a display gap (hitch / app backgrounded) occurred during the trial.
    public var timingValid = true

    public var id: Int { index }
    public var isComplete: Bool { end != nil }
    public var tapped: Bool { tapTime != nil }
    /// Reaction time in seconds.
    public var rt: Double? { tapTime.map { $0 - onset } }
    public var rtMs: Double? { rt.map { $0 * 1000 } }
    public var stimulusDuration: Double? { maskOnset.map { $0 - onset } }
    /// Tap on the no-go digit.
    public var commission: Bool { isNoGo && tapped }
    /// No tap on a go digit (only meaningful once the trial is complete).
    public var omission: Bool { !isNoGo && !tapped && isComplete }
    public var isError: Bool { commission || omission }

    public init(index: Int, digit: Int, fontSize: Double, isNoGo: Bool, onset: Double, maskOnset: Double? = nil,
                tapTime: Double? = nil, end: Double? = nil, timingValid: Bool = true) {
        self.index = index; self.digit = digit; self.fontSize = fontSize; self.isNoGo = isNoGo; self.onset = onset
        self.maskOnset = maskOnset; self.tapTime = tapTime; self.end = end; self.timingValid = timingValid
    }
}

public enum SARTDisplay: Equatable, Sendable {
    case blank
    case digit(Int, fontSize: Double)
    case mask
    case finished
}

public struct SARTResult: Codable, Equatable, Sendable {
    public var config: SARTConfig
    public var trials: [SARTTrial]

    public init(config: SARTConfig, trials: [SARTTrial]) { self.config = config; self.trials = trials }

    public var goCount: Int { trials.filter { !$0.isNoGo }.count }
    public var noGoCount: Int { trials.filter(\.isNoGo).count }
    public var commissionCount: Int { trials.filter(\.commission).count }
    public var omissionCount: Int { trials.filter(\.omission).count }
    /// Go-trial RTs in seconds.
    public var goRTs: [Double] { trials.filter { !$0.isNoGo }.compactMap(\.rt) }
    public var meanRT: Double? { CalibrationStats.mean(goRTs) }
    public var rtCV: Double? { CalibrationStats.coefficientOfVariation(goRTs) }
    public var start: Double? { trials.first?.onset }
    public var end: Double? { trials.last?.end }
}

public final class SARTEngine {
    public let config: SARTConfig
    /// Media time (CACurrentMediaTime clock) that corresponds to session time 0.
    public let clockOrigin: Double
    /// Pre-generated (digit, fontSize) sequence.
    public let sequence: [(digit: Int, fontSize: Double)]

    public private(set) var trials: [SARTTrial] = []
    public private(set) var display: SARTDisplay = .blank
    public private(set) var isFinished = false
    /// Called once, on the tick that ends the last trial.
    public var onFinish: ((SARTResult) -> Void)?

    private enum Phase { case notStarted, stimulus(Int), mask(Int), finished }
    private var phase: Phase = .notStarted
    private var lastFrame: Double?

    public init(config: SARTConfig = SARTConfig(), clockOrigin: Double = 0) {
        self.config = config
        self.clockOrigin = clockOrigin
        var rng = config.seed.map { CalibrationSplitMix64(seed: $0) } ?? CalibrationSplitMix64()
        sequence = Self.makeSequence(count: config.trialCount, noGoDigit: config.noGoDigit,
                                     fontSizes: config.fontSizes, using: &rng)
    }

    /// Digits are drawn in shuffled blocks of 1-9 (each digit equally often, as in the standard SART),
    /// truncated to `count`, so the no-go ratio is ~1/9. Font sizes are drawn uniformly from `fontSizes`.
    public static func makeSequence<G: RandomNumberGenerator>(count: Int, noGoDigit: Int = 3,
                                                              fontSizes: [Double] = SARTConfig().fontSizes,
                                                              using rng: inout G) -> [(digit: Int, fontSize: Double)] {
        var out: [(digit: Int, fontSize: Double)] = []
        out.reserveCapacity(count)
        var previous: Int?
        while out.count < count {
            var block = Array(1...9).shuffled(using: &rng)
            // Avoid an immediate repeat across the block boundary.
            if block.first == previous, block.count > 1 { block.swapAt(0, Int.random(in: 1..<block.count, using: &rng)) }
            for d in block where out.count < count {
                let size = fontSizes.isEmpty ? 96 : fontSizes[Int.random(in: 0..<fontSizes.count, using: &rng)]
                out.append((d, size))
            }
            previous = block.last
        }
        return out
    }

    public var result: SARTResult { SARTResult(config: config, trials: trials.filter(\.isComplete)) }
    public var currentTrialIndex: Int? {
        switch phase { case let .stimulus(i), let .mask(i): i; default: nil }
    }

    /// Advance the state machine for a display frame that will be shown at `frameTime` (media clock;
    /// pass `CADisplayLink.targetTimestamp`). Returns what that frame must show.
    @discardableResult
    public func tick(frameTime: Double) -> SARTDisplay {
        let t = frameTime - clockOrigin
        if let last = lastFrame, t - last > config.maxFrameGap, let i = currentTrialIndex {
            trials[i].timingValid = false
        }
        lastFrame = t
        let tol = config.frameTolerance
        switch phase {
        case .notStarted:
            if config.trialCount > 0 { startTrial(0, at: t) } else { finish(at: t) }
        case let .stimulus(i):
            if t - trials[i].onset >= config.stimulusDuration - tol {
                trials[i].maskOnset = t
                phase = .mask(i)
                display = .mask
            }
        case let .mask(i):
            if t - trials[i].onset >= config.trialDuration - tol {
                trials[i].end = t
                if i + 1 < sequence.count { startTrial(i + 1, at: t) } else { finish(at: t) }
            }
        case .finished:
            break
        }
        return display
    }

    /// Register a touch at `mediaTime` (pass `UITouch.timestamp`). The tap belongs to the latest trial whose
    /// onset is at or before it, within that trial's response window. Only the first tap per trial counts.
    public func tap(at mediaTime: Double) {
        let t = mediaTime - clockOrigin
        guard let i = trials.lastIndex(where: { $0.onset <= t }) else { return }
        if let end = trials[i].end, t >= end { return }
        if trials[i].tapTime == nil { trials[i].tapTime = t }
    }

    private func startTrial(_ i: Int, at t: Double) {
        let s = sequence[i]
        trials.append(SARTTrial(index: i, digit: s.digit, fontSize: s.fontSize, isNoGo: s.digit == config.noGoDigit,
                                onset: t))
        phase = .stimulus(i)
        display = .digit(s.digit, fontSize: s.fontSize)
    }

    private func finish(at t: Double) {
        phase = .finished
        display = .finished
        isFinished = true
        onFinish?(result)
    }
}

// MARK: - CSV for the forecaster

public enum SARTTrialCSV {
    public static let header = "trial,onset_s,digit,is_nogo,tapped,rt_ms,commission,omission"

    /// `trial` is 1-based; `onset_s` is session time; `rt_ms` empty when there was no tap.
    public static func make(_ trials: [SARTTrial], isSynthetic: Bool) -> String {
        var s = isSynthetic ? "# FocusLoop SART trials v1 (SYNTHETIC)\n" : "# FocusLoop SART trials v1 (REAL device data)\n"
        s += header + "\n"
        for tr in trials {
            let rt = tr.rtMs.map { String(format: "%.1f", $0) } ?? ""
            s += "\(tr.index + 1),\(String(format: "%.4f", tr.onset)),\(tr.digit),\(tr.isNoGo ? 1 : 0),"
            s += "\(tr.tapped ? 1 : 0),\(rt),\(tr.commission ? 1 : 0),\(tr.omission ? 1 : 0)\n"
        }
        return s
    }

    /// Writes `FocusLoop_sart_trials.csv` (prefix lets deleteAll sweep it) to the temporary directory (or `directory`) and returns its URL.
    public static func write(_ trials: [SARTTrial], isSynthetic: Bool, directory: URL? = nil) throws -> URL {
        let url = (directory ?? FileManager.default.temporaryDirectory).appendingPathComponent("FocusLoop_sart_trials.csv")
        try make(trials, isSynthetic: isSynthetic).write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}
