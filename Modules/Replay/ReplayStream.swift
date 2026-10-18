import Foundation

/// A recorded (or SYNTHETIC) 10 Hz feature stream loaded from `<name>.replay.csv`.
/// CSV columns: t, label, lapse, then `FeatureFrame.featureNames` in order. Lines starting with `#` are comments.
public struct ReplayStream: Sendable {
    public let frames: [FeatureFrame]
    /// Ground-truth labels when present (synthetic generator / exported sessions).
    public let labels: [AttentionState?]
    public let lapse: [Bool]
    /// True if the file header says SYNTHETIC.
    public let isSynthetic: Bool

    public enum LoadError: Error { case missingResource, badHeader(String) }

    public static func load(_ name: String = "synthetic_session", bundle: Bundle = .main) throws -> ReplayStream {
        guard let url = bundle.url(forResource: name, withExtension: "replay.csv") else { throw LoadError.missingResource }
        return try parse(String(contentsOf: url, encoding: .utf8))
    }

    public static func parse(_ text: String) throws -> ReplayStream {
        var synthetic = false
        var header: [String]?
        var frames: [FeatureFrame] = [], labels: [AttentionState?] = [], lapse: [Bool] = []
        for line in text.split(whereSeparator: \.isNewline) {
            if line.hasPrefix("#") { if line.contains("SYNTHETIC") { synthetic = true }; continue }
            let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            guard let h = header else {
                let n = FeatureFrame.featureNames.count
                // Extra trailing columns (e.g. session_id,source,window_id from app exports) are ignored.
                guard cols.count >= 3 + n, cols[0] == "t", Array(cols[3..<(3 + n)]) == FeatureFrame.featureNames else {
                    throw LoadError.badHeader(String(line.prefix(120)))
                }
                header = cols; continue
            }
            guard cols.count == h.count else { continue }
            let v = cols[3..<(3 + FeatureFrame.featureNames.count)].map { Float($0) ?? 0 }
            frames.append(FeatureFrame(t: Double(cols[0]) ?? 0, vector: Array(v)))
            labels.append(AttentionState(rawValue: cols[1]))
            lapse.append(cols[2] == "1")
        }
        return ReplayStream(frames: frames, labels: labels, lapse: lapse, isSynthetic: synthetic)
    }

    public var duration: Double { (frames.last?.t ?? 0) + 1 / FeatureFrame.rateHz }
}

/// Plays a `ReplayStream` in real time (x `speed`), looping, as a `FeatureFrameSource`.
/// Synthesises a display pulse waveform from the replayed heart rate (SYNTHETIC).
public final class ReplayFeatureSource: FeatureFrameSource {
    public let stream: ReplayStream
    public var speed: Double
    public var loop: Bool
    public var isSynthetic: Bool { stream.isSynthetic }

    public let frames: AsyncStream<FeatureFrame>
    public let heartRate: AsyncStream<HeartRateSample>
    private let frameCont: AsyncStream<FeatureFrame>.Continuation
    private let hrCont: AsyncStream<HeartRateSample>.Continuation
    private var task: Task<Void, Never>?
    private let pauseState = ReplayPauseState()

    public init(stream: ReplayStream, speed: Double = 1, loop: Bool = true) {
        self.stream = stream; self.speed = speed; self.loop = loop
        (frames, frameCont) = AsyncStream.makeStream(of: FeatureFrame.self, bufferingPolicy: .bufferingNewest(64))
        (heartRate, hrCont) = AsyncStream.makeStream(of: HeartRateSample.self, bufferingPolicy: .bufferingNewest(8))
    }

    public func start() {
        guard task == nil, !stream.frames.isEmpty else { return }
        let stream = self.stream, speed = self.speed, loop = self.loop
        let fc = frameCont, hc = hrCont, pauseState = self.pauseState
        task = Task.detached(priority: .userInitiated) {
            let dt = 1 / FeatureFrame.rateHz
            let clock = ContinuousClock()
            var next = clock.now
            var i = 0, lap = 0.0
            while !Task.isCancelled {
                if pauseState.isPaused {
                    // Suspended (app in the background): hold the stream, then continue without a burst.
                    try? await Task.sleep(for: .milliseconds(50))
                    next = clock.now
                    continue
                }
                if i >= stream.frames.count {
                    guard loop else { break }
                    i = 0; lap += stream.duration
                }
                var f = stream.frames[i]
                f.t += lap
                fc.yield(f)
                if i % Int(FeatureFrame.rateHz) == 0, f.heartRate > 0 {
                    hc.yield(Self.syntheticHR(for: f))
                }
                i += 1
                next = next.advanced(by: .milliseconds(Int(dt * 1000 / speed)))
                try? await clock.sleep(until: next)
            }
            fc.finish(); hc.finish()
        }
    }

    public func stop() { task?.cancel(); task = nil }

    /// Holds the stream (e.g. while the app is in the background) until `resume()`.
    public func pause() { pauseState.isPaused = true }
    public func resume() { pauseState.isPaused = false }

    /// SYNTHETIC display waveform matching the replayed BPM (3 s at 30 Hz).
    static func syntheticHR(for f: FeatureFrame) -> HeartRateSample {
        let hz: Float = 30, beat = f.heartRate / 60
        let wave = (0..<90).map { k -> Float in
            let ph = 2 * Float.pi * beat * (Float(k) / hz + Float(f.t))
            return 0.8 * sin(ph) + 0.2 * sin(2 * ph + 0.6)
        }
        return HeartRateSample(t: f.t, bpm: f.heartRate, quality: f.heartRateQuality, snrDB: 10 * f.heartRateQuality,
                               motionFlagged: f.headMotion > 0.6, baselineBPM: f.heartRate - f.heartRateDelta,
                               waveform: wave)
    }
}

/// Thread-safe pause flag shared with the replay task.
final class ReplayPauseState: @unchecked Sendable {
    private let lock = NSLock()
    private var paused = false
    var isPaused: Bool {
        get { lock.withLock { paused } }
        set { lock.withLock { paused = newValue } }
    }
}

/// SYNTHETIC. Plays a `ReplayStream` locked to an external clock (the Study Mode lecture playhead) instead
/// of wall time, for the one-tap scripted demo: frame `i` is emitted when the lecture reaches `i / 10` s, so
/// the replay timeline always matches the lecture time. When the video pauses (a recall card, the user,
/// the app going to the background) the stream pauses with it, and nothing is ever emitted ahead of the
/// lecture. Frame times are a monotonic session clock (10 Hz), so a seek never makes time run backwards:
/// seeking back replays from the new position; seeking forward skips ahead (at most `maxCatchUpFrames`
/// are emitted in one poll). Polling and emission happen on the main actor, where the playhead lives.
public final class LectureClockReplaySource: FeatureFrameSource, @unchecked Sendable {
    public let stream: ReplayStream
    public var isSynthetic: Bool { stream.isSynthetic }
    public let frames: AsyncStream<FeatureFrame>
    public let heartRate: AsyncStream<HeartRateSample>
    /// Lecture-frame index of the next frame to emit (exposed for tests).
    public private(set) var nextIndex = 0
    public static let maxCatchUpFrames = 20

    private let clock: @MainActor () -> Double
    private let pollInterval: Duration
    private let frameCont: AsyncStream<FeatureFrame>.Continuation
    private let hrCont: AsyncStream<HeartRateSample>.Continuation
    private var task: Task<Void, Never>?
    private var emitted = 0

    /// - Parameters:
    ///   - clock: the lecture playhead in seconds (read on the main actor every `pollInterval`).
    ///   - pollInterval: seconds between polls (0.05 s in the app; tests use less to run fast).
    public init(stream: ReplayStream, pollInterval: Double = 0.05, clock: @escaping @MainActor () -> Double) {
        self.stream = stream
        self.clock = clock
        self.pollInterval = .milliseconds(max(1, Int(pollInterval * 1000)))
        (frames, frameCont) = AsyncStream.makeStream(of: FeatureFrame.self, bufferingPolicy: .bufferingNewest(64))
        (heartRate, hrCont) = AsyncStream.makeStream(of: HeartRateSample.self, bufferingPolicy: .bufferingNewest(8))
    }

    public func start() {
        guard task == nil, !stream.frames.isEmpty else { return }
        task = Task { @MainActor [weak self] in
            while !Task.isCancelled, let self {
                self.poll()
                let interval = self.pollInterval
                try? await Task.sleep(for: interval)
            }
        }
    }

    public func stop() {
        task?.cancel(); task = nil
        frameCont.finish(); hrCont.finish()
    }

    /// Emits every frame the lecture has reached since the last poll.
    @MainActor func poll() {
        let hz = FeatureFrame.rateHz
        let target = min(stream.frames.count - 1, Int((clock() * hz + 1e-6).rounded(.down)))
        guard target >= 0 else { return }
        if target < nextIndex - 1 { nextIndex = target }                                 // seeked back
        if target - nextIndex >= Self.maxCatchUpFrames { nextIndex = target - Self.maxCatchUpFrames + 1 }  // seeked forward
        while nextIndex <= target {
            var f = stream.frames[nextIndex]
            f.t = Double(emitted) / hz
            frameCont.yield(f)
            if emitted % Int(hz) == 0, f.heartRate > 0 { hrCont.yield(ReplayFeatureSource.syntheticHR(for: f)) }
            emitted += 1
            nextIndex += 1
        }
    }
}
