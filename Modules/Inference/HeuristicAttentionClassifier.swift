import Foundation

/// Rule-based 4-state classifier with the same interface as the Core ML model (spec §1.2).
///
/// It turns a 10 s window into three evidence scores (0 = none, 0.5 = decision point, 1 = strong)
/// and a soft probability vector (softmax of `gain * evidence - offset` against a constant
/// `focused` logit), plus a fixed 16-dim engineered embedding (`embeddingNames`) so the personal
/// head can learn even without the model.
///
/// * distracted: gaze outside the work zone (fraction of the last 2 s, dwell time), phone pickup.
/// * drowsy: heavy eyelids (median eye closure over the last 3 s), long blinks (closure > 0.8 for
///   >= 0.6 s), yawns (jawOpen > 0.6), head nods (pitch drop then recovery).
/// * stressed: heart rate above the personal baseline (only frames with good rPPG quality) plus
///   brow tension / squint.
/// * focused: none of the above.
///
/// Window-local signals are preferred over the trailing-60 s counters in `FeatureFrame`
/// (`perclos`, `yawnCount`...), because those counters stay high for a minute after the event.
///
/// Missing face: when the face is tracked in fewer than `minFaceFraction` of the last 3 s, the
/// face-based evidence is unavailable, so the window is reported as distracted with moderate
/// evidence (`missingFaceDistractedEvidence`, ~0.68 probability): the user is not looking at the
/// work. A phone pickup on top of that raises it to full evidence (~0.86). Drowsiness cannot be
/// seen without a face, so it is never inferred then.
public final class HeuristicAttentionClassifier: AttentionClassifier {

    public struct Config: Codable, Equatable, Sendable {
        /// "Recent" span for closure / yawn / nod / face presence (frames; 3 s at 10 Hz).
        public var recentFrames = 30
        /// Span for gaze, pickup and heart rate (2 s).
        public var gazeFrames = 20
        /// Span for brow tension / squint (1 s; the signal is already smoothed).
        public var browFrames = 10
        /// Look-back for long-blink counting (5 s).
        public var longBlinkLookback = 50

        public var minFaceFraction: Float = 0.5
        public var missingFaceDistractedEvidence: Float = 0.75

        /// Off-zone dwell giving full distracted evidence, seconds.
        public var dwellFullSeconds: Float = 3

        public var closureOnset: Float = 0.1
        public var closureSpan: Float = 0.25
        public var longBlinkClosure: Float = 0.8
        public var longBlinkFrames = 6
        public var longBlinkWeight: Float = 0.25
        public var yawnJawOpen: Float = 0.6
        public var yawnWeight: Float = 0.5
        public var nodDrop: Float = 0.2
        public var nodRecovery: Float = 0.15
        public var nodWeight: Float = 0.5

        public var browOnset: Float = 0.15
        public var browSpan: Float = 0.3
        public var squintOnset: Float = 0.15
        public var squintSpan: Float = 0.2
        public var squintWeight: Float = 0.7
        /// Minimum rPPG quality for a heart-rate frame to count.
        public var heartRateQualityMin: Float = 0.4
        public var minHeartRateFrames = 5
        public var heartRateDeltaOnset: Float = 6
        public var heartRateDeltaSpan: Float = 8
        public var stressBrowWeight: Float = 0.55
        public var stressHeartRateWeight: Float = 0.55

        /// logit_k = gain * evidence_k - offset; focused logit is constant.
        public var logitGain: Float = 4
        public var logitOffset: Float = 1
        public var focusedLogit: Float = 1
        public var maxEvidence: Float = 1.5

        public init() {}
    }

    /// Evidence scores behind one estimate (0 none, 0.5 decision point, 1 strong).
    public struct Evidence: Equatable, Sendable {
        public var distracted: Float
        public var drowsy: Float
        public var stressed: Float
        /// Fraction of the recent span (3 s) with a tracked face.
        public var recentFaceFraction: Float
        /// True when the face was missing for most of the recent span.
        public var faceMissing: Bool
        /// True when at least `minHeartRateFrames` good-quality HR frames were available.
        public var heartRateUsable: Bool
    }

    /// Names of the engineered embedding dimensions (order of `StateEstimate.embedding`).
    public static let embeddingNames = [
        "faceFraction", "offZoneRecent", "offZoneWindow", "gazeDeviationMean", "offZoneDwellNorm",
        "phonePickupRecent", "closureMedianRecent", "closureMeanWindow", "localPerclos", "longBlinksNorm",
        "yawn", "nod", "heartRateDeltaNorm", "heartRateUsable", "browTensionRecent", "squintRecent",
    ]
    public static var embeddingDimension: Int { embeddingNames.count }

    public var config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    public func classify(window: [FeatureFrame]) -> StateEstimate {
        let t = window.last?.t ?? 0
        guard !window.isEmpty else {
            return StateEstimate(t: t, probabilities: [0.25, 0.25, 0.25, 0.25], source: .heuristic,
                                 embedding: Array(repeating: 0, count: Self.embeddingDimension))
        }
        let (evidence, embedding) = analyze(window: window)
        return StateEstimate(t: t, probabilities: probabilities(for: evidence), source: .heuristic, embedding: embedding)
    }

    /// Soft probabilities (`AttentionState.allCases` order) from evidence scores.
    public func probabilities(for e: Evidence) -> [Float] {
        let c = config
        func logit(_ x: Float) -> Float { c.logitGain * min(max(x, 0), c.maxEvidence) - c.logitOffset }
        return InferenceMath.softmax([c.focusedLogit, logit(e.distracted), logit(e.drowsy), logit(e.stressed)])
    }

    /// Evidence and the engineered embedding for a non-empty window (oldest first).
    public func analyze(window: [FeatureFrame]) -> (Evidence, [Float]) {
        let c = config
        let recent = window.suffix(c.recentFrames)
        let gazeSpan = window.suffix(c.gazeFrames)

        let recentFace = InferenceMath.mean(recent.map { $0.faceDetected > 0.5 ? 1 : 0 })
        let windowFace = InferenceMath.mean(window.map { $0.faceDetected > 0.5 ? 1 : 0 })
        let faceMissing = recentFace < c.minFaceFraction

        // --- distracted ---
        let gazeFaces = gazeSpan.filter { $0.faceDetected > 0.5 }
        let windowFaces = window.filter { $0.faceDetected > 0.5 }
        var offRecent: Float = 0, dwell: Float = 0
        var distracted: Float = 0
        if gazeFaces.count >= 5 {
            offRecent = 1 - InferenceMath.mean(gazeFaces.map(\.inWorkZone))
            dwell = gazeFaces.last?.offZoneDwell ?? 0
            distracted = max(offRecent, InferenceMath.clamp01(dwell / c.dwellFullSeconds))
        }
        let offWindow = windowFaces.isEmpty ? 0 : 1 - InferenceMath.mean(windowFaces.map(\.inWorkZone))
        let deviationMean = InferenceMath.mean(windowFaces.map(\.gazeDeviation))
        let pickup = InferenceMath.mean(gazeSpan.map { $0.phonePickup > 0.5 ? 1 : 0 })
        distracted = max(distracted, pickup)
        if faceMissing { distracted = max(distracted, c.missingFaceDistractedEvidence) }

        // --- drowsy / stressed (need a face) ---
        var drowsy: Float = 0, stressed: Float = 0
        var closureMedian: Float = 0, closureMean: Float = 0, localPerclos: Float = 0
        var longBlinks = 0, yawn: Float = 0, nod: Float = 0, brow: Float = 0, squint: Float = 0
        if windowFaces.count >= 10 {
            let closure = windowFaces.map { min($0.eyeBlinkLeft, $0.eyeBlinkRight) }
            closureMedian = InferenceMath.median(Array(closure.suffix(c.recentFrames)))
            closureMean = InferenceMath.mean(closure)
            localPerclos = InferenceMath.mean(closure.map { $0 > c.longBlinkClosure ? 1 : 0 })
            longBlinks = Self.runCount(closure.suffix(c.longBlinkLookback), above: c.longBlinkClosure,
                                       minLength: c.longBlinkFrames)
            let recentFaces = windowFaces.suffix(c.recentFrames)
            yawn = (recentFaces.map(\.jawOpen).max() ?? 0) > c.yawnJawOpen ? 1 : 0
            let pitch = windowFaces.map(\.headPitch)
            let recentPitch = pitch.suffix(c.recentFrames)
            let pitchMin = recentPitch.min() ?? 0
            let pitchMedian = InferenceMath.median(pitch)
            if pitchMedian - pitchMin > c.nodDrop, let lastPitch = pitch.last, lastPitch > pitchMin + c.nodRecovery {
                nod = 1
            }
            let eClosure = InferenceMath.clamp01((closureMedian - c.closureOnset) / c.closureSpan)
            drowsy = min(c.maxEvidence, eClosure + c.longBlinkWeight * Float(longBlinks)
                         + c.yawnWeight * yawn + c.nodWeight * nod)

            let browSpan = windowFaces.suffix(c.browFrames)
            brow = InferenceMath.mean(browSpan.map(\.browTension))
            squint = InferenceMath.mean(browSpan.map { ($0.eyeSquintLeft + $0.eyeSquintRight) / 2 })
            let eBrow = max(InferenceMath.clamp01((brow - c.browOnset) / c.browSpan),
                            c.squintWeight * InferenceMath.clamp01((squint - c.squintOnset) / c.squintSpan))
            stressed = c.stressBrowWeight * eBrow
        }

        // Heart rate: only good-quality rPPG frames count.
        let goodHR = gazeSpan.filter { $0.heartRate > 0 && $0.heartRateQuality >= c.heartRateQualityMin }
        let hrUsable = goodHR.count >= c.minHeartRateFrames
        var hrDelta: Float = 0
        if hrUsable {
            hrDelta = InferenceMath.mean(goodHR.map(\.heartRateDelta))
            if windowFaces.count >= 10 {
                let eHR = InferenceMath.clamp01((hrDelta - c.heartRateDeltaOnset) / c.heartRateDeltaSpan)
                stressed += c.stressHeartRateWeight * eHR
            }
        }
        stressed = min(stressed, c.maxEvidence)

        let evidence = Evidence(distracted: distracted, drowsy: drowsy, stressed: stressed,
                                recentFaceFraction: recentFace, faceMissing: faceMissing, heartRateUsable: hrUsable)
        let embedding: [Float] = [
            windowFace, offRecent, offWindow, min(deviationMean, 2),
            InferenceMath.clamp01(dwell / 10), pickup,
            closureMedian, closureMean, localPerclos, min(Float(longBlinks) / 3, 1),
            yawn, nod, max(-1, min(1, hrDelta / 20)), hrUsable ? 1 : 0, brow, squint,
        ]
        return (evidence, embedding)
    }

    /// Number of runs of consecutive values > `threshold` lasting at least `minLength` samples.
    static func runCount<C: Collection>(_ xs: C, above threshold: Float, minLength: Int) -> Int where C.Element == Float {
        var runs = 0, run = 0
        for x in xs {
            if x > threshold {
                run += 1
            } else {
                if run >= minLength { runs += 1 }
                run = 0
            }
        }
        if run >= minLength { runs += 1 }
        return runs
    }
}

/// Small numeric helpers shared by the Inference module.
enum InferenceMath {
    static func mean(_ xs: [Float]) -> Float {
        xs.isEmpty ? 0 : xs.reduce(0, +) / Float(xs.count)
    }

    static func median(_ xs: [Float]) -> Float {
        guard !xs.isEmpty else { return 0 }
        let s = xs.sorted()
        let m = s.count / 2
        return s.count % 2 == 1 ? s[m] : (s[m - 1] + s[m]) / 2
    }

    static func clamp01(_ x: Float) -> Float { min(max(x, 0), 1) }

    static func softmax(_ z: [Float]) -> [Float] {
        guard let mx = z.max(), mx.isFinite else {
            return Array(repeating: 1 / Float(max(z.count, 1)), count: z.count)
        }
        let e = z.map { exp($0 - mx) }
        let s = e.reduce(0, +)
        return e.map { $0 / s }
    }

    /// Clean a 4-vector into a probability distribution: softmax if it looks like logits
    /// (negative entries), renormalise otherwise; uniform if unusable.
    static func normalizedProbabilities(_ raw: [Float], count: Int = AttentionState.allCases.count) -> [Float] {
        let uniform = Array(repeating: 1 / Float(count), count: count)
        guard raw.count >= count else { return uniform }
        let p = Array(raw.prefix(count))
        guard p.allSatisfy(\.isFinite) else { return uniform }
        if p.contains(where: { $0 < 0 }) { return softmax(p) }
        let s = p.reduce(0, +)
        guard s > 1e-6 else { return uniform }
        return p.map { $0 / s }
    }
}
