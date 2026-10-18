import Foundation

/// Tunables for `POSHeartRateEstimator`. Defaults are the shipped values.
struct RPPGConfig: Equatable {
    /// Uniform resampling rate for the colour trace (Hz).
    var sampleRate: Double = 30
    /// POS overlap-add window length (s). Wang et al. 2017 use 1.6 s (l = 48 at 30 fps).
    var posWindow: Double = 1.6
    /// Sliding spectral analysis window (s).
    var analysisWindow: Double = 10
    /// Pass band (Hz): 0.7-3.0 Hz = 42-180 BPM.
    var bandLow: Double = 0.7
    var bandHigh: Double = 3.0
    /// Half-width (Hz) of the "signal" bins around the peak and its first harmonic (de Haan & Jeanne 2013).
    var snrHalfWidth: Double = 0.1
    /// FFT length after zero-padding (power of two).
    var fftLength: Int = 2048
    /// Seconds between estimate updates.
    var updateInterval: Double = 1
    /// Samples further apart than this restart the signal buffer (face lost / app paused).
    var maxGap: Double = 0.5
    /// Head angular speed (rad/s) above which readings are flagged and the last good value held.
    var motionThreshold: Float = 0.6
    /// Readings stay flagged for this long after the last over-threshold motion sample (s).
    var motionHold: Double = 2
    /// Head speed (rad/s) below which the user counts as calm for the resting baseline.
    var calmThreshold: Float = 0.2
    /// SNR (dB) mapped to quality 0 and 1 (linear in between).
    var snrForZeroQuality: Float = -2
    var snrForFullQuality: Float = 4
    /// Minimum quality for a reading to be "good" (held during motion, used for the baseline).
    var goodQuality: Float = 0.5
    /// Good calm readings needed before a first baseline is established.
    var baselineWarmup: Int = 15
    /// EMA weight per good calm reading (~1 Hz, so a time constant of ~1/alpha seconds).
    var baselineAlpha: Float = 0.02
    /// Length of the display waveform (s).
    var waveformSeconds: Double = 3
}

/// Camera rPPG heart-rate estimator: POS (Wang, den Brinker, Stuijk & de Haan 2017, "Algorithmic
/// Principles of Remote PPG", IEEE TBME 64(7)) reimplemented from the paper, followed by linear
/// detrend, zero-phase Butterworth band-pass (0.7-3.0 Hz), Hann-windowed zero-padded FFT over a
/// sliding 10 s window with parabolic peak interpolation, and an SNR quality index after
/// de Haan & Jeanne 2013 (IEEE TBME 60(10)).
///
/// Wellness signal only (not a medical device). Not thread-safe: call from one actor/queue.
final class POSHeartRateEstimator: HeartRateEstimator {
    let config: RPPGConfig

    private(set) var latest: HeartRateSample?
    /// Personal resting baseline in BPM. Set it from persisted storage before a session and read
    /// it back afterwards; `reset()` does not clear it.
    var restingBaseline: Float?

    // Uniformly resampled RGB trace and the POS overlap-added pulse signal H (same indexing).
    private var rgb: [SIMD3<Double>] = []
    private var pulse: [Double] = []
    private var nextGridT: Double?
    private var lastRaw: (t: Double, rgb: SIMD3<Double>)?
    private var lastUpdateT: Double?

    // Motion and baseline state.
    private var lastMotionT = -Double.infinity
    private var lastUncalmT = -Double.infinity
    private var lastGood: HeartRateSample?
    private var warmupSum: Float = 0
    private var warmupCount = 0

    private let bandSections: [RPPGBiquad]
    private let spectrum: RPPGSpectrum
    private let posLength: Int
    private let analysisLength: Int
    private let maxStored: Int

    init(config: RPPGConfig = RPPGConfig(), restingBaseline: Float? = nil) {
        self.config = config
        self.restingBaseline = restingBaseline
        bandSections = [RPPGBiquad.butterworthHighPass(cutoff: config.bandLow, fs: config.sampleRate),
                        RPPGBiquad.butterworthLowPass(cutoff: config.bandHigh, fs: config.sampleRate)]
        spectrum = RPPGSpectrum(fftLength: config.fftLength)
        posLength = Int((config.posWindow * config.sampleRate).rounded(.up))
        analysisLength = Int((config.analysisWindow * config.sampleRate).rounded())
        maxStored = analysisLength + Int(config.sampleRate * 4)
        rgb.reserveCapacity(maxStored + 64)
        pulse.reserveCapacity(maxStored + 64)
    }

    // MARK: HeartRateEstimator

    func ingest(rgb sample: SIMD3<Float>, t: Double, headMotion: Float) {
        if headMotion.isFinite {
            if headMotion > config.motionThreshold { lastMotionT = t }
            if headMotion > config.calmThreshold { lastUncalmT = t }
        }
        guard t.isFinite, sample.x.isFinite, sample.y.isFinite, sample.z.isFinite,
              sample.x > 0, sample.y > 0, sample.z > 0 else { return }
        let x = SIMD3<Double>(Double(sample.x), Double(sample.y), Double(sample.z))

        if let prev = lastRaw {
            guard t > prev.t else { return }               // drop out-of-order duplicates
            if t - prev.t > config.maxGap {                // gap: restart the trace
                clearSignal()
            }
        }
        guard let prev = lastRaw else {
            lastRaw = (t, x)
            appendUniform(t: t, x)
            nextGridT = t + 1 / config.sampleRate
            return
        }
        // Linear interpolation onto the uniform grid.
        var g = nextGridT ?? t
        let dt = t - prev.t
        while g <= t {
            let f = (g - prev.t) / dt
            appendUniform(t: g, prev.rgb + (x - prev.rgb) * f)
            g += 1 / config.sampleRate
        }
        nextGridT = g
        lastRaw = (t, x)
    }

    func reset() {
        clearSignal()
        latest = nil
        lastGood = nil
        lastMotionT = -.infinity
        lastUncalmT = -.infinity
        warmupSum = 0
        warmupCount = 0
    }

    // MARK: Internals

    private func clearSignal() {
        rgb.removeAll(keepingCapacity: true)
        pulse.removeAll(keepingCapacity: true)
        nextGridT = nil
        lastRaw = nil
        lastUpdateT = nil
    }

    /// Appends one uniform sample and runs one POS overlap-add step.
    private func appendUniform(t: Double, _ x: SIMD3<Double>) {
        rgb.append(x)
        pulse.append(0)
        let n = rgb.count
        if n >= posLength { posStep(end: n) }
        if n > maxStored + 60 {                       // amortised trim
            let drop = n - maxStored
            rgb.removeFirst(drop); pulse.removeFirst(drop)
        }
        if rgb.count >= analysisLength, t - (lastUpdateT ?? -.infinity) >= config.updateInterval - 1e-9 {
            lastUpdateT = t
            update(t: t)
        }
    }

    /// POS (Wang et al. 2017, Algorithm 1) on the window [end - l, end):
    /// temporal normalisation Cn = C / mean(C); projection S = P * Cn with P = [[0, 1, -1], [-2, 1, 1]];
    /// alpha tuning h = S1 + (std S1 / std S2) * S2; overlap-add of the zero-mean h into H.
    private func posStep(end: Int) {
        let start = end - posLength
        var mean = SIMD3<Double>(repeating: 0)
        for i in start..<end { mean += rgb[i] }
        mean /= Double(posLength)
        guard mean.x > 0, mean.y > 0, mean.z > 0 else { return }
        var s1 = [Double](repeating: 0, count: posLength)
        var s2 = [Double](repeating: 0, count: posLength)
        var m1 = 0.0, m2 = 0.0
        for k in 0..<posLength {
            let c = rgb[start + k] / mean
            s1[k] = c.y - c.z
            s2[k] = -2 * c.x + c.y + c.z
            m1 += s1[k]; m2 += s2[k]
        }
        m1 /= Double(posLength); m2 /= Double(posLength)
        var v1 = 0.0, v2 = 0.0
        for k in 0..<posLength {
            v1 += (s1[k] - m1) * (s1[k] - m1); v2 += (s2[k] - m2) * (s2[k] - m2)
        }
        let alpha = v2 > 1e-24 ? (v1 / v2).squareRoot() : 0
        var hMean = 0.0
        for k in 0..<posLength {
            s1[k] += alpha * s2[k]
            hMean += s1[k]
        }
        hMean /= Double(posLength)
        for k in 0..<posLength { pulse[start + k] += s1[k] - hMean }
    }

    /// Spectral estimate over the last `analysisWindow` seconds of H.
    private func update(t: Double) {
        let fs = config.sampleRate
        var h = Array(pulse.suffix(analysisLength))
        RPPGDSP.detrendLinear(&h)
        let filtered = RPPGDSP.filtfilt(h, sections: bandSections, padLength: Int(fs * 1.5))
        let p = spectrum.power(of: filtered)
        let df = fs / Double(spectrum.fftLength)
        let lo = max(1, Int((config.bandLow / df).rounded(.up)))
        let hi = min(p.count - 2, Int((config.bandHigh / df).rounded(.down)))
        guard hi > lo + 2 else { return }

        var k0 = lo
        for k in lo...hi where p[k] > p[k0] { k0 = k }
        var peakBin = Double(k0)
        if k0 > lo, k0 < hi, p[k0 - 1] > 0, p[k0] > 0, p[k0 + 1] > 0 {
            peakBin += RPPGDSP.parabolicOffset(log(p[k0 - 1]), log(p[k0]), log(p[k0 + 1]))
        }
        let f0 = peakBin * df
        let bpm = Float(f0 * 60)

        // SNR: power within +-0.1 Hz of f0 and 2*f0 vs. the rest of the pass band.
        let w = config.snrHalfWidth
        var sig = 0.0, noise = 0.0
        for k in lo...hi {
            let f = Double(k) * df
            if abs(f - f0) <= w || abs(f - 2 * f0) <= w { sig += p[k] } else { noise += p[k] }
        }
        let snrDB: Float = (sig > 0 && noise > 0) ? Float(10 * log10(sig / noise)) : (sig > 0 ? 30 : -30)
        let quality = max(0, min(1, (snrDB - config.snrForZeroQuality)
                                    / (config.snrForFullQuality - config.snrForZeroQuality)))

        // Waveform: last 3 s of the band-passed pulse, normalised to -1...1.
        let wfCount = min(filtered.count, Int(config.waveformSeconds * fs))
        let tail = filtered.suffix(wfCount)
        let peakAbs = tail.reduce(0) { max($0, abs($1)) }
        let waveform = tail.map { peakAbs > 0 ? Float($0 / peakAbs) : 0 }

        let flagged = t - lastMotionT < config.motionHold
        if flagged {
            // Hold the last good value; quality halves to tell consumers it is stale.
            let held = lastGood
            latest = HeartRateSample(t: t, bpm: held?.bpm ?? bpm, quality: (held?.quality ?? 0) * 0.5,
                                     snrDB: snrDB, motionFlagged: true, baselineBPM: restingBaseline,
                                     waveform: waveform)
            return
        }

        let calmWholeWindow = t - lastUncalmT >= config.analysisWindow
        if quality >= config.goodQuality, calmWholeWindow {
            updateBaseline(with: bpm)
        }
        let sample = HeartRateSample(t: t, bpm: bpm, quality: quality, snrDB: snrDB, motionFlagged: false,
                                     baselineBPM: restingBaseline, waveform: waveform)
        if quality >= config.goodQuality { lastGood = sample }
        latest = sample
    }

    private func updateBaseline(with bpm: Float) {
        if let b = restingBaseline {
            restingBaseline = b + config.baselineAlpha * (bpm - b)
            return
        }
        warmupSum += bpm
        warmupCount += 1
        if warmupCount >= config.baselineWarmup {
            restingBaseline = warmupSum / Float(warmupCount)
        }
    }
}
