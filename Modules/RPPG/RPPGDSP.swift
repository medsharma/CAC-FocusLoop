import Accelerate
import Foundation

// Signal-processing helpers for the rPPG pipeline. Pure Swift + Accelerate, no ARKit, so all of
// it runs (and is tested) in the Simulator. Not thread-safe; owned by one estimator instance.

/// One second-order IIR section (Direct Form I), coefficients normalised so a0 == 1.
struct RPPGBiquad: Equatable {
    var b0: Double, b1: Double, b2: Double, a1: Double, a2: Double

    /// 2nd-order Butterworth low-pass (bilinear transform with pre-warping, Q = 1/sqrt 2).
    static func butterworthLowPass(cutoff: Double, fs: Double) -> RPPGBiquad {
        let w0 = 2 * Double.pi * cutoff / fs
        let c = cos(w0), alpha = sin(w0) / (2 * (1 / 2.0.squareRoot()))
        let a0 = 1 + alpha
        return RPPGBiquad(b0: (1 - c) / 2 / a0, b1: (1 - c) / a0, b2: (1 - c) / 2 / a0,
                          a1: -2 * c / a0, a2: (1 - alpha) / a0)
    }

    /// 2nd-order Butterworth high-pass (bilinear transform with pre-warping, Q = 1/sqrt 2).
    static func butterworthHighPass(cutoff: Double, fs: Double) -> RPPGBiquad {
        let w0 = 2 * Double.pi * cutoff / fs
        let c = cos(w0), alpha = sin(w0) / (2 * (1 / 2.0.squareRoot()))
        let a0 = 1 + alpha
        return RPPGBiquad(b0: (1 + c) / 2 / a0, b1: -(1 + c) / a0, b2: (1 + c) / 2 / a0,
                          a1: -2 * c / a0, a2: (1 - alpha) / a0)
    }

    /// Filters `x` in place, starting from rest.
    func apply(_ x: inout [Double]) {
        var x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0
        for i in x.indices {
            let x0 = x[i]
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1; x1 = x0; y2 = y1; y1 = y0
            x[i] = y0
        }
    }

    /// Magnitude response at `f` Hz (used by tests).
    func gain(at f: Double, fs: Double) -> Double {
        let w = 2 * Double.pi * f / fs
        // H(e^jw) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)
        let nr = b0 + b1 * cos(w) + b2 * cos(2 * w), ni = -(b1 * sin(w) + b2 * sin(2 * w))
        let dr = 1 + a1 * cos(w) + a2 * cos(2 * w), di = -(a1 * sin(w) + a2 * sin(2 * w))
        return ((nr * nr + ni * ni) / (dr * dr + di * di)).squareRoot()
    }
}

enum RPPGDSP {
    /// Removes the least-squares straight line from `x` in place.
    static func detrendLinear(_ x: inout [Double]) {
        let n = x.count
        guard n > 2 else { return }
        let nD = Double(n)
        let meanI = (nD - 1) / 2
        var meanX = 0.0
        for v in x { meanX += v }
        meanX /= nD
        var sxy = 0.0, sxx = 0.0
        for i in 0..<n {
            let di = Double(i) - meanI
            sxy += di * (x[i] - meanX); sxx += di * di
        }
        let slope = sxy / sxx
        for i in 0..<n { x[i] -= meanX + slope * (Double(i) - meanI) }
    }

    /// Zero-phase filtering (forward + backward pass of every section) with odd reflection padding,
    /// like scipy's `filtfilt(padtype="odd")`, so the waveform is not delayed.
    static func filtfilt(_ x: [Double], sections: [RPPGBiquad], padLength: Int) -> [Double] {
        let n = x.count
        guard n > 1 else { return x }
        let pad = min(padLength, n - 1)
        var y = [Double]()
        y.reserveCapacity(n + 2 * pad)
        for k in stride(from: pad, through: 1, by: -1) { y.append(2 * x[0] - x[k]) }
        y.append(contentsOf: x)
        if pad > 0 { for k in 1...pad { y.append(2 * x[n - 1] - x[n - 1 - k]) } }
        for s in sections { s.apply(&y) }
        y.reverse()
        for s in sections { s.apply(&y) }
        y.reverse()
        return Array(y[pad..<(pad + n)])
    }

    /// Parabolic interpolation of a peak at index `k` from three samples; returns the fractional
    /// offset in -0.5...0.5.
    static func parabolicOffset(_ ym1: Double, _ y0: Double, _ yp1: Double) -> Double {
        let denom = ym1 - 2 * y0 + yp1
        guard denom != 0, denom.isFinite else { return 0 }
        return max(-0.5, min(0.5, 0.5 * (ym1 - yp1) / denom))
    }
}

/// Hann-windowed, zero-padded real FFT power spectrum via vDSP. Reuses its buffers.
final class RPPGSpectrum {
    let fftLength: Int
    private let log2n: vDSP_Length
    private let setup: FFTSetup
    private var real: [Float]
    private var imag: [Float]
    private var input: [Float]
    private var window: [Float] = []

    init(fftLength: Int = 2048) {
        precondition(fftLength > 0 && fftLength & (fftLength - 1) == 0, "power of two")
        self.fftLength = fftLength
        log2n = vDSP_Length(Int(log2(Double(fftLength))))
        guard let s = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("vDSP_create_fftsetup failed")
        }
        setup = s
        real = [Float](repeating: 0, count: fftLength / 2)
        imag = [Float](repeating: 0, count: fftLength / 2)
        input = [Float](repeating: 0, count: fftLength)
    }

    deinit { vDSP_destroy_fftsetup(setup) }

    /// Power spectrum (bins 0...N/2-1, bin k at k*fs/N Hz) of `x` after a Hann window and
    /// zero-padding to `fftLength`. Absolute scale is arbitrary; only ratios are used.
    func power(of x: [Double]) -> [Double] {
        let n = min(x.count, fftLength)
        if window.count != n {
            window = [Float](repeating: 0, count: n)
            vDSP_hann_window(&window, vDSP_Length(n), Int32(vDSP_HANN_DENORM))
        }
        for i in 0..<fftLength { input[i] = 0 }
        for i in 0..<n { input[i] = Float(x[x.count - n + i]) * window[i] }
        let half = fftLength / 2
        var out = [Double](repeating: 0, count: half)
        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                input.withUnsafeBufferPointer { inp in
                    inp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
                        vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
                    }
                }
                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                // Packed format: realp[0] = DC, imagp[0] = Nyquist.
                out[0] = Double(rp[0]) * Double(rp[0])
                for k in 1..<half {
                    let r = Double(rp[k]), i = Double(ip[k])
                    out[k] = r * r + i * i
                }
            }
        }
        return out
    }
}
