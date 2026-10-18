import CoreVideo
import Foundation

/// Averages skin pixels of a camera `CVPixelBuffer` inside convex image-space polygons, returning mean
/// **linear** RGB in 0...1. Accepts bi-planar YCbCr 4:2:0 (ARKit `capturedImage`, 420f; 420v also
/// accepted) and 32BGRA (e.g. decoded video frames).
///
/// Privacy: pixels are read in place under a read-only lock and only summed into three accumulators;
/// nothing is copied, stored or written anywhere. No ARKit dependency, so it is unit-tested in the
/// Simulator with synthetic pixel buffers.
struct RPPGSkinPixelAverager {
    /// Upper bound on sampled pixels per frame (sub-sampling step adapts to the ROI area).
    var maxSamples: Int = 3000
    /// Fewer valid skin pixels than this -> nil (ROI too small, off-screen or not skin).
    var minValidPixels: Int = 40
    /// Skin mask in full-range 8-bit YCbCr. Cb/Cr ranges follow Chai & Ngan (1999), widened a little
    /// because fixed chroma ranges are known to under-include some skin tones; Y bounds drop very dark
    /// and specular (saturated) pixels.
    var yRange: ClosedRange<Int> = 25...245
    var cbRange: ClosedRange<Int> = 72...132
    var crRange: ClosedRange<Int> = 128...178

    /// sRGB-encoded 0...1 -> linear, 1024-entry lookup.
    private static let linearLUT: [Float] = (0..<1024).map { i in
        let c = Float(i) / 1023
        return c <= 0.04045 ? c / 12.92 : powf((c + 0.055) / 1.055, 2.4)
    }

    private enum Layout { case yuvFull, yuvVideo, bgra }

    /// Pixel formats `average` understands.
    static func supports(_ format: OSType) -> Bool {
        format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            || format == kCVPixelFormatType_32BGRA
    }

    /// Mean linear RGB of skin pixels inside the union of `polygons` (pixel coordinates of the luma
    /// plane, or of the image for BGRA; each polygon is replaced by its convex hull). nil if too few skin pixels.
    func average(pixelBuffer: CVPixelBuffer, polygons: [[SIMD2<Float>]]) -> SIMD3<Float>? {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let layout: Layout
        switch format {
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange: layout = .yuvFull
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: layout = .yuvVideo
        case kCVPixelFormatType_32BGRA: layout = .bgra
        default: return nil
        }
        if layout != .bgra { guard CVPixelBufferGetPlaneCount(pixelBuffer) == 2 else { return nil } }

        let hulls = polygons.map(RPPGGeometry.convexHull).filter { $0.count >= 3 }
        guard !hulls.isEmpty else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let width: Int, height: Int, cWidth: Int, cHeight: Int, yStride: Int, cStride: Int
        let yPtr: UnsafeMutablePointer<UInt8>, cPtr: UnsafeMutablePointer<UInt8>
        if layout == .bgra {
            guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
            width = CVPixelBufferGetWidth(pixelBuffer); height = CVPixelBufferGetHeight(pixelBuffer)
            cWidth = width; cHeight = height
            yStride = CVPixelBufferGetBytesPerRow(pixelBuffer); cStride = yStride
            yPtr = base.assumingMemoryBound(to: UInt8.self); cPtr = yPtr
        } else {
            guard let yBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
                  let cBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) else { return nil }
            width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
            height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
            cWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1)
            cHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)
            yStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
            cStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)
            yPtr = yBase.assumingMemoryBound(to: UInt8.self)
            cPtr = cBase.assumingMemoryBound(to: UInt8.self)
        }

        // Adaptive sub-sampling step from the total (clipped) hull area.
        var area: Float = 0
        for h in hulls { area += RPPGGeometry.area(h) }
        guard area.isFinite else { return nil }
        let step = max(1, Int(min(4096, (area / Float(maxSamples)).squareRoot()).rounded(.up)))

        let lut = Self.linearLUT
        var sum = SIMD3<Float>(repeating: 0)
        var count = 0
        for hull in hulls {
            var minY = Float.greatestFiniteMagnitude, maxY = -Float.greatestFiniteMagnitude
            for p in hull { minY = min(minY, p.y); maxY = max(maxY, p.y) }
            // Clamp before converting: a degenerate ROI far off the image must not overflow `Int`.
            let fh = Float(height)
            let y0 = max(0, Int(max(-1, min(fh, minY)).rounded(.up))), y1 = min(height - 1, Int(max(-1, min(fh, maxY)).rounded(.down)))
            guard y0 <= y1 else { continue }
            var row = y0
            while row <= y1 {
                guard let span = RPPGGeometry.scanlineSpan(hull, y: Float(row)) else { row += step; continue }
                let fw = Float(width)
                let x0 = max(0, Int(max(-1, min(fw, span.lowerBound)).rounded(.up)))
                let x1 = min(width - 1, Int(max(-1, min(fw, span.upperBound)).rounded(.down)))
                let cRow = min(cHeight - 1, row >> 1)
                var col = x0
                while col <= x1 {
                    // Normalise to full-range YCbCr (yF 0...1, cb/cr -0.5...0.5).
                    let yF: Float, cb: Float, cr: Float
                    switch layout {
                    case .bgra:
                        let o = row * yStride + 4 * col
                        let bv = Float(yPtr[o]) / 255, gv = Float(yPtr[o + 1]) / 255, rv = Float(yPtr[o + 2]) / 255
                        yF = 0.299 * rv + 0.587 * gv + 0.114 * bv
                        cb = (bv - yF) / 1.772
                        cr = (rv - yF) / 1.402
                    case .yuvFull, .yuvVideo:
                        let yv = Int(yPtr[row * yStride + col])
                        let cCol = min(cWidth - 1, col >> 1)
                        let cbv = Int(cPtr[cRow * cStride + 2 * cCol])
                        let crv = Int(cPtr[cRow * cStride + 2 * cCol + 1])
                        if layout == .yuvFull {
                            yF = Float(yv) / 255; cb = Float(cbv - 128) / 255; cr = Float(crv - 128) / 255
                        } else {
                            yF = Float(yv - 16) / 219; cb = Float(cbv - 128) / 224; cr = Float(crv - 128) / 224
                        }
                    }
                    let yFull = Int(yF * 255), cbFull = Int(cb * 255) + 128, crFull = Int(cr * 255) + 128
                    if yRange.contains(yFull), cbRange.contains(cbFull), crRange.contains(crFull) {
                        // BT.601 full-range YCbCr -> sRGB (same matrix as Apple's ARKit Metal sample).
                        let r = yF + 1.402 * cr
                        let g = yF - 0.344136 * cb - 0.714136 * cr
                        let b = yF + 1.772 * cb
                        sum.x += lut[Int(max(0, min(1, r)) * 1023)]
                        sum.y += lut[Int(max(0, min(1, g)) * 1023)]
                        sum.z += lut[Int(max(0, min(1, b)) * 1023)]
                        count += 1
                    }
                    col += step
                }
                row += step
            }
        }
        guard count >= minValidPixels else { return nil }
        return sum / Float(count)
    }
}

/// Small 2-D helpers for convex ROIs.
enum RPPGGeometry {
    /// Andrew's monotone chain; returns the hull counter-clockwise (y down = clockwise on screen).
    static func convexHull(_ points: [SIMD2<Float>]) -> [SIMD2<Float>] {
        let pts = points.filter { $0.x.isFinite && $0.y.isFinite }
            .sorted { $0.x == $1.x ? $0.y < $1.y : $0.x < $1.x }
        guard pts.count >= 3 else { return pts }
        func cross(_ o: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        var lower: [SIMD2<Float>] = []
        for p in pts {
            while lower.count >= 2, cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 { lower.removeLast() }
            lower.append(p)
        }
        var upper: [SIMD2<Float>] = []
        for p in pts.reversed() {
            while upper.count >= 2, cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 { upper.removeLast() }
            upper.append(p)
        }
        return Array(lower.dropLast() + upper.dropLast())
    }

    static func area(_ poly: [SIMD2<Float>]) -> Float {
        guard poly.count >= 3 else { return 0 }
        var a: Float = 0
        for i in poly.indices {
            let p = poly[i], q = poly[(i + 1) % poly.count]
            a += p.x * q.y - q.x * p.y
        }
        return abs(a) / 2
    }

    /// Horizontal extent of a convex polygon at scanline `y`, or nil if the line misses it.
    static func scanlineSpan(_ poly: [SIMD2<Float>], y: Float) -> ClosedRange<Float>? {
        var lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
        for i in poly.indices {
            let a = poly[i], b = poly[(i + 1) % poly.count]
            if (a.y <= y && y <= b.y) || (b.y <= y && y <= a.y) {
                let x: Float = a.y == b.y ? a.x : a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y)
                lo = min(lo, x); hi = max(hi, x)
                if a.y == b.y { lo = min(lo, b.x); hi = max(hi, b.x) }
            }
        }
        return lo <= hi ? lo...hi : nil
    }
}
