import CoreVideo
import XCTest
@testable import FocusLoop

/// Uses synthetic in-memory pixel buffers only (no camera, nothing written to disk).
final class RPPGSkinPixelAveragerTests: XCTestCase {

    /// Bi-planar 420f buffer; `fill(x, y)` returns (Y, Cb, Cr) for luma pixel (x, y).
    private func makeBuffer(width: Int = 1280, height: Int = 720,
                            fill: (Int, Int) -> (UInt8, UInt8, UInt8)) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
        let status = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                         attrs, &pb)
        let buffer = try XCTUnwrap(pb)
        XCTAssertEqual(status, kCVReturnSuccess)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let yp = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!.assumingMemoryBound(to: UInt8.self)
        let cp = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!.assumingMemoryBound(to: UInt8.self)
        let ys = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0), cs = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        for y in 0..<height {
            for x in 0..<width {
                let (lv, cb, cr) = fill(x, y)
                yp[y * ys + x] = lv
                if x % 2 == 0 && y % 2 == 0 {
                    cp[(y / 2) * cs + x] = cb
                    cp[(y / 2) * cs + x + 1] = cr
                }
            }
        }
        return buffer
    }

    private func srgbToLinear(_ c: Float) -> Float {
        c <= 0.04045 ? c / 12.92 : powf((c + 0.055) / 1.055, 2.4)
    }

    private let square: [SIMD2<Float>] = [[100, 100], [300, 100], [300, 300], [100, 300], [200, 200]]

    func testAveragesUniformSkinToExpectedLinearRGB() throws {
        let (yv, cb, cr): (UInt8, UInt8, UInt8) = (150, 110, 150)   // typical skin chroma
        let buffer = try makeBuffer { _, _ in (yv, cb, cr) }
        let rgb = try XCTUnwrap(RPPGSkinPixelAverager().average(pixelBuffer: buffer, polygons: [square]))
        let yF = Float(yv) / 255, cbF = Float(Int(cb) - 128) / 255, crF = Float(Int(cr) - 128) / 255
        let expected = SIMD3<Float>(srgbToLinear(yF + 1.402 * crF),
                                    srgbToLinear(yF - 0.344136 * cbF - 0.714136 * crF),
                                    srgbToLinear(yF + 1.772 * cbF))
        XCTAssertEqual(rgb.x, expected.x, accuracy: 0.004)
        XCTAssertEqual(rgb.y, expected.y, accuracy: 0.004)
        XCTAssertEqual(rgb.z, expected.z, accuracy: 0.004)
        XCTAssertGreaterThan(rgb.x, rgb.y)   // skin: R > G > B
        XCTAssertGreaterThan(rgb.y, rgb.z)
    }

    func testNonSkinIsMaskedOut() throws {
        // Saturated blue (Cb high, Cr low) is outside the skin range everywhere.
        let buffer = try makeBuffer { _, _ in (80, 200, 100) }
        XCTAssertNil(RPPGSkinPixelAverager().average(pixelBuffer: buffer, polygons: [square]))
    }

    func testOnlyPixelsInsideRegionAndSkinCount() throws {
        // Left half skin, right half blue background: a polygon straddling both still returns skin.
        let buffer = try makeBuffer { x, _ in x < 200 ? (150, 110, 150) : (80, 200, 100) }
        let rgb = try XCTUnwrap(RPPGSkinPixelAverager().average(pixelBuffer: buffer, polygons: [square]))
        XCTAssertGreaterThan(rgb.x, rgb.z)
        // Polygon fully outside the image -> nil.
        let off: [SIMD2<Float>] = [[-500, -500], [-400, -500], [-400, -400]]
        XCTAssertNil(RPPGSkinPixelAverager().average(pixelBuffer: buffer, polygons: [off]))
    }

    func testConvexHullAndScanline() {
        let hull = RPPGGeometry.convexHull(square)
        XCTAssertEqual(hull.count, 4, "interior point dropped")
        XCTAssertEqual(RPPGGeometry.area(hull), 40_000, accuracy: 1)
        let span = RPPGGeometry.scanlineSpan(hull, y: 150)
        XCTAssertEqual(span?.lowerBound ?? 0, 100, accuracy: 1e-4)
        XCTAssertEqual(span?.upperBound ?? 0, 300, accuracy: 1e-4)
        XCTAssertNil(RPPGGeometry.scanlineSpan(hull, y: 400))
    }

    func testAveragingIsFastForLargeROIs() throws {
        let buffer = try makeBuffer { x, y in (UInt8(120 + (x + y) % 40), 110, 150) }
        // Three large regions roughly the size of forehead + cheeks on a close face.
        let rois: [[SIMD2<Float>]] = [
            [[400, 100], [880, 100], [880, 260], [400, 260]],
            [[350, 350], [560, 350], [560, 560], [350, 560]],
            [[720, 350], [930, 350], [930, 560], [720, 560]]
        ]
        let avg = RPPGSkinPixelAverager()
        var times: [Double] = []
        for _ in 0..<30 {
            let s = CFAbsoluteTimeGetCurrent()
            XCTAssertNotNil(avg.average(pixelBuffer: buffer, polygons: rois))
            times.append((CFAbsoluteTimeGetCurrent() - s) * 1000)
        }
        let median = times.sorted()[times.count / 2]
        print(String(format: "RPPG_PERF roi_average_ms_median=%.3f", median))
        XCTAssertLessThan(median, 2.0)
    }
}
