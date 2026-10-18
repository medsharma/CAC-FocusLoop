import Foundation

#if !targetEnvironment(simulator)
import ARKit
import UIKit

/// Projects forehead and both-cheek face-mesh vertices into `ARFrame.capturedImage` and returns the
/// mean linear RGB of the skin pixels inside those three convex ROIs (input to `POSHeartRateEstimator`).
///
/// Plug into Perception as its `skinSampler`:
/// ```
/// let roi = SkinROIExtractor()
/// source.skinSampler = roi.sample(frame:anchor:)
/// ```
/// Privacy: the camera image is only read under a read-only lock and summed into three accumulators
/// (see `RPPGSkinPixelAverager`). No pixel is copied, stored or written to disk.
/// Cost: ~1-2 k sampled pixels + ~150 projected vertices per frame (well under 2 ms on A15+).
/// Not thread-safe; call from the ARSession delegate queue only.
final class SkinROIExtractor {
    /// Normalised face-mesh boxes. u = (x - centreX) / meshWidth in -0.5...0.5 (face-local x),
    /// v = (y - minY) / meshHeight in 0 (chin) ... 1 (top of forehead).
    struct Region {
        var v: ClosedRange<Float>
        var absU: ClosedRange<Float>
        /// -1: u < 0 only, +1: u > 0 only, 0: both sides.
        var side: Int
    }

    static let forehead = Region(v: 0.80...0.95, absU: 0...0.22, side: 0)
    static let cheekA = Region(v: 0.38...0.56, absU: 0.17...0.34, side: -1)
    static let cheekB = Region(v: 0.38...0.56, absU: 0.17...0.34, side: 1)

    var averager = RPPGSkinPixelAverager()
    /// Wall-clock cost of the most recent `sample` call (ms), for profiling.
    private(set) var lastDurationMs: Double = 0

    private var regionIndices: [[Int]] = []
    private var cachedVertexCount = 0

    init() {}

    func sample(frame: ARFrame, anchor: ARFaceAnchor) -> SIMD3<Float>? {
        let start = CACurrentMediaTime()
        defer { lastDurationMs = (CACurrentMediaTime() - start) * 1000 }
        guard anchor.isTracked else { return nil }
        let vertices = anchor.geometry.vertices
        if vertices.count != cachedVertexCount || regionIndices.isEmpty {
            regionIndices = Self.selectRegions(vertices: vertices)
            cachedVertexCount = vertices.count
        }
        let camera = frame.camera
        let size = camera.imageResolution
        let transform = anchor.transform
        var polygons: [[SIMD2<Float>]] = []
        polygons.reserveCapacity(regionIndices.count)
        for indices in regionIndices where indices.count >= 3 {
            var pts: [SIMD2<Float>] = []
            pts.reserveCapacity(indices.count)
            for i in indices {
                let v = vertices[i]
                let world = transform * SIMD4<Float>(v.x, v.y, v.z, 1)
                let p = camera.projectPoint(SIMD3<Float>(world.x, world.y, world.z),
                                            orientation: .landscapeRight, viewportSize: size)
                pts.append(SIMD2<Float>(Float(p.x), Float(p.y)))
            }
            polygons.append(pts)
        }
        return averager.average(pixelBuffer: frame.capturedImage, polygons: polygons)
    }

    /// Vertex indices for [forehead, cheek, cheek], chosen once from the mesh's own extent.
    static func selectRegions(vertices: [SIMD3<Float>]) -> [[Int]] {
        guard !vertices.isEmpty else { return [] }
        var lo = vertices[0], hi = vertices[0]
        for v in vertices { lo = simd_min(lo, v); hi = simd_max(hi, v) }
        let width = max(hi.x - lo.x, 1e-4), height = max(hi.y - lo.y, 1e-4), cx = (lo.x + hi.x) / 2
        return [forehead, cheekA, cheekB].map { r in
            vertices.indices.filter { i in
                let u = (vertices[i].x - cx) / width, v = (vertices[i].y - lo.y) / height
                guard r.v.contains(v), r.absU.contains(abs(u)) else { return false }
                return r.side == 0 || (r.side < 0 ? u < 0 : u > 0)
            }
        }
    }
}
#endif
