import MetalKit
import UIKit

/// Renders a still cortex frame (non-rotating, default viewing angle) at given activations, for the
/// report export (`ImageRenderer` cannot capture a live Metal view).
///
/// It drives the frozen `BrainRenderer` (same app target) against an offscreen `MTKView` and reads
/// the resolved drawable texture back into a `UIImage` with a transparent background. The image is
/// kept in memory only; it reaches disk only embedded in a report the user explicitly exports.
///
/// The renderer eases activations with `HemodynamicSmoother` using wall-clock `dt` (clamped to
/// 0.05 s per frame), so the snapshot warms up for `warmUpSeconds` (~2.5 s, ~96% converged)
/// before the final frame. Call it ahead of time (e.g. when the report screen appears).
/// Returns nil if Metal or the brain mesh is unavailable.
@MainActor
public enum CortexSnapshotter {
    public static var warmUpSeconds: Double = 2.5

    public static func snapshot(activations: [Float], size: CGSize = CGSize(width: 480, height: 360),
                                scale: CGFloat = 2, gain: Float = 0.85) async -> UIImage? {
        guard size.width >= 1, size.height >= 1,
              let device = MTLCreateSystemDefaultDevice(), let mesh = BrainMesh.shared else { return nil }
        let view = MTKView(frame: CGRect(origin: .zero, size: size), device: device)
        guard let renderer = BrainRenderer(view: view, mesh: mesh) else { return nil }
        view.framebufferOnly = false          // allow blitting from the drawable
        view.isPaused = true
        view.enableSetNeedsDisplay = false
        view.autoResizeDrawable = false
        view.contentScaleFactor = scale
        view.drawableSize = CGSize(width: (size.width * scale).rounded(), height: (size.height * scale).rounded())
        view.delegate = renderer
        renderer.autoRotate = false
        renderer.gain = gain
        renderer.vertexMix = 0
        renderer.targets = (0..<7).map { $0 < activations.count ? max(0, min(1, activations[$0])) : 0 }

        // Warm-up: let the smoother converge (each frame advances it by <= 0.05 s of real time).
        let frames = max(1, Int(warmUpSeconds / 0.05))
        for _ in 0..<frames {
            view.draw()
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        // Final frame drawn directly, so the view keeps its drawable for read-back.
        renderer.draw(in: view)
        guard let texture = view.currentDrawable?.texture else { return nil }
        // The renderer commits on its own queue; give the GPU time to finish before reading back.
        try? await Task.sleep(nanoseconds: 120_000_000)
        return image(from: texture, device: device, scale: scale)
    }

    /// Copies a BGRA8 texture to a CPU buffer (blit, so private/managed storage works) -> UIImage.
    static func image(from texture: MTLTexture, device: MTLDevice, scale: CGFloat) -> UIImage? {
        let w = texture.width, h = texture.height, bpr = w * 4
        guard texture.pixelFormat == .bgra8Unorm || texture.pixelFormat == .bgra8Unorm_srgb,
              let buffer = device.makeBuffer(length: bpr * h, options: .storageModeShared),
              let queue = device.makeCommandQueue(), let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1), to: buffer, destinationOffset: 0,
                  destinationBytesPerRow: bpr, destinationBytesPerImage: bpr * h)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }

        let data = Data(bytes: buffer.contents(), count: bpr * h)
        guard let provider = CGDataProvider(data: data as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bpr,
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue
                                                        | CGImageAlphaInfo.premultipliedFirst.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        else { return nil }
        return UIImage(cgImage: cg, scale: scale, orientation: .up)
    }

    /// Fraction of pixels with non-zero alpha (0 = blank). Used by tests and as a sanity check.
    public static func coverage(of image: UIImage) -> Double {
        guard let cg = image.cgImage else { return 0 }
        let w = cg.width, h = cg.height
        var px = [UInt8](repeating: 0, count: w * h * 4)
        let drawn: Bool = px.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { return 0 }
        var n = 0
        for i in stride(from: 3, to: px.count, by: 4) where px[i] > 8 { n += 1 }
        return Double(n) / Double(w * h)
    }
}
