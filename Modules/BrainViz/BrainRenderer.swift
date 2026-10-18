import MetalKit
import simd
import UIKit

/// Draws the cortex with the Axonal Labs look. All state is touched on the main thread
/// (MTKView calls `draw(in:)` on the main run loop by default).
final class BrainRenderer: NSObject, MTKViewDelegate {
    struct Uniforms {
        var proj: simd_float4x4
        var rot: simd_float4x4
        var act0: SIMD4<Float>
        var act1: SIMD4<Float>
        var params: SIMD4<Float> // time, shimmer, gain, vertexMix
    }

    // Inputs
    var targets = [Float](repeating: 0, count: 7)
    var gain: Float = 0.85
    var vertexMix: Float = 0
    var autoRotate = true

    // Camera / interaction (same constants as the website)
    var yaw: Float = -1.3
    var pitch: Float = -0.12
    var yawVelocity: Float = 0
    var isDragging = false
    var lastInteraction: CFTimeInterval = 0

    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let vertexBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    private let activityBuffer: MTLBuffer
    private let indexCount: Int
    private let vertexCount: Int
    private var smoother = HemodynamicSmoother(count: 7)
    private var lastTime = CACurrentMediaTime()
    private let startTime = CACurrentMediaTime()
    private let reduceMotion = UIAccessibility.isReduceMotionEnabled

    init?(view: MTKView, mesh: BrainMesh) {
        guard let device = view.device ?? MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return nil }
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = 4
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        view.isOpaque = false
        view.backgroundColor = .clear

        let library: MTLLibrary
        do { library = try device.makeLibrary(source: BrainShaderSource.metal, options: nil) }
        catch { assertionFailure("Brain shader failed to compile: \(error)"); return nil }

        let vd = MTLVertexDescriptor()
        let formats: [(MTLVertexFormat, Int, Int)] = [
            (.float3, 0, 0), (.float3, 12, 0), (.uchar4Normalized, 24, 0),
            (.uchar4Normalized, 28, 0), (.float, 0, 1),
        ]
        for (i, f) in formats.enumerated() {
            vd.attributes[i].format = f.0
            vd.attributes[i].offset = f.1
            vd.attributes[i].bufferIndex = f.2
        }
        vd.layouts[0].stride = BrainMesh.vertexStride
        vd.layouts[1].stride = MemoryLayout<Float>.stride

        let pd = MTLRenderPipelineDescriptor()
        pd.vertexFunction = library.makeFunction(name: "brainVertex")
        pd.fragmentFunction = library.makeFunction(name: "brainFragment")
        pd.vertexDescriptor = vd
        pd.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pd.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        pd.rasterSampleCount = view.sampleCount

        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true

        guard let pipeline = try? device.makeRenderPipelineState(descriptor: pd),
              let depthState = device.makeDepthStencilState(descriptor: dd),
              let vb = mesh.vertexBytes.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count) }),
              let ib = mesh.indices.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count) }),
              let ab = device.makeBuffer(length: mesh.vertexCount * MemoryLayout<Float>.stride, options: .storageModeShared)
        else { return nil }

        self.queue = queue
        self.pipeline = pipeline
        self.depthState = depthState
        self.vertexBuffer = vb
        self.indexBuffer = ib
        self.activityBuffer = ab
        self.indexCount = mesh.indices.count
        self.vertexCount = mesh.vertexCount
        super.init()
    }

    /// Per-vertex activity (0...1) in fsaverage5 order, e.g. raw brain-model output.
    /// Updated at ~1 Hz, so a plain copy into the shared buffer is fine.
    func setVertexActivity(_ values: [Float]) {
        let n = min(values.count, vertexCount)
        let p = activityBuffer.contents().bindMemory(to: Float.self, capacity: vertexCount)
        values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: n) }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let dt = Float(min(0.05, max(0.001, now - lastTime)))
        lastTime = now

        let a = smoother.step(targets: targets, dt: dt)
        if autoRotate && !isDragging && !reduceMotion && now - lastInteraction > 3 {
            yaw += 0.12 * dt
        }
        yaw += yawVelocity
        yawVelocity *= 0.92

        guard let pass = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }

        let size = view.drawableSize
        var u = Uniforms(
            proj: Self.perspective(fovy: 0.58, aspect: Float(size.width / max(1, size.height)), near: 0.1, far: 20),
            rot: Self.rotation(yaw: yaw, pitch: pitch),
            act0: SIMD4(a[0], a[1], a[2], a[3]),
            act1: SIMD4(a[4], a[5], a[6], 0),
            params: SIMD4(Float(now - startTime), reduceMotion ? 0 : 0.045, gain, vertexMix))

        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(depthState)
        enc.setCullMode(.none)
        enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        enc.setVertexBuffer(activityBuffer, offset: 0, index: 1)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 2)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: indexCount, indexType: .uint16,
                                  indexBuffer: indexBuffer, indexBufferOffset: 0)
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }

    /// Right-handed perspective with Metal's 0...1 clip depth.
    static func perspective(fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let f = 1 / tan(fovy / 2)
        return simd_float4x4(columns: (
            SIMD4(f / aspect, 0, 0, 0),
            SIMD4(0, f, 0, 0),
            SIMD4(0, 0, far / (near - far), -1),
            SIMD4(0, 0, near * far / (near - far), 0)))
    }

    /// Rx(pitch) · Ry(yaw), identical to the website's column-major matrix.
    static func rotation(yaw: Float, pitch: Float) -> simd_float4x4 {
        let cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch)
        return simd_float4x4(columns: (
            SIMD4(cy, sy * sp, -sy * cp, 0),
            SIMD4(0, cp, sp, 0),
            SIMD4(sy, -cy * sp, cy * cp, 0),
            SIMD4(0, 0, 0, 1)))
    }
}
