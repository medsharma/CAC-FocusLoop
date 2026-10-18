import MetalKit
import SwiftUI

/// Rotating 3D cortex with the Axonal Labs fMRI heat overlay.
///
///     BrainCortexView(activations: timeline.activations(at: playhead))
///
/// - `activations`: 7 values in `BrainNetwork` order, 0...1 (0 = rest, 1 = strongly active).
/// - `vertexActivity`: optional 20,484 values in fsaverage5 order (0...1). When provided it
///   replaces the network view, so raw per-vertex brain-model output can be shown directly.
/// Drag to rotate; auto-rotates after 3 s idle; honours Reduce Motion. Works in the Simulator.
public struct BrainCortexView: UIViewRepresentable {
    public var activations: [Float]
    public var vertexActivity: [Float]?
    public var autoRotate: Bool
    public var gain: Float

    public init(activations: [Float], vertexActivity: [Float]? = nil,
                autoRotate: Bool = true, gain: Float = 0.85) {
        self.activations = activations
        self.vertexActivity = vertexActivity
        self.autoRotate = autoRotate
        self.gain = gain
    }

    public final class Coordinator: NSObject {
        var renderer: BrainRenderer?   // MTKView.delegate is weak; the coordinator keeps it alive
        var lastVertexActivity: [Float]?

        @objc func pan(_ g: UIPanGestureRecognizer) {
            guard let r = renderer else { return }
            switch g.state {
            case .began: r.isDragging = true
            case .ended, .cancelled, .failed: r.isDragging = false
            default: break
            }
            let t = g.translation(in: g.view)
            g.setTranslation(.zero, in: g.view)
            let dx = Float(t.x), dy = Float(t.y)
            r.yaw += dx * 0.008
            r.yawVelocity = dx * 0.0015
            r.pitch = max(-0.9, min(0.6, r.pitch + dy * 0.005))
            r.lastInteraction = CACurrentMediaTime()
        }
    }

    public func makeCoordinator() -> Coordinator { Coordinator() }

    public func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.preferredFramesPerSecond = 60
        view.backgroundColor = .clear
        view.isAccessibilityElement = true
        view.accessibilityLabel = "3D brain showing predicted network activity"
        if let mesh = BrainMesh.shared, let r = BrainRenderer(view: view, mesh: mesh) {
            context.coordinator.renderer = r
            view.delegate = r
        } else {
            // Don't crash previews/debug builds; the view just stays empty.
            print("BrainCortexView: BrainMesh.bin missing from Bundle.main (add Resources/Brain as a group, not a folder reference) or Metal unavailable")
        }
        view.addGestureRecognizer(UIPanGestureRecognizer(target: context.coordinator,
                                                         action: #selector(Coordinator.pan(_:))))
        return view
    }

    public func updateUIView(_ view: MTKView, context: Context) {
        guard let r = context.coordinator.renderer else { return }
        r.targets = (0..<7).map { $0 < activations.count ? max(0, min(1, activations[$0])) : 0 }
        r.autoRotate = autoRotate
        r.gain = gain
        if let va = vertexActivity {
            if va != context.coordinator.lastVertexActivity {
                r.setVertexActivity(va)
                context.coordinator.lastVertexActivity = va
            }
            r.vertexMix = 1
        } else {
            r.vertexMix = 0
        }
    }
}
