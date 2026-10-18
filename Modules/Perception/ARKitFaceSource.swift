import Foundation
import CoreGraphics
import simd
import ARKit

public enum PerceptionError: Error, Equatable {
    case faceTrackingUnsupported
}

/// Blendshapes read from `ARFaceAnchor` (spec 1.1), keyed by `BlendShapeLocation.rawValue`.
public let perceptionBlendShapeNames: [String] = [
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawOpen", "browDownLeft", "browDownRight", "browInnerUp", "mouthClose",
]

#if !targetEnvironment(simulator)

/// ARKit face tracking (TrueDepth) as a `FaceSignalSource`.
///
/// * Runs `ARFaceTrackingConfiguration` with its delegate on a dedicated serial queue.
/// * Emits one `FaceSample` per camera frame (numbers only). Timestamps are `ARFrame.timestamp`
///   (seconds, device uptime clock) minus `timeOrigin`; set `timeOrigin` to align with the session clock.
/// * `skinSampler` (rPPG ROI extractor) and `onFrame` (face-mesh overlay) are called synchronously on
///   the ARKit queue; the `ARFrame` must not be retained beyond the call. Set both before `start()`.
/// * Nothing is ever written to disk.
public final class ARKitFaceSource: NSObject, FaceSignalSource, ARSessionDelegate {
    public var isAvailable: Bool { ARFaceTrackingConfiguration.isSupported }
    public let samples: AsyncStream<FaceSample>

    /// rPPG hook: mean linear RGB of the skin ROI for this frame, or nil if invalid.
    public var skinSampler: ((ARFrame, ARFaceAnchor) -> SIMD3<Float>?)?
    /// Face-mesh overlay hook: projected face-mesh vertices in normalized view coordinates (0...1,
    /// origin top-left, portrait), throttled to `meshMaxRate`. Called on the ARKit queue.
    public var onFrame: (([CGPoint]) -> Void)?
    /// Width / height of the overlay view, so projected points match an aspect-fill camera layout.
    public var overlayAspect: CGFloat = 9.0 / 19.5
    /// Max overlay update rate (Hz) and vertex stride (1220 vertices / stride points per update).
    public var meshMaxRate: Double = 15
    public var meshVertexStride: Int = 3
    /// Subtracted from `ARFrame.timestamp` to produce `FaceSample.t`.
    public var timeOrigin: Double = 0

    private let session = ARSession()
    private let queue = DispatchQueue(label: "org.focusloop.perception.arkit", qos: .userInteractive)
    private let continuation: AsyncStream<FaceSample>.Continuation
    private var lastMeshEmit: Double = -.infinity
    private var hadFace = false

    public override init() {
        var c: AsyncStream<FaceSample>.Continuation!
        samples = AsyncStream(bufferingPolicy: .bufferingNewest(120)) { c = $0 }
        continuation = c
        super.init()
        session.delegate = self
        session.delegateQueue = queue
    }

    deinit {
        session.pause()
        continuation.finish()
    }

    public func start() throws {
        guard ARFaceTrackingConfiguration.isSupported else { throw PerceptionError.faceTrackingUnsupported }
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = false
        config.maximumNumberOfTrackedFaces = 1
        if let fmt = ARFaceTrackingConfiguration.supportedVideoFormats.first(where: { $0.framesPerSecond == 30 }) {
            config.videoFormat = fmt
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    public func stop() {
        session.pause()
    }

    // MARK: ARSessionDelegate (on `queue`)

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let t = frame.timestamp - timeOrigin
        guard let face = frame.anchors.lazy.compactMap({ $0 as? ARFaceAnchor }).first, face.isTracked else {
            if hadFace { hadFace = false; onFrame?([]) }
            continuation.yield(FaceSample(t: t, faceDetected: false))
            return
        }
        hadFace = true

        var shapes: [String: Float] = [:]
        shapes.reserveCapacity(perceptionBlendShapeNames.count)
        for name in perceptionBlendShapeNames {
            if let v = face.blendShapes[ARFaceAnchor.BlendShapeLocation(rawValue: name)] {
                shapes[name] = v.floatValue
            }
        }

        let viewMatrix = frame.camera.viewMatrix(for: .portrait)
        let pose = PerceptionPoseMath.headPose(faceInView: viewMatrix * face.transform)
        var look = face.lookAtPoint
        if simd_length(look) < 1e-5 {
            look = PerceptionPoseMath.lookAtFallback(leftEye: face.leftEyeTransform, rightEye: face.rightEyeTransform)
        }
        let rgb = skinSampler?(frame, face)

        continuation.yield(FaceSample(t: t, faceDetected: true, blendShapes: shapes, lookAtPoint: look,
                                      headYaw: pose.yaw, headPitch: pose.pitch, headRoll: pose.roll, skinRGB: rgb))

        if let onFrame, frame.timestamp - lastMeshEmit >= 1 / meshMaxRate {
            lastMeshEmit = frame.timestamp
            onFrame(projectMesh(face: face, camera: frame.camera))
        }
    }

    private func projectMesh(face: ARFaceAnchor, camera: ARCamera) -> [CGPoint] {
        let h: CGFloat = 1000
        let viewport = CGSize(width: h * overlayAspect, height: h)
        let verts = face.geometry.vertices
        let stride = max(1, meshVertexStride)
        var out: [CGPoint] = []
        out.reserveCapacity(verts.count / stride + 1)
        let m = face.transform
        var i = 0
        while i < verts.count {
            let v = verts[i]
            let w = m * SIMD4<Float>(v.x, v.y, v.z, 1)
            let p = camera.projectPoint(SIMD3<Float>(w.x, w.y, w.z), orientation: .portrait, viewportSize: viewport)
            out.append(CGPoint(x: p.x / viewport.width, y: p.y / viewport.height))
            i += stride
        }
        return out
    }
}

#else

/// Simulator stub: ARKit face tracking is unavailable, so `isAvailable == false` and `start()` throws.
/// Use `MockFaceSource` (SYNTHETIC) or the Replay stream instead.
public final class ARKitFaceSource: NSObject, FaceSignalSource {
    public var isAvailable: Bool { false }
    public let samples: AsyncStream<FaceSample>
    public var skinSampler: ((ARFrame, ARFaceAnchor) -> SIMD3<Float>?)?
    public var onFrame: (([CGPoint]) -> Void)?
    public var overlayAspect: CGFloat = 9.0 / 19.5
    public var meshMaxRate: Double = 15
    public var meshVertexStride: Int = 3
    public var timeOrigin: Double = 0
    private let continuation: AsyncStream<FaceSample>.Continuation

    public override init() {
        var c: AsyncStream<FaceSample>.Continuation!
        samples = AsyncStream(bufferingPolicy: .bufferingNewest(120)) { c = $0 }
        continuation = c
        super.init()
    }

    deinit { continuation.finish() }

    public func start() throws { throw PerceptionError.faceTrackingUnsupported }
    public func stop() {}
}

#endif
