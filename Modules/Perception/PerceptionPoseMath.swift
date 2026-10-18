import Foundation
import simd

/// Pure head-pose math (no ARKit types), so it is testable in the Simulator.
public enum PerceptionPoseMath {
    /// Head yaw/pitch/roll (radians) from the face anchor transform expressed in portrait view space
    /// (`camera.viewMatrix(for: .portrait) * faceAnchor.transform`): x right on screen, y up, z toward
    /// the viewer. The face anchor's +z points out of the face, +x to the face's own left.
    ///
    /// * yaw: + = user's left (forward vector rotates toward +x)
    /// * pitch: + = up
    /// * roll: rotation of the face's x axis about the view axis (+ = counter-clockwise on screen)
    /// A face looking straight into the camera gives (0, 0, 0).
    public static func headPose(faceInView m: simd_float4x4) -> (yaw: Float, pitch: Float, roll: Float) {
        let fwd = simd_normalize(SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z))
        let xAxis = simd_normalize(SIMD3<Float>(m.columns.0.x, m.columns.0.y, m.columns.0.z))
        let yaw = atan2(fwd.x, fwd.z)
        let pitch = asin(min(1, max(-1, fwd.y)))
        let roll = atan2(xAxis.y, xAxis.x)
        return (yaw, pitch, roll)
    }

    /// Eye-in-head gaze point fallback from the eye transforms (face space): a point 1 m along the mean
    /// of both eyes' +z axes. Used only when `lookAtPoint` is degenerate.
    public static func lookAtFallback(leftEye: simd_float4x4, rightEye: simd_float4x4) -> SIMD3<Float> {
        let l = SIMD3<Float>(leftEye.columns.2.x, leftEye.columns.2.y, leftEye.columns.2.z)
        let r = SIMD3<Float>(rightEye.columns.2.x, rightEye.columns.2.y, rightEye.columns.2.z)
        let d = l + r
        let len = simd_length(d)
        return len > 1e-5 ? d / len : SIMD3<Float>(0, 0, 1)
    }
}
