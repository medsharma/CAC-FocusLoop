import Foundation
import simd

/// Work-zone geometry: how gaze + head pose are combined into one direction, how a `WorkZone`
/// is calibrated from fixations, and how far a direction lies outside the zone.
///
/// Conventions (match `ml/schema/feature_schema.json` and `ml/synth/make_replay.py`):
///   * yaw: + = user's left, pitch: + = up, radians.
///   * combined direction = gaze + `headWeight` x head.
///   * deviation = Euclidean distance outside the rectangular (yaw x pitch) cone, 0 inside.
public enum WorkZoneMath {
    /// Weight of head pose added to the gaze direction for the combined work-zone direction.
    public static let headWeight: Float = 0.3
    /// Absolute margin (rad) added to each calibrated half-width.
    public static let defaultMargin: Float = 0.05
    /// Relative margin added to each calibrated half-width (fraction of the measured extent).
    public static let defaultRelativeMargin: Float = 0.15
    /// Half-width clamps (rad) so a sloppy calibration can't produce a degenerate or huge zone.
    public static let minHalfWidth: Float = 0.08
    public static let maxHalfWidth: Float = 0.9

    // MARK: Directions

    /// Eye-in-head gaze direction from `ARFaceAnchor.lookAtPoint` (face space: +x = face's own left,
    /// +y up, +z out of the face). Returns (yaw, pitch) in radians, or nil for a degenerate point.
    public static func eyeDirection(lookAtPoint p: SIMD3<Float>) -> (yaw: Float, pitch: Float)? {
        guard p.z > 1e-4 || abs(p.x) > 1e-4 || abs(p.y) > 1e-4 else { return nil }
        let z = max(p.z, 1e-4)
        return (atan2(p.x, z), atan2(p.y, (p.x * p.x + z * z).squareRoot()))
    }

    /// Gaze direction in camera space (+ = user's left / up): head pose plus eye-in-head angle.
    /// Small-angle composition, which is accurate well beyond the work-zone range.
    public static func gazeDirection(lookAtPoint: SIMD3<Float>, headYaw: Float, headPitch: Float) -> (yaw: Float, pitch: Float) {
        let eye = eyeDirection(lookAtPoint: lookAtPoint) ?? (0, 0)
        return (headYaw + eye.yaw, headPitch + eye.pitch)
    }

    /// Combined gaze + head direction used for the work zone (same formula as the synthetic generator).
    public static func combinedDirection(gazeYaw: Float, gazePitch: Float, headYaw: Float, headPitch: Float) -> (yaw: Float, pitch: Float) {
        (gazeYaw + headWeight * headYaw, gazePitch + headWeight * headPitch)
    }

    /// Combined direction for one face sample, or nil when no face is tracked. The Calibration
    /// module should feed exactly these values into `calibrate(center:corners:isPhoneScreen:)`.
    public static func combinedDirection(for sample: FaceSample) -> (yaw: Float, pitch: Float)? {
        guard sample.faceDetected else { return nil }
        let g = gazeDirection(lookAtPoint: sample.lookAtPoint, headYaw: sample.headYaw, headPitch: sample.headPitch)
        return combinedDirection(gazeYaw: g.yaw, gazePitch: g.pitch, headYaw: sample.headYaw, headPitch: sample.headPitch)
    }

    // MARK: Calibration

    /// Builds a `WorkZone` from 5 fixations: the centre and 4 corners, each a list of combined
    /// (yaw, pitch) samples (see `combinedDirection(for:)`). Each fixation is reduced with a
    /// per-axis median (robust to blinks/glances); half-widths are the largest corner offset from
    /// the centre plus `relativeMargin` and `margin`, clamped to `minHalfWidth...maxHalfWidth`.
    /// Empty inputs fall back to `WorkZone.default` values for that part.
    public static func calibrate(center: [(yaw: Float, pitch: Float)],
                                 corners: [[(yaw: Float, pitch: Float)]],
                                 isPhoneScreen: Bool,
                                 margin: Float = defaultMargin,
                                 relativeMargin: Float = defaultRelativeMargin,
                                 calibratedAt: Date = Date()) -> WorkZone {
        let cornerMedians = corners.compactMap { robustCenter($0) }
        let c: (yaw: Float, pitch: Float)
        if let m = robustCenter(center) {
            c = m
        } else if !cornerMedians.isEmpty {
            c = (median(cornerMedians.map(\.yaw)), median(cornerMedians.map(\.pitch)))
        } else {
            c = (WorkZone.default.centerYaw, WorkZone.default.centerPitch)
        }
        func halfWidth(_ offsets: [Float], fallback: Float) -> Float {
            guard let extent = offsets.max() else { return fallback }
            return min(maxHalfWidth, max(minHalfWidth, extent * (1 + relativeMargin) + margin))
        }
        let hy = halfWidth(cornerMedians.map { abs($0.yaw - c.yaw) }, fallback: WorkZone.default.halfWidthYaw)
        let hp = halfWidth(cornerMedians.map { abs($0.pitch - c.pitch) }, fallback: WorkZone.default.halfWidthPitch)
        return WorkZone(centerYaw: c.yaw, centerPitch: c.pitch, halfWidthYaw: hy, halfWidthPitch: hp,
                        calibratedAt: calibratedAt, isPhoneScreen: isPhoneScreen)
    }

    /// Per-axis median of a fixation's samples (non-finite samples dropped), nil if empty.
    public static func robustCenter(_ samples: [(yaw: Float, pitch: Float)]) -> (yaw: Float, pitch: Float)? {
        let s = samples.filter { $0.yaw.isFinite && $0.pitch.isFinite }
        guard !s.isEmpty else { return nil }
        return (median(s.map(\.yaw)), median(s.map(\.pitch)))
    }

    // MARK: Deviation

    /// Angular distance (rad) of a combined direction outside the zone's rectangular cone; 0 inside.
    public static func deviation(yaw: Float, pitch: Float, zone: WorkZone) -> Float {
        let dy = max(0, abs(yaw - zone.centerYaw) - zone.halfWidthYaw)
        let dp = max(0, abs(pitch - zone.centerPitch) - zone.halfWidthPitch)
        return (dy * dy + dp * dp).squareRoot()
    }

    public static func isInside(yaw: Float, pitch: Float, zone: WorkZone) -> Bool {
        deviation(yaw: yaw, pitch: pitch, zone: zone) == 0
    }

    static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let s = values.sorted()
        let n = s.count
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
    }
}
