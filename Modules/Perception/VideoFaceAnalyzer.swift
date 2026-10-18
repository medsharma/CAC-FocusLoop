import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import simd
import Vision

/// Turns one decoded video frame into a `FaceSample` with Vision, so a recorded video of a real person can
/// stand in for the TrueDepth camera. Used by `VideoFaceSource` (real-time playback), by the Mac face-track
/// tool (`tools/facetrack.sh`) and by tests (offline decoding).
///
/// Two modes:
/// * `.landmarks` (devices, macOS): Vision face rectangles (head pose) + face landmarks (eyes, brows, lips,
///   pupils). Every blendshape is an approximation calibrated **per video** from rolling statistics:
///   - eyeBlinkLeft/Right: eye aspect ratio (EAR = lid opening / eye width, face-aligned) vs an open-eye
///     baseline (rolling 90th percentile). EAR >= `openFraction` (70%) of the baseline reads 0, <= 35% reads 1.
///   - eyeLook*: pupil offset from the eye-corner midpoint (in eye half-widths), relative to its rolling median.
///   - jawOpen: inner-lip gap / face height above its rolling 20th percentile; mouthClose follows it.
///   - browDown / browInnerUp: eyebrow-to-eye distance (inter-ocular units) vs its rolling median.
///   - eyeSquint / eyeWide: partial closure / extra opening vs the EAR baseline.
/// * `.poseOnly`: face rectangles only (head pose, face presence, skin ROIs placed from the face box);
///   blendshapes stay neutral. Used in the Simulator, where Vision's face-landmark model returns garbage
///   (values that grow without bound within a few frames, on every compute device and revision we tried),
///   and as an automatic fallback if landmarks ever fail their sanity checks. In the Simulator, blinks and
///   yawns come from a face-track sidecar computed on the Mac by this same class (see `FaceTrack`).
///
/// Shared by both modes:
/// * head pose: `VNFaceObservation` yaw / pitch / roll (rectangles request, revision 3), re-signed to
///   `PerceptionPoseMath` (yaw + = user's left, pitch + = up, roll + = counter-clockwise on screen), lightly
///   smoothed, then centred on the video's resting pose (rolling median): a phone beside a laptop sees the
///   student's "looking at my work" pose off-axis.
/// * lookAtPoint: eye-in-head direction from the pupils (face space, +x = face's own left); straight ahead
///   in `.poseOnly`.
/// * skinRGB: mean linear RGB of a forehead and two cheek ROIs (from the eyes / brows / contour, or from the
///   face box), averaged by `RPPGSkinPixelAverager` in buffer coordinates (420f, 420v or BGRA).
///
/// Footage is assumed unmirrored (as the iPhone Camera app saves front-camera video by default): the user's
/// left is the image's right. Set `mirrored` for selfie-mirrored footage.
///
/// Privacy: pixels are only read in place (Vision and the skin averager); only numbers leave `analyze`.
/// Not thread-safe: call from one queue.
final class VideoFaceAnalyzer {
    enum Mode: String, Sendable { case landmarks, poseOnly }

    struct Config {
        /// Open-eye EAR baseline = this percentile of recent EAR samples.
        var earBaselinePercentile: Float = 0.9
        /// EAR as a fraction of the baseline at which the eye reads fully closed (1).
        var closedFraction: Float = 0.35
        /// EAR fraction at or above which the eye reads fully open (0) (dead zone for landmark jitter).
        var openFraction: Float = 0.7
        /// Rolling window for the eye / brow / mouth / pose baselines (s).
        var baselineSeconds: Double = 45
        /// Baseline samples needed before closure / brow / pose centring are reported (~3 s).
        var minBaselineSamples = 72
        /// Baselines are recomputed at most this often (s).
        var baselineRefresh: Double = 0.5
        /// Eye-in-head rotation (rad) per eye half-width of pupil offset.
        var eyeYawPerUnit: Float = 0.9
        var eyePitchPerUnit: Float = 0.9
        /// Inner-lip gap / face height above the resting gap that reads jawOpen = 1.
        var jawFullRatio: Float = 0.22
        /// Relative eyebrow lowering / raising (fraction of the resting brow-eye distance) that adds 1.
        var browDownSpan: Float = 0.5
        var browUpSpan: Float = 0.5
        /// Smoothing time constants (s) for head pose, pupil offsets and brows.
        var poseTau: Double = 0.2
        var eyeTau: Double = 0.1
        var browTau: Double = 0.25
        /// Centre head pose on the rolling median (the video's resting pose).
        var centerPose = true
        /// `.poseOnly`: run face detection on every n-th frame and hold it in between (CPU budget).
        var poseOnlyDetectEvery = 1
        /// Consecutive frames of implausible landmarks before switching to `.poseOnly` for good.
        var badLandmarkLimit = 12
        /// Face-box geometry for `.poseOnly` ROIs (measured on Vision landmarks): inter-ocular distance as a
        /// fraction of the box width, eye line and mouth as fractions of the box height from its top.
        var iodPerBoxWidth: Float = 0.36
        var eyeLinePerBoxHeight: Float = 0.28
        var mouthPerBoxHeight: Float = 0.70
    }

    /// One analysed frame.
    struct Result {
        var sample: FaceSample
        /// Landmark points normalized to the upright video frame (0...1, origin top-left); in `.poseOnly`, an
        /// outline of the face box plus the estimated eye and mouth centres. Empty if no face.
        var mesh: [CGPoint]
        /// Raw measurements (for tuning / logs).
        var debug: Debug?
    }

    struct Debug {
        var earLeft: Float, earRight: Float, earBaselineLeft: Float, earBaselineRight: Float
        var lipRatio: Float, browDistance: Float, rawYaw: Float, rawPitch: Float
        var pupilU: Float, pupilV: Float
    }

    var config: Config
    /// True for selfie-mirrored footage (user's left on the image's left).
    var mirrored = false
    /// Current mode (`.landmarks` may fall back to `.poseOnly` for good if landmarks are implausible).
    private(set) var mode: Mode
    var averager = RPPGSkinPixelAverager()

    /// Vision face landmarks are unusable in the iOS Simulator runtime (see type doc).
    static var landmarksAvailable: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return true
        #endif
    }

    // Rolling statistics (face frames only).
    private var earL = RollingStat(), earR = RollingStat()
    private var lipStat = RollingStat(), browStat = RollingStat(), browInnerStat = RollingStat()
    private var yawStat = RollingStat(), pitchStat = RollingStat()
    private var pupilUStat = RollingStat(), pupilVStat = RollingStat()
    private var baselines = Baselines()
    private var lastBaselineT = -Double.infinity

    private var smoothYaw: Float?, smoothPitch: Float?, smoothRoll: Float?
    private var smoothU: Float?, smoothV: Float?, smoothBrow: Float?, smoothBrowInner: Float?
    private var lastT: Double?
    private var frameIndex = 0
    private var heldFace: VNFaceObservation?
    private var badLandmarkFrames = 0

    private struct Baselines {
        var earL: Float?, earR: Float?, lip: Float?, brow: Float?, browInner: Float?
        var yaw: Float?, pitch: Float?, pupilU: Float?, pupilV: Float?
    }

    /// A face-aligned frame in upright pixel coordinates: origin between the eyes, x along the eye line
    /// (image-left -> image-right), y down, unit = 1 inter-ocular distance.
    private struct FaceFrame {
        var mid: SIMD2<Float>, ex: SIMD2<Float>, ey: SIMD2<Float>, iod: Float
        func align(_ p: SIMD2<Float>) -> SIMD2<Float> { let d = p - mid; return SIMD2(simd_dot(d, ex), simd_dot(d, ey)) / iod }
        func unalign(_ q: SIMD2<Float>) -> SIMD2<Float> { mid + (ex * q.x + ey * q.y) * iod }
    }

    init(config: Config = Config(), mode: Mode? = nil) {
        self.config = config
        self.mode = mode ?? (Self.landmarksAvailable ? .landmarks : .poseOnly)
        #if targetEnvironment(simulator)
        if mode == nil { self.config.poseOnlyDetectEvery = 2 }   // ~40 ms per CPU detection in the Simulator
        #endif
    }

    /// Fresh requests for every frame (cheap objects; keeps no state between frames).
    private static func makeRectRequest() -> VNDetectFaceRectanglesRequest {
        let rect = VNDetectFaceRectanglesRequest()
        rect.revision = VNDetectFaceRectanglesRequestRevision3
        #if targetEnvironment(simulator)
        // The Simulator can't create a GPU inference context for Vision ("Could not create inference context").
        forceCPU(rect)
        #endif
        return rect
    }

    private static func makeLandmarksRequest() -> VNDetectFaceLandmarksRequest {
        let landmarks = VNDetectFaceLandmarksRequest()
        landmarks.revision = VNDetectFaceLandmarksRequestRevision3
        landmarks.constellation = .constellation76Points
        #if targetEnvironment(simulator)
        forceCPU(landmarks)
        #endif
        return landmarks
    }

    #if targetEnvironment(simulator)
    private static func forceCPU(_ request: VNRequest) {
        if #available(iOS 17.0, *) {
            if let devices = try? request.supportedComputeStageDevices {
                for (stage, list) in devices {
                    if let cpu = list.first(where: { if case .cpu = $0 { return true } else { return false } }) {
                        request.setComputeDevice(cpu, for: stage)
                    }
                }
            }
        }
    }
    #endif

    func reset() {
        earL = RollingStat(); earR = RollingStat(); lipStat = RollingStat(); browStat = RollingStat()
        browInnerStat = RollingStat(); yawStat = RollingStat(); pitchStat = RollingStat()
        pupilUStat = RollingStat(); pupilVStat = RollingStat()
        baselines = Baselines(); lastBaselineT = -.infinity
        clearSmoothing(); lastT = nil; frameIndex = 0; heldFace = nil
    }

    private func clearSmoothing() {
        smoothYaw = nil; smoothPitch = nil; smoothRoll = nil; smoothU = nil; smoothV = nil
        smoothBrow = nil; smoothBrowInner = nil
    }

    // MARK: Analysis

    /// Analyses one frame. `orientation` maps the stored buffer to the upright picture (from the video
    /// track's `preferredTransform`). `t` is the sample time (seconds, the caller's clock).
    func analyze(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation, t: Double,
                 wantMesh: Bool = true, wantDebug: Bool = false) -> Result {
        let bufW = CVPixelBufferGetWidth(pixelBuffer), bufH = CVPixelBufferGetHeight(pixelBuffer)
        let rotated = orientation.swapsAxes
        let upright = CGSize(width: rotated ? bufH : bufW, height: rotated ? bufW : bufH)
        frameIndex += 1
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation)

        // Face rectangles (head pose). In `.poseOnly` they may be held for `poseOnlyDetectEvery - 1` frames.
        let face: VNFaceObservation?
        if mode == .poseOnly, config.poseOnlyDetectEvery > 1, frameIndex % config.poseOnlyDetectEvery != 0, let held = heldFace {
            face = held
        } else {
            let rect = Self.makeRectRequest()
            try? handler.perform([rect])
            face = (rect.results ?? []).max(by: { $0.boundingBox.width < $1.boundingBox.width })
            heldFace = face
        }
        guard let face else { return noFace(t) }

        let ctx = FrameContext(pixelBuffer: pixelBuffer, orientation: orientation, upright: upright,
                               bufW: Float(bufW), bufH: Float(bufH), t: t, wantMesh: wantMesh, wantDebug: wantDebug)
        if mode == .landmarks {
            let request = Self.makeLandmarksRequest()
            request.inputFaceObservations = [face]
            try? handler.perform([request])
            if let obs = request.results?.first, let result = landmarkResult(obs, pose: face, ctx: ctx) {
                badLandmarkFrames = 0
                return result
            }
            badLandmarkFrames += 1
            if badLandmarkFrames >= config.badLandmarkLimit { mode = .poseOnly; clearSmoothing() }
        }
        return poseOnlyResult(face, ctx: ctx)
    }

    private struct FrameContext {
        var pixelBuffer: CVPixelBuffer
        var orientation: CGImagePropertyOrientation
        var upright: CGSize
        var bufW: Float, bufH: Float
        var t: Double
        var wantMesh: Bool, wantDebug: Bool
    }

    // MARK: Landmarks mode

    /// nil if the landmarks are missing or implausible (the caller falls back to pose only).
    private func landmarkResult(_ obs: VNFaceObservation, pose: VNFaceObservation, ctx: FrameContext) -> Result? {
        guard let lm = obs.landmarks, let le = lm.leftEye, let re = lm.rightEye, le.pointCount >= 4, re.pointCount >= 4 else { return nil }
        let upright = ctx.upright
        func pts(_ r: VNFaceLandmarkRegion2D?) -> [SIMD2<Float>] {
            guard let r else { return [] }
            return r.pointsInImage(imageSize: upright).map { SIMD2(Float($0.x), Float(upright.height - $0.y)) }
        }
        // Image-left / image-right eyes (Vision's own left/right naming is not relied on).
        var eyeA = pts(le), eyeB = pts(re)
        var browA = pts(lm.leftEyebrow), browB = pts(lm.rightEyebrow)
        var pupilA = pts(lm.leftPupil).first, pupilB = pts(lm.rightPupil).first
        if Self.centroid(eyeA).x > Self.centroid(eyeB).x {
            swap(&eyeA, &eyeB); swap(&browA, &browB); swap(&pupilA, &pupilB)
        }
        let cA = Self.centroid(eyeA), cB = Self.centroid(eyeB)
        let iod = simd_length(cB - cA)
        let boxW = Float(pose.boundingBox.width * upright.width)
        // Sanity: a real face has its eyes 0.2-0.6 face-box widths apart.
        guard iod.isFinite, boxW > 1, iod > 4, iod / boxW > 0.2, iod / boxW < 0.6 else { return nil }
        let ex = (cB - cA) / iod
        let f = FaceFrame(mid: (cA + cB) / 2, ex: ex, ey: SIMD2<Float>(-ex.y, ex.x), iod: iod)

        let aEyeA = eyeA.map(f.align), aEyeB = eyeB.map(f.align)
        let earA = Self.aspectRatio(aEyeA), earB = Self.aspectRatio(aEyeB)
        let inner = pts(lm.innerLips).map(f.align)
        let lipGap = inner.isEmpty ? 0 : (inner.map(\.y).max()! - inner.map(\.y).min()!)
        let aBrowA = browA.map(f.align), aBrowB = browB.map(f.align)
        let aBrows = aBrowA + aBrowB
        // More sanity: eyes no taller than wide, lips and brows where a face has them.
        guard earA < 1.2, earB < 1.2, lipGap < 1.5, aBrows.allSatisfy({ abs($0.x) < 1.5 && $0.y > -1.2 && $0.y < 0.3 }) else { return nil }

        // In unmirrored footage the image-left eye is the user's right eye.
        let (earUserL, earUserR) = mirrored ? (earA, earB) : (earB, earA)
        let faceH = Float(obs.boundingBox.height * upright.height) / iod
        let lipRatio = faceH > 0 ? lipGap / faceH : 0
        // Eyebrow height above the eye line (positive), whole brow and inner ends.
        let browDist = aBrows.isEmpty ? 0 : -aBrows.map(\.y).reduce(0, +) / Float(aBrows.count)
        let innerBrow = (aBrowA.max(by: { $0.x < $1.x }).map { [$0] } ?? []) + (aBrowB.min(by: { $0.x < $1.x }).map { [$0] } ?? [])
        let browInnerDist = innerBrow.isEmpty ? browDist : -innerBrow.map(\.y).reduce(0, +) / Float(innerBrow.count)

        // Pupil offsets from each eye's corner midpoint, in eye half-widths (x: image right, y: image up).
        var us: [Float] = [], vs: [Float] = []
        for (eye, pupil) in [(aEyeA, pupilA.map(f.align)), (aEyeB, pupilB.map(f.align))] {
            guard let p = pupil, let l = eye.min(by: { $0.x < $1.x }), let r = eye.max(by: { $0.x < $1.x }) else { continue }
            let half = (r.x - l.x) / 2
            guard half > 0.05 else { continue }
            let c = (l + r) / 2
            us.append((p.x - c.x) / half)
            vs.append(-(p.y - c.y) / half)
        }
        let rawU = us.isEmpty ? 0 : us.reduce(0, +) / Float(us.count)
        let rawV = vs.isEmpty ? 0 : vs.reduce(0, +) / Float(vs.count)

        let t = ctx.t
        let dt = advanceClock(t)
        let pose = smoothedPose(pose, dt: dt, t: t, eyeLineRoll: -atan2(ex.y, ex.x))
        let uS = Self.ema(&smoothU, rawU, dt: dt, tau: config.eyeTau)
        let vS = Self.ema(&smoothV, rawV, dt: dt, tau: config.eyeTau)
        let browS = Self.ema(&smoothBrow, browDist, dt: dt, tau: config.browTau)
        let browInnerS = Self.ema(&smoothBrowInner, browInnerDist, dt: dt, tau: config.browTau)

        // Rolling statistics + baselines.
        let window = config.baselineSeconds
        earL.add(earUserL, t: t, window: window); earR.add(earUserR, t: t, window: window)
        lipStat.add(lipRatio, t: t, window: window)
        browStat.add(browS, t: t, window: window); browInnerStat.add(browInnerS, t: t, window: window)
        pupilUStat.add(uS, t: t, window: window); pupilVStat.add(vS, t: t, window: window)
        refreshBaselines(t)

        // Blendshape approximations.
        var s: [String: Float] = [:]
        func closure(_ ear: Float, _ base: Float?) -> Float {
            guard let base, base > 1e-3 else { return 0 }
            return Self.clamp01((config.openFraction - ear / base) / (config.openFraction - config.closedFraction))
        }
        func wide(_ ear: Float, _ base: Float?) -> Float {
            guard let base, base > 1e-3 else { return 0.05 }
            return Self.clamp01(0.05 + (ear / base - 1.05) / 0.3)
        }
        let blinkL = closure(earUserL, baselines.earL), blinkR = closure(earUserR, baselines.earR)
        s["eyeBlinkLeft"] = blinkL
        s["eyeBlinkRight"] = blinkR
        // Squint: partial lid closure (a blink's full closure is not a squint).
        s["eyeSquintLeft"] = Self.clamp01(0.1 + 0.5 * max(0, min(blinkL, 0.7) - 0.15))
        s["eyeSquintRight"] = Self.clamp01(0.1 + 0.5 * max(0, min(blinkR, 0.7) - 0.15))
        s["eyeWideLeft"] = wide(earUserL, baselines.earL)
        s["eyeWideRight"] = wide(earUserR, baselines.earR)

        // Eye-in-head direction (+ yaw = user's left, + pitch = up).
        let sign: Float = mirrored ? -1 : 1
        let eyeYaw = max(-0.6, min(0.6, sign * (uS - (baselines.pupilU ?? uS)) * config.eyeYawPerUnit))
        let eyePitch = max(-0.5, min(0.5, (vS - (baselines.pupilV ?? vS)) * config.eyePitchPerUnit))
        Self.setGaze(&s, yaw: eyeYaw, pitch: eyePitch)

        let jaw = Self.clamp01(0.03 + (lipRatio - (baselines.lip ?? lipRatio)) / config.jawFullRatio)
        s["jawOpen"] = jaw
        s["mouthClose"] = Self.clamp01(0.2 - 0.2 * jaw)
        if let b = baselines.brow, b > 1e-3 {
            let v = Self.clamp01(0.1 + (b - browS) / b / config.browDownSpan)
            s["browDownLeft"] = v; s["browDownRight"] = v
        } else {
            s["browDownLeft"] = 0.1; s["browDownRight"] = 0.1
        }
        if let b = baselines.browInner, b > 1e-3 {
            s["browInnerUp"] = Self.clamp01(0.1 + (browInnerS - b) / b / config.browUpSpan)
        } else {
            s["browInnerUp"] = 0.1
        }
        let look = SIMD3<Float>(sin(eyeYaw) * cos(eyePitch), sin(eyePitch), cos(eyeYaw) * cos(eyePitch))

        // rPPG skin ROIs: forehead above the brows, cheeks below the eyes beside the nose, kept inside the contour.
        let browTop = aBrows.map(\.y).min() ?? -0.35
        let contour = pts(lm.faceContour).map(f.align)
        let rgb = skinRGB(frame: f, browTop: browTop, contourX: contour.isEmpty ? nil : (contour.map(\.x).min()!, contour.map(\.x).max()!), ctx: ctx)

        var mesh: [CGPoint] = []
        if ctx.wantMesh, let all = lm.allPoints {
            let w = Float(upright.width), h = Float(upright.height)
            mesh = all.pointsInImage(imageSize: upright).map { CGPoint(x: CGFloat(Float($0.x) / w), y: CGFloat(1 - Float($0.y) / h)) }
        }
        let sample = FaceSample(t: t, faceDetected: true, blendShapes: s, lookAtPoint: look,
                                headYaw: pose.yaw, headPitch: pose.pitch, headRoll: pose.roll, skinRGB: rgb)
        let debug = ctx.wantDebug ? Debug(earLeft: earUserL, earRight: earUserR, earBaselineLeft: baselines.earL ?? 0,
                                          earBaselineRight: baselines.earR ?? 0, lipRatio: lipRatio, browDistance: browDist,
                                          rawYaw: pose.rawYaw, rawPitch: pose.rawPitch, pupilU: rawU, pupilV: rawV) : nil
        return Result(sample: sample, mesh: mesh, debug: debug)
    }

    // MARK: Pose-only mode

    private func poseOnlyResult(_ face: VNFaceObservation, ctx: FrameContext) -> Result {
        let up = ctx.upright
        let bb = face.boundingBox
        let boxW = Float(bb.width * up.width), boxH = Float(bb.height * up.height)
        guard boxW > 4, boxH > 4 else { return noFace(ctx.t) }
        let visionRoll = face.roll?.floatValue ?? 0
        // Box centre and axes (top-left origin); the eye line sits `eyeLinePerBoxHeight` below the box top.
        let center = SIMD2<Float>(Float(bb.midX * up.width), Float((1 - bb.midY) * up.height))
        let ex = SIMD2<Float>(cos(-visionRoll), sin(-visionRoll))
        let ey = SIMD2<Float>(-ex.y, ex.x)
        let mid = center + ey * ((config.eyeLinePerBoxHeight - 0.5) * boxH)
        let f = FaceFrame(mid: mid, ex: ex, ey: ey, iod: config.iodPerBoxWidth * boxW)

        let t = ctx.t
        let dt = advanceClock(t)
        let pose = smoothedPose(face, dt: dt, t: t, eyeLineRoll: visionRoll)
        refreshBaselines(t)

        var s = Self.neutralShapes()
        Self.setGaze(&s, yaw: 0, pitch: 0)
        let mouthY = (config.mouthPerBoxHeight - config.eyeLinePerBoxHeight) * boxH / f.iod
        let halfW = 0.5 * boxW / f.iod
        let rgb = skinRGB(frame: f, browTop: -0.35, contourX: (-halfW, halfW), ctx: ctx)

        var mesh: [CGPoint] = []
        if ctx.wantMesh {
            let w = Float(up.width), h = Float(up.height)
            func norm(_ p: SIMD2<Float>) -> CGPoint { CGPoint(x: CGFloat(p.x / w), y: CGFloat(p.y / h)) }
            let n = 32
            for k in 0..<n {   // face-box ellipse
                let a = Float(k) / Float(n) * 2 * .pi
                mesh.append(norm(center + ex * (cos(a) * boxW / 2) + ey * (sin(a) * boxH / 2)))
            }
            mesh.append(norm(f.unalign([-0.5, 0]))); mesh.append(norm(f.unalign([0.5, 0])))   // eye centres
            mesh.append(norm(f.unalign([0, mouthY])))                                         // mouth
        }
        let sample = FaceSample(t: t, faceDetected: true, blendShapes: s, lookAtPoint: SIMD3(0, 0, 1),
                                headYaw: pose.yaw, headPitch: pose.pitch, headRoll: pose.roll, skinRGB: rgb)
        let debug = ctx.wantDebug ? Debug(earLeft: 0, earRight: 0, earBaselineLeft: 0, earBaselineRight: 0, lipRatio: 0,
                                          browDistance: 0, rawYaw: pose.rawYaw, rawPitch: pose.rawPitch, pupilU: 0, pupilV: 0) : nil
        return Result(sample: sample, mesh: mesh, debug: debug)
    }

    // MARK: Shared pieces

    private func advanceClock(_ t: Double) -> Double {
        let dt = lastT.map { max(0, t - $0) } ?? 0
        lastT = t
        return dt
    }

    private static func ema(_ s: inout Float?, _ x: Float, dt: Double, tau: Double) -> Float {
        guard let prev = s, dt > 0, dt < 0.5 else { s = x; return x }
        let a = Float(1 - exp(-dt / tau))
        let v = prev + a * (x - prev)
        s = v
        return v
    }

    /// Vision pose re-signed (yaw + = user's left, pitch + = up, roll + = counter-clockwise), smoothed, centred.
    private func smoothedPose(_ face: VNFaceObservation, dt: Double, t: Double, eyeLineRoll: Float)
        -> (yaw: Float, pitch: Float, roll: Float, rawYaw: Float, rawPitch: Float) {
        // Vision yaw + = nose toward the image's right (= user's left when unmirrored); Vision pitch + = head down.
        let sign: Float = mirrored ? -1 : 1
        let rawYaw = sign * (face.yaw?.floatValue ?? 0)
        let rawPitch = -(face.pitch?.floatValue ?? 0)
        let rawRoll = sign * eyeLineRoll
        let yawS = Self.ema(&smoothYaw, rawYaw, dt: dt, tau: config.poseTau)
        let pitchS = Self.ema(&smoothPitch, rawPitch, dt: dt, tau: config.poseTau)
        let rollS = Self.ema(&smoothRoll, rawRoll, dt: dt, tau: config.poseTau)
        yawStat.add(yawS, t: t, window: config.baselineSeconds)
        pitchStat.add(pitchS, t: t, window: config.baselineSeconds)
        var yaw = yawS, pitch = pitchS
        if config.centerPose {
            yaw -= baselines.yaw ?? yawS
            pitch -= baselines.pitch ?? pitchS
        }
        return (yaw, pitch, rollS, rawYaw, rawPitch)
    }

    private func refreshBaselines(_ t: Double) {
        guard t - lastBaselineT >= config.baselineRefresh || t < lastBaselineT else { return }
        lastBaselineT = t
        let n = config.minBaselineSamples
        baselines.earL = earL.percentile(config.earBaselinePercentile, minCount: n)
        baselines.earR = earR.percentile(config.earBaselinePercentile, minCount: n)
        baselines.lip = lipStat.percentile(0.2, minCount: n)
        baselines.brow = browStat.percentile(0.5, minCount: n)
        baselines.browInner = browInnerStat.percentile(0.5, minCount: n)
        baselines.yaw = yawStat.percentile(0.5, minCount: n)
        baselines.pitch = pitchStat.percentile(0.5, minCount: n)
        baselines.pupilU = pupilUStat.percentile(0.5, minCount: n)
        baselines.pupilV = pupilVStat.percentile(0.5, minCount: n)
    }

    /// Forehead (above `browTop`) and both cheeks, in the aligned frame, averaged in buffer coordinates.
    private func skinRGB(frame f: FaceFrame, browTop: Float, contourX: (Float, Float)?, ctx: FrameContext) -> SIMD3<Float>? {
        guard Self.isSkinCapable(ctx.pixelBuffer) else { return nil }
        var polys: [[SIMD2<Float>]] = [
            [[-0.42, browTop - 0.42], [0.42, browTop - 0.42], [0.42, browTop - 0.08], [-0.42, browTop - 0.08]],
            [[-0.72, 0.32], [-0.28, 0.32], [-0.28, 0.72], [-0.72, 0.72]],
            [[0.28, 0.32], [0.72, 0.32], [0.72, 0.72], [0.28, 0.72]],
        ]
        if let (minX, maxX) = contourX {
            polys[1] = polys[1].map { SIMD2(max($0.x, minX + 0.08), $0.y) }
            polys[2] = polys[2].map { SIMD2(min($0.x, maxX - 0.08), $0.y) }
        }
        let buffer = polys.map { $0.map { Self.bufferPoint(f.unalign($0), orientation: ctx.orientation, bufW: ctx.bufW, bufH: ctx.bufH) } }
        return averager.average(pixelBuffer: ctx.pixelBuffer, polygons: buffer)
    }

    /// Neutral blendshapes (same resting values as the synthetic generator).
    static func neutralShapes() -> [String: Float] {
        var s: [String: Float] = [:]
        for name in FaceTrack.blendShapeNames { s[name] = 0 }
        s["eyeSquintLeft"] = 0.1; s["eyeSquintRight"] = 0.1
        s["eyeWideLeft"] = 0.05; s["eyeWideRight"] = 0.05
        s["browDownLeft"] = 0.1; s["browDownRight"] = 0.1; s["browInnerUp"] = 0.1
        s["mouthClose"] = 0.2; s["jawOpen"] = 0.03
        return s
    }

    private static func setGaze(_ s: inout [String: Float], yaw: Float, pitch: Float) {
        s["eyeLookOutLeft"] = clamp01(yaw * 1.2); s["eyeLookInRight"] = clamp01(yaw * 1.2)
        s["eyeLookInLeft"] = clamp01(-yaw * 1.2); s["eyeLookOutRight"] = clamp01(-yaw * 1.2)
        s["eyeLookUpLeft"] = clamp01(pitch * 1.5); s["eyeLookUpRight"] = clamp01(pitch * 1.5)
        s["eyeLookDownLeft"] = clamp01(-pitch * 1.5); s["eyeLookDownRight"] = clamp01(-pitch * 1.5)
    }

    private func noFace(_ t: Double) -> Result {
        clearSmoothing()
        return Result(sample: FaceSample(t: t, faceDetected: false), mesh: [], debug: nil)
    }


    // MARK: Geometry helpers

    static func clamp01(_ x: Float) -> Float { x.isFinite ? max(0, min(1, x)) : 0 }

    static func centroid(_ p: [SIMD2<Float>]) -> SIMD2<Float> {
        p.isEmpty ? .zero : p.reduce(.zero, +) / Float(p.count)
    }

    /// Lid opening / eye width of an eye contour in aligned coordinates.
    static func aspectRatio(_ p: [SIMD2<Float>]) -> Float {
        guard let minX = p.map(\.x).min(), let maxX = p.map(\.x).max(),
              let minY = p.map(\.y).min(), let maxY = p.map(\.y).max(), maxX - minX > 1e-4 else { return 0 }
        return (maxY - minY) / (maxX - minX)
    }

    static func distance(_ a: CGRect, _ b: CGRect) -> CGFloat { hypot(a.midX - b.midX, a.midY - b.midY) }

    static func isSkinCapable(_ pb: CVPixelBuffer) -> Bool {
        RPPGSkinPixelAverager.supports(CVPixelBufferGetPixelFormatType(pb))
    }

    /// Maps a point of the upright picture (pixels, origin top-left) to stored-buffer pixel coordinates.
    static func bufferPoint(_ p: SIMD2<Float>, orientation: CGImagePropertyOrientation, bufW: Float, bufH: Float) -> SIMD2<Float> {
        let u = p.x, v = p.y
        switch orientation {
        case .up: return SIMD2(u, v)
        case .upMirrored: return SIMD2(bufW - 1 - u, v)
        case .down: return SIMD2(bufW - 1 - u, bufH - 1 - v)
        case .downMirrored: return SIMD2(u, bufH - 1 - v)
        case .right: return SIMD2(v, bufH - 1 - u)
        case .rightMirrored: return SIMD2(bufW - 1 - v, bufH - 1 - u)
        case .left: return SIMD2(bufW - 1 - v, u)
        case .leftMirrored: return SIMD2(v, u)
        @unknown default: return SIMD2(u, v)
        }
    }

    /// Orientation that turns a video track's stored frames upright, from its `preferredTransform`.
    static func orientation(for transform: CGAffineTransform) -> CGImagePropertyOrientation {
        let a = transform.a, b = transform.b, c = transform.c, d = transform.d
        func near(_ x: CGFloat, _ y: CGFloat) -> Bool { abs(x - y) < 0.01 }
        if near(a, 0), near(b, 1), near(c, -1), near(d, 0) { return .right }
        if near(a, 0), near(b, -1), near(c, 1), near(d, 0) { return .left }
        if near(a, -1), near(b, 0), near(c, 0), near(d, -1) { return .down }
        if near(a, -1), near(b, 0), near(c, 0), near(d, 1) { return .upMirrored }
        if near(a, 1), near(b, 0), near(c, 0), near(d, -1) { return .downMirrored }
        if near(a, 0), near(b, 1), near(c, 1), near(d, 0) { return .leftMirrored }
        if near(a, 0), near(b, -1), near(c, -1), near(d, 0) { return .rightMirrored }
        return .up
    }
}

extension CGImagePropertyOrientation {
    /// True when the upright picture is the stored buffer turned by 90 degrees.
    var swapsAxes: Bool {
        switch self {
        case .left, .right, .leftMirrored, .rightMirrored: return true
        default: return false
        }
    }
}

/// Values within a trailing time window, with order statistics. Small (<= ~1500 entries at 30 Hz).
struct RollingStat {
    private var items: [(t: Double, v: Float)] = []
    private var head = 0

    mutating func add(_ v: Float, t: Double, window: Double) {
        guard v.isFinite else { return }
        if let last = items.last, t < last.t { items.removeAll(); head = 0 }   // clock went backwards (loop)
        items.append((t, v))
        while head < items.count, items[head].t < t - window { head += 1 }
        if head > 512 { items.removeFirst(head); head = 0 }
    }

    var count: Int { items.count - head }

    func percentile(_ q: Float, minCount: Int) -> Float? {
        guard count >= max(1, minCount) else { return nil }
        let sorted = items[head...].map(\.v).sorted()
        let i = Int((Float(sorted.count - 1) * q).rounded())
        return sorted[max(0, min(sorted.count - 1, i))]
    }
}
