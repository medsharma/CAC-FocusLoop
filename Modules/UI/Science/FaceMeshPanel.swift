import SwiftUI

/// Stylised SYNTHETIC face landmarks driven by replayed feature values (blinks, jaw, gaze, head pose).
/// Used only in Replay so the Science view shows a face-mesh animation; always labeled SYNTHETIC.
public enum SyntheticFaceMesh {
    public struct Inputs: Equatable {
        public var blinkL: Double = 0, blinkR: Double = 0, jaw: Double = 0, brow: Double = 0
        public var gazeYaw: Double = 0, gazePitch: Double = 0
        public var yaw: Double = 0, pitch: Double = 0, roll: Double = 0
        public init() {}
        public init(frame f: FeatureFrame) {
            blinkL = Double(f.eyeBlinkLeft); blinkR = Double(f.eyeBlinkRight); jaw = Double(f.jawOpen)
            brow = Double((f.browDownLeft + f.browDownRight) / 2)
            gazeYaw = Double(f.gazeYaw); gazePitch = Double(f.gazePitch)
            yaw = Double(f.headYaw); pitch = Double(f.headPitch); roll = Double(f.headRoll)
        }
    }

    /// Polylines (each closed or open) in normalized 0...1 coordinates, plus loose mesh vertices.
    public struct Shape {
        public var loops: [[CGPoint]]
        public var lines: [[CGPoint]]
        public var vertices: [CGPoint]
        public var pupils: [CGPoint]
    }

    public static func shape(_ i: Inputs, t: Double) -> Shape {
        let c = CGPoint(x: 0.5, y: 0.5)
        func ellipse(_ cx: Double, _ cy: Double, _ rx: Double, _ ry: Double, _ n: Int, depth: Double) -> [CGPoint] {
            (0..<n).map { k in
                let a = Double(k) / Double(n) * 2 * .pi
                return project(cx + rx * cos(a), cy + ry * sin(a), depth: depth)
            }
        }
        func project(_ x: Double, _ y: Double, depth: Double) -> CGPoint {
            // Simple parallax: nearer features (depth 1) shift more with head yaw/pitch; then roll about centre.
            let px = x + sin(i.yaw) * 0.22 * depth - (x - 0.5) * (1 - cos(i.yaw)) * 0.8
            let py = y - sin(i.pitch) * 0.2 * depth
            let dx = px - c.x, dy = py - c.y
            let cr = cos(i.roll), sr = sin(i.roll)
            return CGPoint(x: c.x + dx * cr - dy * sr, y: c.y + dx * sr + dy * cr)
        }
        let breathe = 0.003 * sin(t * 1.3)
        var loops: [[CGPoint]] = []
        // Face oval + concentric mesh rings.
        for (k, s) in [1.0, 0.78, 0.55].enumerated() {
            loops.append(ellipse(0.5, 0.53, 0.29 * s + breathe, 0.37 * s + breathe, [40, 30, 22][k], depth: 0.4 + 0.3 * Double(k)))
        }
        // Eyes (close with blinks).
        let eyeL = ellipse(0.39, 0.45, 0.065, max(0.003, 0.028 * (1 - i.blinkL)), 16, depth: 0.9)
        let eyeR = ellipse(0.61, 0.45, 0.065, max(0.003, 0.028 * (1 - i.blinkR)), 16, depth: 0.9)
        loops += [eyeL, eyeR]
        // Mouth (opens with jaw).
        loops.append(ellipse(0.5, 0.71 + 0.02 * i.jaw, 0.085, 0.012 + 0.055 * i.jaw, 18, depth: 0.9))
        // Brows (lower with tension).
        let bDrop = 0.03 * i.brow
        func brow(_ cx: Double, _ dir: Double) -> [CGPoint] {
            (0..<6).map { k in
                let u = Double(k) / 5
                return project(cx + (u - 0.5) * 0.13 * dir, 0.385 + bDrop - 0.018 * sin(u * .pi), depth: 0.9)
            }
        }
        // Nose bridge + base.
        let nose = [(0.5, 0.44), (0.5, 0.5), (0.5, 0.57), (0.47, 0.6), (0.5, 0.615), (0.53, 0.6)].map { project($0.0, $0.1, depth: 1.1) }
        let lines = [brow(0.39, 1), brow(0.61, 1), nose]
        // Pupils follow gaze.
        let gx = max(-0.035, min(0.035, -i.gazeYaw * 0.06)), gy = max(-0.012, min(0.012, -i.gazePitch * 0.05))
        let pupils = [project(0.39 + gx, 0.45 + gy, depth: 1), project(0.61 + gx, 0.45 + gy, depth: 1)]
        // Loose vertices across the cheeks / forehead for a mesh feel.
        var verts: [CGPoint] = []
        for row in 0..<7 {
            for col in 0..<7 {
                let u = Double(col) / 6, v = Double(row) / 6
                let x = 0.5 + (u - 0.5) * 0.46 * (1 - 0.35 * pow(v - 0.45, 2) * 4)
                let y = 0.24 + v * 0.6
                let e = pow((x - 0.5) / 0.29, 2) + pow((y - 0.53) / 0.37, 2)
                if e < 0.92 { verts.append(project(x, y, depth: 0.7)) }
            }
        }
        return Shape(loops: loops, lines: lines, vertices: verts, pupils: pupils)
    }
}

/// Face-mesh panel: live landmarks from the services (device), drawn over the recorded video when the session is
/// driven by one, or the stylised SYNTHETIC animation (Replay). Framed as a Quiet Dark card.
struct FaceMeshPanel: View {
    let model: LiveSessionModel
    let services: UIServices
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var shape: RoundedRectangle { RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous) }

    var body: some View {
        // Read observable session state so the panel re-renders when a recorded-video session starts.
        let video = (model.isRecorded || model.phase == .running) ? services.faceVideoView() : nil
        ZStack(alignment: .topLeading) {
            FLColor.surface
            if let video {
                // The live video the pipeline is analysing (never stored), aspect-filled under the mesh.
                Color.clear
                    .overlay { video }
                    .clipped()
                    .overlay(FLColor.bg.opacity(0.35))
                    .accessibilityHidden(true)
            } else {
                RadialGradient(colors: [FLColor.accent.opacity(0.05), .clear], center: .center, startRadius: 0, endRadius: 160)
                    .accessibilityHidden(true)
            }
            TimelineView(.animation(minimumInterval: 1.0 / 30)) { ctx in
                let t = ctx.date.timeIntervalSinceReferenceDate
                let live = services.faceMeshPoints()
                Canvas { gc, size in
                    let rect = CGRect(origin: .zero, size: size).insetBy(dx: 16, dy: 16)
                    if video == nil { drawGrid(&gc, size: size) }
                    if let pts = live, !pts.isEmpty {
                        // On a recorded video, draw the mesh exactly on the face (same aspect-fill as the video).
                        let mapped = video != nil && services.faceVideoAspectRatio() != nil
                            ? FLAspectFill.map(pts, videoAspect: services.faceVideoAspectRatio()!, into: size)
                            : FaceMeshMapper.map(pts, into: rect)
                        for p in mapped {
                            gc.fill(Path(ellipseIn: CGRect(x: p.x - 0.9, y: p.y - 0.9, width: 1.8, height: 1.8)), with: .color(FLColor.accent.opacity(0.9)))
                        }
                    } else if model.isReplay, let f = model.latestFrame, f.faceDetected > 0.5 {
                        drawSynthetic(&gc, rect: rect, inputs: .init(frame: f), t: t)
                    }
                    if !reduceMotion {
                        let y = (t.truncatingRemainder(dividingBy: 4) / 4) * size.height
                        gc.fill(Path(CGRect(x: 0, y: y - 24, width: size.width, height: 48)),
                                with: .linearGradient(Gradient(colors: [.clear, FLColor.accent.opacity(0.06), .clear]),
                                                      startPoint: CGPoint(x: 0, y: y - 24), endPoint: CGPoint(x: 0, y: y + 24)))
                        var line = Path(); line.move(to: CGPoint(x: 0, y: y)); line.addLine(to: CGPoint(x: size.width, y: y))
                        gc.stroke(line, with: .color(FLColor.accent.opacity(0.18)), lineWidth: 0.5)
                    }
                }
                .overlay {
                    if live == nil && video == nil && !(model.isReplay && (model.latestFrame?.faceDetected ?? 0) > 0.5) {
                        Text("No face in view").flText(.caption)
                    }
                }
            }
        }
        .clipShape(shape)
        .overlay(shape.strokeBorder(FLColor.hairline, lineWidth: 0.5))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(video != nil ? "Your recorded video with live face-mesh landmarks"
                            : model.isReplay ? "Synthetic face-mesh animation driven by replayed signals" : "Live face-mesh landmarks")
    }

    private func drawGrid(_ gc: inout GraphicsContext, size: CGSize) {
        var p = Path()
        let step: CGFloat = 20
        var x: CGFloat = step
        while x < size.width { p.move(to: CGPoint(x: x, y: 0)); p.addLine(to: CGPoint(x: x, y: size.height)); x += step }
        var y: CGFloat = step
        while y < size.height { p.move(to: CGPoint(x: 0, y: y)); p.addLine(to: CGPoint(x: size.width, y: y)); y += step }
        gc.stroke(p, with: .color(.white.opacity(0.03)), lineWidth: 0.5)
    }

    private func drawSynthetic(_ gc: inout GraphicsContext, rect: CGRect, inputs: SyntheticFaceMesh.Inputs, t: Double) {
        let s = SyntheticFaceMesh.shape(inputs, t: t)
        func m(_ pts: [CGPoint]) -> [CGPoint] { FaceMeshMapper.map(pts, into: rect) }
        let tint = FLColor.accent
        let verts = m(s.vertices)
        // Light triangulation between neighbours for a mesh look.
        var mesh = Path()
        for (i, a) in verts.enumerated() {
            for b in verts[(i + 1)...].prefix(8) where hypot(a.x - b.x, a.y - b.y) < rect.width * 0.1 {
                mesh.move(to: a); mesh.addLine(to: b)
            }
        }
        gc.stroke(mesh, with: .color(tint.opacity(0.10)), lineWidth: 0.5)
        for v in verts { gc.fill(Path(ellipseIn: CGRect(x: v.x - 0.8, y: v.y - 0.8, width: 1.6, height: 1.6)), with: .color(tint.opacity(0.45))) }
        for (k, loop) in s.loops.enumerated() {
            var p = Path(); let q = m(loop)
            p.addLines(q); p.closeSubpath()
            gc.stroke(p, with: .color(tint.opacity(k < 3 ? 0.30 - 0.07 * Double(k) : 0.85)), lineWidth: k < 3 ? 0.6 : 1)
            if k >= 3 {
                for pt in q { gc.fill(Path(ellipseIn: CGRect(x: pt.x - 0.9, y: pt.y - 0.9, width: 1.8, height: 1.8)), with: .color(tint)) }
            }
        }
        for line in s.lines {
            var p = Path(); p.addLines(m(line))
            gc.stroke(p, with: .color(tint.opacity(0.7)), style: StrokeStyle(lineWidth: 1, lineCap: .round, lineJoin: .round))
        }
        for p in m(s.pupils) {
            gc.fill(Path(ellipseIn: CGRect(x: p.x - 2.5, y: p.y - 2.5, width: 5, height: 5)), with: .color(FLColor.textPrimary))
            gc.stroke(Path(ellipseIn: CGRect(x: p.x - 5.5, y: p.y - 5.5, width: 11, height: 11)), with: .color(FLColor.textPrimary.opacity(0.35)), lineWidth: 0.5)
        }
    }
}

/// Thin (2pt) ring meter with a light numeral inside and a caption label below.
/// Used for blink rate, PERCLOS and gaze deviation.
struct FLRingGauge: View {
    var title: String
    var value: Double          // 0...1
    var valueText: String
    var unit: String
    var color: Color
    static let diameter: CGFloat = 76
    var body: some View {
        VStack(spacing: FLSpacing.xs) {
            ZStack {
                Circle().stroke(FLColor.hairlineStrong, lineWidth: 2)
                Circle().trim(from: 0, to: max(0.004, value))
                    .stroke(color, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.easeOut(duration: 0.4), value: value)
                VStack(spacing: 0) {
                    Text(valueText).font(FLFont.scaled(22, weight: .light, relativeTo: .title2, monospacedDigits: true))
                        .tracking(-0.5)
                        .foregroundStyle(FLColor.textPrimary)
                        .lineLimit(1).minimumScaleFactor(0.6)
                    Text(unit).font(FLFont.scaled(10, weight: .regular, relativeTo: .caption2))
                        .foregroundStyle(FLColor.textTertiary).lineLimit(1).minimumScaleFactor(0.8)
                }
                .padding(.horizontal, 8)
            }
            .frame(width: Self.diameter, height: Self.diameter)
            GaugeLabel(title)
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title): \(valueText) \(unit)")
    }
}

/// Two-line centred caption under a gauge.
private struct GaugeLabel: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text).font(FLFont.caption).foregroundStyle(FLColor.textSecondary)
            .multilineTextAlignment(.center)
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// Head-pose crosshair in a thin ring: dot = yaw/pitch, tilted line = roll.
struct HeadPoseGauge: View {
    var yaw: Float, pitch: Float, roll: Float
    var body: some View {
        let d = FLRingGauge.diameter
        VStack(spacing: FLSpacing.xs) {
            ZStack {
                Circle().stroke(FLColor.hairlineStrong, lineWidth: 2)
                Circle().stroke(FLColor.hairline, lineWidth: 0.5).scaleEffect(0.5)
                Rectangle().fill(FLColor.hairline).frame(width: 0.5)
                Rectangle().fill(FLColor.hairline).frame(height: 0.5)
                Capsule().fill(FLColor.textTertiary).frame(width: d * 0.5, height: 1)
                    .rotationEffect(.radians(Double(roll)))
                Circle().fill(FLColor.accent).frame(width: 7, height: 7)
                    .offset(x: CGFloat(GaugeScale.headAngle(-yaw) - 0.5) * d * 0.8,
                            y: CGFloat(GaugeScale.headAngle(-pitch) - 0.5) * d * 0.8)
                    .animation(.easeOut(duration: 0.2), value: yaw)
            }
            .frame(width: d, height: d)
            GaugeLabel("Head pose")
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Head pose: yaw \(UIFormat.degrees(yaw)), pitch \(UIFormat.degrees(pitch)), roll \(UIFormat.degrees(roll))")
    }
}
