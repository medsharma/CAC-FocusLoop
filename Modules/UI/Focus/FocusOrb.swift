import SwiftUI

/// The signature focus orb: a soft luminous core, two hairline rings and a halo.
/// Colour = attention state (blended by probability). Motion = state: slow breathing when focused, a restless drift
/// when distracted, slow sinking when drowsy, a quicker pulse when stressed. Halo intensity = predicted lapse risk,
/// shifting toward amber as risk rises. Fully static under Reduce Motion. Drawn with gradients only (no blur filters).
public struct FocusOrb: View {
    /// State probabilities in `AttentionState.allCases` order (blended so transitions are smooth).
    var probabilities: [Float]
    /// P(lapse within the horizon), 0...1.
    var lapseRisk: Float
    var riskAlert: Bool
    var paused: Bool

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var smoother = OrbSmoother()

    /// When false the luminous core is omitted (a face video sits inside the rings instead).
    var showsCore = true

    public init(probabilities: [Float], lapseRisk: Float, riskAlert: Bool = false, paused: Bool = false,
                showsCore: Bool = true) {
        self.probabilities = probabilities; self.lapseRisk = lapseRisk; self.riskAlert = riskAlert; self.paused = paused
        self.showsCore = showsCore
    }

    public var body: some View {
        // The canvas overdraws its layout frame (halo and rings extend past the orb) so nothing is clipped square.
        GeometryReader { g in
            TimelineView(.animation(minimumInterval: 1.0 / 30, paused: paused || reduceMotion)) { ctx in
                Canvas { gc, size in
                    let now = ctx.date.timeIntervalSinceReferenceDate
                    smoother.advance(to: probabilities, risk: lapseRisk, now: now, snap: reduceMotion)
                    draw(&gc, size: size, t: reduceMotion ? 0 : now)
                }
            }
            .frame(width: g.size.width * Self.overdraw, height: g.size.height * Self.overdraw)
            .position(x: g.size.width / 2, y: g.size.height / 2)
            .allowsHitTesting(false)
        }
        .accessibilityElement()
        .accessibilityLabel(accessibilityText)
    }

    static let overdraw: CGFloat = 1.7

    private var accessibilityText: String {
        let i = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
        let s = AttentionState.allCases[min(i, 3)]
        return "Focus orb. \(s.displayName). Lapse risk \(UIFormat.percent(Double(lapseRisk)))."
    }

    private func draw(_ gc: inout GraphicsContext, size: CGSize, t: Double) {
        let w = smoother.weights
        let motion = OrbMotion.blend(probabilities: w.map(Float.init))
        let color = blendedColor(w)
        let rm = reduceMotion
        let side = min(size.width, size.height) / Self.overdraw
        let baseR = side * 0.30
        let center = CGPoint(x: size.width / 2, y: size.height / 2)

        // Breathing (accumulated phase so frequency changes don't jump). Static mid-breath under Reduce Motion.
        let phase = smoother.phase
        let breath: Double
        if rm {
            breath = 0.5
        } else if motion.period < 1.5 {
            // Heartbeat-like double bump for the stressed pulse.
            let x = phase.truncatingRemainder(dividingBy: 1)
            breath = exp(-pow((x - 0.15) / 0.07, 2)) + 0.6 * exp(-pow((x - 0.38) / 0.08, 2))
        } else {
            breath = 0.5 + 0.5 * sin(phase * 2 * .pi)
        }
        let scale = 1 + motion.amplitude * (breath * 2 - 1)
        // The outer ring follows the breath a little later and less, so the orb feels layered.
        let lagBreath = rm ? 0.5 : 0.5 + 0.5 * sin((phase - 0.12) * 2 * .pi)

        // Restless drift (incommensurate sines) and slow sinking, both scaled down to stay restrained.
        let j = rm ? 0 : motion.jitter * 0.6
        let jx = j * (sin(t * 3.1) * 0.5 + sin(t * 5.3 + 1) * 0.3 + sin(t * 1.7 + 2) * 0.4)
        let jy = j * (cos(t * 2.7) * 0.5 + sin(t * 4.9 + 0.5) * 0.3 + cos(t * 1.3 + 1) * 0.3)
        let sinkCycle = 0.5 - 0.5 * cos(t * 2 * .pi / 9)
        let sy = rm ? 0 : motion.sink * 0.6 * sinkCycle
        let c = CGPoint(x: center.x + jx, y: center.y + jy + sy)
        let r = baseR * scale

        // 1. Halo: opacity = lapse risk; hue drifts from the state colour toward amber as risk rises.
        let risk = smoother.risk
        let haloOpacity = RiskHalo.opacity(probability: Float(risk))
        let haloColor = mix(color, FLColor.risk, Double(min(1, max(0, risk))) * 0.8)
        let alertPulse = riskAlert && !rm ? 1 + 0.06 * sin(t * 2.4) : 1
        let haloR = baseR * RiskHalo.scale(probability: Float(risk)) * 1.95 * alertPulse
        gc.fill(circle(center, haloR),
                with: .radialGradient(Gradient(stops: [
                    .init(color: haloColor.opacity(haloOpacity * 0.55), location: 0),
                    .init(color: haloColor.opacity(haloOpacity * 0.32), location: 0.42),
                    .init(color: haloColor.opacity(haloOpacity * 0.10), location: 0.72),
                    .init(color: haloColor.opacity(0), location: 1),
                ]), center: center, startRadius: 0, endRadius: haloR))

        // 2. Rings: an outer hairline, and an inner ring with a slow travelling highlight.
        let outerR = baseR * (1.62 + 0.05 * (lagBreath * 2 - 1))
        gc.stroke(circle(c, outerR), with: .color(color.opacity(0.16)), lineWidth: 0.75)

        let innerR = r * 1.26
        gc.stroke(circle(c, innerR), with: .color(color.opacity(0.22)), lineWidth: 0.75)
        let spin = Angle.radians(t * 2 * .pi / 28)
        gc.stroke(circle(c, innerR),
                  with: .conicGradient(Gradient(stops: [
                      .init(color: color.opacity(0), location: 0),
                      .init(color: color.opacity(0.0), location: 0.55),
                      .init(color: color.opacity(0.75), location: 0.92),
                      .init(color: color.opacity(0), location: 1),
                  ]), center: c, angle: spin),
                  lineWidth: 1)

        // 3. Core: a soft luminous sphere, brightest slightly above centre, feathered edge.
        guard showsCore else { return }
        let glow = min(1.1, motion.glow)
        gc.fill(circle(c, r * 1.08),
                with: .radialGradient(Gradient(stops: [
                    .init(color: color.opacity(0.62 * glow), location: 0),
                    .init(color: color.opacity(0.42 * glow), location: 0.55),
                    .init(color: color.opacity(0.14 * glow), location: 0.84),
                    .init(color: color.opacity(0), location: 1),
                ]), center: c, startRadius: 0, endRadius: r * 1.08))
        let hc = CGPoint(x: c.x, y: c.y - r * 0.12)
        gc.fill(circle(hc, r * 0.78),
                with: .radialGradient(Gradient(stops: [
                    .init(color: mix(color, .white, 0.6).opacity(min(1, 1.0 * glow)), location: 0),
                    .init(color: color.opacity(0.7 * glow), location: 0.42),
                    .init(color: color.opacity(0), location: 1),
                ]), center: hc, startRadius: 0, endRadius: r * 0.78))
    }

    private func circle(_ c: CGPoint, _ r: CGFloat) -> Path {
        Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: r * 2, height: r * 2))
    }

    private func mix(_ a: Color, _ b: Color, _ t: Double) -> Color {
        var ar: CGFloat = 0, ag: CGFloat = 0, ab: CGFloat = 0, aa: CGFloat = 0
        var br: CGFloat = 0, bg: CGFloat = 0, bb: CGFloat = 0, ba: CGFloat = 0
        UIColor(a).getRed(&ar, green: &ag, blue: &ab, alpha: &aa)
        UIColor(b).getRed(&br, green: &bg, blue: &bb, alpha: &ba)
        let k = CGFloat(t)
        return Color(.sRGB, red: Double(ar + (br - ar) * k), green: Double(ag + (bg - ag) * k), blue: Double(ab + (bb - ab) * k))
    }

    private func blendedColor(_ w: [Double]) -> Color {
        var r = 0.0, g = 0.0, b = 0.0
        let total = max(0.0001, w.reduce(0, +))
        for (i, s) in AttentionState.allCases.enumerated() {
            let c = UIColor(s.color)
            var cr: CGFloat = 0, cg: CGFloat = 0, cb: CGFloat = 0, ca: CGFloat = 0
            c.getRed(&cr, green: &cg, blue: &cb, alpha: &ca)
            let k = (i < w.count ? w[i] : 0) / total
            r += k * Double(cr); g += k * Double(cg); b += k * Double(cb)
        }
        return Color(.sRGB, red: r, green: g, blue: b)
    }
}

/// Per-frame exponential smoothing of state weights / risk, plus an accumulated breathing phase.
final class OrbSmoother {
    var weights: [Double] = [1, 0, 0, 0]
    var risk: Double = 0
    var phase: Double = 0
    private var last: Double?

    func advance(to probabilities: [Float], risk target: Float, now: Double, snap: Bool = false) {
        let dt = min(0.1, max(0, now - (last ?? now)))
        last = now
        let k = snap ? 1 : 1 - exp(-dt / 0.6)   // ~0.6 s time constant; Reduce Motion snaps (cross-fade-free)
        if probabilities.count == weights.count {
            for i in weights.indices { weights[i] += k * (Double(probabilities[i]) - weights[i]) }
        }
        risk += k * (Double(target) - risk)
        let m = OrbMotion.blend(probabilities: weights.map(Float.init))
        phase += dt / m.period
    }
}

#Preview {
    VStack {
        FocusOrb(probabilities: [1, 0, 0, 0], lapseRisk: 0.2).frame(height: 300)
        FocusOrb(probabilities: [0, 0, 0, 1], lapseRisk: 0.8, riskAlert: true).frame(height: 300)
    }
    .background(FLColor.bg)
}
