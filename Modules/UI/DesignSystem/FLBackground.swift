import SwiftUI

/// Quiet Dark background: solid `bg`, plus at most ONE very soft static radial glow behind the hero element.
/// No moving gradients. `FLBackground()` is solid black; `FLBackground(glow: color, center: …)` adds the glow.
public struct FLBackground: View {
    var glow: Color?
    var center: UnitPoint
    var intensity: Double
    var radiusFraction: CGFloat

    /// - Parameters:
    ///   - glow: glow colour (accent or a state colour), or nil for solid `bg`.
    ///   - center: glow centre in unit coordinates of the screen (put it behind the hero).
    ///   - intensity: peak opacity, 0.06–0.10 per spec.
    ///   - radius: glow radius as a fraction of the screen's longer side (~0.6 of the width per spec).
    public init(glow: Color? = nil, center: UnitPoint = UnitPoint(x: 0.5, y: 0.38),
                intensity: Double = 0.08, radius: CGFloat = 0.45) {
        self.glow = glow; self.center = center; self.intensity = intensity; self.radiusFraction = radius
    }

    /// Legacy initialiser: a faint static glow in `tint` near the top-centre. `animated` is ignored.
    public init(tint: Color, animated: Bool = true) {
        self.init(glow: tint, center: UnitPoint(x: 0.5, y: 0.3), intensity: 0.06)
    }

    public var body: some View {
        GeometryReader { geo in
            let r = max(geo.size.width, geo.size.height) * radiusFraction
            ZStack {
                FLColor.bg
                if let glow {
                    RadialGradient(colors: [glow.opacity(intensity), glow.opacity(intensity * 0.4), glow.opacity(0)],
                                   center: center, startRadius: 0, endRadius: r)
                }
            }
        }
        .accessibilityHidden(true)
    }
}

#Preview { FLBackground(glow: FLColor.accent) }
