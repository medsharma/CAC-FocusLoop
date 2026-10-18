import SwiftUI

/// Intervention roles mapped onto the Quiet Dark tokens (no custom colours).
enum InterventionStyle {
    /// Focus-restoring glow, breathing ring, stretch ring: the single accent.
    static var calm: Color { FLColor.accent }
    /// Pre-emptive glow (softer, cooler): the drowsy violet.
    static var preemptive: Color { FLColor.drowsy }
    /// Cool-light pulse (EXPERIMENTAL arm): near-white with a faint steel edge.
    static var coolLight: Color { FLColor.textPrimary }
    static var coolEdge: Color { FLColor.accentBlue }

    /// Smooth 0...1 ease-in-out.
    static func ease(_ x: Double) -> Double {
        let c = min(max(x, 0), 1)
        return c * c * (3 - 2 * c)
    }
}

/// Minimal nudge card: `bgElevated` with a strong hairline, radius 20, no material, no shadow.
struct InterventionPanel<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous)
        content
            .padding(FLSpacing.l)
            .background(FLColor.bgElevated, in: shape)
            .overlay(shape.strokeBorder(FLColor.hairlineStrong, lineWidth: 0.5))
            .environment(\.colorScheme, .dark)
    }
}
