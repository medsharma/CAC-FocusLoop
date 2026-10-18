import SwiftUI

/// Study Mode's semantic roles, mapped onto the Quiet Dark tokens (`FLColor` / `FLFont`).
/// No custom colours live here any more; this only names what each token means inside the player.
enum StudyModeStyle {
    /// Predicted-valley tint (soft amber, calm rather than alarming).
    static var valley: Color { FLColor.distracted }
    /// Opacity of the valley bands behind the scrubber (spec: amber 25% bands).
    static let valleyBandOpacity: Double = 0.25
    /// Corner radius of the video inset and the corner cortex.
    static let stageRadius: CGFloat = FLRadius.card

    /// Lapse-risk ring colour: calm below 70% of the personal threshold, amber near it, coral above it.
    static func riskColor(_ p: Float, threshold: Float) -> Color {
        if p >= threshold { return FLColor.stressed }
        if p >= threshold * 0.7 { return FLColor.distracted }
        return FLColor.focused
    }
}
