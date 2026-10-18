import SwiftUI
import UIKit

/// FocusLoop design tokens: "Quiet Dark".
/// Near-black surfaces, one mint accent, four attention-state colours used only where they carry meaning.
/// Never put raw hex in views; use these tokens.
public enum FLColor {
    // MARK: Surfaces
    /// App background (near-OLED black, very slightly cool).
    public static let bg = Color(hex: 0x050507)
    /// Sheets, full-screen covers.
    public static let bgElevated = Color(hex: 0x0C0C10)
    /// Cards / grouped rows (only where grouping is needed).
    public static let surface = Color(hex: 0x111116)
    /// Pressed rows, inputs, chips, secondary buttons.
    public static let surfaceHigh = Color(hex: 0x18181F)
    /// Card borders, dividers (0.5pt).
    public static let hairline = Color.white.opacity(0.08)
    /// Focused / selected outlines, zero lines.
    public static let hairlineStrong = Color.white.opacity(0.14)

    // MARK: Text
    public static let textPrimary = Color(hex: 0xF5F5F7)
    public static let textSecondary = Color(hex: 0xA1A1AA)
    /// Captions, axis labels. Slightly lighter than the spec's #71717A so it holds ≥ 4.5:1 on `surface` too.
    public static let textTertiary = Color(hex: 0x7E7E87)

    // MARK: Accent
    /// Mint. Primary CTA fill, selection, focused state, links.
    public static let accent = Color(hex: 0x7CF2D0)
    /// Text / symbols on an accent fill.
    public static let onAccent = Color(hex: 0x04110D)
    public static let danger = Color(hex: 0xFF5A5F)

    // MARK: Attention states (full strength only on small elements; large fills use 12–18% tints)
    public static let focused = Color(hex: 0x7CF2D0)
    public static let distracted = Color(hex: 0xF5B454)
    public static let drowsy = Color(hex: 0x9B8CFF)
    public static let stressed = Color(hex: 0xFF7A85)
    /// rPPG waveform and BPM only.
    public static let heart = Color(hex: 0xFF6B81)

    // MARK: Tag tints
    /// Preview-model tag (formerly MOCK): neutral grey, never alarming.
    public static let mock = Color(hex: 0xA1A1AA)
    public static let synthetic = Color(hex: 0xB4A8FF)
    public static let demo = Color(hex: 0xFF7A85)

    // MARK: Legacy names (mapped onto Quiet Dark so older screens restyle automatically)
    public static var background: Color { bg }
    public static var backgroundRaised: Color { bgElevated }
    public static var surfaceStrong: Color { surfaceHigh }
    public static var stroke: Color { hairline }
    public static var strokeStrong: Color { hairlineStrong }
    /// Former secondary brand blue; now a quiet steel used as a neutral data-series colour.
    public static let accentBlue = Color(hex: 0xA9B4CC)
    /// Former gradient; Quiet Dark has no gradients, so this is the flat accent.
    public static let accentGradient = LinearGradient(colors: [accent, accent], startPoint: .top, endPoint: .bottom)
    /// Lapse risk (same soft amber as `distracted`, never a shouting orange).
    public static let risk = Color(hex: 0xF5B454)

    public static func state(_ s: AttentionState) -> Color {
        switch s {
        case .focused: focused
        case .distracted: distracted
        case .drowsy: drowsy
        case .stressed: stressed
        }
    }
}

public extension AttentionState {
    var color: Color { FLColor.state(self) }
    /// SF Symbol for the state (regular-weight outline symbols).
    var symbol: String {
        switch self {
        case .focused: "scope"
        case .distracted: "eye.trianglebadge.exclamationmark"
        case .drowsy: "moon.zzz"
        case .stressed: "waveform.path.ecg"
        }
    }
    /// Short supportive caption (wellness framing).
    var caption: String {
        switch self {
        case .focused: "In the zone"
        case .distracted: "Attention drifting"
        case .drowsy: "Energy dipping"
        case .stressed: "Tension rising"
        }
    }
}

public extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: opacity)
    }
}

/// Typography: SF Pro (system) only, all sizes scale with Dynamic Type.
/// For the full spec (font + tracking + colour) prefer `Text(...).flText(.largeTitle)`.
public enum FLFont {
    /// A system font at a fixed design size that scales with Dynamic Type relative to `style`.
    public static func scaled(_ size: CGFloat, weight: UIFont.Weight, relativeTo style: UIFont.TextStyle,
                              monospacedDigits: Bool = false) -> Font {
        let base = monospacedDigits ? UIFont.monospacedDigitSystemFont(ofSize: size, weight: weight)
                                    : UIFont.systemFont(ofSize: size, weight: weight)
        return Font(UIFontMetrics(forTextStyle: style).scaledFont(for: base))
    }

    /// 56pt thin, monospaced digits: the timer and hero numbers.
    public static var hero: Font { scaled(56, weight: .thin, relativeTo: .largeTitle, monospacedDigits: true) }
    /// 34pt semibold: screen titles. (Legacy name `display` maps here.)
    public static let largeTitle = Font.system(.largeTitle, weight: .semibold)
    public static var display: Font { largeTitle }
    /// 22pt semibold.
    public static let title = Font.system(.title2, weight: .semibold)
    /// 17pt semibold.
    public static let headline = Font.system(.headline, weight: .semibold)
    /// 17pt regular (use with `textSecondary`).
    public static let body = Font.system(.body)
    /// 15pt regular.
    public static let callout = Font.system(.subheadline)
    /// 13pt regular (use with `textTertiary`).
    public static let caption = Font.system(.footnote)
    /// 11pt semibold; render UPPERCASE with tracking 1.2 (or use `.flText(.eyebrow)`).
    public static let eyebrow = Font.system(.caption2, weight: .semibold)
    /// 34pt light, monospaced digits: metric values.
    public static var metric: Font { scaled(34, weight: .light, relativeTo: .largeTitle, monospacedDigits: true) }
    /// 15pt: metric units.
    public static let unit = Font.system(.subheadline)
    /// 10pt semibold: tag labels.
    public static var tag: Font { scaled(10, weight: .semibold, relativeTo: .caption2) }

    /// Numbers inside running text, rows and small gauges. Heavy weights are softened to keep numerals light.
    public static func numeral(_ style: Font.TextStyle, weight: Font.Weight = .regular) -> Font {
        let w: Font.Weight = (weight == .bold || weight == .heavy || weight == .black || weight == .semibold) ? .medium : weight
        return .system(style, weight: w).monospacedDigit()
    }
}

/// Text roles: font + tracking + colour in one call: `Text("Home").flText(.largeTitle)`.
public enum FLTextStyle {
    case hero, largeTitle, title, headline, body, callout, caption, eyebrow, metric
}

public extension View {
    @ViewBuilder
    func flText(_ style: FLTextStyle, color: Color? = nil) -> some View {
        switch style {
        case .hero: font(FLFont.hero).tracking(-1.5).foregroundStyle(color ?? FLColor.textPrimary)
        case .largeTitle: font(FLFont.largeTitle).tracking(-0.6).foregroundStyle(color ?? FLColor.textPrimary)
        case .title: font(FLFont.title).tracking(-0.3).foregroundStyle(color ?? FLColor.textPrimary)
        case .headline: font(FLFont.headline).foregroundStyle(color ?? FLColor.textPrimary)
        case .body: font(FLFont.body).foregroundStyle(color ?? FLColor.textSecondary)
        case .callout: font(FLFont.callout).foregroundStyle(color ?? FLColor.textSecondary)
        case .caption: font(FLFont.caption).foregroundStyle(color ?? FLColor.textTertiary)
        case .eyebrow: font(FLFont.eyebrow).tracking(1.2).textCase(.uppercase).foregroundStyle(color ?? FLColor.textTertiary)
        case .metric: font(FLFont.metric).tracking(-0.8).foregroundStyle(color ?? FLColor.textPrimary)
        }
    }
}

/// Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64. Screen gutter 20.
public enum FLSpacing {
    public static let xxs: CGFloat = 4
    public static let xs: CGFloat = 8
    public static let s: CGFloat = 12
    public static let m: CGFloat = 16
    public static let l: CGFloat = 24
    public static let xl: CGFloat = 32
    public static let xxl: CGFloat = 48
    public static let xxxl: CGFloat = 64
    /// Screen side gutter.
    public static let gutter: CGFloat = 20
    /// Card inner padding.
    public static let card: CGFloat = 20
    /// Legacy alias for `FLRadius.card`.
    public static let cardRadius: CGFloat = FLRadius.card
}

public enum FLRadius {
    public static let card: CGFloat = 20
    public static let small: CGFloat = 12
    /// Bar marks in charts.
    public static let bar: CGFloat = 3
}

/// Icon sizes (SF Symbols, regular weight, hierarchical).
public enum FLIconSize {
    public static let inline: CGFloat = 17
    public static let row: CGFloat = 20
    public static let toolbar: CGFloat = 24
}

/// Motion tokens. Springs 200–450 ms; respect Reduce Motion at the call site.
public enum FLMotion {
    public static let press = Animation.easeOut(duration: 0.12)
    public static let quick = Animation.spring(response: 0.25, dampingFraction: 0.9)
    public static let standard = Animation.spring(response: 0.4, dampingFraction: 0.88)
    public static let fade = Animation.easeOut(duration: 0.3)
}

/// Haptics for UI taps only (never used as a nudge; nudge haptics belong to the Interventions module).
@MainActor
public enum FLHaptics {
    public static func tap() { UIImpactFeedbackGenerator(style: .soft).impactOccurred(intensity: 0.7) }
    public static func select() { UISelectionFeedbackGenerator().selectionChanged() }
    public static func success() { UINotificationFeedbackGenerator().notificationOccurred(.success) }
    public static func warning() { UINotificationFeedbackGenerator().notificationOccurred(.warning) }
}
