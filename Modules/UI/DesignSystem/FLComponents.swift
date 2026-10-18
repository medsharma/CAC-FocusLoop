import SwiftUI
import Charts

// MARK: - Card

/// Quiet Dark card: `surface` + 0.5pt hairline, radius 20, no shadow, no material.
public struct FLCardModifier: ViewModifier {
    var padding: CGFloat
    var tint: Color?
    public func body(content: Content) -> some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(FLColor.surface, in: RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous)
                    .strokeBorder(FLColor.hairline, lineWidth: 0.5)
            }
    }
}

public extension View {
    /// Standard card. `tint` is accepted for source compatibility and intentionally ignored (no tinted surfaces).
    func flCard(padding: CGFloat = FLSpacing.card, tint: Color? = nil) -> some View {
        modifier(FLCardModifier(padding: padding, tint: tint))
    }

    /// A card for a group of `FLRow`s: no inner padding so rows and inset dividers run to the edges.
    func flRowGroup() -> some View {
        modifier(FLCardModifier(padding: 0, tint: nil))
    }
}

// MARK: - Tags (honesty / status)

/// Data-provenance tags. Required by the honesty guardrails whenever mock/synthetic/demo data is on screen.
public enum FLBadgeKind: String, CaseIterable, Sendable {
    case mock = "MOCK"
    case synthetic = "SYNTHETIC"
    case demo = "DEMO"
    case experimental = "EXPERIMENTAL"
    case preview = "PREVIEW"
    /// Session driven by the user's own recorded face video (not the live camera, not synthetic).
    case recorded = "RECORDED"

    /// The visible tag text. `.mock` reads "PREVIEW": brain predictions come from a placeholder (preview) model.
    public var label: String {
        switch self {
        case .mock: "PREVIEW"
        default: rawValue
        }
    }

    public var color: Color {
        switch self {
        case .mock: FLColor.textSecondary
        case .recorded: FLColor.accentBlue
        case .synthetic: FLColor.synthetic
        case .demo: FLColor.demo
        case .experimental: FLColor.drowsy
        case .preview: FLColor.textSecondary
        }
    }

    public var accessibilityText: String {
        switch self {
        case .mock: "Preview model: placeholder brain predictions"
        case .synthetic: "Synthetic data: signals come from a synthetic replay stream, not a real person"
        case .demo: "Demo Director mode: demonstration only"
        case .experimental: "Experimental arm under test"
        case .preview: "Preview"
        case .recorded: "Recorded video: signals come from your recorded face video, analysed on this device"
        }
    }
}

/// Quiet outline pill: 0.5pt border at tint 55%, fill at tint 10%, 10pt semibold uppercase label. Height 20.
/// Provenance kinds `.mock`, `.synthetic` and `.preview` render NOTHING (product decision: no data tags in the UI;
/// disclose with `FLDataFootnote` / `FLPreviewInfoButton` instead). `.demo` and `.recorded` still render.
public struct FLTag: View {
    let text: String
    let tint: Color
    let accessibilityText: String?
    let hidden: Bool
    public init(_ text: String, tint: Color = FLColor.textSecondary, accessibilityText: String? = nil) {
        self.text = text; self.tint = tint; self.accessibilityText = accessibilityText; self.hidden = false
    }
    public init(_ kind: FLBadgeKind) {
        self.text = kind.label; self.tint = kind.color; self.accessibilityText = kind.accessibilityText
        self.hidden = !kind.isVisibleTag
    }
    public var body: some View {
        if !hidden {
            Text(text.uppercased())
                .font(FLFont.tag)
                .tracking(1)
                .foregroundStyle(tint.mix(with: .white, by: 0.25))
                .lineLimit(1)
                .padding(.horizontal, 8)
                .frame(minHeight: 20)
                .background(tint.opacity(0.10), in: Capsule())
                .overlay(Capsule().strokeBorder(tint.opacity(0.55), lineWidth: 0.5))
                .fixedSize()
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(accessibilityText ?? text)
        }
    }
}

public extension FLBadgeKind {
    /// Whether this kind is shown as a tag at all (mock / synthetic / preview are not).
    var isVisibleTag: Bool {
        switch self {
        case .mock, .synthetic, .preview: false
        default: true
        }
    }
}

/// Legacy name for a provenance tag; renders an `FLTag` (nothing for mock / synthetic / preview).
public struct FLBadge: View {
    let kind: FLBadgeKind
    public init(_ kind: FLBadgeKind, compact: Bool = false) { self.kind = kind }
    public var body: some View { FLTag(kind) }
}

/// A row of provenance tags. Only `.demo` is ever shown (when Demo Director is on); renders nothing otherwise.
public struct FLBadgeRow: View {
    let kinds: [FLBadgeKind]
    public init(mock: Bool = false, synthetic: Bool = false, demo: Bool = false, recorded: Bool = false, compact: Bool = false) {
        kinds = demo ? [.demo] : []
    }
    public var body: some View {
        if !kinds.isEmpty {
            HStack(spacing: 6) { ForEach(kinds, id: \.self) { FLTag($0) } }
        }
    }
}

/// One quiet caption line disclosing data provenance, for the very bottom of reports and Science.
/// e.g. "Includes demo data · Brain predictions from a preview model". Renders nothing when all flags are false.
public struct FLDataFootnote: View {
    let mock: Bool, synthetic: Bool, demo: Bool
    public init(mock: Bool = false, synthetic: Bool = false, demo: Bool = false) {
        self.mock = mock; self.synthetic = synthetic; self.demo = demo
    }
    public static func text(mock: Bool, synthetic: Bool, demo: Bool) -> String? {
        var parts: [String] = []
        switch (demo, synthetic) {
        case (true, true): parts.append("Includes demo and simulated data")
        case (true, false): parts.append("Includes demo data")
        case (false, true): parts.append("Includes simulated data")
        default: break
        }
        if mock { parts.append("Brain predictions from a preview model") }
        return parts.isEmpty ? nil : parts.joined(separator: " \u{00B7} ")
    }
    public var body: some View {
        if let t = Self.text(mock: mock, synthetic: synthetic, demo: demo) {
            Text(t)
                .flText(.caption)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity)
        }
    }
}

/// Small "i" button (44pt hit area) explaining that brain predictions come from a placeholder preview model.
public struct FLPreviewInfoButton: View {
    @State private var shown = false
    public init() {}
    public static let message = "Brain predictions come from a placeholder preview model until the Axonal Labs model is connected. They show an average brain, not yours."
    public var body: some View {
        Button { FLHaptics.tap(); shown = true } label: {
            Image(systemName: "info.circle")
                .font(.system(size: FLIconSize.inline, weight: .regular))
                .foregroundStyle(FLColor.textTertiary)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(FLPressableStyle())
        .accessibilityLabel("About brain predictions")
        .accessibilityHint(Self.message)
        .popover(isPresented: $shown) {
            Text(Self.message)
                .flText(.callout, color: FLColor.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: 280, alignment: .leading)
                .padding(FLSpacing.m)
                .presentationCompactAdaptation(.popover)
                .presentationBackground(FLColor.surfaceHigh)
        }
    }
}

private extension Color {
    /// Linear blend toward `other` (iOS 17 safe; `Color.mix` is iOS 18+).
    func mix(with other: Color, by amount: Double) -> Color {
        let a = UIColor(self), b = UIColor(other)
        var r1: CGFloat = 0, g1: CGFloat = 0, b1: CGFloat = 0, a1: CGFloat = 0
        var r2: CGFloat = 0, g2: CGFloat = 0, b2: CGFloat = 0, a2: CGFloat = 0
        a.getRed(&r1, green: &g1, blue: &b1, alpha: &a1)
        b.getRed(&r2, green: &g2, blue: &b2, alpha: &a2)
        let t = CGFloat(amount)
        return Color(.sRGB, red: Double(r1 + (r2 - r1) * t), green: Double(g1 + (g2 - g1) * t),
                     blue: Double(b1 + (b2 - b1) * t), opacity: Double(a1 + (a2 - a1) * t))
    }
}

// MARK: - Chips

/// Small capsule chip: tint text on a 12% tint fill.
public struct FLChip: View {
    let text: String
    let symbol: String?
    let color: Color
    public init(_ text: String, symbol: String? = nil, color: Color = FLColor.accent) {
        self.text = text; self.symbol = symbol; self.color = color
    }
    public var body: some View {
        HStack(spacing: 6) {
            if let symbol { Image(systemName: symbol).font(.caption).accessibilityHidden(true) }
            Text(text).font(.system(.footnote, weight: .medium))
        }
        .foregroundStyle(color)
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(color.opacity(0.12), in: Capsule())
    }
}

/// State indicator: a small state-colour dot plus the state name on a neutral chip.
public struct FLStateChip: View {
    let state: AttentionState
    public init(_ state: AttentionState) { self.state = state }
    public var body: some View {
        HStack(spacing: 8) {
            Circle().fill(state.color).frame(width: 6, height: 6)
            Text(state.displayName).font(.system(.footnote, weight: .medium))
        }
        .foregroundStyle(FLColor.textPrimary)
        .padding(.horizontal, 12).padding(.vertical, 6)
        .background(FLColor.surfaceHigh, in: Capsule())
        .overlay(Capsule().strokeBorder(FLColor.hairline, lineWidth: 0.5))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Current state: \(state.displayName)")
    }
}

// MARK: - Section header

/// Eyebrow (11pt uppercase, tertiary) above a title. `large` uses the 22pt title role.
public struct FLSectionHeader: View {
    let eyebrow: String?
    let title: String
    let large: Bool
    let trailing: AnyView?
    public init(_ title: String, eyebrow: String? = nil, large: Bool = false) {
        self.title = title; self.eyebrow = eyebrow; self.large = large; trailing = nil
    }
    public init<T: View>(_ title: String, eyebrow: String? = nil, large: Bool = false, @ViewBuilder trailing: () -> T) {
        self.title = title; self.eyebrow = eyebrow; self.large = large; self.trailing = AnyView(trailing())
    }
    public var body: some View {
        HStack(alignment: .lastTextBaseline) {
            VStack(alignment: .leading, spacing: 6) {
                if let eyebrow { Text(eyebrow).flText(.eyebrow) }
                Text(title).flText(large ? .title : .headline)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 8)
            if let trailing { trailing }
        }
    }
}

/// Just an eyebrow line (e.g. "SENSE", "LAST SESSION").
public struct FLEyebrow: View {
    let text: String
    public init(_ text: String) { self.text = text }
    public var body: some View {
        Text(text).flText(.eyebrow).accessibilityAddTraits(.isHeader)
    }
}

/// 0.5pt hairline divider, optionally inset from the leading edge.
public struct FLDivider: View {
    let inset: CGFloat
    let strong: Bool
    public init(inset: CGFloat = 0, strong: Bool = false) { self.inset = inset; self.strong = strong }
    public var body: some View {
        Rectangle().fill(strong ? FLColor.hairlineStrong : FLColor.hairline)
            .frame(height: 0.5)
            .padding(.leading, inset)
            .accessibilityHidden(true)
    }
}

// MARK: - Buttons

/// Primary CTA: full-width accent capsule, 56pt, `onAccent` 17pt semibold. At most one per screen.
public struct FLPrimaryButtonStyle: ButtonStyle {
    var color: Color
    @Environment(\.isEnabled) private var isEnabled
    public init(color: Color = FLColor.accent) { self.color = color }
    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(.headline, weight: .semibold))
            .foregroundStyle(FLColor.onAccent)
            .labelStyle(FLButtonLabelStyle())
            .frame(maxWidth: .infinity, minHeight: 56)
            .background(Capsule().fill(color))
            .contentShape(Capsule())
            .opacity(isEnabled ? (configuration.isPressed ? 0.6 : 1) : 0.35)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(FLMotion.press, value: configuration.isPressed)
    }
}

/// Secondary: `surfaceHigh` capsule with `textPrimary`, 48pt (≥ 44 target). `fill: false` gives an accent text button.
public struct FLSecondaryButtonStyle: ButtonStyle {
    var fullWidth: Bool
    @Environment(\.isEnabled) private var isEnabled
    public init(fullWidth: Bool = true) { self.fullWidth = fullWidth }
    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(.body, weight: .medium))
            .foregroundStyle(FLColor.textPrimary)
            .labelStyle(FLButtonLabelStyle())
            .padding(.horizontal, FLSpacing.l)
            .frame(maxWidth: fullWidth ? .infinity : nil, minHeight: 48)
            .background(Capsule().fill(FLColor.surfaceHigh))
            .overlay(Capsule().strokeBorder(FLColor.hairline, lineWidth: 0.5))
            .contentShape(Capsule())
            .opacity(isEnabled ? (configuration.isPressed ? 0.6 : 1) : 0.35)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(FLMotion.press, value: configuration.isPressed)
    }
}

/// Text-only button in the accent colour (min 44pt tall).
public struct FLTextButtonStyle: ButtonStyle {
    var color: Color
    public init(color: Color = FLColor.accent) { self.color = color }
    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(.body, weight: .medium))
            .foregroundStyle(color)
            .frame(minHeight: 44)
            .contentShape(Rectangle())
            .opacity(configuration.isPressed ? 0.6 : 1)
            .animation(FLMotion.press, value: configuration.isPressed)
    }
}

/// Label layout for buttons: icon (regular weight) + title, 8pt apart.
public struct FLButtonLabelStyle: LabelStyle {
    public func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 8) {
            configuration.icon.font(.system(.body, weight: .regular)).imageScale(.small)
            configuration.title
        }
    }
}

/// Press feedback for rows and card-style buttons: opacity 0.6 + scale 0.98, 120 ms, no layout shift.
public struct FLPressableStyle: ButtonStyle {
    public init() {}
    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .contentShape(Rectangle())
            .opacity(configuration.isPressed ? 0.6 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(FLMotion.press, value: configuration.isPressed)
    }
}

/// Round 44pt icon button used on immersive screens and headers.
public struct FLIconButton: View {
    let symbol: String
    let label: String
    let action: () -> Void
    let identifier: String?
    public init(_ symbol: String, label: String, identifier: String? = nil, action: @escaping () -> Void) {
        self.symbol = symbol; self.label = label; self.identifier = identifier; self.action = action
    }
    public var body: some View {
        Button {
            FLHaptics.tap(); action()
        } label: {
            Image(systemName: symbol)
                .font(.system(size: FLIconSize.inline, weight: .regular))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(FLColor.textPrimary)
                .frame(width: 44, height: 44)
                .background(FLColor.surface, in: Circle())
                .overlay(Circle().strokeBorder(FLColor.hairline, lineWidth: 0.5))
        }
        .buttonStyle(FLPressableStyle())
        .accessibilityLabel(label)
        .accessibilityIdentifier(identifier ?? "")
    }
}

// MARK: - Row

/// List / settings row: 56pt, leading SF Symbol in `textSecondary`, title 17, optional subtitle,
/// trailing value in `textTertiary` and a chevron. Wrap in a `Button`/`NavigationLink` with `FLPressableStyle`,
/// stack inside `.flRowGroup()` and separate with `FLDivider(inset: FLRow.dividerInset)`.
public struct FLRow<Trailing: View>: View {
    let title: String
    let subtitle: String?
    let symbol: String?
    let value: String?
    let chevron: Bool
    let trailing: Trailing

    /// Leading inset for dividers between rows that have a symbol.
    public static var dividerInset: CGFloat { 16 + 28 + 12 }

    public init(_ title: String, subtitle: String? = nil, symbol: String? = nil, value: String? = nil,
                chevron: Bool = true, @ViewBuilder trailing: () -> Trailing) {
        self.title = title; self.subtitle = subtitle; self.symbol = symbol; self.value = value
        self.chevron = chevron; self.trailing = trailing()
    }

    public var body: some View {
        HStack(spacing: 12) {
            if let symbol {
                Image(systemName: symbol)
                    .font(.system(size: FLIconSize.row, weight: .regular))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(FLColor.textSecondary)
                    .frame(width: 28)
                    .accessibilityHidden(true)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(FLFont.body).foregroundStyle(FLColor.textPrimary)
                    .multilineTextAlignment(.leading)
                if let subtitle {
                    Text(subtitle).font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            trailing
            if let value {
                Text(value).font(FLFont.callout).monospacedDigit().foregroundStyle(FLColor.textTertiary)
                    .lineLimit(1)
            }
            if chevron {
                Image(systemName: "chevron.right")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(FLColor.textTertiary)
                    .accessibilityHidden(true)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, minHeight: 56, alignment: .leading)
        .contentShape(Rectangle())
    }
}

public extension FLRow where Trailing == EmptyView {
    init(_ title: String, subtitle: String? = nil, symbol: String? = nil, value: String? = nil, chevron: Bool = true) {
        self.init(title, subtitle: subtitle, symbol: symbol, value: value, chevron: chevron) { EmptyView() }
    }
}

// MARK: - Metric

/// Eyebrow above a 34pt light number, unit in 15pt tertiary, optional delta caption.
public struct FLMetric: View {
    let title: String
    let value: String
    let unit: String?
    let color: Color
    let delta: String?
    public init(_ title: String, value: String, unit: String? = nil, color: Color = FLColor.textPrimary, delta: String? = nil) {
        self.title = title; self.value = value; self.unit = unit; self.color = color; self.delta = delta
    }
    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).flText(.eyebrow)
                .lineLimit(1).minimumScaleFactor(0.7)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value).flText(.metric, color: color)
                    .lineLimit(1).minimumScaleFactor(0.5)
                if let unit { Text(unit).font(FLFont.unit).foregroundStyle(FLColor.textTertiary).lineLimit(1) }
            }
            if let delta { Text(delta).flText(.caption) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

/// A row of metrics separated by hairlines (editorial metrics strip).
public struct FLMetricStrip: View {
    let metrics: [FLMetric]
    public init(_ metrics: [FLMetric]) { self.metrics = metrics }
    public var body: some View {
        HStack(alignment: .top, spacing: 0) {
            ForEach(Array(metrics.enumerated()), id: \.offset) { i, m in
                if i > 0 {
                    Rectangle().fill(FLColor.hairline).frame(width: 0.5).padding(.horizontal, FLSpacing.m)
                }
                m
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}

/// Thin progress / risk line (default 3pt) with a hairline track.
public struct FLLineMeter: View {
    let value: Double
    let color: Color
    let height: CGFloat
    let marker: Double?
    public init(value: Double, color: Color = FLColor.accent, height: CGFloat = 3, marker: Double? = nil) {
        self.value = value; self.color = color; self.height = height; self.marker = marker
    }
    public var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(FLColor.hairlineStrong)
                Capsule().fill(color).frame(width: max(height, geo.size.width * min(1, max(0, value))))
                if let marker {
                    Rectangle().fill(FLColor.textTertiary).frame(width: 1, height: height + 6)
                        .offset(x: geo.size.width * marker)
                }
            }
        }
        .frame(height: height)
        .accessibilityHidden(true)
    }
}

// MARK: - Screen chrome

public extension View {
    /// Pushed-screen chrome: solid `bg`, large title in `textPrimary` (nav bar styling comes from `FLAppearance`).
    func flScreen(_ title: String, inline: Bool = false) -> some View {
        self
            .scrollContentBackground(.hidden)
            .background { FLColor.bg.ignoresSafeArea() }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(inline ? .inline : .large)
    }

    /// Sheet chrome: `bgElevated` background with a grabber.
    func flSheet() -> some View {
        self
            .presentationBackground(FLColor.bgElevated)
            .presentationDragIndicator(.visible)
            .presentationCornerRadius(28)
    }
}

/// UIKit appearance for navigation chrome: transparent bars, large titles in `textPrimary`, minimal back button.
@MainActor
public enum FLAppearance {
    private static var applied = false
    public static func apply() {
        guard !applied else { return }
        applied = true
        let large = UIFont.systemFont(ofSize: 34, weight: .semibold)
        let inline = UIFont.systemFont(ofSize: 17, weight: .semibold)
        let primary = UIColor(FLColor.textPrimary)

        func configured(_ a: UINavigationBarAppearance) -> UINavigationBarAppearance {
            a.largeTitleTextAttributes = [.foregroundColor: primary, .kern: -0.6,
                                          .font: UIFontMetrics(forTextStyle: .largeTitle).scaledFont(for: large)]
            a.titleTextAttributes = [.foregroundColor: primary,
                                     .font: UIFontMetrics(forTextStyle: .headline).scaledFont(for: inline)]
            let back = UIBarButtonItemAppearance(style: .plain)
            back.normal.titleTextAttributes = [.foregroundColor: UIColor.clear]
            back.highlighted.titleTextAttributes = [.foregroundColor: UIColor.clear]
            a.backButtonAppearance = back
            let chevron = UIImage(systemName: "chevron.backward",
                                  withConfiguration: UIImage.SymbolConfiguration(pointSize: 17, weight: .medium))
            a.setBackIndicatorImage(chevron, transitionMaskImage: chevron)
            return a
        }

        let edge = UINavigationBarAppearance()
        edge.configureWithTransparentBackground()
        let standard = UINavigationBarAppearance()
        standard.configureWithTransparentBackground()
        standard.backgroundEffect = UIBlurEffect(style: .systemUltraThinMaterialDark)
        standard.backgroundColor = UIColor(FLColor.bg).withAlphaComponent(0.72)
        standard.shadowColor = UIColor.white.withAlphaComponent(0.08)

        let bar = UINavigationBar.appearance()
        bar.standardAppearance = configured(standard)
        bar.compactAppearance = configured(standard)
        bar.scrollEdgeAppearance = configured(edge)
        bar.tintColor = primary
    }
}

// MARK: - Charts

/// Swift Charts styling: 1.5pt lines, hairline grid, caption axis labels, accent area fade, 3pt bar radius.
public enum FLChart {
    public static let line = StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round)
    public static let barRadius: CGFloat = FLRadius.bar
    /// Area fill: colour @ 12% fading to clear.
    public static func area(_ color: Color = FLColor.accent) -> LinearGradient {
        LinearGradient(colors: [color.opacity(0.12), color.opacity(0)], startPoint: .top, endPoint: .bottom)
    }
}

public extension View {
    /// Quiet axes for a Swift Charts `Chart`: hairline gridlines, no ticks, `caption` labels in `textTertiary`.
    func flChartAxes() -> some View {
        self
            .chartXAxis {
                AxisMarks { _ in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(FLColor.hairline)
                    AxisValueLabel().font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { _ in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(FLColor.hairline)
                    AxisValueLabel().font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                }
            }
            .chartLegend(.hidden)
    }
}

// MARK: - Launch-argument scrolling (demo filming / screenshots)

/// `-scroll <anchor>` scrolls long screens to a section on appear (e.g. `-screen science -scroll brain`).
public struct FLLaunchScroll: ViewModifier {
    let proxy: ScrollViewProxy
    public func body(content: Content) -> some View {
        content.task {
            let args = ProcessInfo.processInfo.arguments
            guard let i = args.firstIndex(of: "-scroll"), i + 1 < args.count else { return }
            try? await Task.sleep(for: .milliseconds(900))
            proxy.scrollTo(args[i + 1], anchor: .top)
        }
    }
}
