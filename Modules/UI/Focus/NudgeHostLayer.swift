import SwiftUI

/// Renders `model.activePresentation` above the focus / study screens. Uses the services' nudge view
/// (Trial agent) when provided, else the minimal fallbacks below.
public struct NudgeHostLayer: View {
    let model: LiveSessionModel
    let services: UIServices
    /// Skip `.card` presentations (Study Mode renders cards in its own overlay).
    var excludeCards: Bool

    public init(model: LiveSessionModel, services: UIServices, excludeCards: Bool = false) {
        self.model = model; self.services = services; self.excludeCards = excludeCards
    }

    private var shown: NudgePresentation? {
        guard let p = model.activePresentation else { return nil }
        if excludeCards, case .card = p { return nil }
        return p
    }

    public var body: some View {
        ZStack {
            if let p = shown {
                Group {
                    if let custom = services.nudgeView(p, model: model) {
                        custom
                    } else {
                        PlaceholderNudgeView(presentation: p, model: model)
                    }
                }
                .transition(.opacity.combined(with: .scale(scale: 0.98)))
                .id(Self.identity(p))
            }
        }
        .animation(.easeInOut(duration: 0.45), value: model.activePresentation)
    }

    static func identity(_ p: NudgePresentation) -> String {
        switch p {
        case .glow(_, let pre): "glow\(pre)"
        case .card(let c): "card\(c.id)"
        case .breathing: "breathing"
        case .stretch: "stretch"
        case .lightPulse: "lightPulse"
        }
    }
}

/// Minimal fallback nudge views. The orchestrator swaps in the Trial agent's views via `UIServices.nudgeView`.
public struct PlaceholderNudgeView: View {
    let presentation: NudgePresentation
    let model: LiveSessionModel

    public var body: some View {
        switch presentation {
        case let .glow(intensity, preemptive):
            FallbackGlowNudge(intensity: Double(intensity), color: preemptive ? FLColor.accent : FLColor.distracted)
        case let .card(card):
            FallbackCardNudge(card: card, model: model)
        case let .breathing(bpm):
            FallbackBreathingNudge(breathsPerMinute: bpm, model: model)
        case let .stretch(seconds):
            FallbackStretchNudge(seconds: seconds, model: model)
        case .lightPulse:
            FallbackLightPulseNudge(model: model)
        }
    }
}

struct FallbackGlowNudge: View {
    var intensity: Double
    var color: Color
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var on = false

    var body: some View {
        RoundedRectangle(cornerRadius: 54, style: .continuous)
            .strokeBorder(color, lineWidth: 26)
            .blur(radius: 26)
            .opacity((on ? 1 : 0.45) * intensity)
            .ignoresSafeArea()
            .allowsHitTesting(false)
            .onAppear {
                guard !reduceMotion else { on = true; return }
                withAnimation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true)) { on = true }
            }
            .accessibilityLabel("Soft screen glow nudge")
    }
}

struct FallbackCardNudge: View {
    let card: NudgeCard
    let model: LiveSessionModel
    @State private var picked: Int?

    var body: some View {
        VStack {
            Spacer()
            VStack(alignment: .leading, spacing: FLSpacing.s) {
                HStack {
                    Image(systemName: icon).font(.title3).foregroundStyle(tint)
                    Text(card.title).font(FLFont.title).foregroundStyle(FLColor.textPrimary)
                    Spacer()
                    Button { model.actions.dismissNudge() } label: {
                        Image(systemName: "xmark").font(.subheadline.weight(.bold)).foregroundStyle(FLColor.textSecondary)
                            .frame(width: 32, height: 32).background(FLColor.surfaceStrong, in: Circle())
                    }
                    .accessibilityLabel("Dismiss")
                }
                Text(card.body).font(FLFont.body).foregroundStyle(FLColor.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                if !card.choices.isEmpty {
                    VStack(spacing: FLSpacing.xs) {
                        ForEach(Array(card.choices.enumerated()), id: \.offset) { i, c in
                            Button {
                                FLHaptics.select()
                                picked = i
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.9) { model.actions.answerRecall(i) }
                            } label: {
                                HStack {
                                    Text(c).font(FLFont.callout.weight(.medium))
                                    Spacer()
                                    if picked != nil, i == card.correctIndex {
                                        Image(systemName: "checkmark.circle.fill").foregroundStyle(FLColor.focused)
                                    }
                                }
                                .foregroundStyle(FLColor.textPrimary)
                                .padding(.horizontal, 14).frame(minHeight: 46)
                                .background(choiceFill(i), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                            }
                            .disabled(picked != nil)
                        }
                    }
                } else {
                    Button(card.kind == .putMeDown ? "Back to studying" : "Got it") {
                        FLHaptics.tap(); model.actions.dismissNudge()
                    }
                    .buttonStyle(FLPrimaryButtonStyle(color: tint))
                    .padding(.top, 4)
                }
            }
            .padding(FLSpacing.l)
            .background(FLColor.bgElevated, in: RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous).strokeBorder(FLColor.hairline, lineWidth: 0.5))
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.bottom, FLSpacing.l)
        }
        .accessibilityElement(children: .contain)
    }

    private var tint: Color { card.kind == .putMeDown ? FLColor.distracted : FLColor.accent }
    private var icon: String {
        switch card.kind {
        case .putMeDown: "iphone.slash"
        case .recap: "text.book.closed.fill"
        case .recall: "lightbulb.fill"
        }
    }
    private func choiceFill(_ i: Int) -> Color {
        guard let picked else { return FLColor.surfaceStrong }
        if i == card.correctIndex { return FLColor.focused.opacity(0.25) }
        return i == picked ? FLColor.stressed.opacity(0.22) : FLColor.surface
    }
}

struct FallbackBreathingNudge: View {
    var breathsPerMinute: Double
    let model: LiveSessionModel
    var body: some View {
        VStack {
            Spacer()
            HStack(spacing: FLSpacing.m) {
                BreathingGuide(breathsPerMinute: breathsPerMinute, compact: true).frame(width: 84, height: 84)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Breathe with the circle").font(FLFont.headline).foregroundStyle(FLColor.textPrimary)
                    Text("\(Int(breathsPerMinute)) breaths a minute. Heart rate \(UIFormat.bpm(model.heartRate?.bpm)) BPM")
                        .font(FLFont.caption).foregroundStyle(FLColor.textSecondary)
                }
                Spacer(minLength: 0)
                Button { model.actions.dismissNudge() } label: {
                    Image(systemName: "xmark").font(.subheadline.weight(.bold)).foregroundStyle(FLColor.textSecondary)
                        .frame(width: 32, height: 32).background(FLColor.surfaceStrong, in: Circle())
                }
                .accessibilityLabel("Dismiss")
            }
            .padding(FLSpacing.m)
            .background(FLColor.bgElevated, in: RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous).strokeBorder(FLColor.hairline, lineWidth: 0.5))
            .padding(.horizontal, FLSpacing.gutter).padding(.bottom, FLSpacing.l)
        }
    }
}

struct FallbackStretchNudge: View {
    let seconds: Int
    let model: LiveSessionModel
    @State private var start = Date()
    var body: some View {
        VStack {
            Spacer()
            TimelineView(.periodic(from: .now, by: 1)) { ctx in
                let left = max(0, seconds - Int(ctx.date.timeIntervalSince(start)))
                HStack(spacing: FLSpacing.m) {
                    ZStack {
                        Circle().stroke(FLColor.drowsy.opacity(0.2), lineWidth: 6)
                        Circle().trim(from: 0, to: CGFloat(left) / CGFloat(max(1, seconds)))
                            .stroke(FLColor.drowsy, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                        Text("\(left)").font(FLFont.numeral(.title3)).foregroundStyle(FLColor.textPrimary)
                    }
                    .frame(width: 64, height: 64)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Stand and stretch").font(FLFont.headline).foregroundStyle(FLColor.textPrimary)
                        Text("Reach up, roll your shoulders, take a few steps.").font(FLFont.caption)
                            .foregroundStyle(FLColor.textSecondary)
                    }
                    Spacer(minLength: 0)
                    Button("Done") { model.actions.dismissNudge() }
                        .font(FLFont.headline).foregroundStyle(FLColor.drowsy)
                }
                .accessibilityElement(children: .combine)
            }
            .padding(FLSpacing.m)
            .background(FLColor.bgElevated, in: RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous).strokeBorder(FLColor.hairline, lineWidth: 0.5))
            .padding(.horizontal, FLSpacing.gutter).padding(.bottom, FLSpacing.l)
        }
    }
}

/// EXPERIMENTAL arm: a bright, cool-light pulse. Under test; never presented as effective.
struct FallbackLightPulseNudge: View {
    let model: LiveSessionModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var on = false
    var body: some View {
        ZStack(alignment: .top) {
            Color(hex: 0xDDEBFF).opacity(on ? 0.55 : 0.08).ignoresSafeArea().allowsHitTesting(false)
            FLBadge(.experimental, compact: true).padding(.top, 8)
        }
        .onAppear {
            withAnimation(reduceMotion ? .easeInOut(duration: 2) : .easeInOut(duration: 1.2).repeatForever(autoreverses: true)) { on = true }
        }
        .accessibilityLabel("Cool light pulse, experimental")
    }
}

/// Fallback experience-sampling prompt ("Were you focused just now?").
struct PlaceholderExperienceSampleView: View {
    let model: LiveSessionModel
    @State private var askReason = false
    var body: some View {
        VStack(alignment: .leading, spacing: FLSpacing.s) {
            Text(askReason ? "What pulled you away?" : "Were you focused just now?")
                .font(FLFont.headline).foregroundStyle(FLColor.textPrimary)
            HStack(spacing: FLSpacing.xs) {
                if askReason {
                    ForEach(NotFocusedReason.allCases, id: \.self) { r in
                        Button(r.rawValue.capitalized) { model.actions.answerExperienceSample(false, r) }
                            .buttonStyle(FLSecondaryButtonStyle())
                    }
                } else {
                    Button("Yes") { model.actions.answerExperienceSample(true, nil) }.buttonStyle(FLPrimaryButtonStyle())
                    Button("Not really") { askReason = true }.buttonStyle(FLSecondaryButtonStyle())
                }
            }
        }
        .flCard()
        .padding(.horizontal, FLSpacing.gutter)
    }
}
