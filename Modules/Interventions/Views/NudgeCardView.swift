import SwiftUI

/// Card for the put-me-down, recap and active-recall nudges.
/// Recall cards show their choices; after a choice the card shows gentle feedback.
public struct NudgeCardView: View {
    public var card: NudgeCard
    public var onDismiss: () -> Void
    public var onChoice: ((Int) -> Void)?

    @State private var chosen: Int?

    public init(card: NudgeCard, onDismiss: @escaping () -> Void, onChoice: ((Int) -> Void)? = nil) {
        self.card = card
        self.onDismiss = onDismiss
        self.onChoice = onChoice
    }

    private var icon: String {
        switch card.kind {
        case .putMeDown: "iphone.slash"
        case .recap: "text.book.closed"
        case .recall: "lightbulb"
        }
    }

    public var body: some View {
        InterventionPanel {
            VStack(alignment: .leading, spacing: FLSpacing.m) {
                HStack(spacing: FLSpacing.xs) {
                    Image(systemName: icon)
                        .font(.system(size: 15, weight: .regular))
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(FLColor.accent)
                        .accessibilityHidden(true)
                    Text(card.title).flText(.eyebrow, color: FLColor.textSecondary)
                        .accessibilityAddTraits(.isHeader)
                    Spacer()
                    Button(action: onDismiss) {
                        Image(systemName: "xmark")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(FLColor.textSecondary)
                            .frame(width: 30, height: 30)
                            .background(FLColor.surfaceHigh, in: Circle())
                            .frame(width: 44, height: 44)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(FLPressableStyle())
                    .padding(.vertical, -FLSpacing.s)
                    .padding(.trailing, -FLSpacing.xs)
                    .accessibilityLabel("Dismiss")
                }
                Text(card.body)
                    .flText(.title)
                    .fixedSize(horizontal: false, vertical: true)

                if !card.choices.isEmpty {
                    VStack(spacing: FLSpacing.xs) {
                        ForEach(card.choices.indices, id: \.self) { i in
                            choiceButton(i)
                        }
                    }
                    if let chosen {
                        Text(feedback(for: chosen))
                            .font(FLFont.callout)
                            .foregroundStyle(chosen == card.correctIndex ? FLColor.focused : FLColor.textSecondary)
                            .transition(.opacity)
                    }
                }

                if card.choices.isEmpty || chosen != nil {
                    Button(action: onDismiss) { Text(primaryLabel) }
                        .buttonStyle(FLPrimaryButtonStyle())
                        .padding(.top, FLSpacing.xxs)
                }
            }
            .animation(.easeInOut(duration: 0.25), value: chosen)
        }
        .frame(maxWidth: 440)
        .padding(.horizontal, FLSpacing.gutter)
    }

    private var primaryLabel: String {
        switch card.kind {
        case .putMeDown: "Putting it down"
        case .recap: "Got it"
        case .recall: "Back to the lecture"
        }
    }

    private func feedback(for i: Int) -> String {
        guard let correct = card.correctIndex else { return "Thanks, nice recall." }
        if i == correct { return "Nice recall." }
        return "Close. The answer was \u{201C}\(card.choices[correct])\u{201D}."
    }

    @ViewBuilder
    private func choiceButton(_ i: Int) -> some View {
        let isChosen = chosen == i
        let isCorrect = card.correctIndex == i
        let reveal = chosen != nil
        let tint: Color? = !reveal ? nil : isCorrect ? FLColor.focused : (isChosen ? FLColor.distracted : nil)
        let shape = RoundedRectangle(cornerRadius: FLRadius.small, style: .continuous)
        Button {
            guard chosen == nil else { return }
            chosen = i
            onChoice?(i)
        } label: {
            HStack(spacing: FLSpacing.s) {
                Text(card.choices[i])
                    .font(FLFont.body)
                    .foregroundStyle(FLColor.textPrimary)
                    .multilineTextAlignment(.leading)
                Spacer(minLength: 0)
                if reveal && isCorrect {
                    Image(systemName: "checkmark").font(.system(size: 13, weight: .semibold)).foregroundStyle(FLColor.focused)
                } else if reveal && isChosen {
                    Image(systemName: "xmark").font(.system(size: 13, weight: .semibold)).foregroundStyle(FLColor.distracted)
                }
            }
            .padding(.horizontal, FLSpacing.m)
            .frame(maxWidth: .infinity, minHeight: 52, alignment: .leading)
            .background(tint.map { $0.opacity(0.12) } ?? FLColor.surfaceHigh, in: shape)
            .overlay(shape.strokeBorder(tint.map { $0.opacity(0.55) } ?? FLColor.hairline, lineWidth: 0.5))
            .opacity(reveal && tint == nil ? 0.5 : 1)
        }
        .buttonStyle(FLPressableStyle())
        .disabled(reveal)
    }
}

#Preview("Cards") {
    ZStack {
        FLColor.bg.ignoresSafeArea()
        VStack(spacing: 16) {
            NudgeCardView(card: PutMeDownNudge.card, onDismiss: {})
            NudgeCardView(card: NudgeCard(kind: .recall, title: "Quick check", body: "Where does the Calvin cycle happen?",
                                          choices: ["Stroma", "Thylakoid membrane", "Mitochondria"], correctIndex: 0),
                          onDismiss: {}, onChoice: { _ in })
        }
    }
}
