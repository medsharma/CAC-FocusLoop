import SwiftUI

/// Minimal, replaceable overlay for `.card` presentations in Study Mode (recap / recall /
/// put-me-down). The Trial module owns the full `NudgeCardView`; swap it in here when merged.
///
/// Recall: tapping a choice shows brief feedback, then calls `onAnswer(choice)`.
/// Other cards: "Continue" calls `onDismiss`.
struct StudyModeCardOverlay: View {
    let card: NudgeCard
    var onAnswer: (Int) -> Void
    var onDismiss: () -> Void
    /// Small print under the card (e.g. the scripted demo's "continues by itself" note).
    var footnote: String? = nil

    @State private var picked: Int?

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous)
        VStack(alignment: .leading, spacing: FLSpacing.m) {
            HStack(spacing: FLSpacing.xs) {
                Image(systemName: icon)
                    .font(.system(size: 15, weight: .regular))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(FLColor.accent)
                    .accessibilityHidden(true)
                Text(card.title).flText(.eyebrow, color: FLColor.textSecondary)
                Spacer()
                if card.kind == .recall {
                    HStack(spacing: 4) {
                        Image(systemName: "pause.fill").font(.system(size: 9))
                        Text("Paused")
                    }
                    .font(FLFont.caption)
                    .foregroundStyle(FLColor.textTertiary)
                }
            }
            Text(card.body)
                .flText(.title)
                .fixedSize(horizontal: false, vertical: true)

            if card.kind == .recall, !card.choices.isEmpty {
                VStack(spacing: FLSpacing.xs) {
                    ForEach(Array(card.choices.enumerated()), id: \.offset) { i, choice in
                        choiceButton(i, choice)
                    }
                }
                if let picked, let correct = card.correctIndex {
                    Label(picked == correct ? "Nice, that's right." : "Answer: \(card.choices[correct])",
                          systemImage: picked == correct ? "checkmark" : "lightbulb")
                        .font(FLFont.callout)
                        .foregroundStyle(picked == correct ? FLColor.focused : StudyModeStyle.valley)
                        .transition(.opacity)
                }
            } else {
                Button(action: onDismiss) { Text("Continue") }
                    .buttonStyle(FLSecondaryButtonStyle())
                    .padding(.top, FLSpacing.xxs)
            }
            if let footnote {
                Text(footnote).flText(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(FLSpacing.l)
        .background(FLColor.bgElevated, in: shape)
        .overlay(shape.strokeBorder(FLColor.hairlineStrong, lineWidth: 0.5))
        .environment(\.colorScheme, .dark)
        .animation(.easeInOut(duration: 0.2), value: picked)
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(.isModal)
    }

    private var icon: String {
        switch card.kind {
        case .recall: "questionmark.bubble"
        case .recap: "text.quote"
        case .putMeDown: "iphone.slash"
        }
    }

    private func choiceButton(_ i: Int, _ choice: String) -> some View {
        let isCorrect = card.correctIndex == i
        let tint: Color? = picked == nil ? nil
            : isCorrect ? FLColor.focused
            : (picked == i ? StudyModeStyle.valley : nil)
        let shape = RoundedRectangle(cornerRadius: FLRadius.small, style: .continuous)
        return Button {
            guard picked == nil else { return }
            picked = i
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.1) { onAnswer(i) }
        } label: {
            HStack(spacing: FLSpacing.s) {
                Text(choice)
                    .font(FLFont.body)
                    .foregroundStyle(FLColor.textPrimary)
                    .multilineTextAlignment(.leading)
                Spacer(minLength: 0)
                if let tint, picked != nil {
                    Image(systemName: isCorrect ? "checkmark" : "xmark")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(tint)
                }
            }
            .padding(.horizontal, FLSpacing.m)
            .frame(maxWidth: .infinity, minHeight: 52, alignment: .leading)
            .background(tint.map { $0.opacity(0.12) } ?? FLColor.surfaceHigh, in: shape)
            .overlay(shape.strokeBorder(tint.map { $0.opacity(0.55) } ?? FLColor.hairline, lineWidth: 0.5))
            .opacity(picked != nil && tint == nil ? 0.5 : 1)
        }
        .buttonStyle(FLPressableStyle())
        .disabled(picked != nil)
        .accessibilityLabel(choice)
    }
}
