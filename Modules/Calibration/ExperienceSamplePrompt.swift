import SwiftUI

/// "Were you focused just now?" -> Yes / No -> (Tired / Distracted / Stressed).
/// Calls `onAnswer(focused, reason)` once (matches `LiveSessionModel.Actions.answerExperienceSample`);
/// `onDismiss` when closed without answering.
public struct ExperienceSamplePrompt: View {
    private let onAnswer: (Bool, NotFocusedReason?) -> Void
    private let onDismiss: () -> Void
    @State private var askingReason = false

    public init(onAnswer: @escaping (Bool, NotFocusedReason?) -> Void, onDismiss: @escaping () -> Void = {}) {
        self.onAnswer = onAnswer
        self.onDismiss = onDismiss
    }

    public var body: some View {
        let shape = RoundedRectangle(cornerRadius: FLRadius.card, style: .continuous)
        VStack(alignment: .leading, spacing: FLSpacing.m) {
            HStack(alignment: .center) {
                Text(askingReason ? "What pulled you away?" : "Were you focused just now?")
                    .flText(.headline)
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
                .accessibilityLabel("Skip")
            }
            if askingReason {
                HStack(spacing: FLSpacing.xs) {
                    choice("Tired", symbol: "moon.zzz") { onAnswer(false, .tired) }
                    choice("Distracted", symbol: "sparkles") { onAnswer(false, .distracted) }
                    choice("Stressed", symbol: "waveform.path.ecg") { onAnswer(false, .stressed) }
                }
                .transition(.move(edge: .trailing).combined(with: .opacity))
            } else {
                HStack(spacing: FLSpacing.xs) {
                    choice("Yes", symbol: "checkmark") { onAnswer(true, nil) }
                    choice("No", symbol: "xmark") { withAnimation(.snappy) { askingReason = true } }
                }
                .transition(.move(edge: .leading).combined(with: .opacity))
            }
        }
        .padding(FLSpacing.card)
        .background(FLColor.bgElevated, in: shape)
        .overlay(shape.strokeBorder(FLColor.hairlineStrong, lineWidth: 0.5))
        .padding(.horizontal, FLSpacing.m)
    }

    private func choice(_ title: String, symbol: String, action: @escaping () -> Void) -> some View {
        let shape = RoundedRectangle(cornerRadius: FLRadius.small, style: .continuous)
        return Button(action: action) {
            VStack(spacing: FLSpacing.xs) {
                Image(systemName: symbol)
                    .font(.system(size: FLIconSize.row, weight: .regular))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(FLColor.textSecondary)
                Text(title).font(.system(.subheadline, weight: .medium)).foregroundStyle(FLColor.textPrimary)
            }
            .frame(maxWidth: .infinity, minHeight: 72)
            .background(FLColor.surfaceHigh, in: shape)
            .overlay(shape.strokeBorder(FLColor.hairline, lineWidth: 0.5))
        }
        .buttonStyle(FLPressableStyle())
    }
}

/// Exports CSV via the share sheet (which includes "Save to Files"). The file is generated lazily on tap.
public struct CSVExportButton: View {
    private let title: String
    private let makeFile: () throws -> URL
    @State private var exported: CSVExportItem?
    @State private var errorText: String?

    public init(_ title: String = "Export labels (CSV)", makeFile: @escaping () throws -> URL) {
        self.title = title
        self.makeFile = makeFile
    }

    /// Convenience for a `LabelSink` (e.g. `CSVLabelSink`).
    public init(_ title: String = "Export labels (CSV)", sink: LabelSink) {
        self.init(title) { try sink.exportCSV() }
    }

    public var body: some View {
        Button {
            do {
                exported = CSVExportItem(url: try makeFile())
                errorText = nil
            } catch {
                errorText = "Export failed: \(error.localizedDescription)"
            }
        } label: {
            Label(title, systemImage: "square.and.arrow.up")
        }
        .sheet(item: $exported) { item in
            CSVActivityView(items: [item.url])
                .presentationDetents([.medium, .large])
        }
        .overlay(alignment: .bottom) {
            if let errorText {
                Text(errorText).font(FLFont.caption).foregroundStyle(FLColor.danger).offset(y: 20)
            }
        }
    }
}

private struct CSVExportItem: Identifiable {
    let url: URL
    var id: String { url.absoluteString }
}

private struct CSVActivityView: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
