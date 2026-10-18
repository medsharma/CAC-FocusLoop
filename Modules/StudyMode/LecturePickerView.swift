import SwiftUI

/// Lists bundled lectures (`LectureInfo.catalog()` + the built-in sample) with subject, duration,
/// and a quiet license/attribution line.
///
/// If `onSelect` is nil, picking a lecture opens `StudyModeView` full screen with `model`.
/// Otherwise the caller handles it (e.g. the engine starts a session, then shows the player).
public struct LecturePickerView: View {
    let model: LiveSessionModel
    var onSelect: ((LectureInfo, ContentTimeline) -> Void)?

    @State private var entries: [StudyLectureEntry] = []
    @State private var playing: StudyLectureEntry?

    public init(model: LiveSessionModel, onSelect: ((LectureInfo, ContentTimeline) -> Void)? = nil) {
        self.model = model
        self.onSelect = onSelect
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FLSpacing.l) {
                VStack(alignment: .leading, spacing: FLSpacing.xs) {
                    Text("Study Mode").flText(.largeTitle)
                        .accessibilityAddTraits(.isHeader)
                    Text("Watch a lecture while FocusLoop predicts where attention tends to dip, using the predicted brain response to this content (average brain).")
                        .flText(.callout)
                        .fixedSize(horizontal: false, vertical: true)
                }

                VStack(alignment: .leading, spacing: FLSpacing.s) {
                    FLEyebrow("Lectures")
                    VStack(spacing: 0) {
                        ForEach(Array(entries.enumerated()), id: \.element.id) { i, entry in
                            if i > 0 { FLDivider(inset: FLSpacing.m + 64 + FLSpacing.m) }
                            Button { select(entry) } label: { StudyLectureRow(entry: entry) }
                                .buttonStyle(FLPressableStyle())
                                .disabled(entry.timeline == nil)
                                .accessibilityIdentifier("lecture.\(entry.id)")
                        }
                    }
                    .flRowGroup()
                }
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.top, FLSpacing.m)
            .padding(.bottom, FLSpacing.xxl)
        }
        .background(FLColor.bg.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .onAppear { if entries.isEmpty { entries = StudyLectureEntry.loadAll() } }
        .fullScreenCover(item: $playing) { entry in
            if let tl = entry.timeline {
                StudyModeView(model: model, lecture: entry.lecture, timeline: tl, onClose: { playing = nil })
            }
        }
    }

    private func select(_ entry: StudyLectureEntry) {
        guard let tl = entry.timeline else { return }
        if let onSelect { onSelect(entry.lecture, tl) } else { playing = entry }
    }
}

struct StudyLectureEntry: Identifiable {
    let lecture: LectureInfo
    let timeline: ContentTimeline?
    var id: String { lecture.id }
    var hasVideo: Bool { StudyLectureLibrary.videoURL(for: lecture) != nil }

    static func loadAll(bundle: Bundle = .main) -> [StudyLectureEntry] {
        StudyLectureLibrary.lectures(bundle: bundle).map {
            StudyLectureEntry(lecture: $0, timeline: StudyLectureLibrary.timeline(for: $0, bundle: bundle))
        }
    }
}

private struct StudyLectureRow: View {
    let entry: StudyLectureEntry

    var body: some View {
        let l = entry.lecture
        HStack(alignment: .top, spacing: FLSpacing.m) {
            // Quiet thumbnail: a neutral tile with one monochrome symbol (no coloured tiles).
            RoundedRectangle(cornerRadius: FLRadius.small, style: .continuous)
                .fill(FLColor.surfaceHigh)
                .overlay(RoundedRectangle(cornerRadius: FLRadius.small, style: .continuous)
                    .strokeBorder(FLColor.hairline, lineWidth: 0.5))
                .overlay {
                    Image(systemName: entry.hasVideo ? "play.fill" : "waveform")
                        .font(.system(size: FLIconSize.inline, weight: .regular))
                        .foregroundStyle(FLColor.textSecondary)
                }
                .frame(width: 64, height: 48)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: FLSpacing.xxs) {
                HStack(alignment: .firstTextBaseline, spacing: FLSpacing.xs) {
                    Text(l.title)
                        .flText(.headline)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: FLSpacing.xxs)
                }
                HStack(spacing: 6) {
                    Text(l.subject)
                    Text("\u{00B7}")
                    Text(Self.duration(l.durationSeconds)).monospacedDigit()
                    if !entry.hasVideo { Text("\u{00B7}"); Text("No video") }
                }
                .font(FLFont.caption)
                .foregroundStyle(FLColor.textSecondary)
                .lineLimit(1)

                // The quiet license line.
                Text("\(l.license) \u{00B7} \(l.attribution)")
                    .font(.caption2)
                    .foregroundStyle(FLColor.textTertiary)
                    .multilineTextAlignment(.leading)
                    .lineLimit(2)
                    .padding(.top, FLSpacing.xxs)

                if entry.timeline == nil {
                    Label("Brain timeline missing", systemImage: "exclamationmark.triangle")
                        .font(FLFont.caption)
                        .foregroundStyle(FLColor.distracted)
                }
            }
            Image(systemName: "chevron.right")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(FLColor.textTertiary)
                .padding(.top, 4)
                .accessibilityHidden(true)
        }
        .padding(FLSpacing.m)
        .frame(maxWidth: .infinity, alignment: .leading)
        .opacity(entry.timeline == nil ? 0.5 : 1)
        .accessibilityElement(children: .combine)
    }

    static func duration(_ s: Double) -> String {
        let m = Int(s) / 60, r = Int(s) % 60
        return r == 0 ? "\(m) min" : "\(m) min \(r) s"
    }
}

#Preview("Lecture picker") {
    LecturePickerView(model: LiveSessionModel())
}
