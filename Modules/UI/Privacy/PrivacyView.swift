import SwiftUI

/// What FocusLoop stores, what it never stores, the limitations, and delete-all.
public struct PrivacyView: View {
    let services: UIServices
    var onDeleted: () -> Void
    @State private var confirm = false
    @State private var deleted = false
    @State private var error: String?

    public init(services: UIServices, onDeleted: @escaping () -> Void = {}) {
        self.services = services; self.onDeleted = onDeleted
    }

    /// Exact categories of locally stored data (numeric features only).
    public static let storedCategories: [(String, String, String)] = [
        ("waveform.path", "1 Hz session ticks", "Attention state, state probabilities, lapse risk, heart rate and its quality, lecture position and the lecture's predicted engagement."),
        ("list.bullet.rectangle", "Session events", "State changes, refocus times, phone pickups, lapse-risk alerts, content valleys, and your \u{201C}Were you focused?\u{201D} answers."),
        ("hand.tap", "Nudge decisions and outcomes", "Which nudge was tried, its randomization probability, and whether you refocused."),
        ("tag", "Labels with numeric feature windows", "Your SART taps and reaction times and your \u{201C}Were you focused?\u{201D} answers, with the numeric features around them (blink, gaze angles, head pose, heart rate)."),
        ("timer", "Sustained-attention (SART) runs", "Every attention-check run: its date, task settings, each digit, whether you tapped and your reaction time, and the computed scores. Numbers only."),
        ("cpu", "Model weights", "Your personal model, trial state, lapse-alert threshold and resting heart rate, so FocusLoop keeps learning what works for you."),
        ("scope", "Calibration cones", "The gaze and head-pose angles of your laptop and phone-screen work zones."),
        ("text.quote", "Insights text", "Short coach summaries generated on your phone from aggregated numbers."),
        ("waveform", "Feature streams (only if you turn them on)", "Settings \u{2192} Record feature streams saves each session's 10 Hz numeric features for retraining. Off by default."),
        ("gearshape", "Settings", "Your preferences, such as Demo Director, Force Replay and whether onboarding is done."),
        ("film", "Demo recording you imported (optional), deleted with Delete all data", "Only if you pick one in Settings \u{2192} Recorded video: your own video file, copied in for demos. FocusLoop itself never records the camera."),
    ]

    public static let limitations: [String] = [
        "FocusLoop is not a medical device. It supports focus and is not intended for any medical purpose.",
        "Lighting and motion affect accuracy.",
        "Camera heart-rate (rPPG) accuracy can vary with skin tone and needs broader testing.",
        "Brain predictions reflect an average brain, not you as an individual.",
        "Studies so far are small.",
    ]

    public var body: some View {
        ScrollViewReader { proxy in
        ScrollView {
            VStack(alignment: .leading, spacing: FLSpacing.xxl) {
                hero
                VStack(alignment: .leading, spacing: FLSpacing.l) {
                    FLSectionHeader("Stored on this phone", eyebrow: "Numbers only", large: true)
                    VStack(spacing: 0) {
                        ForEach(Array(Self.storedCategories.enumerated()), id: \.element.1) { i, item in
                            if i > 0 { FLDivider(inset: 16 + 28 + 12) }
                            HStack(alignment: .firstTextBaseline, spacing: FLSpacing.s) {
                                Image(systemName: item.0)
                                    .font(.system(size: FLIconSize.inline, weight: .regular))
                                    .symbolRenderingMode(.hierarchical)
                                    .foregroundStyle(FLColor.textSecondary)
                                    .frame(width: 28)
                                    .accessibilityHidden(true)
                                VStack(alignment: .leading, spacing: FLSpacing.xxs) {
                                    Text(item.1).font(FLFont.body).foregroundStyle(FLColor.textPrimary)
                                        .fixedSize(horizontal: false, vertical: true)
                                    Text(item.2).flText(.caption)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                                Spacer(minLength: 0)
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 14)
                            .accessibilityElement(children: .combine)
                        }
                    }
                    .flRowGroup()
                }

                VStack(alignment: .leading, spacing: FLSpacing.l) {
                    FLSectionHeader("Never stored, never sent", eyebrow: "By design", large: true)
                    VStack(alignment: .leading, spacing: FLSpacing.m) {
                        ForEach(["Camera video or photos", "Face images or face mesh", "Audio", "Anything over the network: FocusLoop makes no network calls and has no analytics"], id: \.self) { s in
                            HStack(alignment: .firstTextBaseline, spacing: FLSpacing.s) {
                                Image(systemName: "xmark")
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundStyle(FLColor.textTertiary)
                                    .frame(width: 20)
                                    .accessibilityHidden(true)
                                Text(s).font(FLFont.body).foregroundStyle(FLColor.textPrimary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .accessibilityElement(children: .combine)
                            .accessibilityLabel("Never stored: \(s)")
                        }
                    }
                }

                FLDivider()

                VStack(alignment: .leading, spacing: FLSpacing.l) {
                    FLSectionHeader("Limitations", eyebrow: "Honest about accuracy", large: true)
                    VStack(alignment: .leading, spacing: FLSpacing.m) {
                        ForEach(Self.limitations, id: \.self) { s in
                            HStack(alignment: .firstTextBaseline, spacing: FLSpacing.s) {
                                Circle().fill(FLColor.textTertiary).frame(width: 4, height: 4)
                                    .alignmentGuide(.firstTextBaseline) { d in d[.bottom] + 5 }
                                    .frame(width: 20)
                                Text(s).flText(.callout).fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }
                .id("limitations")

                Button(role: .destructive) { confirm = true } label: {
                    Label(deleted ? "All data deleted" : "Delete all data", systemImage: deleted ? "checkmark" : "trash")
                        .labelStyle(FLButtonLabelStyle())
                        .font(.system(.body, weight: .medium))
                        .foregroundStyle(deleted ? FLColor.focused : FLColor.danger)
                        .frame(maxWidth: .infinity, minHeight: 48)
                        .background(FLColor.surfaceHigh, in: Capsule())
                        .overlay(Capsule().strokeBorder(FLColor.hairline, lineWidth: 0.5))
                        .contentShape(Capsule())
                }
                .buttonStyle(FLPressableStyle())
                .disabled(deleted)
                .accessibilityIdentifier("privacy.deleteAll")
                .accessibilityHint("Permanently deletes sessions, labels, model weights, calibration and insights from this phone")
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.top, FLSpacing.xs)
            .padding(.bottom, FLSpacing.xxl)
        }
        .modifier(FLLaunchScroll(proxy: proxy))
        }
        .flScreen("Privacy")
        .confirmationDialog("Delete all FocusLoop data?", isPresented: $confirm, titleVisibility: .visible) {
            Button("Delete everything", role: .destructive) {
                do {
                    try services.deleteAllData()
                    FLHaptics.success(); deleted = true; onDeleted()
                } catch { self.error = error.localizedDescription }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Sessions, labels, feature streams, model weights, calibration, settings and insights will be permanently removed from this phone. This can't be undone.")
        }
        .alert("Couldn't delete", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(error ?? "") }
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: FLSpacing.m) {
            Image(systemName: "lock.shield")
                .font(.system(size: 28, weight: .regular))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(FLColor.accent)
                .accessibilityHidden(true)
            Text("0 bytes of video stored or sent.").flText(.largeTitle)
                .fixedSize(horizontal: false, vertical: true)
            Text("Everything runs on your phone. The camera is read frame by frame, turned into numbers, and the frame is discarded.")
                .flText(.body).fixedSize(horizontal: false, vertical: true)
        }
    }
}
