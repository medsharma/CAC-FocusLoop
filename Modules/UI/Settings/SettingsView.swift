import SwiftUI

/// Settings: Force Replay, experience sampling, Demo (Demo Director, scripted demo, sample history, readiness;
/// DEMONSTRATION ONLY), calibration, export, about the data, debug menu.
/// Quiet Dark: grouped `FLRow` cards on black (a plain `List`, one row per group, so UI tests keep their list).
public struct SettingsView: View {
    let services: UIServices
    var onResetOnboarding: () -> Void
    /// Opens the one-tap scripted demo (demo replay) in Study Mode. Nil hides the row.
    var onRunScriptedDemo: (() -> Void)?
    @State private var csvURL: URL?
    @State private var streamURLs: [URL]?
    @State private var exportError: String?
    @State private var recordStreams = false
    @State private var calibration: CalibrationTask?
    @State private var sampleCount = 0
    @State private var sampleBusy = false
    @State private var confirmRemoveSamples = false
    @State private var demoError: String?
    @State private var push: SettingsPush?
    @State private var showDoctorReport = false

    /// Pushed pages (Buttons + `navigationDestination`, so the List adds no second disclosure chevron).
    enum SettingsPush: Hashable { case readiness, aboutData, brainDemo }

    enum CalibrationTask: String, Identifiable {
        case laptop, phoneScreen, sart
        var id: String { rawValue }
    }

    public init(services: UIServices, onResetOnboarding: @escaping () -> Void, onRunScriptedDemo: (() -> Void)? = nil) {
        self.services = services; self.onResetOnboarding = onResetOnboarding; self.onRunScriptedDemo = onRunScriptedDemo
    }

    public var body: some View {
        ScrollViewReader { proxy in
        List {
            section("Session") {
                SettingsToggleRow(
                    "Force Replay",
                    subtitle: services.replayIsForced
                        ? "This device can't run face tracking, so the demo replay stream is always used."
                        : "Use the demo replay stream instead of the camera.",
                    symbol: "play.circle",
                    isOn: Binding(get: { services.replayIsForced || services.flags.forceReplay },
                                  set: { services.flags.forceReplay = $0 }))
                    .disabled(services.replayIsForced)   // no face tracking here: replay is always on
                    .accessibilityIdentifier("settings.forceReplay")
                FLDivider(inset: FLRow<EmptyView>.dividerInset)
                SettingsToggleRow("Ask \u{201C}Were you focused?\u{201D}", symbol: "questionmark.bubble",
                                  isOn: Binding(get: { services.flags.experienceSampling },
                                                set: { services.flags.experienceSampling = $0 }))
                    .accessibilityIdentifier("settings.experienceSampling")
            }

            demoSection

            section("Calibration",
                    footer: "Study Mode uses the phone-screen work zone; free study uses the laptop one.") {
                Button { calibration = .laptop } label: {
                    FLRow("Recalibrate work zone (laptop)", symbol: "laptopcomputer")
                }
                .buttonStyle(FLPressableStyle())
                FLDivider(inset: FLRow<EmptyView>.dividerInset)
                Button { calibration = .phoneScreen } label: {
                    FLRow("Recalibrate for phone screen", subtitle: "Study Mode", symbol: "iphone")
                }
                .buttonStyle(FLPressableStyle())
                .accessibilityLabel("Recalibrate for phone screen (Study Mode)")
                FLDivider(inset: FLRow<EmptyView>.dividerInset)
                Button { calibration = .sart } label: {
                    FLRow("Run SART again", symbol: "hand.tap")
                }
                .buttonStyle(FLPressableStyle())
            }

            section("Your data", footer: "The summary is made on this phone and only shared if you choose to. The CSV files hold numeric features only, for retraining your models.") {
                Button { showDoctorReport = true } label: {
                    FLRow("Share with your doctor", subtitle: "PDF summary of how your sustained attention behaves", symbol: "doc.text")
                }
                .buttonStyle(FLPressableStyle())
                .accessibilityIdentifier("settings.doctorReport")
                FLDivider(inset: FLRow<EmptyView>.dividerInset)
                if let url = csvURL {
                    ShareLink(item: url) { FLRow("Share labeled windows (CSV)", symbol: "square.and.arrow.up") }
                        .buttonStyle(FLPressableStyle())
                } else {
                    Button {
                        do { csvURL = try services.exportLabelsCSV() } catch { exportError = error.localizedDescription }
                    } label: { FLRow("Export labeled windows (CSV)", symbol: "tablecells", chevron: false) }
                    .buttonStyle(FLPressableStyle())
                }
                FLDivider(inset: FLRow<EmptyView>.dividerInset)
                SettingsToggleRow("Record feature streams",
                                  subtitle: "Saves every session's 10 Hz numeric features (no images) on this phone, for retraining the lapse forecaster.",
                                  symbol: "waveform.path",
                                  isOn: Binding(get: { recordStreams },
                                                set: { recordStreams = $0; services.recordFeatureStreams = $0; streamURLs = nil }))
                    .accessibilityIdentifier("settings.recordStreams")
                if let urls = streamURLs {
                    FLDivider(inset: FLRow<EmptyView>.dividerInset)
                    ShareLink(items: urls) { FLRow("Share feature streams (\(urls.count) files)", symbol: "square.and.arrow.up") }
                        .buttonStyle(FLPressableStyle())
                } else if recordStreams {
                    FLDivider(inset: FLRow<EmptyView>.dividerInset)
                    Button {
                        do { streamURLs = try services.exportFeatureStreams() } catch { exportError = error.localizedDescription }
                    } label: { FLRow("Export feature streams (CSV + manifest)", symbol: "square.and.arrow.down", chevron: false) }
                    .buttonStyle(FLPressableStyle())
                }
            }

            section("About") {
                Button { push = .aboutData } label: {
                    FLRow("About the data", subtitle: "Demo data and the preview brain model", symbol: "info.circle")
                }
                .buttonStyle(FLPressableStyle())
                .accessibilityIdentifier("settings.aboutData")
            }

            section("Debug") {
                Button { push = .brainDemo } label: {
                    FLRow("Brain cortex demo", symbol: "brain")
                }
                .buttonStyle(FLPressableStyle())
                FLDivider(inset: FLRow<EmptyView>.dividerInset)
                Button { onResetOnboarding() } label: {
                    FLRow("Replay onboarding", symbol: "arrow.counterclockwise", chevron: false)
                }
                .buttonStyle(FLPressableStyle())
                if let extra = services.debugMenuItems() {
                    FLDivider(inset: FLRow<EmptyView>.dividerInset)
                    // Rows supplied by the app (links, a toggle, an export); give them row rhythm.
                    VStack(alignment: .leading, spacing: FLSpacing.l) { extra }
                        .font(FLFont.body)
                        .foregroundStyle(FLColor.textPrimary)
                        .tint(FLColor.accent)
                        .buttonStyle(FLPressableStyle())
                        .padding(.horizontal, FLSpacing.m)
                        .padding(.vertical, FLSpacing.m)
                        .labelStyle(SettingsDebugLabelStyle())
                }
            }

            Text("FocusLoop predicts lapses and learns what works for you. It supports focus and is not a medical device.")
                .flText(.caption)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, FLSpacing.xxs)
                .settingsListRow(bottom: FLSpacing.xxl)
        }
        .listStyle(.plain)
        .environment(\.defaultMinListRowHeight, 0)
        .modifier(FLLaunchScroll(proxy: proxy))
        }
        .flScreen("Settings")
        .navigationDestination(isPresented: Binding(get: { push != nil }, set: { if !$0 { push = nil } })) {
            switch push {
            case .readiness: DemoReadinessView(services: services)
            case .aboutData: SettingsAboutDataView()
            case .brainDemo: BrainCortexDemoView().navigationTitle("Brain cortex demo")
            case nil: EmptyView()
            }
        }
        .onAppear {
            recordStreams = services.recordFeatureStreams
            sampleCount = services.sampleHistoryCount
        }
        .sheet(isPresented: $showDoctorReport) { DoctorReportSheet(services: services) }
        .confirmationDialog("Remove the sample history?", isPresented: $confirmRemoveSamples, titleVisibility: .visible) {
            Button("Remove \(sampleCount) sample sessions", role: .destructive) { removeSamples() }
            Button("Cancel", role: .cancel) {}
        } message: { Text("Only the sample sessions are removed. Your real sessions are not touched.") }
        .alert("Sample history", isPresented: Binding(get: { demoError != nil }, set: { if !$0 { demoError = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(demoError ?? "") }
        .fullScreenCover(item: $calibration) { task in
            let done = { calibration = nil }
            ZStack(alignment: .topTrailing) {
                switch task {
                case .laptop:
                    services.calibrationView(isPhoneScreen: false, onFinish: done) ?? AnyView(PlaceholderCalibrationView(onFinish: done))
                case .phoneScreen:
                    services.calibrationView(isPhoneScreen: true, onFinish: done) ?? AnyView(PlaceholderCalibrationView(onFinish: done))
                case .sart:
                    services.sartView(onFinish: done) ?? AnyView(PlaceholderSARTView(onFinish: done))
                }
                FLIconButton("xmark", label: "Close", action: done)
                    .padding(.trailing, FLSpacing.m).padding(.top, FLSpacing.xs)
            }
        }
        .alert("Export unavailable", isPresented: Binding(get: { exportError != nil }, set: { if !$0 { exportError = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(exportError ?? "") }
    }

    /// One settings group: an eyebrow, a `surface` card of rows, an optional caption footer. One list row per group.
    private func section<Content: View>(_ title: String, footer: String? = nil,
                                        @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: FLSpacing.xs) {
            FLEyebrow(title).padding(.horizontal, FLSpacing.xxs)
            VStack(alignment: .leading, spacing: 0) { content() }
                .flRowGroup()
            if let footer {
                Text(footer).flText(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, FLSpacing.xxs)
                    .padding(.top, FLSpacing.xxs)
            }
        }
        .settingsListRow()
        .id(title.lowercased())
    }

    // MARK: Demo (DEMONSTRATION ONLY)

    private var demoSection: some View {
        section("Demo",
                footer: "Everything here is DEMONSTRATION ONLY. The demo and the sample history use generated demo data and never touch your real data.") {
            SettingsToggleRow(
                "Demo Director",
                subtitleText: Text("DEMONSTRATION ONLY. ").fontWeight(.semibold).foregroundStyle(FLColor.textSecondary)
                    + Text("Lower thresholds and shorter pacing, and each state gets a fixed nudge (recall or glow before a dip, \u{201C}Put me down\u{201D} or a glow when distracted, a stretch when drowsy, breathing when stressed). Demo sessions never update your Personal Trial or models and are never used for any result."),
                symbol: "wand.and.stars",
                isOn: Binding(get: { services.flags.demoDirector }, set: { services.flags.demoDirector = $0; FLHaptics.select() }))
                .accessibilityIdentifier("settings.demoDirector")
            if let run = onRunScriptedDemo {
                FLDivider(inset: FLRow<EmptyView>.dividerInset)
                Button(action: run) {
                    FLRow("Run demo",
                          subtitle: "Plays \u{201C}Travel deep inside a leaf\u{201D} with a scripted demo replay and Demo Director on for this session only, then opens the report.",
                          symbol: "play.rectangle")
                }
                .buttonStyle(FLPressableStyle())
                .accessibilityIdentifier("settings.scriptedDemo")
            }
            FLDivider(inset: FLRow<EmptyView>.dividerInset)
            Button { push = .readiness } label: {
                FLRow("Demo readiness", symbol: "checklist")
            }
            .buttonStyle(FLPressableStyle())
            .accessibilityIdentifier("settings.readiness")
            FLDivider(inset: FLRow<EmptyView>.dividerInset)
            if sampleCount > 0 {
                Button(role: .destructive) { confirmRemoveSamples = true } label: {
                    FLRow("Remove sample history", subtitle: "\(sampleCount) sample sessions loaded.",
                          symbol: "trash", chevron: false)
                }
                .buttonStyle(FLPressableStyle())
                .disabled(sampleBusy)
                .accessibilityIdentifier("settings.removeSamples")
            } else {
                Button { loadSamples() } label: {
                    FLRow("Load sample history",
                          subtitle: "Adds about 6 sample sessions so History and the Attention Profile have something to show. Real data is never touched.",
                          symbol: "clock.arrow.circlepath", chevron: false) {
                        if sampleBusy { ProgressView().controlSize(.small).tint(FLColor.textSecondary) }
                    }
                }
                .buttonStyle(FLPressableStyle())
                .disabled(sampleBusy)
                .accessibilityIdentifier("settings.loadSamples")
            }
        }
    }

    private func loadSamples() {
        sampleBusy = true
        Task {
            do { _ = try await services.loadSampleHistory() } catch { demoError = error.localizedDescription }
            sampleCount = services.sampleHistoryCount
            sampleBusy = false
        }
    }

    private func removeSamples() {
        do { _ = try services.removeSampleHistory() } catch { demoError = error.localizedDescription }
        sampleCount = services.sampleHistoryCount
    }
}

// MARK: - Private components

private extension View {
    /// A clear, separator-free list row with the screen gutter.
    func settingsListRow(bottom: CGFloat = FLSpacing.xl) -> some View {
        self
            .listRowInsets(EdgeInsets(top: 0, leading: FLSpacing.gutter, bottom: bottom, trailing: FLSpacing.gutter))
            .listRowBackground(Color.clear)
            .listRowSeparator(.hidden)
    }
}

/// A toggle laid out like an `FLRow`: leading symbol, title, optional caption, trailing switch in the accent.
private struct SettingsToggleRow: View {
    let title: String
    let subtitle: Text?
    let symbol: String?
    @Binding var isOn: Bool
    @Environment(\.isEnabled) private var isEnabled

    init(_ title: String, subtitle: String? = nil, symbol: String? = nil, isOn: Binding<Bool>) {
        self.title = title; self.subtitle = subtitle.map { Text($0) }; self.symbol = symbol; _isOn = isOn
    }

    init(_ title: String, subtitleText: Text, symbol: String? = nil, isOn: Binding<Bool>) {
        self.title = title; self.subtitle = subtitleText; self.symbol = symbol; _isOn = isOn
    }

    var body: some View {
        Toggle(isOn: $isOn) {
            HStack(alignment: .top, spacing: 12) {
                if let symbol {
                    Image(systemName: symbol)
                        .font(.system(size: FLIconSize.row, weight: .regular))
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(FLColor.textSecondary)
                        .frame(width: 28, height: 22)
                        .accessibilityHidden(true)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(FLFont.body).foregroundStyle(FLColor.textPrimary)
                    if let subtitle {
                        subtitle.font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .opacity(isEnabled ? 1 : 0.6)
        }
        .tint(FLColor.accent)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, minHeight: 56, alignment: .leading)
    }
}

/// Labels inside the app-supplied debug rows: regular-weight symbol in `textSecondary`, title in `textPrimary`.
private struct SettingsDebugLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 12) {
            configuration.icon.foregroundStyle(FLColor.textSecondary).frame(width: 28)
            configuration.title.foregroundStyle(FLColor.textPrimary)
        }
        .frame(minHeight: 44)
    }
}

/// Settings → About → About the data: the one place with the full explanation of demo data and the
/// preview brain model.
private struct SettingsAboutDataView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FLSpacing.xl) {
                block("Demo data",
                      "Force Replay, the scripted demo and the sample history use a replay of generated demo signals instead of the camera. They are not measured from a real person and never touch your real sessions.")
                FLDivider()
                block("Preview brain model",
                      "Brain predictions come from a placeholder preview model until the Axonal Labs model is connected. They show an average brain, not yours.")
                FLDivider()
                block("Demo Director",
                      "DEMONSTRATION ONLY. It lowers thresholds and gives each state a fixed nudge. Demo sessions never update your Personal Trial or models and are never used for any result.")
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.top, FLSpacing.m)
            .padding(.bottom, FLSpacing.xxl)
        }
        .flScreen("About the data", inline: true)
    }

    private func block(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: FLSpacing.xs) {
            Text(title).flText(.headline).accessibilityAddTraits(.isHeader)
            Text(text).flText(.body).fixedSize(horizontal: false, vertical: true)
        }
    }
}
