import SwiftUI

/// App root: onboarding gate, home navigation stack, and the immersive session cover.
/// Takes the shared `LiveSessionModel` and a `UIServices` implementation (the orchestrator's DI seam).
public struct FocusLoopRootView: View {
    let model: LiveSessionModel
    let services: UIServices
    /// Optional launch screen for demos / screenshots (`-screen science`, etc.).
    let initialScreen: String?

    @AppStorage(UIPrefs.onboardingCompleteKey) private var onboarded = false
    @State private var path: [FLRoute] = []
    @State private var session: ActiveSession?
    @State private var didApplyLaunch = false

    enum ActiveSession: Identifiable {
        case focus
        case study(LectureInfo)
        case breathing
        var id: String {
            switch self {
            case .focus: "focus"
            case .study(let l): "study-\(l.id)"
            case .breathing: "breathing"
            }
        }
    }

    public init(model: LiveSessionModel, services: UIServices, initialScreen: String? = nil) {
        self.model = model; self.services = services; self.initialScreen = initialScreen
        FLAppearance.apply()
    }

    public var body: some View {
        Group {
            if onboarded {
                NavigationStack(path: $path) {
                    HomeView(model: model, services: services, startFocus: { start(.focus) }, navigate: { path.append($0) })
                        .navigationDestination(for: FLRoute.self, destination: destination)
                }
            } else {
                OnboardingFlow(services: services, startAt: initialScreen == "permissions" ? 2 : 0) {
                    withAnimation(.easeInOut(duration: 0.5)) { onboarded = true }
                }
                .transition(.opacity)
            }
        }
        .tint(FLColor.accent)
        .preferredColorScheme(.dark)
        .fullScreenCover(item: $session) { s in
            ActiveSessionContainer(kind: s, model: model, services: services) { reportID in
                session = nil
                if let id = reportID {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { path.append(.report(id)) }
                }
            }
        }
        .onAppear(perform: applyLaunchScreen)
    }

    @ViewBuilder private func destination(_ r: FLRoute) -> some View {
        switch r {
        case .science: ScienceView(model: model, services: services)
        case .history: HistoryView(services: services) { path.append(.report($0)) }
        case .report(let id): ReportLoader(id: id, services: services)
        case .privacy: PrivacyView(services: services)
        case .settings:
            SettingsView(services: services, onResetOnboarding: { path = []; onboarded = false },
                         onRunScriptedDemo: {
                             path = []
                             DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { startScriptedDemo() }
                         })
        case .lectures: LectureListView(services: services) { start(.study($0)) }
        case .brainDemo: BrainCortexDemoView()
        case .sustainedAttention: SustainedAttentionDetailView(services: services)
        }
    }

    private func start(_ s: ActiveSession) {
        switch s {
        case .focus: services.startSession(.focus)
        case .study(let l): services.startSession(.study(lectureID: l.id))
        case .breathing: if model.phase != .running { services.startSession(.focus) }
        }
        session = s
    }

    /// One-tap scripted demo (SYNTHETIC, Demo Director for this session only): Study Mode on the leaf lecture,
    /// driven by the scripted replay; it closes itself at the end of the lecture and opens the report.
    private func startScriptedDemo() {
        guard session == nil, let lecture = services.beginScriptedDemo() else { return }
        start(.study(lecture))
    }

    /// `-screen onboarding|permissions|home|focus|study|science|report|history|sart|privacy|settings|breathing|brain|scriptedDemo`
    private func applyLaunchScreen() {
        guard !didApplyLaunch, let screen = initialScreen else { return }
        didApplyLaunch = true
        switch screen {
        case "onboarding", "permissions": onboarded = false; return
        default: onboarded = true
        }
        switch screen {
        case "focus": start(.focus)
        case "study": if let l = services.lectures().first { start(.study(l)) }
        case "breathing": start(.breathing)
        case "science": path = [.science]
        case "history": path = [.history]
        case "sart": path = [.history, .sustainedAttention]
        case "privacy": path = [.privacy]
        case "settings": path = [.settings]
        case "brain": path = [.brainDemo]
        case "lectures": path = [.lectures]
        case "scriptedDemo": DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { startScriptedDemo() }
        case "report":
            Task {
                // Wait for any background history loading to settle.
                var last = -1
                for _ in 0..<60 {
                    let n = services.listSessions().count
                    if n > 0 && n == last { break }
                    last = n
                    try? await Task.sleep(for: .milliseconds(400))
                }
                let all = services.listSessions()
                if let id = (all.first { $0.lectureTitle != nil } ?? all.first)?.id { path = [.report(id)] }
            }
        default: break
        }
    }
}

/// Loads a report asynchronously from the services.
struct ReportLoader: View {
    let id: UUID
    let services: UIServices
    @State private var report: SessionReport?
    @State private var missing = false
    @State private var insightsPending = false
    var body: some View {
        Group {
            if let report {
                SessionReportView(report: report, services: services, insightsPending: insightsPending)
            } else if missing {
                Text("This report isn't available.").foregroundStyle(FLColor.textSecondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity).flScreen("Session report", inline: true)
            } else {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity).flScreen("Session report", inline: true)
            }
        }
        .task(id: id) {
            // Show the report at once; on-device coach insights can take a few seconds to write.
            if let preview = services.reportPreview(for: id) {
                report = preview
                insightsPending = preview.insights.isEmpty
            }
            if let full = await services.report(for: id) { report = full }
            insightsPending = false
            missing = report == nil
        }
    }
}

/// Hosts the running session full-screen, with the Science view available as a sheet.
struct ActiveSessionContainer: View {
    let kind: FocusLoopRootView.ActiveSession
    let model: LiveSessionModel
    let services: UIServices
    var onFinish: (UUID?) -> Void
    @State private var showScience = false

    var body: some View {
        Group {
            switch kind {
            case .focus:
                FocusSessionView(model: model, services: services, onFinish: onFinish, onOpenScience: { showScience = true })
            case .study(let l):
                if let custom = services.studyModeView(lecture: l, model: model, onFinish: { finish() }) {
                    custom
                } else {
                    PlaceholderStudyModeView(lecture: l, model: model, services: services, onFinish: onFinish,
                                             onOpenScience: { showScience = true })
                }
            case .breathing:
                services.breathingView(model: model, onClose: { finish() })
                    ?? AnyView(BreathingScreen(model: model, onClose: { finish() }))
            }
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showScience) {
            NavigationStack {
                ScienceView(model: model, services: services, ownsSession: false)
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) { Button("Done") { showScience = false } }
                    }
            }
            .presentationDetents([.large])
            .flSheet()
        }
    }

    private func finish() {
        Task { onFinish(await services.stopSession()) }
    }
}
