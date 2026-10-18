import SwiftUI

@main
struct FocusLoopApp: App {
    // `AppServices` is the DI container: session engine, persistence, models, insights and module views.
    // `-previewServices` swaps in the UI module's SYNTHETIC preview services (screenshots only).
    @State private var model: LiveSessionModel
    @State private var services: any UIServices

    init() {
        let m = LiveSessionModel()
        _model = State(initialValue: m)
        let usePreview = ProcessInfo.processInfo.arguments.contains("-previewServices")
        let s: any UIServices = usePreview ? PreviewUIServices(model: m) : AppServices(model: m)
        // `-resetAllData` (UI tests only): start from a fresh-install state, as Privacy → Delete all data does.
        if ProcessInfo.processInfo.arguments.contains("-resetAllData") {
            try? s.deleteAllData()
            UserDefaults.standard.removeObject(forKey: UIPrefs.onboardingCompleteKey)
        }
        _services = State(initialValue: s)
    }

    var body: some Scene {
        WindowGroup {
            if ProcessInfo.processInfo.arguments.contains("-debugBrainDemo") {
                BrainCortexDemoView()
            } else if ProcessInfo.processInfo.arguments.contains("-debugStudyMode") {
                StudyModeDebugRoot(showPicker: false)
            } else if ProcessInfo.processInfo.arguments.contains("-debugStudyPicker") {
                StudyModeDebugRoot(showPicker: true)
            } else {
                FocusLoopRootView(model: model, services: services, initialScreen: Self.launchScreen)
            }
        }
    }

    /// `-screen <name>` launch argument (demo / screenshot convenience). `-scriptedDemo` starts the one-tap
    /// scripted demo (SYNTHETIC, Demo Director for that session only).
    static var launchScreen: String? {
        let args = ProcessInfo.processInfo.arguments
        if args.contains("-scriptedDemo") { return "scriptedDemo" }
        guard let i = args.firstIndex(of: "-screen"), i + 1 < args.count else { return nil }
        return args[i + 1]
    }
}
