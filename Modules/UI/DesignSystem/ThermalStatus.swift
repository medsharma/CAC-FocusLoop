import Foundation
import Observation

/// The phone's thermal state, observed app-wide. At `.serious` or worse the Metal cortex views pause to a
/// still placeholder (camera tracking + a 60 fps cortex is the app's heaviest load), so a long live demo
/// doesn't heat the phone into throttling the camera pipeline.
@MainActor
@Observable
public final class ThermalStatus {
    public static let shared = ThermalStatus()

    public private(set) var state: ProcessInfo.ThermalState
    @ObservationIgnored private var token: NSObjectProtocol?

    init() {
        state = ProcessInfo.processInfo.thermalState
        token = NotificationCenter.default.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification,
                                                       object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.state = ProcessInfo.processInfo.thermalState }
        }
    }

    /// True when the cortex should hold still to let the phone cool down.
    public var cortexPaused: Bool { Self.shouldPauseCortex(state) }

    public nonisolated static func shouldPauseCortex(_ s: ProcessInfo.ThermalState) -> Bool {
        s == .serious || s == .critical
    }

    public nonisolated static func label(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal: "Normal"
        case .fair: "Warm"
        case .serious: "Hot: the 3D brain pauses to cool down"
        case .critical: "Very hot: let the phone cool before filming"
        @unknown default: "Unknown"
        }
    }
}
