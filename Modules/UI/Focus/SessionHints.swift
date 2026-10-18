import SwiftUI
import UIKit

/// Live-session hints for the device (camera) path, shown on the Focus and Study Mode screens:
/// * camera access denied: a clear message and an "Open Settings" button (nothing can be sensed without it)
/// * the face has not been seen for a few seconds: a gentle "Can't see your face" hint
/// Never shown on the SYNTHETIC replay.
public struct SessionHintsOverlay: View {
    let model: LiveSessionModel
    let services: UIServices
    /// Distance from the top of the screen (keeps the screen's own badges visible).
    var topInset: CGFloat

    public init(model: LiveSessionModel, services: UIServices, topInset: CGFloat = 64) {
        self.model = model; self.services = services; self.topInset = topInset
    }

    public enum Hint: Equatable { case cameraDenied, faceMissing }

    /// Seconds without a tracked face before the hint shows.
    public static let faceMissingSeconds: Double = 4

    /// Which hint applies (exposed for tests).
    public static func hint(isReplay: Bool, running: Bool, cameraDenied: Bool, recentFrames: [FeatureFrame]) -> Hint? {
        guard !isReplay, running else { return nil }
        if cameraDenied { return .cameraDenied }
        let n = Int(faceMissingSeconds * FeatureFrame.rateHz)
        let recent = recentFrames.suffix(n)
        if recent.count >= n, recent.allSatisfy({ $0.faceDetected < 0.5 }) { return .faceMissing }
        return nil
    }

    private var hint: Hint? {
        Self.hint(isReplay: model.isReplay, running: model.phase == .running,
                  cameraDenied: services.permissionStatus(.camera) == .denied, recentFrames: model.recentFrames)
    }

    public var body: some View {
        VStack {
            if let hint {
                card(hint)
                    .padding(.horizontal, FLSpacing.gutter)
                    .padding(.top, topInset)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
            Spacer()
        }
        .animation(.easeInOut(duration: 0.35), value: hint)
    }

    @ViewBuilder private func card(_ hint: Hint) -> some View {
        switch hint {
        case .cameraDenied:
            VStack(alignment: .leading, spacing: 10) {
                Label("Camera access is off", systemImage: "video.slash.fill")
                    .font(FLFont.headline).foregroundStyle(FLColor.textPrimary)
                Text("FocusLoop needs the front camera to notice focus and heart rate. Video stays on the phone and is never saved.")
                    .font(FLFont.caption).foregroundStyle(FLColor.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button {
                    if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                } label: {
                    Label("Open Settings", systemImage: "gear")
                        .font(FLFont.callout.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(FLColor.accent.opacity(0.18), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
                .buttonStyle(.plain)
                .foregroundStyle(FLColor.accent)
            }
            .padding(FLSpacing.m)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).strokeBorder(FLColor.stroke, lineWidth: 1))
        case .faceMissing:
            Label("Can't see your face. Check the phone is facing you and the light is in front of you.",
                  systemImage: "person.crop.circle.badge.questionmark")
                .font(FLFont.caption.weight(.medium))
                .foregroundStyle(FLColor.textPrimary)
                .padding(.horizontal, 14).padding(.vertical, 10)
                .background(.ultraThinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(FLColor.stroke, lineWidth: 1))
                .allowsHitTesting(false)
                .accessibilityAddTraits(.updatesFrequently)
        }
    }
}
