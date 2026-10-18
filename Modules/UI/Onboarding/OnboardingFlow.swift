import SwiftUI

/// Onboarding: mission, privacy promise, permissions, work-zone calibration (slot), SART (slot).
public struct OnboardingFlow: View {
    let services: UIServices
    var onComplete: () -> Void

    enum Step: Int, CaseIterable { case mission, privacy, permissions, calibration, sart, done }
    @State private var step: Step
    @State private var statuses: [UIPermission: UIPermissionStatus] = [:]

    public init(services: UIServices, startAt: Int = 0, onComplete: @escaping () -> Void) {
        self.services = services; self.onComplete = onComplete
        _step = State(initialValue: Step(rawValue: startAt) ?? .mission)
    }

    public var body: some View {
        ZStack {
            FLBackground(glow: step == .mission ? FLColor.accent : nil, center: UnitPoint(x: 0.5, y: 0.42), intensity: 0.09)
                .ignoresSafeArea()
                .animation(FLMotion.fade, value: step)
            VStack(spacing: 0) {
                progress
                    .padding(.top, FLSpacing.xs)
                    .padding(.bottom, FLSpacing.xs)
                Group {
                    switch step {
                    case .mission: mission
                    case .privacy: privacy
                    case .permissions: permissions
                    case .calibration:
                        services.calibrationView(onFinish: { advance() })
                            ?? AnyView(PlaceholderCalibrationView(onFinish: { advance() }))
                    case .sart:
                        services.sartView(onFinish: { advance() }) ?? AnyView(PlaceholderSARTView(onFinish: { advance() }))
                    case .done: done
                    }
                }
                .transition(reduceMotion ? .opacity
                            : .asymmetric(insertion: .offset(x: 24).combined(with: .opacity),
                                          removal: .offset(x: -24).combined(with: .opacity)))
                .id(step)
            }
        }
        .onAppear { for p in UIPermission.allCases { statuses[p] = services.permissionStatus(p) } }
    }

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private func advance() {
        FLHaptics.tap()
        withAnimation(reduceMotion ? FLMotion.fade : .spring(response: 0.45, dampingFraction: 0.9)) {
            step = Step(rawValue: step.rawValue + 1) ?? .done
        }
    }

    /// Five dots; the current one stretches into a short accent capsule.
    private var progress: some View {
        HStack(spacing: 6) {
            ForEach(Step.allCases.dropLast(), id: \.self) { s in
                Capsule()
                    .fill(s == step ? FLColor.accent : (s.rawValue < step.rawValue ? FLColor.textTertiary : FLColor.hairlineStrong))
                    .frame(width: s == step ? 18 : 6, height: 6)
            }
        }
        .frame(height: 20)
        .opacity(step == .done ? 0 : 1)
        .animation(FLMotion.standard, value: step)
        .accessibilityElement()
        .accessibilityLabel("Step \(min(step.rawValue + 1, 5)) of 5")
    }

    /// Bottom-pinned primary CTA with 20pt insets.
    private func cta(_ title: String, action: @escaping () -> Void) -> some View {
        Button(title, action: action)
            .buttonStyle(FLPrimaryButtonStyle())
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.bottom, FLSpacing.xs)
            .padding(.top, FLSpacing.s)
            .background(FLColor.bg.opacity(step == .mission ? 0 : 1))
    }

    // MARK: Pages

    private var mission: some View {
        VStack(spacing: 0) {
            Spacer(minLength: FLSpacing.l)
            FocusOrb(probabilities: [0.9, 0.04, 0.03, 0.03], lapseRisk: 0.15)
                .frame(width: 220, height: 220)
                .accessibilityHidden(true)
            Spacer(minLength: FLSpacing.xl).frame(maxHeight: 120)
            VStack(spacing: FLSpacing.s) {
                FLEyebrow("The attention forecast")
                Text("See the slip before it happens.").flText(.largeTitle)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.isHeader)
                Text("FocusLoop reads the attention signals in your face and the pulse in your skin, forecasts a lapse up to 15 seconds before it happens, then runs a personal experiment to learn what brings you back.")
                    .flText(.body)
                    .multilineTextAlignment(.center)
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
                Text("Computer vision · Camera heart rate · Brain-encoding AI")
                    .flText(.caption)
                    .multilineTextAlignment(.center)
                    .padding(.top, FLSpacing.xs)
            }
            .padding(.horizontal, FLSpacing.xl)
            Spacer(minLength: FLSpacing.l)
        }
        .frame(maxWidth: .infinity)
        .safeAreaInset(edge: .bottom) { cta("Get started") { advance() } }
    }

    private var privacy: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                FLEyebrow("Privacy")
                    .padding(.top, FLSpacing.xl)
                Text("0 bytes of video stored or sent.").flText(.largeTitle)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, FLSpacing.s)
                Text("Everything runs on your phone.").flText(.body)
                    .padding(.top, FLSpacing.s)
                VStack(alignment: .leading, spacing: 0) {
                    promise("camera.metering.none", "Frames become numbers", "Each camera frame is turned into a few numbers (blinks, gaze angle, heart rate) and then thrown away.")
                    FLDivider(inset: 44)
                    promise("wifi.slash", "No network, no analytics", "FocusLoop never connects to the internet. It works in airplane mode.")
                    FLDivider(inset: 44)
                    promise("trash", "You're in control", "See exactly what's stored, and delete it all with one tap in Privacy.")
                }
                .padding(.top, FLSpacing.xl)
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.bottom, FLSpacing.l)
        }
        .scrollBounceBehavior(.basedOnSize)
        .safeAreaInset(edge: .bottom) { cta("Continue") { advance() } }
    }

    private func promise(_ icon: String, _ title: String, _ body: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: FLSpacing.m) {
            Image(systemName: icon)
                .font(.system(size: FLIconSize.row, weight: .regular))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(FLColor.accent)
                .frame(width: 28)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: FLSpacing.xxs) {
                Text(title).flText(.headline)
                Text(body).flText(.callout).lineSpacing(2).fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.vertical, FLSpacing.m)
        .accessibilityElement(children: .combine)
    }

    private var permissions: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                FLEyebrow("Permissions")
                    .padding(.top, FLSpacing.xl)
                Text("A few permissions").flText(.largeTitle)
                    .padding(.top, FLSpacing.s)
                Text("Each one is used on your phone only.").flText(.body)
                    .padding(.top, FLSpacing.s)
                VStack(alignment: .leading, spacing: 0) {
                    permissionRow(.camera, icon: "camera", title: "Camera",
                                  body: "Reads attention signals (gaze, blinks, head pose) and estimates heart rate from subtle skin-color changes. No video is ever saved.")
                    FLDivider(inset: 44)
                    permissionRow(.motion, icon: "iphone.radiowaves.left.and.right", title: "Motion",
                                  body: "Notices when you pick up your phone mid-session.")
                    FLDivider(inset: 44)
                    permissionRow(.health, icon: "heart", title: "Health (optional)",
                                  body: "Reads Apple Watch heart rate to check the camera estimate. Nothing is written to Health.")
                }
                .padding(.top, FLSpacing.xl)
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.bottom, FLSpacing.l)
        }
        .scrollBounceBehavior(.basedOnSize)
        .safeAreaInset(edge: .bottom) { cta("Continue") { advance() } }
    }

    private func permissionRow(_ p: UIPermission, icon: String, title: String, body: String) -> some View {
        let status = statuses[p] ?? .notDetermined
        return HStack(alignment: .firstTextBaseline, spacing: FLSpacing.m) {
            Image(systemName: icon)
                .font(.system(size: FLIconSize.row, weight: .regular))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(FLColor.textSecondary)
                .frame(width: 28)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: FLSpacing.xxs) {
                HStack(alignment: .firstTextBaseline) {
                    Text(title).flText(.headline)
                    Spacer(minLength: FLSpacing.xs)
                    switch status {
                    case .granted:
                        Label("Allowed", systemImage: "checkmark")
                            .font(.system(.subheadline, weight: .medium))
                            .foregroundStyle(FLColor.accent)
                            .labelStyle(FLButtonLabelStyle())
                    case .notDetermined:
                        Button("Allow") {
                            Task { statuses[p] = await services.requestPermission(p) }
                        }
                        .font(.system(.subheadline, weight: .semibold))
                        .foregroundStyle(FLColor.accent)
                        .padding(.horizontal, FLSpacing.m)
                        .frame(minHeight: 32)
                        .background(FLColor.accent.opacity(0.12), in: Capsule())
                        .contentShape(Capsule().inset(by: -6))
                        .buttonStyle(FLPressableStyle())
                    default:
                        EmptyView()
                    }
                }
                Text(body).flText(.callout).lineSpacing(2).fixedSize(horizontal: false, vertical: true)
                switch status {
                case .denied:
                    Text("Off. You can change this in Settings.").flText(.caption, color: FLColor.distracted)
                        .padding(.top, FLSpacing.xxs)
                case .unavailable:
                    Text("Not available on this device").flText(.caption)
                        .padding(.top, FLSpacing.xxs)
                default:
                    EmptyView()
                }
            }
        }
        .padding(.vertical, FLSpacing.m)
    }

    private var done: some View {
        VStack(spacing: 0) {
            Spacer()
            ZStack {
                Circle().stroke(FLColor.accent.opacity(0.25), lineWidth: 0.75).frame(width: 112, height: 112)
                Circle().fill(FLColor.accent.opacity(0.10)).frame(width: 80, height: 80)
                Image(systemName: "checkmark")
                    .font(.system(size: 30, weight: .regular))
                    .foregroundStyle(FLColor.accent)
                    .symbolEffect(.bounce, value: step)
            }
            .accessibilityHidden(true)
            Text("You're set").flText(.largeTitle)
                .padding(.top, FLSpacing.xl)
            Text("Prop your phone next to your laptop with the front camera facing you, and start a session.")
                .flText(.body).multilineTextAlignment(.center).lineSpacing(3)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, FLSpacing.s)
                .padding(.horizontal, FLSpacing.l)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, FLSpacing.gutter)
        .safeAreaInset(edge: .bottom) { cta("Start using FocusLoop") { FLHaptics.success(); onComplete() } }
    }
}
