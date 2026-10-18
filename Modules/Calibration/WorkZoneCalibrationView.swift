import SwiftUI
import QuartzCore

/// Drives one calibration pass from a stream of combined gaze readings (live FeatureFrames, a mock,
/// or the SYNTHETIC simulator), then builds and saves the `WorkZone`.
@MainActor
@Observable
public final class WorkZoneCalibrationModel {
    public enum Phase: Equatable { case intro, collecting, result(WorkZone) }

    public let isPhoneScreen: Bool
    public private(set) var phase: Phase = .intro
    public private(set) var run = WorkZoneCalibrationRun()
    /// True when the last result came from the SYNTHETIC simulator rather than real gaze.
    public private(set) var usedSimulation = false
    public private(set) var lastReading: CalibrationGazeReading?

    private let makeSamples: () -> AsyncStream<CalibrationGazeReading>
    private let builder: any CalibrationWorkZoneBuilder
    private let store: WorkZoneStore
    private var task: Task<Void, Never>?

    public init(isPhoneScreen: Bool, builder: any CalibrationWorkZoneBuilder = CalibrationMedianWorkZoneBuilder(),
                store: WorkZoneStore = WorkZoneStore(),
                samples: @escaping () -> AsyncStream<CalibrationGazeReading>) {
        self.isPhoneScreen = isPhoneScreen
        self.builder = builder
        self.store = store
        self.makeSamples = samples
    }

    public func start(simulated: Bool = false) {
        cancel()
        run = WorkZoneCalibrationRun()
        usedSimulation = simulated
        phase = .collecting
        if simulated {
            let phone = isPhoneScreen
            task = Task { [weak self] in
                var sim = CalibrationGazeSimulator(isPhoneScreen: phone, seed: UInt64(CACurrentMediaTime() * 1000))
                while !Task.isCancelled {
                    guard let self, let target = self.run.currentTarget else { return }
                    self.ingest(sim.reading(for: target, at: CACurrentMediaTime()))
                    try? await Task.sleep(nanoseconds: 33_000_000)
                }
            }
        } else {
            let stream = makeSamples()
            task = Task { [weak self] in
                for await r in stream {
                    guard let self, !Task.isCancelled else { return }
                    self.ingest(r)
                    if self.run.isFinished { return }
                }
            }
        }
    }

    public func ingest(_ r: CalibrationGazeReading) {
        guard phase == .collecting else { return }
        lastReading = r
        if run.ingest(r) { finish() }
    }

    public func cancel() {
        task?.cancel()
        task = nil
        if phase == .collecting { phase = .intro }
    }

    private func finish() {
        task?.cancel()
        task = nil
        let zone = run.makeWorkZone(builder: builder, isPhoneScreen: isPhoneScreen)
        store.save(zone)
        phase = .result(zone)
    }
}

/// Work-zone calibration flow (laptop beside the phone, or the phone screen itself for Study Mode).
public struct WorkZoneCalibrationView: View {
    @State private var model: WorkZoneCalibrationModel
    private let allowSimulation: Bool
    private let onComplete: (WorkZone) -> Void

    /// - Parameters:
    ///   - samples: factory for a fresh stream of combined (gaze+head) readings, e.g.
    ///     `{ CalibrationGazeReading.stream(from: source.frames) }`.
    ///   - allowSimulation: shows the "Simulate calibration (SYNTHETIC)" button (default: Simulator only).
    public init(isPhoneScreen: Bool = false,
                builder: any CalibrationWorkZoneBuilder = CalibrationMedianWorkZoneBuilder(),
                store: WorkZoneStore = WorkZoneStore(),
                allowSimulation: Bool = CalibrationEnvironment.isSimulator,
                samples: @escaping () -> AsyncStream<CalibrationGazeReading>,
                onComplete: @escaping (WorkZone) -> Void) {
        _model = State(initialValue: WorkZoneCalibrationModel(isPhoneScreen: isPhoneScreen, builder: builder,
                                                              store: store, samples: samples))
        self.allowSimulation = allowSimulation
        self.onComplete = onComplete
    }

    public var body: some View {
        ZStack {
            FLColor.bg.ignoresSafeArea()
            switch model.phase {
            case .intro: intro
            case .collecting: collecting
            case let .result(zone): result(zone)
            }
        }
        .preferredColorScheme(.dark)
        .animation(.smooth, value: model.phase)
        .onDisappear { model.cancel() }
    }

    // MARK: Intro

    private var intro: some View {
        VStack(spacing: 0) {
            Spacer()
            Image(systemName: model.isPhoneScreen ? "iphone" : "laptopcomputer.and.iphone")
                .font(.system(size: 48, weight: .ultraLight))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(FLColor.accent)
                .accessibilityHidden(true)
                .padding(.bottom, FLSpacing.xl)
            Text(model.isPhoneScreen ? "Calibrate for this screen" : "Set your work zone")
                .flText(.largeTitle)
                .multilineTextAlignment(.center)
                .padding(.bottom, FLSpacing.s)
            Text(model.isPhoneScreen
                 ? "Hold the phone where you'll watch lectures. Follow the dot to the center and each corner of this screen, about 2 seconds each."
                 : "Prop your phone up next to your laptop with the front camera facing you. Then look at the center of your laptop screen and each of its four corners, about 2 seconds each.")
                .flText(.body)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button("Begin") { model.start() }
                .buttonStyle(FLPrimaryButtonStyle())
            if allowSimulation {
                Button("Simulate calibration") { model.start(simulated: true) }
                    .buttonStyle(FLTextButtonStyle(color: FLColor.textSecondary))
                    .padding(.top, FLSpacing.xs)
            }
        }
        .padding(.horizontal, FLSpacing.gutter)
        .padding(.top, FLSpacing.l)
        .padding(.bottom, FLSpacing.xs)
    }

    // MARK: Collecting

    @ViewBuilder private var collecting: some View {
        if model.isPhoneScreen {
            GeometryReader { geo in
                ZStack {
                    if let target = model.run.currentTarget {
                        CalibrationTargetDot(progress: model.run.progress, isActive: true)
                            .position(position(of: target, in: geo.size, inset: 44))
                            .animation(.smooth(duration: 0.35), value: target)
                    }
                    VStack {
                        Spacer()
                        caption
                    }
                    .padding(.bottom, 40)
                }
            }
            .ignoresSafeArea()
        } else {
            VStack(spacing: FLSpacing.s) {
                Spacer()
                // Fixed slot: one instruction at a time.
                ZStack {
                    Text(model.run.currentTarget?.instruction ?? "")
                        .flText(.title)
                        .multilineTextAlignment(.center)
                        .id(model.run.currentTarget)
                        .transition(.opacity)
                }
                .frame(maxWidth: .infinity, minHeight: 60)
                .animation(FLMotion.fade, value: model.run.currentTarget)
                Text("of your laptop screen")
                    .flText(.body)
                laptopDiagram
                    .frame(height: 220)
                    .padding(.top, FLSpacing.xl)
                Spacer()
                caption
                Button("Cancel") { model.cancel() }
                    .buttonStyle(FLTextButtonStyle(color: FLColor.textSecondary))
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.vertical, FLSpacing.l)
        }
    }

    private var caption: some View {
        HStack(spacing: FLSpacing.xs) {
            if model.usedSimulation { Text("Simulated").font(FLFont.caption).foregroundStyle(FLColor.textTertiary) }
            Text("Target \(min(model.run.targetIndex + 1, CalibrationTarget.allCases.count)) of \(CalibrationTarget.allCases.count)")
                .font(FLFont.numeral(.footnote))
                .foregroundStyle(FLColor.textSecondary)
        }
    }

    /// Laptop screen sketch with all 5 targets; the active one carries the progress ring.
    private var laptopDiagram: some View {
        GeometryReader { geo in
            let screen = CGSize(width: geo.size.width, height: geo.size.height * 0.86)
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: FLRadius.small, style: .continuous)
                    .fill(FLColor.surface)
                    .overlay(RoundedRectangle(cornerRadius: FLRadius.small, style: .continuous)
                        .strokeBorder(FLColor.hairlineStrong, lineWidth: 1))
                    .frame(width: screen.width, height: screen.height)
                Capsule().fill(FLColor.surfaceHigh)
                    .overlay(Capsule().strokeBorder(FLColor.hairline, lineWidth: 0.5))
                    .frame(width: screen.width * 1.08, height: 6)
                    .position(x: screen.width / 2, y: screen.height + 10)
                ForEach(CalibrationTarget.allCases) { t in
                    let active = t == model.run.currentTarget
                    let done = t.rawValue < model.run.targetIndex
                    Group {
                        if active {
                            CalibrationTargetDot(progress: model.run.progress, isActive: true)
                        } else {
                            Image(systemName: done ? "checkmark.circle.fill" : "circle")
                                .font(.system(size: 16, weight: .regular))
                                .foregroundStyle(done ? FLColor.accent : FLColor.textTertiary)
                        }
                    }
                    .position(position(of: t, in: screen, inset: 26))
                }
            }
        }
    }

    private func position(of t: CalibrationTarget, in size: CGSize, inset: CGFloat) -> CGPoint {
        let p = t.unitPosition
        return CGPoint(x: size.width / 2 + CGFloat(p.x) * (size.width / 2 - inset),
                       y: size.height / 2 + CGFloat(p.y) * (size.height / 2 - inset))
    }

    // MARK: Result

    private func result(_ zone: WorkZone) -> some View {
        VStack(spacing: 0) {
            Spacer(minLength: FLSpacing.l)
            VStack(spacing: FLSpacing.s) {
                Image(systemName: "scope")
                    .font(.system(size: 36, weight: .ultraLight))
                    .foregroundStyle(FLColor.accent)
                    .accessibilityHidden(true)
                Text("Work zone saved").flText(.largeTitle)
                if model.usedSimulation {
                    Text("Simulated gaze, not your real calibration")
                        .flText(.caption, color: FLColor.textSecondary)
                }
            }
            .padding(.bottom, FLSpacing.l)
            CalibrationConeView(zone: zone)
                .frame(height: 220)
                .flCard(padding: FLSpacing.m)
            Text(String(format: "Center %+.0f° / %+.0f°   ·   Cone ±%.0f° × ±%.0f°",
                        degrees(zone.centerYaw), degrees(zone.centerPitch),
                        degrees(zone.halfWidthYaw), degrees(zone.halfWidthPitch)))
                .font(FLFont.numeral(.footnote))
                .foregroundStyle(FLColor.textTertiary)
                .padding(.top, FLSpacing.s)
            Spacer()
            Button("Done") { onComplete(zone) }
                .buttonStyle(FLPrimaryButtonStyle())
            Button("Recalibrate") { model.start(simulated: model.usedSimulation) }
                .buttonStyle(FLTextButtonStyle(color: FLColor.textSecondary))
                .padding(.top, FLSpacing.xs)
        }
        .padding(.horizontal, FLSpacing.gutter)
        .padding(.top, FLSpacing.l)
        .padding(.bottom, FLSpacing.xs)
    }

    private func degrees(_ r: Float) -> Double { Double(r) * 180 / .pi }
}

/// Progress ring around a fixation dot.
struct CalibrationTargetDot: View {
    let progress: Double
    let isActive: Bool
    var body: some View {
        ZStack {
            Circle().stroke(FLColor.hairlineStrong, lineWidth: 2)
            Circle().trim(from: 0, to: progress)
                .stroke(FLColor.accent, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Circle().fill(FLColor.textPrimary).frame(width: 8, height: 8)
        }
        .frame(width: 44, height: 44)
        .accessibilityLabel("Calibration target, \(Int(progress * 100)) percent")
    }
}

/// The calibrated cone in yaw/pitch space (±0.8 rad view, yaw + = user's left drawn to the left).
struct CalibrationConeView: View {
    let zone: WorkZone
    var body: some View {
        Canvas { ctx, size in
            let range: CGFloat = 0.8
            let sx = size.width / (2 * range), sy = size.height / (2 * range)
            func pt(_ yaw: Float, _ pitch: Float) -> CGPoint {
                CGPoint(x: size.width / 2 - CGFloat(yaw) * sx, y: size.height / 2 - CGFloat(pitch) * sy)
            }
            var axes = Path()
            axes.move(to: CGPoint(x: 0, y: size.height / 2)); axes.addLine(to: CGPoint(x: size.width, y: size.height / 2))
            axes.move(to: CGPoint(x: size.width / 2, y: 0)); axes.addLine(to: CGPoint(x: size.width / 2, y: size.height))
            ctx.stroke(axes, with: .color(FLColor.hairlineStrong), lineWidth: 0.5)
            let c = pt(zone.centerYaw, zone.centerPitch)
            let rect = CGRect(x: c.x - CGFloat(zone.halfWidthYaw) * sx, y: c.y - CGFloat(zone.halfWidthPitch) * sy,
                              width: CGFloat(zone.halfWidthYaw) * 2 * sx, height: CGFloat(zone.halfWidthPitch) * 2 * sy)
            let cone = Path(ellipseIn: rect)
            ctx.fill(cone, with: .color(FLColor.accent.opacity(0.12)))
            ctx.stroke(cone, with: .color(FLColor.accent), lineWidth: 1.5)
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)), with: .color(FLColor.textPrimary))
            ctx.draw(Text("you").font(FLFont.caption).foregroundStyle(FLColor.textTertiary),
                     at: CGPoint(x: size.width / 2, y: size.height / 2 + 12))
        }
        .accessibilityLabel("Calibrated work zone cone")
    }
}
