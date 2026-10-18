import SwiftUI
import UIKit
import QuartzCore

/// SART screen: intro (optional practice block) -> task -> summary. A thin layer over `SARTEngine`.
///
/// Timing: the task surface is a UIKit view driven by `CADisplayLink`. Each callback ticks the engine with
/// `link.targetTimestamp` (the presentation time of the frame being prepared) and updates layers inside the
/// same callback with implicit animations off, so the recorded onset is the timestamp of the frame on which
/// the digit actually appears. Taps use `UITouch.timestamp`, the same clock (CACurrentMediaTime).
/// `sessionClockOrigin` maps that clock onto session time (session t = media time - origin) so trial times
/// line up with the FeatureFrames recorded meanwhile.
public struct SARTView: View {
    private enum Stage: Equatable { case intro, running(practice: Bool), practiceDone(SARTResult), done(SARTResult) }

    private let config: SARTConfig
    private let sessionClockOrigin: Double
    private let offerPractice: Bool
    private let onComplete: (SARTResult) -> Void
    /// Shows "Skip for now" on the intro and after practice (nil hides it).
    private let onSkip: (() -> Void)?
    @State private var stage: Stage = .intro

    public init(config: SARTConfig = SARTConfig(), sessionClockOrigin: Double = 0, offerPractice: Bool = true,
                onSkip: (() -> Void)? = nil, onComplete: @escaping (SARTResult) -> Void) {
        self.onSkip = onSkip
        self.config = config
        self.sessionClockOrigin = sessionClockOrigin
        self.offerPractice = offerPractice
        self.onComplete = onComplete
    }

    public var body: some View {
        ZStack {
            // Calm chrome around the task; the task surface itself stays plain black (standard SART).
            FLColor.bg.ignoresSafeArea()
            switch stage {
            case .intro:
                intro
            case let .running(practice):
                SARTTaskSurface(config: practice ? practiceConfig : config, clockOrigin: sessionClockOrigin) { r in
                    stage = practice ? .practiceDone(r) : .done(r)
                    if !practice { onComplete(r) }
                }
                .ignoresSafeArea()
                .statusBarHidden()
                .persistentSystemOverlays(.hidden)
            case let .practiceDone(r):
                summary(r, practice: true)
            case let .done(r):
                summary(r, practice: false)
            }
        }
        .preferredColorScheme(.dark)
    }

    private var practiceConfig: SARTConfig {
        var c = SARTConfig.practice
        c.fontSizes = config.fontSizes
        return c
    }

    private var intro: some View {
        VStack(spacing: 0) {
            Spacer(minLength: FLSpacing.l)
            Image(systemName: "hand.tap")
                .font(.system(size: 44, weight: .ultraLight))
                .foregroundStyle(FLColor.accent)
                .accessibilityHidden(true)
                .padding(.bottom, FLSpacing.l)
            Text("Attention check")
                .flText(.largeTitle)
                .padding(.bottom, FLSpacing.xl)
            VStack(alignment: .leading, spacing: 0) {
                SARTInstructionRow(symbol: "number", text: "Digits 1 to 9 will flash in the middle of the screen.")
                FLDivider(inset: 44)
                SARTInstructionRow(symbol: "hand.tap", text: "Tap anywhere for every digit…")
                FLDivider(inset: 44)
                SARTInstructionRow(symbol: "hand.raised", text: "…except 3. Don't tap for 3.")
                FLDivider(inset: 44)
                SARTInstructionRow(symbol: "timer",
                                   text: "Takes about \(Int((config.estimatedDuration / 60).rounded())) minutes. Speed and accuracy both matter.")
            }
            .padding(.horizontal, FLSpacing.m)
            .padding(.vertical, FLSpacing.xxs)
            .flRowGroup()
            Spacer(minLength: FLSpacing.l)
            VStack(spacing: FLSpacing.xxs) {
                Button("Start") { stage = .running(practice: false) }
                    .buttonStyle(FLPrimaryButtonStyle())
                    .padding(.bottom, FLSpacing.xxs)
                if offerPractice {
                    Button("Try a practice round (\(SARTConfig.practice.trialCount) trials)") { stage = .running(practice: true) }
                        .buttonStyle(FLTextButtonStyle())
                }
                skipButton
            }
        }
        .padding(.horizontal, FLSpacing.gutter)
        .padding(.top, FLSpacing.l)
        .padding(.bottom, FLSpacing.xs)
    }

    private func summary(_ r: SARTResult, practice: Bool) -> some View {
        VStack(spacing: 0) {
            Spacer()
            Image(systemName: practice ? "checkmark.circle" : "checkmark.seal")
                .font(.system(size: 44, weight: .ultraLight))
                .foregroundStyle(FLColor.accent)
                .accessibilityHidden(true)
                .padding(.bottom, FLSpacing.l)
            Text(practice ? "Practice done" : "All done")
                .flText(.largeTitle)
                .padding(.bottom, FLSpacing.xl)
            FLMetricStrip([
                FLMetric("Mean response", value: r.meanRT.map { "\(Int(($0 * 1000).rounded()))" } ?? "–",
                         unit: r.meanRT == nil ? nil : "ms"),
                FLMetric("Taps on 3", value: "\(r.commissionCount)/\(r.noGoCount)"),
                FLMetric("Missed", value: "\(r.omissionCount)"),
            ])
            .padding(.vertical, FLSpacing.m)
            .overlay(alignment: .top) { FLDivider() }
            .overlay(alignment: .bottom) { FLDivider() }
            Text(practice ? "Ready for the real thing? Remember: never tap for 3."
                          : "Your responses help FocusLoop learn what your focus looks like. It stays on this device.")
                .flText(.callout)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, FLSpacing.l)
            Spacer()
            if practice {
                Button("Start the task") { stage = .running(practice: false) }
                    .buttonStyle(FLPrimaryButtonStyle())
                    .padding(.bottom, FLSpacing.xxs)
                skipButton
            }
        }
        .padding(.horizontal, FLSpacing.gutter)
        .padding(.top, FLSpacing.l)
        .padding(.bottom, FLSpacing.xs)
    }
}

private extension SARTView {
    @ViewBuilder var skipButton: some View {
        if let onSkip {
            Button("Skip for now", action: onSkip)
                .buttonStyle(FLTextButtonStyle(color: FLColor.textSecondary))
        }
    }
}

private struct SARTInstructionRow: View {
    let symbol: String
    let text: String
    var body: some View {
        HStack(alignment: .center, spacing: FLSpacing.m) {
            Image(systemName: symbol)
                .font(.system(size: FLIconSize.row, weight: .regular))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(FLColor.textSecondary)
                .frame(width: 28)
                .accessibilityHidden(true)
            Text(text)
                .font(FLFont.body)
                .foregroundStyle(FLColor.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 14)
        .frame(minHeight: 56)
    }
}

// MARK: - UIKit task surface

private struct SARTTaskSurface: UIViewRepresentable {
    let config: SARTConfig
    let clockOrigin: Double
    let onFinish: (SARTResult) -> Void

    func makeUIView(context: Context) -> SARTDisplayView {
        let engine = SARTEngine(config: config, clockOrigin: clockOrigin)
        let v = SARTDisplayView(engine: engine)
        v.onFinish = onFinish
        return v
    }

    func updateUIView(_ uiView: SARTDisplayView, context: Context) {}

    static func dismantleUIView(_ uiView: SARTDisplayView, coordinator: ()) { uiView.stop() }
}

/// Weak trampoline so the display link doesn't retain the view.
private final class SARTDisplayLinkProxy: NSObject {
    weak var target: SARTDisplayView?
    init(_ target: SARTDisplayView) { self.target = target }
    @objc func step(_ link: CADisplayLink) { target?.step(link) }
}

final class SARTDisplayView: UIView {
    let engine: SARTEngine
    var onFinish: ((SARTResult) -> Void)?

    private let digitLabel = UILabel()
    private let maskLayer = CAShapeLayer()
    private var link: CADisplayLink?
    private var shown: SARTDisplay = .blank
    private var reported = false

    init(engine: SARTEngine) {
        self.engine = engine
        super.init(frame: .zero)
        backgroundColor = .black
        isMultipleTouchEnabled = false
        digitLabel.textColor = .white
        digitLabel.textAlignment = .center
        digitLabel.isHidden = true
        addSubview(digitLabel)
        maskLayer.strokeColor = UIColor.white.cgColor
        maskLayer.fillColor = UIColor.clear.cgColor
        maskLayer.lineWidth = 6
        maskLayer.isHidden = true
        layer.addSublayer(maskLayer)
        isAccessibilityElement = true
        accessibilityLabel = "Attention task. Tap for every digit except 3."
        accessibilityTraits = .allowsDirectInteraction
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    override func layoutSubviews() {
        super.layoutSubviews()
        digitLabel.frame = bounds
        // Mask: ring with a diagonal cross, sized to the largest digit.
        let d = CGFloat(engine.config.fontSizes.max() ?? 120) * 1.15
        let rect = CGRect(x: bounds.midX - d / 2, y: bounds.midY - d / 2, width: d, height: d)
        let p = UIBezierPath(ovalIn: rect)
        let r = d / 2 * 0.7071
        p.move(to: CGPoint(x: bounds.midX - r, y: bounds.midY - r)); p.addLine(to: CGPoint(x: bounds.midX + r, y: bounds.midY + r))
        p.move(to: CGPoint(x: bounds.midX + r, y: bounds.midY - r)); p.addLine(to: CGPoint(x: bounds.midX - r, y: bounds.midY + r))
        CATransaction.begin(); CATransaction.setDisableActions(true)
        maskLayer.frame = bounds
        maskLayer.path = p.cgPath
        CATransaction.commit()
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window != nil { start() } else { stop() }
    }

    func start() {
        guard link == nil, !engine.isFinished else { return }
        let l = CADisplayLink(target: SARTDisplayLinkProxy(self), selector: #selector(SARTDisplayLinkProxy.step(_:)))
        // >60 Hz on ProMotion iPhones also needs CADisableMinimumFrameDurationOnPhone in Info.plist.
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120, preferred: 120)
        l.add(to: .main, forMode: .common)
        link = l
    }

    func stop() {
        link?.invalidate()
        link = nil
    }

    fileprivate func step(_ link: CADisplayLink) {
        let d = engine.tick(frameTime: link.targetTimestamp)
        if d != shown {
            render(d)
            shown = d
        }
        if engine.isFinished, !reported {
            reported = true
            stop()
            let r = engine.result
            DispatchQueue.main.async { [weak self] in self?.onFinish?(r) }
        }
    }

    private func render(_ d: SARTDisplay) {
        CATransaction.begin(); CATransaction.setDisableActions(true)
        switch d {
        case let .digit(n, size):
            digitLabel.font = .systemFont(ofSize: CGFloat(size), weight: .semibold)
            digitLabel.text = "\(n)"
            digitLabel.isHidden = false
            maskLayer.isHidden = true
        case .mask:
            digitLabel.isHidden = true
            maskLayer.isHidden = false
        case .blank, .finished:
            digitLabel.isHidden = true
            maskLayer.isHidden = true
        }
        CATransaction.commit()
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        for t in touches { engine.tap(at: t.timestamp) }
    }
}
