import SwiftUI

/// A 60 s stand-and-stretch break: a countdown ring plus simple prompts that change as time passes.
public struct StretchCountdownView: View {
    public var seconds: Int
    public var onFinish: (() -> Void)?

    @State private var startDate = Date()
    @State private var finished = false

    public static let prompts = [
        "Stand up and roll your shoulders back a few times.",
        "Reach both arms overhead and stretch tall.",
        "Gently turn your upper body left, then right.",
        "Shake out your hands and take one slow, deep breath.",
    ]

    public init(seconds: Int = 60, onFinish: (() -> Void)? = nil) {
        self.seconds = seconds
        self.onFinish = onFinish
    }

    /// Index of the prompt shown at `elapsed` seconds.
    public static func promptIndex(elapsed: Double, total: Int) -> Int {
        let per = Double(max(total, 1)) / Double(prompts.count)
        return min(prompts.count - 1, max(0, Int(elapsed / per)))
    }

    public var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30)) { context in
            let elapsed = min(Double(seconds), context.date.timeIntervalSince(startDate))
            let remaining = max(0, Double(seconds) - elapsed)
            let fraction = seconds > 0 ? remaining / Double(seconds) : 0
            let idx = Self.promptIndex(elapsed: elapsed, total: seconds)
            VStack(spacing: FLSpacing.xl) {
                FLEyebrow("Stretch break")

                // Thin countdown ring (2pt) on a hairline track.
                ZStack {
                    Circle().stroke(FLColor.hairlineStrong, lineWidth: 2)
                    Circle()
                        .trim(from: 0, to: fraction)
                        .stroke(InterventionStyle.calm, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                    VStack(spacing: FLSpacing.xxs) {
                        Text("\(Int(remaining.rounded(.up)))")
                            .flText(.hero)
                            .contentTransition(.numericText(countsDown: true))
                        Text("seconds").flText(.caption)
                    }
                }
                .frame(width: 240, height: 240)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(Int(remaining.rounded(.up))) seconds left")

                VStack(spacing: FLSpacing.m) {
                    Text(Self.prompts[idx])
                        .flText(.title)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(minHeight: 60, alignment: .top)
                        .id(idx)
                        .transition(.opacity)

                    HStack(spacing: 6) {
                        ForEach(Self.prompts.indices, id: \.self) { i in
                            Capsule()
                                .fill(i <= idx ? FLColor.textPrimary : FLColor.hairlineStrong)
                                .frame(width: i == idx ? 16 : 6, height: 6)
                        }
                    }
                    .accessibilityHidden(true)
                }
                .animation(.easeInOut(duration: 0.5), value: idx)

                if let onFinish {
                    Button(remaining > 0 ? "Skip" : "Back to it", action: onFinish)
                        .buttonStyle(FLSecondaryButtonStyle(fullWidth: false))
                        .padding(.top, FLSpacing.xs)
                }
            }
            .padding(.horizontal, FLSpacing.l)
        }
        // On black: opaque, so nothing underneath bleeds through.
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(FLColor.bg.ignoresSafeArea())
        .environment(\.colorScheme, .dark)
        .task {
            try? await Task.sleep(nanoseconds: UInt64(max(0, seconds)) * 1_000_000_000)
            if !Task.isCancelled, !finished {
                finished = true
                onFinish?()
            }
        }
    }
}

#Preview("Stretch") {
    ZStack {
        FLColor.bg.ignoresSafeArea()
        StretchCountdownView(seconds: 60, onFinish: {})
    }
}
