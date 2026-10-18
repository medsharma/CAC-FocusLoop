import AVFoundation
import SwiftUI
import UIKit

/// Bare `AVPlayerLayer` host (custom controls live in SwiftUI; no system chrome).
struct StudyVideoSurface: UIViewRepresentable {
    let player: AVPlayer

    final class PlayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }

    func makeUIView(context: Context) -> PlayerView {
        let v = PlayerView()
        v.backgroundColor = .black
        v.playerLayer.videoGravity = .resizeAspect
        v.playerLayer.player = player
        v.isAccessibilityElement = true
        v.accessibilityLabel = "Lecture video"
        return v
    }

    func updateUIView(_ view: PlayerView, context: Context) {
        if view.playerLayer.player !== player { view.playerLayer.player = player }
    }
}

/// Calm "no video" stage for timer-driven lectures: title, transcript-like recap text, progress.
struct StudyNoVideoStage: View {
    let lecture: LectureInfo
    let recap: LectureInfo.TimedText?
    let progress: Double
    let isPlaying: Bool

    var body: some View {
        ZStack {
            FLColor.surface

            VStack(alignment: .leading, spacing: FLSpacing.s) {
                Text(lecture.title)
                    .flText(.headline)
                    .lineLimit(2)

                Spacer(minLength: 0)

                Text(recap?.text ?? "The lecture is about to begin.")
                    .font(FLFont.body)
                    .foregroundStyle(FLColor.textPrimary.opacity(0.9))
                    .lineSpacing(3)
                    .lineLimit(4)
                    .minimumScaleFactor(0.85)
                    .id(recap?.start ?? -1)
                    .transition(.opacity.combined(with: .offset(y: 6)))

                FLLineMeter(value: progress, color: FLColor.textSecondary, height: 2)
            }
            .padding(FLSpacing.card)
            .animation(.easeInOut(duration: 0.6), value: recap?.start)
        }
        .accessibilityElement(children: .combine)
    }
}
