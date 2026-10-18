import SwiftUI

/// Scrubber with predicted valley markers, a predicted-engagement sparkline and the playhead.
/// Tap or drag anywhere to seek.
struct StudyScrubber: View {
    let duration: Double
    let playhead: Double
    let valleys: [(start: Double, end: Double)]
    /// Predicted engagement 0...1, evenly sampled over the lecture.
    let engagement: [Float]
    var onSeek: (Double) -> Void
    var onScrubbingChanged: (Bool) -> Void = { _ in }

    @State private var isScrubbing = false

    private let sparkHeight: CGFloat = 32
    /// Centre of the 3pt track.
    private let trackY: CGFloat = 42
    private let trackHeight: CGFloat = 3
    private let totalHeight: CGFloat = 52

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let px = StudyScrubberGeometry.x(for: playhead, duration: duration, width: w)
            let segments = StudyScrubberGeometry.valleySegments(valleys, duration: duration, width: w)
            let activeValley = StudyScrubberGeometry.valleyIndex(containing: playhead, in: valleys)

            ZStack(alignment: .topLeading) {
                sparkline(width: w, segments: segments, activeValley: activeValley, playheadX: px)

                // Track (3pt): hairline remainder, elapsed in textPrimary.
                Capsule().fill(FLColor.hairlineStrong)
                    .frame(width: w, height: trackHeight)
                    .offset(y: trackY - trackHeight / 2)
                Capsule().fill(FLColor.textPrimary)
                    .frame(width: max(trackHeight, px), height: trackHeight)
                    .offset(y: trackY - trackHeight / 2)

                // Predicted valleys on the track (amber; the one under the playhead at full strength).
                ForEach(Array(segments.enumerated()), id: \.offset) { i, seg in
                    Capsule()
                        .fill(StudyModeStyle.valley.opacity(i == activeValley ? 1 : 0.7))
                        .frame(width: max(trackHeight, seg.upperBound - seg.lowerBound), height: trackHeight)
                        .offset(x: seg.lowerBound, y: trackY - trackHeight / 2)
                }

                // Playhead: hairline through the sparkline, small knob on the track.
                Rectangle().fill(FLColor.textPrimary.opacity(0.45))
                    .frame(width: 1, height: sparkHeight)
                    .offset(x: px - 0.5, y: 0)
                Circle().fill(FLColor.textPrimary)
                    .frame(width: 12, height: 12)
                    .scaleEffect(isScrubbing ? 1.4 : 1)
                    .offset(x: px - 6, y: trackY - 6)
                    .animation(FLMotion.quick, value: isScrubbing)
            }
            .frame(width: w, height: totalHeight, alignment: .topLeading)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { g in
                        if !isScrubbing { isScrubbing = true; onScrubbingChanged(true) }
                        onSeek(StudyScrubberGeometry.time(forX: g.location.x, duration: duration, width: w))
                    }
                    .onEnded { g in
                        onSeek(StudyScrubberGeometry.time(forX: g.location.x, duration: duration, width: w))
                        isScrubbing = false
                        onScrubbingChanged(false)
                    }
            )
        }
        .frame(height: totalHeight)
        .accessibilityElement()
        .accessibilityLabel("Lecture position")
        .accessibilityIdentifier("study.scrubber")
        .accessibilityValue("\(StudyScrubberGeometry.timeLabel(playhead)) of \(StudyScrubberGeometry.timeLabel(duration)), \(valleys.count) predicted low-engagement stretches")
        .accessibilityAdjustableAction { dir in
            switch dir {
            case .increment: onSeek(min(duration, playhead + 10))
            case .decrement: onSeek(max(0, playhead - 10))
            @unknown default: break
            }
        }
    }

    private func sparkline(width w: CGFloat, segments: [ClosedRange<CGFloat>], activeValley: Int?,
                           playheadX: CGFloat) -> some View {
        let h = sparkHeight
        let pts = StudyScrubberGeometry.sparklinePoints(engagement, width: w, height: h - 6)
            .map { CGPoint(x: $0.x, y: $0.y + 3) }
        return Canvas { ctx, _ in
            // Predicted valleys: amber 25% bands from the top of the sparkline down to the track.
            for (i, seg) in segments.enumerated() {
                let r = CGRect(x: seg.lowerBound, y: 0, width: max(2, seg.upperBound - seg.lowerBound), height: trackY)
                let opacity = StudyModeStyle.valleyBandOpacity * (i == activeValley ? 1.4 : 1)
                ctx.fill(Path(roundedRect: r, cornerRadius: 3), with: .color(StudyModeStyle.valley.opacity(opacity)))
            }
            guard pts.count >= 2 else { return }
            var line = Path()
            line.addLines(pts)
            var area = line
            area.addLine(to: CGPoint(x: pts.last!.x, y: h))
            area.addLine(to: CGPoint(x: pts.first!.x, y: h))
            area.closeSubpath()
            // Elapsed part: accent line with a 12% fade below; the rest stays quiet.
            ctx.drawLayer { layer in
                layer.clip(to: Path(CGRect(x: 0, y: 0, width: playheadX, height: h)))
                layer.fill(area, with: .linearGradient(
                    Gradient(colors: [FLColor.accent.opacity(0.12), FLColor.accent.opacity(0)]),
                    startPoint: .zero, endPoint: CGPoint(x: 0, y: h)))
                layer.stroke(line, with: .color(FLColor.accent), style: FLChart.line)
            }
            ctx.drawLayer { layer in
                layer.clip(to: Path(CGRect(x: playheadX, y: 0, width: max(0, w - playheadX), height: h)))
                layer.stroke(line, with: .color(FLColor.textTertiary.opacity(0.7)), style: FLChart.line)
            }
        }
        .frame(width: w, height: trackY)
        .allowsHitTesting(false)
    }
}
