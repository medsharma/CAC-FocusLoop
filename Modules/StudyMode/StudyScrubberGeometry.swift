import CoreGraphics
import Foundation

/// Pure geometry for the Study Mode scrubber (time <-> x, valley segments, sparkline).
public enum StudyScrubberGeometry {
    /// x position (0...width) for time `t` on a `duration`-long track.
    public static func x(for t: Double, duration: Double, width: CGFloat) -> CGFloat {
        guard duration > 0, width > 0 else { return 0 }
        return CGFloat(min(max(t / duration, 0), 1)) * width
    }

    /// Time for an x position (clamped to 0...duration).
    public static func time(forX x: CGFloat, duration: Double, width: CGFloat) -> Double {
        guard duration > 0, width > 0 else { return 0 }
        return Double(min(max(x / width, 0), 1)) * duration
    }

    /// Valley segments as x ranges, clamped to the track; empty/out-of-range valleys are dropped.
    public static func valleySegments(_ valleys: [(start: Double, end: Double)], duration: Double,
                                      width: CGFloat) -> [ClosedRange<CGFloat>] {
        guard duration > 0, width > 0 else { return [] }
        return valleys.compactMap { v in
            let a = x(for: min(v.start, v.end), duration: duration, width: width)
            let b = x(for: max(v.start, v.end), duration: duration, width: width)
            return b > a ? a...b : nil
        }
    }

    /// Index of the valley containing `t`, if any (half-open [start, end)).
    public static func valleyIndex(containing t: Double, in valleys: [(start: Double, end: Double)]) -> Int? {
        valleys.firstIndex { t >= $0.start && t < $0.end }
    }

    /// Evenly spaced samples of `value` over 0...duration (`count` >= 2 points, endpoints included).
    public static func samples(count: Int, duration: Double, value: (Double) -> Float) -> [Float] {
        let n = max(2, count)
        return (0..<n).map { value(duration * Double($0) / Double(n - 1)) }
    }

    /// Sparkline points: samples spread across `width`; value 0...1 maps bottom...top of `height`.
    public static func sparklinePoints(_ samples: [Float], width: CGFloat, height: CGFloat) -> [CGPoint] {
        guard samples.count >= 2 else {
            return samples.map { CGPoint(x: 0, y: height * (1 - CGFloat(min(max($0, 0), 1)))) }
        }
        let step = width / CGFloat(samples.count - 1)
        return samples.enumerated().map { i, v in
            CGPoint(x: CGFloat(i) * step, y: height * (1 - CGFloat(min(max(v, 0), 1))))
        }
    }

    /// "m:ss" time label.
    public static func timeLabel(_ seconds: Double) -> String {
        let s = max(0, Int(seconds.rounded(.down)))
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
