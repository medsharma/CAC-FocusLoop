import SwiftUI

/// Player-chrome indicator: current attention state (colored dot + name) and a lapse-risk ring
/// (fill = predicted P(lapse), tick = personal threshold). Shows "--" before the first forecast.
struct StudyStateIndicator: View {
    let state: AttentionState
    let hasEstimate: Bool
    let risk: LapseRisk?

    var body: some View {
        HStack(spacing: 10) {
            HStack(spacing: 8) {
                Circle()
                    .fill(hasEstimate ? state.color : FLColor.textTertiary)
                    .frame(width: 6, height: 6)
                // Fixed slot (widest state name), one label at a time.
                ZStack(alignment: .leading) {
                    Text("Distracted").hidden()
                    Text(hasEstimate ? state.displayName : "Waiting")
                        .id(hasEstimate ? state.displayName : "Waiting")
                        .transition(.opacity)
                }
                .font(.system(.footnote, weight: .medium))
                .foregroundStyle(FLColor.textPrimary)
                .lineLimit(1)
            }
            ring
        }
        .padding(.leading, 12)
        .padding(.trailing, 5)
        .padding(.vertical, 5)
        .background(FLColor.surfaceHigh, in: Capsule())
        .overlay(Capsule().strokeBorder(FLColor.hairline, lineWidth: 0.5))
        .animation(.easeInOut(duration: 0.35), value: state)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Attention \(hasEstimate ? state.displayName : "not yet estimated")")
        .accessibilityValue(risk.map { "Predicted lapse risk \(Int(($0.probability * 100).rounded())) percent" } ?? "")
    }

    /// Thin lapse-risk ring (2pt): fill = P(lapse), tick = personal threshold.
    private var ring: some View {
        let p = CGFloat(min(max(risk?.probability ?? 0, 0), 1))
        let th = CGFloat(min(max(risk?.threshold ?? 0.5, 0), 1))
        let color = risk.map { StudyModeStyle.riskColor($0.probability, threshold: $0.threshold) } ?? FLColor.textTertiary
        return ZStack {
            Circle().stroke(FLColor.hairlineStrong, lineWidth: 2)
            Circle().trim(from: 0, to: p)
                .stroke(color, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeOut(duration: 0.4), value: p)
            // Threshold tick
            Capsule().fill(FLColor.textSecondary)
                .frame(width: 1.5, height: 5)
                .offset(y: -13)
                .rotationEffect(.degrees(Double(th) * 360))
            Text(risk.map { "\(Int(($0.probability * 100).rounded()))" } ?? "--")
                .font(.system(size: 10, weight: .medium).monospacedDigit())
                .foregroundStyle(FLColor.textPrimary)
        }
        .frame(width: 28, height: 28)
    }
}
