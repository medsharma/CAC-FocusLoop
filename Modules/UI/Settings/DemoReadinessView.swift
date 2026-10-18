import SwiftUI

/// Settings → Demo → Demo readiness: a live checklist to run through before filming (camera permission,
/// face tracking support, face seen right now, heart-rate quality, work-zone calibration, lecture files,
/// models and whether they were trained on generated demo data, Demo Director). On a device it starts a short camera probe while
/// open so the face and heart-rate rows are live; nothing is stored.
public struct DemoReadinessView: View {
    let services: UIServices
    @State private var items: [DemoReadinessItem] = []

    public init(services: UIServices) { self.services = services }

    public var body: some View {
        let problems = items.filter { $0.status == .problem }.count
        let warnings = items.filter { $0.status == .warning }.count
        let summary = problems > 0 ? "\(problems) to fix before a live demo"
            : warnings > 0 ? "Ready, with \(warnings) to check" : "Ready for a live demo"
        let summaryStatus: DemoReadinessItem.Status = problems > 0 ? .problem : warnings > 0 ? .warning : .ok
        List {
            // Summary first: one quiet status line with a state dot.
            VStack(alignment: .leading, spacing: FLSpacing.xs) {
                HStack(spacing: FLSpacing.xs) {
                    Circle().fill(Self.color(summaryStatus)).frame(width: 8, height: 8)
                    Text(items.isEmpty ? "Checking\u{2026}" : summary).flText(.title)
                }
                .accessibilityElement(children: .combine)
                Text("No camera here? Use Settings \u{2192} Demo \u{2192} Run demo instead.")
                    .flText(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .readinessRow(bottom: FLSpacing.l)

            VStack(alignment: .leading, spacing: FLSpacing.xs) {
                FLEyebrow("Checklist").padding(.horizontal, FLSpacing.xxs)
                VStack(spacing: 0) {
                    ForEach(Array(items.enumerated()), id: \.element.id) { i, item in
                        if i > 0 { FLDivider(inset: 16 + 22 + 12) }
                        row(item)
                    }
                }
                .flRowGroup()
                Text("Rows update live. On an iPhone the camera runs only while this screen is open, to check your face and heart-rate signal. Nothing is recorded.")
                    .flText(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, FLSpacing.xxs)
                    .padding(.top, FLSpacing.xxs)
            }
            .readinessRow(bottom: FLSpacing.xxl)
        }
        .listStyle(.plain)
        .environment(\.defaultMinListRowHeight, 0)
        .flScreen("Demo readiness", inline: true)
        .task {
            services.startReadinessProbe()
            while !Task.isCancelled {
                items = services.demoReadiness()
                try? await Task.sleep(for: .milliseconds(500))
            }
        }
        .onDisappear { services.stopReadinessProbe() }
    }

    private func row(_ item: DemoReadinessItem) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: Self.symbol(item.status))
                .foregroundStyle(Self.color(item.status))
                .font(.system(size: 17, weight: .regular))
                .frame(width: 22, height: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title).font(FLFont.body).foregroundStyle(FLColor.textPrimary)
                Text(item.detail).font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, minHeight: 56, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(item.title): \(item.detail)")
    }

    static func symbol(_ s: DemoReadinessItem.Status) -> String {
        switch s {
        case .ok: "checkmark.circle"
        case .warning: "exclamationmark.triangle"
        case .problem: "xmark.octagon"
        case .info: "info.circle"
        }
    }

    static func color(_ s: DemoReadinessItem.Status) -> Color {
        switch s {
        case .ok: FLColor.focused
        case .warning: FLColor.distracted
        case .problem: FLColor.stressed
        case .info: FLColor.textSecondary
        }
    }
}

private extension View {
    func readinessRow(bottom: CGFloat) -> some View {
        self
            .listRowInsets(EdgeInsets(top: FLSpacing.xs, leading: FLSpacing.gutter, bottom: bottom, trailing: FLSpacing.gutter))
            .listRowBackground(Color.clear)
            .listRowSeparator(.hidden)
    }
}
