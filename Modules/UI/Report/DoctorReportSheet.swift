import SwiftUI

/// "Share with your doctor": explains what the PDF contains, offers the demo-data toggle, makes the PDF on
/// this phone and opens the share sheet. Nothing is sent automatically. Reachable from the Attention Profile and
/// Settings → Your data.
public struct DoctorReportSheet: View {
    let services: UIServices
    @Environment(\.dismiss) private var dismiss
    @State private var counts = DoctorReportCounts()
    @State private var includeDemo = false
    @State private var working = false
    @State private var error: String?
    @State private var sharing: ShareItem?

    public init(services: UIServices) { self.services = services }

    public static let intro = "A printable PDF of how your sustained attention behaves: your sessions, attention-check runs, camera heart rate, what helped you refocus and your own check-in answers. It is made on this phone, and you choose who to share it with. Nothing is sent automatically."

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: FLSpacing.xl) {
                    VStack(alignment: .leading, spacing: FLSpacing.m) {
                        Image(systemName: "doc.text")
                            .font(.system(size: 28, weight: .regular))
                            .symbolRenderingMode(.hierarchical)
                            .foregroundStyle(FLColor.accent)
                            .accessibilityHidden(true)
                        Text("Share with your doctor").flText(.largeTitle).fixedSize(horizontal: false, vertical: true)
                        Text(Self.intro).flText(.body).fixedSize(horizontal: false, vertical: true)
                    }

                    HStack(alignment: .top, spacing: FLSpacing.s) {
                        Rectangle().fill(FLColor.textTertiary).frame(width: 2)
                        Text(DoctorReportContent.statement).flText(.callout, color: FLColor.textPrimary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityElement(children: .combine)
                    .accessibilityIdentifier("doctor.statement")

                    VStack(alignment: .leading, spacing: FLSpacing.xs) {
                        FLEyebrow("What goes in").padding(.horizontal, FLSpacing.xxs)
                        VStack(spacing: 0) {
                            FLRow("Real sessions", symbol: "scope", value: "\(counts.realSessions)", chevron: false)
                            FLDivider(inset: FLRow<EmptyView>.dividerInset)
                            FLRow("Attention-check runs", subtitle: "Every run: the taps are real", symbol: "hand.tap",
                                  value: "\(counts.sartRuns)", chevron: false)
                            if counts.demoSessions > 0 {
                                FLDivider(inset: FLRow<EmptyView>.dividerInset)
                                Toggle(isOn: $includeDemo) {
                                    HStack(alignment: .top, spacing: 12) {
                                        Image(systemName: "play.circle")
                                            .font(.system(size: FLIconSize.row, weight: .regular))
                                            .symbolRenderingMode(.hierarchical)
                                            .foregroundStyle(FLColor.textSecondary)
                                            .frame(width: 28, height: 22)
                                            .accessibilityHidden(true)
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text("Include demo sessions (marked as demo data)").font(FLFont.body)
                                                .foregroundStyle(FLColor.textPrimary)
                                                .fixedSize(horizontal: false, vertical: true)
                                            Text("\(counts.demoSessions) demo \(counts.demoSessions == 1 ? "session" : "sessions") (replay, recorded video or Demo Director). Left out by default.")
                                                .font(FLFont.caption).foregroundStyle(FLColor.textTertiary)
                                                .fixedSize(horizontal: false, vertical: true)
                                        }
                                    }
                                }
                                .tint(FLColor.accent)
                                .padding(.horizontal, 16).padding(.vertical, 12)
                                .accessibilityIdentifier("doctor.includeDemo")
                            }
                        }
                        .flRowGroup()
                        if counts.realSessions == 0 {
                            Text("There are no real sessions yet, so the summary will say so plainly and show your attention-check runs.")
                                .flText(.caption).fixedSize(horizontal: false, vertical: true)
                                .padding(.horizontal, FLSpacing.xxs).padding(.top, FLSpacing.xxs)
                        }
                    }

                    Button(action: make) {
                        HStack(spacing: FLSpacing.xs) {
                            if working { ProgressView().tint(FLColor.onAccent) }
                            Label(working ? "Making PDF\u{2026}" : "Create PDF", systemImage: "square.and.arrow.up")
                        }
                    }
                    .buttonStyle(FLPrimaryButtonStyle())
                    .disabled(working)
                    .accessibilityIdentifier("doctor.create")
                }
                .padding(.horizontal, FLSpacing.gutter)
                .padding(.top, FLSpacing.m)
                .padding(.bottom, FLSpacing.xxl)
            }
            .scrollContentBackground(.hidden)
            .background { FLColor.bgElevated.ignoresSafeArea() }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }.accessibilityIdentifier("doctor.done")
                }
            }
        }
        .flSheet()
        .onAppear { counts = services.doctorReportCounts() }
        .sheet(item: $sharing) { item in
            ActivityShareSheet(url: item.url).ignoresSafeArea().presentationDetents([.medium, .large])
        }
        .alert("Couldn\u{2019}t make the PDF", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(error ?? "") }
    }

    private func make() {
        working = true
        Task {
            do { sharing = ShareItem(url: try await services.makeDoctorReport(includeDemo: includeDemo)) }
            catch { self.error = error.localizedDescription }
            working = false
        }
    }
}
