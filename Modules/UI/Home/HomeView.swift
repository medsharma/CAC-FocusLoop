import SwiftUI

public enum FLRoute: Hashable {
    case science, history, report(UUID), privacy, settings, lectures, brainDemo, sustainedAttention
}

/// Home: start a session, Study Mode, History / Attention Profile, Science, Privacy, Settings.
public struct HomeView: View {
    let model: LiveSessionModel
    let services: UIServices
    var startFocus: () -> Void
    var navigate: (FLRoute) -> Void


    public init(model: LiveSessionModel, services: UIServices, startFocus: @escaping () -> Void,
                navigate: @escaping (FLRoute) -> Void) {
        self.model = model; self.services = services; self.startFocus = startFocus; self.navigate = navigate
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                header
                hero
                    .padding(.top, FLSpacing.l)
                VStack(alignment: .leading, spacing: FLSpacing.s) {
                    FLEyebrow("Explore")
                        .padding(.leading, FLSpacing.xxs)
                    VStack(spacing: 0) {
                        row("Study Mode", "Lectures with predicted brain response", "play.rectangle", id: "home.study") { navigate(.lectures) }
                        FLDivider(inset: FLRow<EmptyView>.dividerInset)
                        row("Science", "See the live pipeline", "waveform.path.ecg", id: "home.science") { navigate(.science) }
                        FLDivider(inset: FLRow<EmptyView>.dividerInset)
                        row("Attention Profile", "Trends and what works for you", "chart.xyaxis.line", id: "home.history") { navigate(.history) }
                        FLDivider(inset: FLRow<EmptyView>.dividerInset)
                        row("Privacy", "0 bytes of video stored or sent", "lock.shield", id: "home.privacy") { navigate(.privacy) }
                    }
                    .flRowGroup()
                }
                .padding(.top, FLSpacing.xxl)
                if let latest = services.listSessions().first {
                    recent(latest).padding(.top, FLSpacing.xl)
                }
                Text("FocusLoop supports focus. It is not a medical device.")
                    .flText(.caption)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.top, FLSpacing.xl)
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.bottom, FLSpacing.xl)
        }
        .scrollContentBackground(.hidden)
        .background {
            FLBackground(glow: FLColor.accent, center: UnitPoint(x: 0.5, y: 0.26), intensity: 0.07, radius: 0.4)
                .ignoresSafeArea()
        }
        .toolbar(.hidden, for: .navigationBar)
    }

    private var greeting: String {
        let h = Calendar.current.component(.hour, from: Date())
        return h < 5 ? "Late night focus" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening"
    }

    private var header: some View {
        HStack(alignment: .center, spacing: FLSpacing.s) {
            VStack(alignment: .leading, spacing: FLSpacing.xs) {
                Text(greeting).flText(.eyebrow)
                Text("FocusLoop").flText(.largeTitle)
            }
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)
            Spacer()
            if services.flags.demoDirector { FLBadge(.demo, compact: true) }
            FLIconButton("gearshape", label: "Settings", identifier: "home.settings") { navigate(.settings) }
        }
        .padding(.top, FLSpacing.m)
    }

    private var hero: some View {
        VStack(spacing: 0) {
            FocusOrb(probabilities: [0.92, 0.03, 0.03, 0.02], lapseRisk: 0.1)
                .frame(width: 136, height: 136)
                .padding(.vertical, FLSpacing.l)
                .accessibilityHidden(true)
            Text("Ready when you are").flText(.title)
                .padding(.top, FLSpacing.xs)
            Text("Prop your phone beside your laptop, front camera facing you.")
                .flText(.callout)
                .multilineTextAlignment(.center)
                .lineSpacing(2)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, FLSpacing.xs)
                .padding(.horizontal, FLSpacing.m)
            Button {
                FLHaptics.tap(); startFocus()
            } label: {
                Label("Start focus session", systemImage: "play.fill")
            }
            .buttonStyle(FLPrimaryButtonStyle())
            .accessibilityIdentifier("home.startFocus")
            .padding(.top, FLSpacing.l)
        }
        .frame(maxWidth: .infinity)
    }

    private func row(_ title: String, _ subtitle: String, _ icon: String, id: String, action: @escaping () -> Void) -> some View {
        Button { FLHaptics.tap(); action() } label: {
            FLRow(title, subtitle: subtitle, symbol: icon)
        }
        .buttonStyle(FLPressableStyle())
        .accessibilityIdentifier(id)
    }

    private func recent(_ s: UISessionSummary) -> some View {
        VStack(alignment: .leading, spacing: FLSpacing.s) {
            FLEyebrow("Last session")
                .padding(.leading, FLSpacing.xxs)
            Button { FLHaptics.tap(); navigate(.report(s.id)) } label: {
                FLRow(s.lectureTitle ?? "Focus session",
                      subtitle: "\(UIFormat.percent(s.focusedFraction)) focused \u{00B7} back in \(s.medianRefocusSeconds.map(UIFormat.oneDecimal) ?? "--") s",
                      symbol: "clock")
            }
            .buttonStyle(FLPressableStyle())
            .flRowGroup()
        }
    }
}

/// Study Mode lecture list (slot: the Study Mode agent's player is opened per lecture).
public struct LectureListView: View {
    let services: UIServices
    var open: (LectureInfo) -> Void
    public init(services: UIServices, open: @escaping (LectureInfo) -> Void) { self.services = services; self.open = open }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FLSpacing.s) {
                Text("Watch a lecture while FocusLoop follows along. Amber marks show where the content is predicted to lose an average viewer.")
                    .font(FLFont.callout).foregroundStyle(FLColor.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.bottom, FLSpacing.xs)
                ForEach(services.lectures()) { l in
                    Button { FLHaptics.tap(); open(l) } label: { LectureRow(lecture: l) }
                        .buttonStyle(FLPressableStyle())
                        .accessibilityIdentifier("lecture.\(l.id)")
                }
            }
            .padding(.horizontal, FLSpacing.gutter)
            .padding(.bottom, FLSpacing.xxl)
        }
        .flScreen("Study Mode")
    }
}

private struct LectureRow: View {
    let lecture: LectureInfo
    var isMock: Bool { (try? BrainTimeline.load(lecture.id))?.mock ?? false }
    var body: some View {
        HStack(alignment: .top, spacing: FLSpacing.m) {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(FLColor.surfaceHigh)
                .frame(width: 72, height: 56)
                .overlay(Image(systemName: lecture.videoFile == nil ? "brain" : "play.fill")
                    .font(.system(size: FLIconSize.row, weight: .regular)).foregroundStyle(FLColor.textSecondary))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(FLColor.hairline, lineWidth: 0.5))
            VStack(alignment: .leading, spacing: 4) {
                Text(lecture.title).font(FLFont.headline).foregroundStyle(FLColor.textPrimary).multilineTextAlignment(.leading)
                    .lineLimit(2).fixedSize(horizontal: false, vertical: true)
                Text("\(lecture.subject) \u{00B7} \(UIFormat.timer(lecture.durationSeconds))").font(FLFont.caption).foregroundStyle(FLColor.textSecondary)
                Text("\(lecture.license) \u{00B7} \(lecture.attribution)").font(.caption2).foregroundStyle(FLColor.textTertiary)
                    .lineLimit(3).multilineTextAlignment(.leading)
            }
            Spacer(minLength: 0)
            if isMock { FLBadge(.mock, compact: true) }
        }
        .flCard(padding: FLSpacing.m)
        .accessibilityElement(children: .combine)
    }
}
