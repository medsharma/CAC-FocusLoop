import SwiftUI

/// Drop-in test screen: plays the bundled sample timeline, or drive networks by hand.
/// Add `BrainCortexDemoView()` to any preview or debug menu to check the port on device.
public struct BrainCortexDemoView: View {
    @State private var manual: [Float] = Array(repeating: 0, count: 7)
    @State private var playhead: Double = 0
    @State private var playing = true
    private let timeline = try? BrainTimeline.load("sample_lecture")
    private let tick = Timer.publish(every: 1.0 / 30, on: .main, in: .common).autoconnect()

    public init() {}

    private var activations: [Float] {
        if playing, let tl = timeline { return tl.activations(at: playhead) }
        return manual
    }

    public var body: some View {
        VStack(spacing: 12) {
            ZStack(alignment: .topLeading) {
                BrainCortexView(activations: activations)
                    .frame(height: 320)
                if timeline?.mock == true && playing {
                    Text("MOCK DATA").font(.caption2.bold()).padding(6)
                        .background(.orange.opacity(0.85), in: Capsule()).padding(8)
                }
            }
            Toggle("Play sample timeline", isOn: $playing).disabled(timeline == nil)
            ForEach(BrainNetwork.allCases) { n in
                HStack {
                    Circle().fill(n.color).frame(width: 8, height: 8)
                    Text(n.displayName).font(.caption).frame(width: 170, alignment: .leading)
                    Slider(value: Binding(get: { Double(activations[n.rawValue]) },
                                          set: { playing = false; manual[n.rawValue] = Float($0) }))
                }
            }
        }
        .padding()
        .background(Color.black)
        .preferredColorScheme(.dark)
        .onReceive(tick) { _ in
            guard playing, let tl = timeline, tl.duration > 0 else { return }
            playhead = (playhead + 1.0 / 30).truncatingRemainder(dividingBy: tl.duration)
        }
    }
}

#Preview { BrainCortexDemoView() }
