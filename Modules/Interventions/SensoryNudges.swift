import AVFoundation
import CoreHaptics
import Foundation

/// Distracted: a gentle two-pulse Core Haptics tap. No-op where haptics are unsupported (Simulator, iPad).
@MainActor
public final class HapticNudge: Nudge {
    public let arm: NudgeArm = .haptic
    private var engine: CHHapticEngine?

    public init() {}

    public static var isSupported: Bool { CHHapticEngine.capabilitiesForHardware().supportsHaptics }

    /// The gentle 2-pulse pattern (exposed for tests).
    public static func makePattern() throws -> CHHapticPattern {
        func tap(at time: TimeInterval, intensity: Float) -> CHHapticEvent {
            CHHapticEvent(eventType: .hapticTransient, parameters: [
                CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
                CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.25),
            ], relativeTime: time)
        }
        return try CHHapticPattern(events: [tap(at: 0, intensity: 0.55), tap(at: 0.2, intensity: 0.4)], parameters: [])
    }

    public func start(presenter: NudgePresenting) {
        guard Self.isSupported else { return }
        do {
            let engine = try CHHapticEngine()
            engine.playsHapticsOnly = true
            engine.isAutoShutdownEnabled = true
            try engine.start()
            let player = try engine.makePlayer(with: Self.makePattern())
            try player.start(atTime: CHHapticTimeImmediate)
            self.engine = engine
        } catch {
            engine = nil
        }
    }

    public func cancel() {
        engine?.stop(completionHandler: nil)
        engine = nil
    }
}

/// Distracted: a soft synthesised chime (two decaying sine partials), generated on the fly with
/// AVAudioEngine, so no audio file is bundled. Uses the `.ambient` audio session category, which
/// respects the Ring/Silent switch and mixes with other audio. If another component already set a
/// playback category (e.g. Study Mode video), that category is left alone and the chime mixes into it.
@MainActor
public final class ChimeNudge: Nudge {
    public let arm: NudgeArm = .chime
    private var engine: AVAudioEngine?
    private var player: AVAudioPlayerNode?

    public init() {}

    /// Mono PCM buffer of the chime (exposed for tests).
    public static func makeBuffer(sampleRate: Double = 44_100, duration: Double = 1.4) -> AVAudioPCMBuffer? {
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1) else { return nil }
        let frames = AVAudioFrameCount(sampleRate * duration)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames),
              let data = buffer.floatChannelData?[0] else { return nil }
        buffer.frameLength = frames
        // E5 then B5 (a soft fifth), each with a 10 ms attack and exponential decay.
        let partials: [(freq: Double, delay: Double, amp: Double)] = [(659.25, 0, 0.16), (987.77, 0.12, 0.10)]
        for i in 0..<Int(frames) {
            let t = Double(i) / sampleRate
            var s = 0.0
            for p in partials where t >= p.delay {
                let tt = t - p.delay
                let env = min(1, tt / 0.01) * exp(-tt * 3.2)
                s += p.amp * env * sin(2 * Double.pi * p.freq * tt)
            }
            // Fade the last 50 ms to avoid a click.
            let fade = min(1, (duration - t) / 0.05)
            data[i] = Float(s * fade)
        }
        return buffer
    }

    public func start(presenter: NudgePresenting) {
        let session = AVAudioSession.sharedInstance()
        if session.category != .playback && session.category != .playAndRecord {
            try? session.setCategory(.ambient, mode: .default, options: [.mixWithOthers])
        }
        try? session.setActive(true)
        guard let buffer = Self.makeBuffer() else { return }
        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: buffer.format)
        do {
            try engine.start()
        } catch {
            return
        }
        self.engine = engine
        self.player = player
        player.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            DispatchQueue.main.async { self?.cancel() }
        }
        player.play()
    }

    public func cancel() {
        player?.stop()
        engine?.stop()
        player = nil
        engine = nil
    }
}
