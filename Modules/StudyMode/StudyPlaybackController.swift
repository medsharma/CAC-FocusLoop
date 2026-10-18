import AVFoundation
import Foundation
import Observation
import QuartzCore

/// Drives lecture playback for Study Mode: an `AVPlayer` when the lecture has a bundled video,
/// otherwise a 10 Hz timer ("no video" playback). Either way `playhead` updates every ~0.1 s and
/// `onTick` fires so the view can forward it to `model.actions.setPlayhead`.
@MainActor
@Observable
public final class StudyPlaybackController {
    public static let tickInterval: Double = 0.1

    public let duration: Double
    public private(set) var playhead: Double = 0
    public private(set) var isPlaying = false

    @ObservationIgnored public let player: AVPlayer?
    @ObservationIgnored public var onTick: ((Double) -> Void)?
    @ObservationIgnored private let resources = StudyPlaybackResources()
    @ObservationIgnored private var lastTimerTick: CFTimeInterval = 0

    public var hasVideo: Bool { player != nil }
    public var progress: Double { duration > 0 ? min(max(playhead / duration, 0), 1) : 0 }
    public var isAtEnd: Bool { playhead >= duration - 0.05 }

    public init(lecture: LectureInfo, bundle: Bundle = .main) {
        duration = max(1, lecture.durationSeconds)
        if let url = StudyLectureLibrary.videoURL(for: lecture, bundle: bundle) {
            let p = AVPlayer(url: url)
            p.actionAtItemEnd = .pause
            player = p
        } else {
            player = nil
        }
        installPlayerObservers()
    }

    // MARK: Transport

    public func play() {
        guard !isPlaying else { return }
        if isAtEnd { seek(to: 0) }
        isPlaying = true
        if let player {
            StudyAudioSession.activateForLecture()
            player.play()
        } else {
            startTimer()
        }
    }

    public func pause() {
        guard isPlaying else { return }
        isPlaying = false
        player?.pause()
        resources.timer?.invalidate()
        resources.timer = nil
    }

    public func toggle() { isPlaying ? pause() : play() }

    public func seek(to seconds: Double) {
        let t = min(max(seconds, 0), duration)
        setPlayhead(t)
        player?.seek(to: CMTime(seconds: t, preferredTimescale: 600),
                     toleranceBefore: CMTime(value: 1, timescale: 10),
                     toleranceAfter: CMTime(value: 1, timescale: 10))
        lastTimerTick = CACurrentMediaTime()
    }

    public func skip(by delta: Double) { seek(to: playhead + delta) }

    /// Stops playback and releases the timer / time observer. Safe to call repeatedly.
    public func tearDown() {
        pause()
        resources.invalidate()
    }

    // MARK: Internals

    private func setPlayhead(_ t: Double) {
        playhead = t
        onTick?(t)
    }

    private func startTimer() {
        resources.timer?.invalidate()
        lastTimerTick = CACurrentMediaTime()
        let timer = Timer(timeInterval: Self.tickInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.timerTick() }
        }
        RunLoop.main.add(timer, forMode: .common)
        resources.timer = timer
    }

    private func timerTick() {
        guard isPlaying else { return }
        let now = CACurrentMediaTime()
        let next = playhead + (now - lastTimerTick)
        lastTimerTick = now
        if next >= duration {
            setPlayhead(duration)
            pause()
        } else {
            setPlayhead(next)
        }
    }

    private func installPlayerObservers() {
        guard let player else { return }
        resources.player = player
        let interval = CMTime(seconds: Self.tickInterval, preferredTimescale: 600)
        resources.timeObserver = player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            MainActor.assumeIsolated {
                guard let self, self.isPlaying, time.isNumeric else { return }
                self.setPlayhead(min(max(time.seconds, 0), self.duration))
            }
        }
        resources.endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime, object: player.currentItem, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.isPlaying = false
                self.setPlayhead(self.duration)
            }
        }
    }
}

/// Non-isolated holder so timers/observers are released even if `tearDown()` is never called.
final class StudyPlaybackResources {
    var timer: Timer?
    var player: AVPlayer?
    var timeObserver: Any?
    var endObserver: NSObjectProtocol?

    func invalidate() {
        timer?.invalidate()
        timer = nil
        if let timeObserver, let player { player.removeTimeObserver(timeObserver) }
        timeObserver = nil
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        endObserver = nil
        player?.pause()
    }

    deinit { invalidate() }
}

/// Audio session handling for Study Mode's lecture video vs the chime nudge.
///
/// The lecture plays in `.playback` / `.moviePlayback` (audible with the Ring/Silent switch on silent, like any
/// video). The chime (`ChimeNudge`) mixes into whatever category is active, so a chime during a lecture plays
/// over the video instead of interrupting it. When Study Mode closes, the session goes back to `.ambient`
/// (mixes with other audio, respects the silent switch), so a later Focus-session chime doesn't inherit the
/// video's category and play through the silent switch.
enum StudyAudioSession {
    static func activateForLecture() {
        let s = AVAudioSession.sharedInstance()
        try? s.setCategory(.playback, mode: .moviePlayback)
        try? s.setActive(true)
    }

    static func restoreAfterLecture() {
        let s = AVAudioSession.sharedInstance()
        guard s.category == .playback else { return }
        try? s.setActive(false, options: [.notifyOthersOnDeactivation])
        try? s.setCategory(.ambient, mode: .default, options: [.mixWithOthers])
    }
}
