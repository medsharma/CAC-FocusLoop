import AVFoundation
import CoreGraphics
import Foundation
import QuartzCore

public enum VideoFaceSourceError: Error, Equatable {
    case fileMissing
    case noVideoTrack
}

/// A recorded video of a real person used as the camera (the Simulator has no TrueDepth camera): plays the
/// file in real time with `AVPlayer` (muted, looping) and analyses the frames being shown, so what is on
/// screen and what the pipeline sees stay in sync.
///
/// * `AVPlayerItemVideoOutput` hands out 420f pixel buffers; a ~30 Hz timer on a private serial queue pulls
///   each new frame and runs `VideoFaceAnalyzer` (Vision) on it, emitting one `FaceSample`.
/// * In the Simulator, where Vision's face-landmark model is broken, the frame's numbers come from the video's
///   face track (`<video>.facetrack.csv`, computed on the Mac by the same analyzer, see `FaceTrack`) when one
///   is present; otherwise the live analysis runs in head-pose-only mode (no blinks / yawns).
/// * `FaceSample.t` is the host time the frame is displayed at (device uptime clock) minus `timeOrigin`, so
///   frame spacing follows the video's own timing (as the rPPG estimator needs) across loops and pauses.
/// * `onFrame` receives every landmark point (normalized to the upright video frame, origin top-left),
///   throttled to `meshMaxRate`, on the analysis queue. Numbers only.
/// * Never writes frames or images anywhere; pixel buffers are read in place and released.
/// * Thread-safety: analysis state lives on a private serial queue; `start`/`stop` are called from one actor.
public final class VideoFaceSource: NSObject, FaceSignalSource, @unchecked Sendable {
    public let url: URL
    public var isAvailable: Bool { FileManager.default.fileExists(atPath: url.path) }
    public let samples: AsyncStream<FaceSample>
    /// The player whose frames are analysed. Show it with an `AVPlayerLayer` to display exactly those frames.
    public let player: AVPlayer
    /// Mesh-overlay hook (see type doc). Set before `start()`.
    public var onFrame: (([CGPoint]) -> Void)?
    public var meshMaxRate: Double = 15
    /// Subtracted from the host time (seconds of `CACurrentMediaTime`) to produce `FaceSample.t`.
    public var timeOrigin: Double = 0
    /// Polling rate for new frames (Hz).
    public var pollRate: Double = 30
    /// Selfie-mirrored footage (default: unmirrored, as the iPhone Camera app saves it).
    public var mirrored: Bool {
        get { queue.sync { analyzer.mirrored } }
        set { queue.sync { analyzer.mirrored = newValue } }
    }
    /// Upright video width / height (1 until the track has loaded).
    public private(set) var videoAspect: CGFloat = 1
    /// Face track used instead of live analysis (see type doc). Found automatically next to the video in the
    /// Simulator; set it explicitly (before `start()`) to force it anywhere.
    var faceTrack: FaceTrack?
    /// Where the numbers come from, for Settings / logs.
    var analysisDescription: String {
        if faceTrack != nil { return "face track computed on the Mac (Vision landmarks)" }
        return analyzer.mode == .landmarks ? "live Vision face landmarks" : "live Vision head pose only (no blinks or yawns)"
    }

    private let continuation: AsyncStream<FaceSample>.Continuation
    private let queue = DispatchQueue(label: "org.focusloop.perception.video", qos: .userInteractive)
    private let analyzer = VideoFaceAnalyzer()
    private var output: AVPlayerItemVideoOutput?
    private var timer: DispatchSourceTimer?
    private var orientation: CGImagePropertyOrientation = .up
    private var endObserver: NSObjectProtocol?
    private var lastMeshEmit = -Double.infinity
    private var lastItemTime: CMTime = .invalid
    private var hadFace = false
    private var prepared = false

    public init(url: URL) {
        self.url = url
        player = AVPlayer()
        player.isMuted = true
        player.actionAtItemEnd = .none
        player.preventsDisplaySleepDuringVideoPlayback = true
        var c: AsyncStream<FaceSample>.Continuation!
        samples = AsyncStream(bufferingPolicy: .bufferingNewest(120)) { c = $0 }
        continuation = c
        super.init()
    }

    deinit {
        timer?.cancel()
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        player.pause()
        continuation.finish()
    }

    /// Starts playback + analysis. Throws if the file is missing; the track loads asynchronously, and if it
    /// turns out to have no video the sample stream simply finishes (`onFailure` is called).
    public func start() throws {
        guard isAvailable else { throw VideoFaceSourceError.fileMissing }
        running = true
        guard prepared else {
            guard prepareTask == nil else { return }
            prepareTask = Task { [weak self] in
                guard let self else { return }
                do {
                    try await self.prepare()
                    await MainActor.run { if self.running { self.begin() } }
                } catch {
                    self.continuation.finish()
                    self.onFailure?(error)
                }
            }
            return
        }
        begin()
    }

    /// Called if the video can't be loaded (no video track / unreadable).
    public var onFailure: ((Error) -> Void)?

    private var running = false
    private var prepareTask: Task<Void, Never>?

    private func begin() {
        player.play()
        guard timer == nil else { return }
        let tm = DispatchSource.makeTimerSource(queue: queue)
        tm.schedule(deadline: .now(), repeating: 1 / pollRate, leeway: .milliseconds(2))
        tm.setEventHandler { [weak self] in self?.poll() }
        timer = tm
        tm.resume()
    }

    /// Pauses playback and analysis (resume with `start()`).
    public func stop() {
        running = false
        player.pause()
        timer?.cancel()
        timer = nil
    }

    private func prepare() async throws {
        if faceTrack == nil, !VideoFaceAnalyzer.landmarksAvailable, let sidecar = FaceTrack.locate(for: url) {
            faceTrack = try? FaceTrack.load(sidecar)
        }
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw VideoFaceSourceError.noVideoTrack }
        let (transform, size) = try await track.load(.preferredTransform, .naturalSize)
        let o = VideoFaceAnalyzer.orientation(for: transform)
        let aspect = o.swapsAxes ? size.height / max(size.width, 1) : size.width / max(size.height, 1)
        let out = AVPlayerItemVideoOutput(pixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        ])
        let item = AVPlayerItem(asset: asset)
        item.add(out)
        queue.sync {
            orientation = o
            output = out
        }
        await MainActor.run {
            videoAspect = aspect
            player.replaceCurrentItem(with: item)
            endObserver = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: item,
                                                                 queue: .main) { [weak self] _ in
                // Loop: the session clock keeps running; only the video starts over.
                self?.player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
                if self?.running == true { self?.player.play() }
            }
            prepared = true
        }
    }

    /// On `queue`: pull the newest frame (if any) and analyse it.
    private func poll() {
        guard let output else { return }
        let host = CACurrentMediaTime()
        let itemNow = output.itemTime(forHostTime: host)
        guard output.hasNewPixelBuffer(forItemTime: itemNow) else { return }
        var display = CMTime.invalid
        guard let pb = output.copyPixelBuffer(forItemTime: itemNow, itemTimeForDisplay: &display) else { return }
        if display.isValid, lastItemTime.isValid, display == lastItemTime { return }
        lastItemTime = display
        // Host time at which this frame is shown (item clock runs at the player's rate).
        let rate = Double(player.rate)
        var frameHost = host
        if display.isValid, itemNow.isValid, rate > 0 {
            frameHost = host - (itemNow.seconds - display.seconds) / rate
        }
        let t = frameHost - timeOrigin
        let wantMesh = onFrame != nil && frameHost - lastMeshEmit >= 1 / meshMaxRate
        let r: VideoFaceAnalyzer.Result
        if let track = faceTrack {
            // Simulator: this displayed frame's numbers from the face track (pixels are not needed).
            let f = track.frame(at: display.isValid ? display.seconds : itemNow.seconds)
            let sample = f?.sample(t: t) ?? FaceSample(t: t, faceDetected: false)
            r = VideoFaceAnalyzer.Result(sample: sample, mesh: wantMesh && !(f?.mesh.isEmpty ?? true) ? f?.mesh ?? [] : [], debug: nil)
        } else {
            r = analyzer.analyze(pixelBuffer: pb, orientation: orientation, t: t, wantMesh: wantMesh)
        }
        continuation.yield(r.sample)
        if let onFrame {
            if r.sample.faceDetected {
                hadFace = true
                if wantMesh, !r.mesh.isEmpty { lastMeshEmit = frameHost; onFrame(r.mesh) }
            } else if hadFace {
                hadFace = false
                onFrame([])
            }
        }
    }
}
