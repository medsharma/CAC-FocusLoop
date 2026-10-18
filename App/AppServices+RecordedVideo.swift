import AVFoundation
import Foundation
import SwiftUI
import UIKit

// RECORDED video as the camera (demonstrations, and the Simulator, which has no camera): a video the user
// recorded of themselves studying drives the real perception pipeline frame by frame. The video comes from,
// in order: the `-recordedVideo <absolute path>` launch argument, a video imported in Settings (copied to
// Application Support/FocusLoop-demo-recording.<ext>), or a bundled Resources/Demo/demo_face.mov|mp4.
// Recorded sessions are labelled RECORDED and run on the SYNTHETIC/demo learner set, so a clip replayed many
// times never trains the real trial, personal head, forecaster threshold or labels.

/// Where the demo recording lives. Nothing here is ever written by the app except the user's own import.
enum RecordedVideoLibrary {
    /// File-name prefix of an imported recording (also what Delete all data removes).
    static let importedPrefix = "FocusLoop-demo-recording"
    static let bundledName = "demo_face"
    static let enabledKey = "focusloop.recordedVideo.enabled"
    static let launchArgument = "-recordedVideo"

    static var directory: URL? {
        try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    }

    /// The imported recording, if any.
    static func importedURL(in dir: URL? = directory) -> URL? {
        guard let dir, let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return nil }
        return names.filter { $0.hasPrefix(importedPrefix) && !$0.hasSuffix(FaceTrack.fileSuffix) }.sorted().first
            .map { dir.appendingPathComponent($0) }
    }

    /// Copies a face track (`*.facetrack.csv`, made on the Mac by tools/facetrack.sh) next to the imported video.
    @discardableResult
    static func importFaceTrack(from source: URL, into dir: URL? = directory) throws -> URL {
        guard let dir else { throw CocoaError(.fileNoSuchFile) }
        _ = try FaceTrack.load(source)   // reject anything that isn't a face track
        let dest = dir.appendingPathComponent(importedPrefix + FaceTrack.fileSuffix)
        if FileManager.default.fileExists(atPath: dest.path) { try FileManager.default.removeItem(at: dest) }
        try FileManager.default.copyItem(at: source, to: dest)
        return dest
    }

    static func isFaceTrack(_ url: URL) -> Bool { url.lastPathComponent.lowercased().hasSuffix(FaceTrack.fileSuffix) }

    /// `Resources/Demo/demo_face.mov` (or `.mp4`) if it was bundled at build time.
    static func bundledURL(bundle: Bundle = .main) -> URL? {
        bundle.url(forResource: bundledName, withExtension: "mov") ?? bundle.url(forResource: bundledName, withExtension: "mp4")
            ?? bundle.url(forResource: bundledName, withExtension: "m4v")
    }

    /// `-recordedVideo <absolute path>` launch argument (if the file exists).
    static func launchArgumentURL(_ args: [String] = ProcessInfo.processInfo.arguments) -> URL? {
        guard let i = args.firstIndex(of: launchArgument), i + 1 < args.count else { return nil }
        let url = URL(fileURLWithPath: args[i + 1])
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Copies a picked video to Application Support/FocusLoop-demo-recording.<ext>, replacing any earlier import.
    @discardableResult
    static func importVideo(from source: URL, into dir: URL? = directory) throws -> URL {
        guard let dir else { throw CocoaError(.fileNoSuchFile) }
        try removeImported(in: dir)
        let ext = source.pathExtension.isEmpty ? "mov" : source.pathExtension.lowercased()
        let dest = dir.appendingPathComponent("\(importedPrefix).\(ext)")
        try FileManager.default.copyItem(at: source, to: dest)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var d = dest
        try? d.setResourceValues(values)
        return dest
    }

    /// Deletes every imported recording (also called by Delete all data).
    static func removeImported(in dir: URL? = directory) throws {
        guard let dir, let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return }
        for n in names where n.hasPrefix(importedPrefix) {
            try FileManager.default.removeItem(at: dir.appendingPathComponent(n))
        }
    }
}

extension AppServices: RecordedVideoSettingsControlling {

    /// Sessions use the recorded video (Settings toggle, or forced by the launch argument).
    var useRecordedVideo: Bool {
        get {
            if RecordedVideoLibrary.launchArgumentURL() != nil { return true }
            if UserDefaults.standard.object(forKey: RecordedVideoLibrary.enabledKey) != nil {
                return UserDefaults.standard.bool(forKey: RecordedVideoLibrary.enabledKey)
            }
            // Default: with no live face-tracking camera (e.g. the Simulator), use a bundled demo recording if one exists.
            return !LiveFeatureFrameSource.isSupported && RecordedVideoLibrary.bundledURL() != nil
        }
        set {
            UserDefaults.standard.set(newValue, forKey: RecordedVideoLibrary.enabledKey)
            if !engine.isRunning { selectLearners(replay: willUseReplay || willUseRecordedVideo) }
        }
    }

    /// The video the next session would analyse (launch argument > imported > bundled), regardless of the toggle.
    var recordedVideoURL: URL? {
        RecordedVideoLibrary.launchArgumentURL() ?? RecordedVideoLibrary.importedURL() ?? RecordedVideoLibrary.bundledURL()
    }

    /// The next session runs on the RECORDED video (never during the SYNTHETIC scripted demo).
    var willUseRecordedVideo: Bool { useRecordedVideo && recordedVideoURL != nil && !scriptedDemoActive }

    var recordedVideoDescription: String? {
        if let u = RecordedVideoLibrary.launchArgumentURL() { return "Launch argument: \(u.lastPathComponent)" }
        if let u = RecordedVideoLibrary.importedURL() { return "Imported: \(Self.describe(u))" }
        if let u = RecordedVideoLibrary.bundledURL() { return "Bundled: \(u.lastPathComponent)" }
        return nil
    }

    var hasImportedRecording: Bool { RecordedVideoLibrary.importedURL() != nil }

    func importRecordedVideo(from url: URL) throws {
        if RecordedVideoLibrary.isFaceTrack(url) {
            try RecordedVideoLibrary.importFaceTrack(from: url)
            return
        }
        try RecordedVideoLibrary.importVideo(from: url)
        useRecordedVideo = true
    }

    /// How the recorded video is analysed here (the Simulator needs a face track for blinks and yawns).
    var recordedVideoAnalysisNote: String? {
        guard !VideoFaceAnalyzer.landmarksAvailable, let url = recordedVideoURL else { return nil }
        return FaceTrack.locate(for: url) != nil
            ? "Face track found: blinks, yawns and gaze come from it (computed on the Mac by the same analyzer)."
            : "No face track: in the Simulator only head pose, looking away and heart rate are measured. Run tools/facetrack.sh on the Mac and add the .facetrack.csv."
    }

    func removeImportedRecording() throws {
        try RecordedVideoLibrary.removeImported()
        if recordedVideoURL == nil { useRecordedVideo = false }
    }

    private static func describe(_ url: URL) -> String {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return "\(url.lastPathComponent) (\(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)))"
    }

    /// A source for the recorded video (the work zone keeps the saved size, re-centred on the video's own pose).
    func makeRecordedVideoSource(url: URL) -> RecordedVideoFrameSource {
        RecordedVideoFrameSource(url: url, workZone: WorkZoneStore().current(isPhoneScreen: model.studyMode))
    }

    // MARK: UIServices slots

    func faceVideoView() -> AnyView? {
        guard let s = engine.currentSource as? RecordedVideoFrameSource else { return nil }
        return AnyView(RecordedVideoPlayerView(player: s.player, gravity: .resizeAspectFill)
            .accessibilityLabel("Recorded video being analysed"))
    }

    /// Width / height of the upright recorded video, so overlays can map normalized mesh points under aspect-fill.
    func faceVideoAspectRatio() -> CGFloat? {
        guard let s = engine.currentSource as? RecordedVideoFrameSource,
              let size = s.player.currentItem?.presentationSize, size.width > 0, size.height > 0 else { return nil }
        return size.width / size.height
    }
}

/// Shows an `AVPlayer` (the one the pipeline analyses) with an `AVPlayerLayer`. Muted; no controls.
struct RecordedVideoPlayerView: UIViewRepresentable {
    let player: AVPlayer
    var gravity: AVLayerVideoGravity = .resizeAspect

    final class PlayerUIView: UIView {
        override static var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }

    func makeUIView(context: Context) -> PlayerUIView {
        let v = PlayerUIView()
        v.backgroundColor = .black
        v.playerLayer.player = player
        v.playerLayer.videoGravity = gravity
        return v
    }

    func updateUIView(_ v: PlayerUIView, context: Context) {
        if v.playerLayer.player !== player { v.playerLayer.player = player }
        v.playerLayer.videoGravity = gravity
    }
}
