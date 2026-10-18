import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// What the "Use my recorded video" settings need from the app (implemented by `AppServices`).
@MainActor
public protocol RecordedVideoSettingsControlling: AnyObject {
    /// Sessions analyse the recorded video instead of the camera / SYNTHETIC replay.
    var useRecordedVideo: Bool { get set }
    /// Which video would be used ("Imported: …", "Bundled: …", "Launch argument: …"), nil if none.
    var recordedVideoDescription: String? { get }
    var hasImportedRecording: Bool { get }
    /// How the video is analysed on this device, when that limits what is measured (Simulator); nil otherwise.
    var recordedVideoAnalysisNote: String? { get }
    /// Copies the picked file into the app (Application Support/FocusLoop-demo-recording.<ext>). A
    /// `.facetrack.csv` (from tools/facetrack.sh) is stored next to it.
    func importRecordedVideo(from url: URL) throws
    func removeImportedRecording() throws
}

/// Self-contained Settings section: pick a video of yourself studying (Photos or Files) and use it as the camera
/// for demos. The real perception pipeline runs on it frame by frame; such sessions are labelled RECORDED and
/// never train your personal models. Embed inside a `List`.
public struct RecordedVideoSettingsSection: View {
    let controller: RecordedVideoSettingsControlling
    @State private var enabled = false
    @State private var detail: String?
    @State private var hasImport = false
    @State private var note: String?
    @State private var photoItem: PhotosPickerItem?
    @State private var showFiles = false
    @State private var busy = false
    @State private var error: String?

    public init(controller: RecordedVideoSettingsControlling) { self.controller = controller }

    public var body: some View {
        Section {
            Toggle(isOn: Binding(get: { enabled }, set: { enabled = $0; controller.useRecordedVideo = $0 })) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) { Text("Use my recorded video"); FLTag(.recorded) }
                    Text(detail ?? "No video yet. Pick one below.")
                        .font(.caption).foregroundStyle(FLColor.textSecondary)
                }
            }
            .disabled(detail == nil)
            .accessibilityIdentifier("settings.recordedVideo.toggle")

            PhotosPicker(selection: $photoItem, matching: .videos, photoLibrary: .shared()) {
                Label(hasImport ? "Replace from Photos" : "Choose from Photos", systemImage: "photo.on.rectangle")
            }
            .disabled(busy)
            .accessibilityIdentifier("settings.recordedVideo.photos")

            Button { showFiles = true } label: { Label("Choose from Files (video and/or face track)", systemImage: "folder") }
                .disabled(busy)

            if let note {
                Text(note).font(.caption).foregroundStyle(FLColor.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if busy { HStack(spacing: 8) { ProgressView(); Text("Copying video\u{2026}").foregroundStyle(FLColor.textSecondary) } }

            if hasImport {
                Button(role: .destructive) {
                    do { try controller.removeImportedRecording() } catch { self.error = error.localizedDescription }
                    refresh()
                } label: { Label("Remove imported video", systemImage: "trash") }
            }
        } header: {
            Text("Recorded video (demo camera)")
        } footer: {
            Text("Record yourself studying, then pick the video. Sessions analyse it frame by frame on this phone, exactly like the camera. They are labelled RECORDED and never train your personal models. The imported copy is removed with Delete all data.")
        }
        .onAppear(perform: refresh)
        .onChange(of: photoItem) { _, item in
            guard let item else { return }
            busy = true
            Task {
                defer { busy = false; photoItem = nil; refresh() }
                do {
                    guard let movie = try await item.loadTransferable(type: PickedMovie.self) else { return }
                    defer { try? FileManager.default.removeItem(at: movie.url) }
                    try controller.importRecordedVideo(from: movie.url)
                } catch {
                    self.error = error.localizedDescription
                }
            }
        }
        .fileImporter(isPresented: $showFiles, allowedContentTypes: [.movie, .quickTimeMovie, .mpeg4Movie, .commaSeparatedText],
                      allowsMultipleSelection: true) { result in
            switch result {
            case .success(let urls):
                // The video first (importing a video replaces any earlier face track), then its face track.
                func isCSV(_ u: URL) -> Bool { u.pathExtension.lowercased() == "csv" }
                let ordered = urls.filter { !isCSV($0) } + urls.filter(isCSV)
                for url in ordered {
                    let scoped = url.startAccessingSecurityScopedResource()
                    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                    do { try controller.importRecordedVideo(from: url) } catch { self.error = error.localizedDescription }
                }
            case .failure(let e):
                error = e.localizedDescription
            }
            refresh()
        }
        .alert("Couldn't use that video", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(error ?? "") }
    }

    private func refresh() {
        enabled = controller.useRecordedVideo
        detail = controller.recordedVideoDescription
        hasImport = controller.hasImportedRecording
        note = controller.recordedVideoAnalysisNote
    }
}

/// A movie picked from Photos, copied to a temporary file (FocusLoop prefix, swept by Delete all data).
struct PickedMovie: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { SentTransferredFile($0.url) } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("FocusLoop-import-\(UUID().uuidString).\(ext)")
            try FileManager.default.copyItem(at: received.file, to: dest)
            return PickedMovie(url: dest)
        }
    }
}
