import Foundation

/// Opt-in recording of the full 10 Hz numeric feature stream of each session, so the LapseForecaster can
/// be retrained on real data (`ml/forecast/train.py --manifest ...`). The labelled-window CSV only holds
/// 10 s windows around labels, which is not enough for the forecaster's 20 s history + 15 s horizon.
///
/// Numbers only (the 42 `FeatureFrame` values, session time and lecture playhead), never images.
/// Off by default (Settings → Your data). Files live in Application Support/FocusLoop/streams/ and are
/// removed by Delete all data. The export writes a `manifest.csv` in the format `ml/forecast/README.md`
/// documents, plus copies of the bundled `<lecture>.brain.json` files the sessions used.
public final class FeatureStreamStore {
    public static let enabledKey = "focusloop.recordFeatureStreams"
    public static let manifestName = "manifest.csv"
    public static let manifestColumns = ["subject", "session_id", "session_csv", "sart_csv", "es_csv",
                                         "lecture_json", "playhead_offset", "synthetic"]
    public static var sessionColumns: [String] { ["t", "playhead"] + FeatureFrame.featureNames }

    public let directory: URL
    private let defaults: UserDefaults

    public init(directory: URL = FeatureStreamStore.defaultDirectory, defaults: UserDefaults = .standard) {
        self.directory = directory
        self.defaults = defaults
    }

    public static var defaultDirectory: URL {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                 appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("FocusLoop", isDirectory: true).appendingPathComponent("streams", isDirectory: true)
    }

    public var isEnabled: Bool {
        get { defaults.bool(forKey: Self.enabledKey) }
        set { defaults.set(newValue, forKey: Self.enabledKey) }
    }

    /// One session being recorded. Append frames as they arrive, then `finish`.
    public final class Recording {
        public let sessionID: UUID
        public let isSynthetic: Bool
        public let lectureID: String?
        let url: URL
        private var buffer = ""
        private var handle: FileHandle?
        private var count = 0

        init(sessionID: UUID, isSynthetic: Bool, lectureID: String?, url: URL) throws {
            self.sessionID = sessionID; self.isSynthetic = isSynthetic; self.lectureID = lectureID; self.url = url
            let header = (isSynthetic ? "# FocusLoop feature stream v1 (SYNTHETIC replay)\n"
                                      : "# FocusLoop feature stream v1 (REAL device data)\n")
                + FeatureStreamStore.sessionColumns.joined(separator: ",") + "\n"
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(header.utf8).write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
            handle = try FileHandle(forWritingTo: url)
            handle?.seekToEndOfFile()
        }

        public var frameCount: Int { count }

        public func append(_ f: FeatureFrame, playhead: Double?) {
            let ph = playhead.map { String(format: "%.2f", $0) } ?? ""
            buffer += String(format: "%.3f", f.t) + "," + ph + ","
                + f.vector.map { String(format: "%.4f", $0) }.joined(separator: ",") + "\n"
            count += 1
            if count % 50 == 0 { flush() }
        }

        func flush() {
            guard !buffer.isEmpty, let handle else { return }
            handle.write(Data(buffer.utf8))
            buffer = ""
        }

        /// Writes the remaining frames and the experience-sampling answers (`t,focused,reason`).
        public func finish(experienceSamples: [(t: Double, focused: Bool, reason: NotFocusedReason?)]) {
            flush()
            try? handle?.close()
            handle = nil
            guard !experienceSamples.isEmpty else { return }
            var es = "t,focused,reason\n"
            for s in experienceSamples {
                es += String(format: "%.3f", s.t) + ",\(s.focused ? 1 : 0),\(s.reason?.rawValue ?? "")\n"
            }
            try? Data(es.utf8).write(to: FeatureStreamStore.esURL(for: url),
                                     options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        }
    }

    /// Starts a recording if the user turned recording on (nil otherwise).
    public func begin(sessionID: UUID, startedAt: Date, isSynthetic: Bool, lectureID: String?,
                      kind: String = "session") -> Recording? {
        guard isEnabled else { return nil }
        let url = directory.appendingPathComponent(Self.baseName(startedAt: startedAt, sessionID: sessionID, kind: kind,
                                                                 lectureID: lectureID, isSynthetic: isSynthetic) + ".csv")
        return try? Recording(sessionID: sessionID, isSynthetic: isSynthetic, lectureID: lectureID, url: url)
    }

    /// Saves SART trials next to a SART recording (same session clock), as `<stream>.sart.csv`.
    public func attachSART(_ csv: String, to recording: Recording) {
        try? Data(csv.utf8).write(to: Self.sartURL(for: recording.url),
                                  options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
    }

    public var sessionFiles: [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return items.filter { $0.pathExtension == "csv" && !$0.lastPathComponent.contains(".es.")
            && !$0.lastPathComponent.contains(".sart.") && $0.lastPathComponent != Self.manifestName }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Writes `manifest.csv` (plus the brain timelines it references) and returns every file to share.
    public func prepareExport(subject: String = "me", bundle: Bundle = .main) throws -> [URL] {
        let fm = FileManager.default
        var rows = [Self.manifestColumns.joined(separator: ",")]
        var files: [URL] = []
        for url in sessionFiles {
            let name = url.lastPathComponent
            let es = Self.esURL(for: url), sart = Self.sartURL(for: url)
            let synthetic = name.contains("-SYNTHETIC")
            var lectureJSON = ""
            if let id = Self.lectureID(fromFileName: name), let src = bundle.url(forResource: id, withExtension: "brain.json") {
                let dst = directory.appendingPathComponent("\(id).brain.json")
                if !fm.fileExists(atPath: dst.path) { try? fm.copyItem(at: src, to: dst) }
                if fm.fileExists(atPath: dst.path) { lectureJSON = dst.lastPathComponent; files.append(dst) }
            }
            let hasES = fm.fileExists(atPath: es.path), hasSART = fm.fileExists(atPath: sart.path)
            rows.append([subject, url.deletingPathExtension().lastPathComponent, name,
                         hasSART ? sart.lastPathComponent : "", hasES ? es.lastPathComponent : "",
                         lectureJSON, "0", synthetic ? "1" : "0"].joined(separator: ","))
            files.append(url)
            if hasES { files.append(es) }
            if hasSART { files.append(sart) }
        }
        guard !files.isEmpty else { throw ExportError.nothingRecorded }
        let manifest = directory.appendingPathComponent(Self.manifestName)
        try Data((rows.joined(separator: "\n") + "\n").utf8).write(to: manifest, options: .atomic)
        return [manifest] + Array(Set(files)).sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    public enum ExportError: LocalizedError {
        case nothingRecorded
        public var errorDescription: String? {
            "No feature streams recorded yet. Turn on \u{201C}Record feature streams\u{201D} and run a session, then export."
        }
    }

    /// Removes every recorded stream (Delete all data).
    public func removeAll() {
        try? FileManager.default.removeItem(at: directory)
    }

    // MARK: Naming

    /// `stream-20261025-1530-<id8>[-lecture_<id>][-sart][-SYNTHETIC]`
    static func baseName(startedAt: Date, sessionID: UUID, kind: String, lectureID: String?, isSynthetic: Bool) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd-HHmmss"
        var name = "stream-\(f.string(from: startedAt))-\(sessionID.uuidString.prefix(8))"
        if let lectureID { name += "-lecture_\(lectureID)" }
        if kind != "session" { name += "-\(kind)" }
        if isSynthetic { name += "-SYNTHETIC" }
        return name
    }

    static func lectureID(fromFileName name: String) -> String? {
        guard let r = name.range(of: "-lecture_") else { return nil }
        var rest = String(name[r.upperBound...])
        if rest.hasSuffix(".csv") { rest.removeLast(4) }
        for suffix in ["-SYNTHETIC", "-sart"] where rest.hasSuffix(suffix) { rest.removeLast(suffix.count) }
        return rest.isEmpty ? nil : rest
    }

    static func esURL(for url: URL) -> URL {
        url.deletingPathExtension().appendingPathExtension("es.csv")
    }

    static func sartURL(for url: URL) -> URL {
        url.deletingPathExtension().appendingPathExtension("sart.csv")
    }
}
