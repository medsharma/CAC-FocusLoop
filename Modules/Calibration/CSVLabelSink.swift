import Foundation

/// `LabelSink` that keeps labelled windows in memory, persists them (numbers only) as JSON in
/// Application Support, and exports CSV in the replay format.
///
/// CSV format (one row per frame): `t,label,lapse,<FeatureFrame.featureNames...>,session_id,source,window_id`.
/// The first 45 columns are identical to `Resources/Replay/synthetic_session.replay.csv`; the three trailing
/// columns are extra, so ML loaders must select columns by name (or ignore trailing extras).
/// `t` is written with 3 decimals (real sessions aren't on a 0.1 s grid), features with 4.
/// The first line is a comment: `# FocusLoop export v1 (REAL device data)` or `# FocusLoop export v1 (SYNTHETIC replay)`.
/// Thread-safe (internal lock).
public final class CSVLabelSink: LabelSink {
    public static let realHeaderComment = "# FocusLoop export v1 (REAL device data)"
    public static let syntheticHeaderComment = "# FocusLoop export v1 (SYNTHETIC replay)"
    public static let extraColumns = ["session_id", "source", "window_id"]
    public static var columns: [String] { ["t", "label", "lapse"] + FeatureFrame.featureNames + extraColumns }
    public static let exportFileName = "focusloop_labels.csv"

    /// Set by the caller: true when the windows came from the SYNTHETIC replay stream.
    public var isSyntheticData: Bool {
        get { lock.withLock { synthetic } }
        set { lock.withLock { synthetic = newValue } }
    }

    private let lock = NSLock()
    private var stored: [LabeledWindow] = []
    private var synthetic: Bool
    private let storeURL: URL?

    /// - Parameter storeURL: JSON persistence file; nil keeps windows in memory only (tests).
    public init(storeURL: URL? = CSVLabelSink.defaultStoreURL, isSyntheticData: Bool = false) {
        self.storeURL = storeURL
        synthetic = isSyntheticData
        if let storeURL, let data = try? Data(contentsOf: storeURL),
           let w = try? JSONDecoder().decode([LabeledWindow].self, from: data) {
            stored = w
        }
    }

    public static var defaultStoreURL: URL? {
        guard let dir = try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                     appropriateFor: nil, create: true) else { return nil }
        return dir.appendingPathComponent("FocusLoop", isDirectory: true).appendingPathComponent("labeled_windows.json")
    }

    public var windows: [LabeledWindow] { lock.withLock { stored } }

    public func add(_ window: LabeledWindow) {
        let snapshot = lock.withLock { () -> [LabeledWindow] in
            stored.append(window)
            return stored
        }
        persist(snapshot)
    }

    public func add(contentsOf windows: [LabeledWindow]) {
        let snapshot = lock.withLock { () -> [LabeledWindow] in
            stored.append(contentsOf: windows)
            return stored
        }
        persist(snapshot)
    }

    public func removeAll() {
        lock.withLock { stored.removeAll() }
        if let storeURL { try? FileManager.default.removeItem(at: storeURL) }
    }

    private func persist(_ windows: [LabeledWindow]) {
        guard let storeURL else { return }
        do {
            try FileManager.default.createDirectory(at: storeURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(windows).write(to: storeURL, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        } catch {
            // Persistence is best effort; windows remain in memory for export.
        }
    }

    public func exportCSV() throws -> URL {
        try exportCSV(isSynthetic: isSyntheticData)
    }

    public func exportCSV(isSynthetic: Bool, directory: URL? = nil) throws -> URL {
        let url = (directory ?? FileManager.default.temporaryDirectory).appendingPathComponent(Self.exportFileName)
        try Self.csv(windows, isSynthetic: isSynthetic).write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    public static func csv(_ windows: [LabeledWindow], isSynthetic: Bool) -> String {
        var lines: [String] = [isSynthetic ? syntheticHeaderComment : realHeaderComment, columns.joined(separator: ",")]
        for w in windows {
            let tail = "\(w.sessionID.uuidString),\(w.source.rawValue),\(w.id.uuidString)"
            let head = "\(w.label.rawValue),\(w.isLapse ? 1 : 0)"
            for f in w.frames {
                let feats = f.vector.map { String(format: "%.4f", $0) }.joined(separator: ",")
                lines.append("\(String(format: "%.3f", f.t)),\(head),\(feats),\(tail)")
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }
}
