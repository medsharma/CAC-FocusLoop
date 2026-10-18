import Foundation

/// The bundled lectures (`lectures.json` + `<id>.brain.json` + video + optional `<id>.vertices.f16`,
/// all at the bundle root).
///
/// Brain timelines are the predicted brain response to the content (average brain), never the
/// student's own brain. While they are MOCK (`isMock`), the UI must show a MOCK badge.
/// Not thread-safe (caches); use from one actor.
public final class LectureLibrary {
    public let lectures: [LectureInfo]
    public let bundle: Bundle

    private var timelineCache: [String: BrainContentTimeline] = [:]

    /// Loads `LectureInfo.catalog(bundle:)`.
    public convenience init(bundle: Bundle = .main) {
        self.init(lectures: LectureInfo.catalog(bundle: bundle), bundle: bundle)
    }

    public init(lectures: [LectureInfo], bundle: Bundle = .main) {
        self.lectures = lectures
        self.bundle = bundle
    }

    public func lecture(id: String) -> LectureInfo? { lectures.first { $0.id == id } }

    /// `<id>.brain.json` wrapped as a `ContentTimeline` (cached), or nil if missing/invalid.
    public func timeline(for id: String) -> BrainContentTimeline? {
        if let t = timelineCache[id] { return t }
        guard let t = try? BrainContentTimeline(named: id, bundle: bundle) else { return nil }
        timelineCache[id] = t
        return t
    }

    /// Bundle URL of the lecture video; nil for timer-driven lectures (videoFile == nil) or a missing file.
    public func videoURL(for id: String) -> URL? {
        guard let file = lecture(id: id)?.videoFile, !file.isEmpty else { return nil }
        let ns = file as NSString
        return bundle.url(forResource: ns.deletingPathExtension, withExtension: ns.pathExtension)
    }

    /// True when `<id>.vertices.f16` is bundled.
    public func hasVertexPlayback(for id: String) -> Bool { vertexFileURL(for: id) != nil }

    public func vertexFileURL(for id: String) -> URL? {
        bundle.url(forResource: id, withExtension: "vertices.f16")
    }

    /// Per-second fsaverage5 vertex activity (0...1) for `BrainCortexView(vertexActivity:)`, or nil if
    /// the lecture has no `.vertices.f16`. `frames[s]` has `BrainMesh.expectedVertexCount` values.
    /// ~14 MB of Floats for a 3-minute lecture: load on demand and drop it when the lecture closes.
    public func vertexFrames(for id: String) -> [[Float]]? {
        guard let url = vertexFileURL(for: id), let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            return nil
        }
        return Self.decodeVertexFrames(data)
    }

    public func valleyTracker(for id: String, approachHorizon: Double = 10) -> ValleyTracker? {
        timeline(for: id).map { ValleyTracker(timeline: $0, approachHorizon: approachHorizon) }
    }

    /// Recap covering playhead `t` (or the latest one that ended before `t`), for the `recapCard` arm.
    public func recap(for id: String, at t: Double) -> LectureInfo.TimedText? {
        guard let recaps = lecture(id: id)?.recaps else { return nil }
        return recaps.first { t >= $0.start && t < $0.end } ?? recaps.last { $0.end <= t }
    }

    /// Latest recall question whose content has been covered (`after <= t`) and that hasn't been asked yet.
    public func recallQuestion(for id: String, at t: Double,
                               excluding asked: Set<Int> = []) -> (index: Int, question: LectureInfo.RecallQuestion)? {
        guard let qs = lecture(id: id)?.recallQuestions else { return nil }
        guard let i = qs.indices.last(where: { qs[$0].after <= t && !asked.contains($0) }) else { return nil }
        return (i, qs[i])
    }

    // MARK: vertices.f16

    /// Raw little-endian float16, no header, row-major [second][vertex]. Returns nil if the size is not
    /// a whole number of frames.
    public static func decodeVertexFrames(_ data: Data, vertexCount: Int = BrainMesh.expectedVertexCount) -> [[Float]]? {
        let frameBytes = vertexCount * 2
        guard vertexCount > 0, !data.isEmpty, data.count % frameBytes == 0 else { return nil }
        let frameCount = data.count / frameBytes
        var frames: [[Float]] = []
        frames.reserveCapacity(frameCount)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for f in 0..<frameCount {
                let base = f * frameBytes
                frames.append([Float](unsafeUninitializedCapacity: vertexCount) { buf, count in
                    for v in 0..<vertexCount {
                        let bits = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: base + v * 2, as: UInt16.self))
                        buf[v] = Self.halfToFloat(bits)
                    }
                    count = vertexCount
                })
            }
        }
        return frames
    }

    /// IEEE 754 binary16 -> Float (portable; no reliance on the `Float16` type).
    @inline(__always)
    static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = Int((h >> 10) & 0x1F)
        let mant = UInt32(h & 0x3FF)
        if exp == 0 {
            if mant == 0 { return Float(bitPattern: sign) }
            let value = Float(mant) * Float(sign: .plus, exponent: -24, significand: 1)   // subnormal
            return sign == 0 ? value : -value
        }
        if exp == 0x1F { return Float(bitPattern: sign | 0x7F80_0000 | (mant << 13)) }      // inf / NaN
        return Float(bitPattern: sign | (UInt32(exp + 112) << 23) | (mant << 13))
    }
}
