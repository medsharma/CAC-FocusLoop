import Foundation

/// The Axonal Labs cortical surface (fsaverage5 pial, 20,484 vertices) with Yeo-7 network labels.
///
/// Loaded from `BrainMesh.bin`, written by `ml/brain/export_brain_mesh.py`. Vertex `i` is
/// fsaverage5 vertex `i` (left hemisphere 0–10241, right 10242–20483), so per-vertex model
/// output on fsaverage5 can be passed straight to `BrainCortexView(vertexActivity:)`.
public struct BrainMesh {
    public static let expectedVertexCount = 20_484
    /// Interleaved GPU layout, 32 bytes per vertex:
    /// float3 position | float3 normal | uchar4 weights[0..3] | uchar3 weights[4..6] + uchar ao
    public static let vertexStride = 32

    public let vertexCount: Int
    public let vertexBytes: Data
    public let indices: [UInt16]
    /// 0 = medial wall, 1...7 = `BrainNetwork.allCases[id - 1]`
    public let network: [UInt8]
    /// 0 = medial wall, 1...400 = Schaefer-400 parcel (LH 1–200, RH 201–400)
    public let parcel: [UInt16]

    public enum LoadError: Error { case missingResource, badMagic, truncated }

    /// Decoded once and shared; the file is ~1 MB.
    public static let shared: BrainMesh? = try? load()

    public static func load(bundle: Bundle = .main, name: String = "BrainMesh") throws -> BrainMesh {
        guard let url = bundle.url(forResource: name, withExtension: "bin") else {
            throw LoadError.missingResource
        }
        return try BrainMesh(data: Data(contentsOf: url))
    }

    public init(data: Data) throws {
        guard data.count >= 16, data.prefix(4) == Data("FLBM".utf8) else { throw LoadError.badMagic }

        let header: (n: Int, ni: Int) = data.withUnsafeBytes { raw in
            (Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self))),
             Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self))))
        }
        let n = header.n, ni = header.ni
        let posOff = 16
        let nrmOff = posOff + n * 12
        let aoOff = nrmOff + n * 12
        let netOff = aoOff + n
        let parcelOff = netOff + n
        let weightOff = parcelOff + n * 2
        let idxOff = weightOff + n * 8
        guard data.count >= idxOff + ni * 2 else { throw LoadError.truncated }

        var interleaved = Data(count: n * Self.vertexStride)
        var idx = [UInt16](repeating: 0, count: ni)
        var net = [UInt8](repeating: 0, count: n)
        var par = [UInt16](repeating: 0, count: n)

        data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            let s = src.baseAddress!
            interleaved.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
                let d = dst.baseAddress!
                for i in 0..<n {
                    let o = i * Self.vertexStride
                    (d + o).copyMemory(from: s + posOff + i * 12, byteCount: 12)
                    (d + o + 12).copyMemory(from: s + nrmOff + i * 12, byteCount: 12)
                    (d + o + 24).copyMemory(from: s + weightOff + i * 8, byteCount: 7) // 7 networks
                    dst[o + 31] = src[aoOff + i]                                       // ao in slot 8
                }
            }
            idx.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[idxOff ..< idxOff + ni * 2])) }
            net.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[netOff ..< netOff + n])) }
            par.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[parcelOff ..< parcelOff + n * 2])) }
        }

        vertexCount = n
        vertexBytes = interleaved
        indices = idx
        network = net
        parcel = par
    }
}
