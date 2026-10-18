import Accelerate
import Foundation

/// On-device multinomial logistic regression over the classifier embedding (spec §1.2
/// "Personalization").
///
/// * Embeddings are standardised with the mean / std of the stored labelled samples.
/// * Each `update` stores the sample in a replay buffer (newest `replayCapacity` kept) and runs
///   `stepsPerUpdate` mini-batch gradient steps (softmax cross-entropy + L2) over the buffer,
///   always including the newest sample. Matrix maths uses Accelerate (vDSP).
/// * `predict` returns nil until `minLabels` labels covering `minClasses` distinct states exist.
/// * The input dimension is fixed by the first embedding. If a later embedding has a different
///   dimension (e.g. the Core ML model replaced the heuristic), the head resets and starts over,
///   because the old weights live in a different embedding space. Empty embeddings are ignored.
/// * `serialized()` / `restore(from:)` persist everything as JSON (weights, standardisation,
///   replay buffer, label count).
///
/// Thread-safe (internal lock): the session pipeline predicts/updates while the main actor
/// serialises, restores or resets it.
public final class LogisticPersonalHead: PersonalHead, @unchecked Sendable {
    private let lock = NSRecursiveLock()
    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock(); defer { lock.unlock() }
        return try body()
    }

    public struct Config: Codable, Equatable, Sendable {
        public var minLabels = 8
        public var minClasses = 2
        public var replayCapacity = 500
        public var learningRate: Float = 0.3
        public var l2: Float = 1e-3
        public var stepsPerUpdate = 20
        public var batchSize = 64
        public init() {}
    }

    public enum PersistenceError: Error { case unsupportedVersion(Int), corrupt(String) }

    private struct State: Codable {
        var version = 1
        var config: Config
        var labelCount: Int
        var dimension: Int?
        /// D x K, row-major.
        var weights: [Float]
        var bias: [Float]
        var mean: [Float]
        var std: [Float]
        /// Flattened N x D.
        var replayX: [Float]
        var replayY: [Int]
    }

    public static let classCount = AttentionState.allCases.count

    public private(set) var config: Config
    public private(set) var labelCount = 0
    /// Embedding dimension, fixed by the first update (nil before).
    public private(set) var dimension: Int?

    private var weights: [Float] = []   // D x K
    private var bias: [Float] = Array(repeating: 0, count: LogisticPersonalHead.classCount)
    private var mean: [Float] = []
    private var std: [Float] = []
    private var replayX: [Float] = []   // N x D
    private var replayY: [Int] = []
    private var rng = InferenceSplitMix64(seed: 0x5EED_F0C5)

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Number of stored samples per state, `AttentionState.allCases` order.
    public var storedClassCounts: [Int] { locked { classCounts() } }

    private func classCounts() -> [Int] {
        var c = Array(repeating: 0, count: Self.classCount)
        for y in replayY { c[y] += 1 }
        return c
    }

    public var isReady: Bool { locked { ready() } }

    private func ready() -> Bool {
        labelCount >= config.minLabels && classCounts().filter { $0 > 0 }.count >= config.minClasses
    }

    public func predict(embedding: [Float]) -> [Float]? { locked { predictLocked(embedding: embedding) } }

    private func predictLocked(embedding: [Float]) -> [Float]? {
        guard ready(), let d = dimension, embedding.count == d else { return nil }
        let x = standardize(embedding)
        let k = Self.classCount
        var z = [Float](repeating: 0, count: k)
        // z (1 x K) = x (1 x D) * W (D x K)
        vDSP_mmul(x, 1, weights, 1, &z, 1, 1, vDSP_Length(k), vDSP_Length(d))
        return InferenceMath.softmax(vDSP.add(z, bias))
    }

    public func update(embedding: [Float], label: AttentionState) { locked { updateLocked(embedding: embedding, label: label) } }

    private func updateLocked(embedding: [Float], label: AttentionState) {
        guard !embedding.isEmpty, embedding.allSatisfy(\.isFinite) else { return }
        if let d = dimension, d != embedding.count { resetLocked() }
        if dimension == nil { initialise(dimension: embedding.count) }
        guard let d = dimension else { return }

        replayX.append(contentsOf: embedding)
        replayY.append(label.index)
        let overflow = replayY.count - config.replayCapacity
        if overflow > 0 {
            replayX.removeFirst(overflow * d)
            replayY.removeFirst(overflow)
        }
        labelCount += 1
        recomputeStandardization()
        train(steps: config.stepsPerUpdate)
    }

    public func serialized() throws -> Data {
        let s = locked { State(config: config, labelCount: labelCount, dimension: dimension, weights: weights, bias: bias,
                      mean: mean, std: std, replayX: replayX, replayY: replayY) }
        return try JSONEncoder().encode(s)
    }

    public func restore(from data: Data) throws {
        let s = try JSONDecoder().decode(State.self, from: data)
        guard s.version == 1 else { throw PersistenceError.unsupportedVersion(s.version) }
        let k = Self.classCount
        guard s.bias.count == k, s.replayY.allSatisfy({ (0..<k).contains($0) }) else {
            throw PersistenceError.corrupt("bias/labels")
        }
        if let d = s.dimension {
            guard d > 0, s.weights.count == d * k, s.mean.count == d, s.std.count == d,
                  s.replayX.count == s.replayY.count * d else { throw PersistenceError.corrupt("shapes") }
        } else {
            guard s.replayY.isEmpty else { throw PersistenceError.corrupt("replay without dimension") }
        }
        locked {
            config = s.config; labelCount = s.labelCount; dimension = s.dimension
            weights = s.weights; bias = s.bias; mean = s.mean; std = s.std
            replayX = s.replayX; replayY = s.replayY
        }
    }

    public func reset() { locked { resetLocked() } }

    private func resetLocked() {
        labelCount = 0
        dimension = nil
        weights = []; mean = []; std = []
        bias = Array(repeating: 0, count: Self.classCount)
        replayX = []; replayY = []
    }

    // MARK: - Training

    private func initialise(dimension d: Int) {
        dimension = d
        weights = Array(repeating: 0, count: d * Self.classCount)
        bias = Array(repeating: 0, count: Self.classCount)
        mean = Array(repeating: 0, count: d)
        std = Array(repeating: 1, count: d)
    }

    private func standardize(_ x: [Float]) -> [Float] {
        vDSP.divide(vDSP.subtract(x, mean), std)
    }

    private func recomputeStandardization() {
        guard let d = dimension, !replayY.isEmpty else { return }
        let n = replayY.count
        replayX.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            for j in 0..<d {
                var m: Float = 0, ms: Float = 0
                vDSP_meanv(base + j, vDSP_Stride(d), &m, vDSP_Length(n))
                vDSP_measqv(base + j, vDSP_Stride(d), &ms, vDSP_Length(n))
                let sd = sqrt(max(ms - m * m, 0))
                mean[j] = m
                std[j] = sd > 1e-3 ? sd : 1
            }
        }
    }

    private func train(steps: Int) {
        guard let d = dimension, !replayY.isEmpty else { return }
        let k = Self.classCount
        let n = replayY.count
        let b = min(config.batchSize, n)
        let lr = config.learningRate, l2 = config.l2

        var x = [Float](repeating: 0, count: b * d)      // B x D
        var xt = [Float](repeating: 0, count: d * b)     // D x B
        var z = [Float](repeating: 0, count: b * k)      // B x K
        var gw = [Float](repeating: 0, count: d * k)     // D x K
        var indices = Array(0..<n)

        for _ in 0..<steps {
            // Mini-batch: newest sample + a random subset of the replay buffer.
            var batch: [Int]
            if b == n {
                batch = indices
            } else {
                indices.shuffle(using: &rng)
                batch = Array(indices.prefix(b))
                if !batch.contains(n - 1) { batch[0] = n - 1 }
            }
            for (r, i) in batch.enumerated() {
                let row = standardize(Array(replayX[(i * d)..<((i + 1) * d)]))
                x.replaceSubrange((r * d)..<((r + 1) * d), with: row)
            }
            // Z = X W + b, softmax rows, G = P - Y
            vDSP_mmul(x, 1, weights, 1, &z, 1, vDSP_Length(b), vDSP_Length(k), vDSP_Length(d))
            for r in 0..<b {
                var p = InferenceMath.softmax(vDSP.add(z[(r * k)..<((r + 1) * k)], bias))
                p[replayY[batch[r]]] -= 1
                z.replaceSubrange((r * k)..<((r + 1) * k), with: p)
            }
            // dW = X^T G / B + l2 W ; db = colsum(G) / B
            vDSP_mtrans(x, 1, &xt, 1, vDSP_Length(d), vDSP_Length(b))
            vDSP_mmul(xt, 1, z, 1, &gw, 1, vDSP_Length(d), vDSP_Length(k), vDSP_Length(b))
            let grad = vDSP.add(vDSP.multiply(1 / Float(b), gw), vDSP.multiply(l2, weights))
            weights = vDSP.subtract(weights, vDSP.multiply(lr, grad))
            for c in 0..<k {
                var s: Float = 0
                z.withUnsafeBufferPointer { buf in
                    if let base = buf.baseAddress { vDSP_sve(base + c, vDSP_Stride(k), &s, vDSP_Length(b)) }
                }
                bias[c] -= lr * s / Float(b)
            }
        }
    }
}

/// Deterministic PRNG (SplitMix64) so personal-head training is reproducible.
struct InferenceSplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
