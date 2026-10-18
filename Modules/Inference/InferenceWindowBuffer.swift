import Foundation

/// Fixed-capacity ring buffer used to hold the most recent frames of a stream
/// (e.g. 10 s of 10 Hz `FeatureFrame`s for the state classifier).
///
/// Appending is O(1); `elements` returns the contents oldest first, which is the order
/// `AttentionClassifier.classify(window:)` expects. Not thread-safe (value type; the session
/// engine owns one per actor).
public struct InferenceWindowBuffer<Element> {
    public let capacity: Int
    private var storage: [Element] = []
    /// Index of the oldest element once the buffer is full (always 0 before that).
    private var head = 0

    public init(capacity: Int) {
        precondition(capacity > 0, "InferenceWindowBuffer capacity must be positive")
        self.capacity = capacity
        storage.reserveCapacity(capacity)
    }

    public var count: Int { storage.count }
    public var isEmpty: Bool { storage.isEmpty }
    public var isFull: Bool { storage.count == capacity }

    public mutating func append(_ element: Element) {
        if storage.count < capacity {
            storage.append(element)
        } else {
            storage[head] = element
            head = (head + 1) % capacity
        }
    }

    public mutating func append<S: Sequence>(contentsOf elements: S) where S.Element == Element {
        for e in elements { append(e) }
    }

    /// Contents, oldest first.
    public var elements: [Element] {
        head == 0 ? storage : Array(storage[head...]) + Array(storage[..<head])
    }

    /// Most recently appended element.
    public var last: Element? {
        storage.isEmpty ? nil : storage[(head + storage.count - 1) % storage.count]
    }

    /// Oldest element still held.
    public var first: Element? {
        storage.isEmpty ? nil : storage[head]
    }

    public mutating func removeAll() {
        storage.removeAll(keepingCapacity: true)
        head = 0
    }
}

public extension InferenceWindowBuffer where Element == FeatureFrame {
    /// 10 s x 10 Hz = 100 frames: the state-classifier window.
    static func stateWindow() -> InferenceWindowBuffer<FeatureFrame> {
        InferenceWindowBuffer(capacity: Int((FeatureFrame.stateWindowSeconds * FeatureFrame.rateHz).rounded()))
    }

    /// 20 s x 10 Hz = 200 frames: the lapse-forecaster window.
    static func forecastWindow() -> InferenceWindowBuffer<FeatureFrame> {
        InferenceWindowBuffer(capacity: Int((FeatureFrame.forecastWindowSeconds * FeatureFrame.rateHz).rounded()))
    }
}
