import Foundation
@testable import FocusLoop

/// Hand-built 10 Hz windows for Inference tests.
enum InferenceTestWindows {
    /// A calm, focused baseline frame.
    static func focusedFrame(t: Double) -> FeatureFrame {
        var f = FeatureFrame(t: t)
        f.faceDetected = 1
        f.inWorkZone = 1
        f.eyeBlinkLeft = 0.04; f.eyeBlinkRight = 0.05
        f.eyeSquintLeft = 0.1; f.eyeSquintRight = 0.1
        f.browDownLeft = 0.1; f.browDownRight = 0.1; f.browTension = 0.1
        f.jawOpen = 0.03
        f.blinkRate = 14; f.blinkMeanDuration = 0.2
        f.heartRate = 68; f.heartRateQuality = 0.8; f.heartRateDelta = 0
        return f
    }

    /// `n` frames ending at t = n/10, with `modify(frame, index)` applied to the focused baseline.
    static func make(_ n: Int = 100, _ modify: (inout FeatureFrame, Int) -> Void = { _, _ in }) -> [FeatureFrame] {
        (0..<n).map { i in
            var f = focusedFrame(t: Double(i) / 10)
            modify(&f, i)
            return f
        }
    }

    static func focused() -> [FeatureFrame] {
        make { f, i in
            if i % 40 == 5 || i % 40 == 6 { f.eyeBlinkLeft = 0.95; f.eyeBlinkRight = 0.95 } // normal blinks
        }
    }

    static func distractedGaze() -> [FeatureFrame] {
        make { f, i in
            if i >= 60 {
                f.inWorkZone = 0; f.gazeYaw = 0.8; f.gazeDeviation = 0.4
                f.offZoneDwell = Float(i - 59) / 10
            }
        }
    }

    static func phonePickup() -> [FeatureFrame] {
        make { f, i in
            if i >= 75 { f.phonePickup = 1; f.motionMagnitude = 0.15 }
        }
    }

    static func drowsy() -> [FeatureFrame] {
        make { f, i in
            f.eyeBlinkLeft = 0.45; f.eyeBlinkRight = 0.47 // heavy lids
            if (70..<80).contains(i) { f.eyeBlinkLeft = 0.95; f.eyeBlinkRight = 0.95 } // 1 s long blink
            f.perclos = 0.2; f.longBlinkCount = 3
        }
    }

    static func yawnAndNod() -> [FeatureFrame] {
        make { f, i in
            if (75..<95).contains(i) { f.jawOpen = 0.8; f.yawnActive = 1 }
            if (80..<90).contains(i) { f.headPitch = -0.35 }
        }
    }

    static func stressed() -> [FeatureFrame] {
        make { f, _ in
            f.heartRate = 85; f.heartRateDelta = 17
            f.browDownLeft = 0.55; f.browDownRight = 0.55; f.browTension = 0.55
            f.eyeSquintLeft = 0.35; f.eyeSquintRight = 0.35
        }
    }

    static func faceMissing(pickup: Bool = false) -> [FeatureFrame] {
        make { f, i in
            if i >= 50 {
                f = FeatureFrame(t: f.t) // all zeros: no face, no blendshapes, no HR
                if pickup { f.phonePickup = 1 }
            }
        }
    }
}

/// Classifier stub returning fixed probabilities / embedding.
final class InferenceStubClassifier: AttentionClassifier {
    var probabilities: [Float]
    var embedding: [Float]
    private(set) var calls = 0
    init(probabilities: [Float], embedding: [Float] = [1, 2, 3]) {
        self.probabilities = probabilities
        self.embedding = embedding
    }
    func classify(window: [FeatureFrame]) -> StateEstimate {
        calls += 1
        return StateEstimate(t: window.last?.t ?? 0, probabilities: probabilities, source: .model, embedding: embedding)
    }
}

/// Personal head stub with fixed output.
final class InferenceStubHead: PersonalHead {
    var labelCount: Int
    var output: [Float]?
    init(labelCount: Int, output: [Float]?) { self.labelCount = labelCount; self.output = output }
    func predict(embedding: [Float]) -> [Float]? { output }
    func update(embedding: [Float], label: AttentionState) { labelCount += 1 }
    func serialized() throws -> Data { Data() }
    func restore(from data: Data) throws {}
    func reset() { labelCount = 0 }
}
