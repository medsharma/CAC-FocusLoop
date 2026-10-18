import XCTest
import simd
@testable import FocusLoop

// All inputs below are constructed / SYNTHETIC sequences.

private func face(_ t: Double, closure: Float = 0, jaw: Float = 0.03, pitch: Float = 0, yaw: Float = 0,
                  look: SIMD3<Float> = SIMD3(0, 0, 0.5), brow: Float = 0.1, detected: Bool = true) -> FaceSample {
    FaceSample(t: t, faceDetected: detected,
               blendShapes: ["eyeBlinkLeft": closure, "eyeBlinkRight": closure, "jawOpen": jaw,
                             "browDownLeft": brow, "browDownRight": brow],
               lookAtPoint: look, headYaw: yaw, headPitch: pitch)
}

/// Runs face samples at `faceHz` through the extractor and collects frames polled at each sample.
private func perceptionRun(_ ex: AttentionFeatureExtractor, from t0: Double = 0, to t1: Double, faceHz: Double = 30,
                 _ make: (Double) -> FaceSample) -> [FeatureFrame] {
    var frames: [FeatureFrame] = []
    let n = Int(((t1 - t0) * faceHz).rounded())
    for i in 0..<n {
        let t = t0 + Double(i) / faceHz
        ex.ingest(face: make(t))
        if let f = ex.frame(at: t) { frames.append(f) }
    }
    return frames
}

final class PerceptionBlinkTests: XCTestCase {
    func testHysteresisDetector() {
        var d = PerceptionBlinkDetector()
        XCTAssertNil(d.update(closure: 0.1, t: 0))
        XCTAssertNil(d.update(closure: 0.6, t: 0.1))   // onset
        XCTAssertNil(d.update(closure: 0.4, t: 0.2))   // between thresholds: still closed
        XCTAssertNil(d.update(closure: 0.45, t: 0.3))
        let b = d.update(closure: 0.2, t: 0.4)         // offset
        XCTAssertEqual(b?.onset, 0.1)
        XCTAssertEqual(b?.duration ?? 0, 0.3, accuracy: 1e-9)
        // Flicker between 0.3 and 0.5 never starts a blink.
        for i in 0..<10 { XCTAssertNil(d.update(closure: i % 2 == 0 ? 0.45 : 0.35, t: 1 + Double(i) * 0.1)) }
        XCTAssertFalse(d.isClosed)
    }

    func testBlinkCountRateAndDuration() {
        // 10 blinks of 0.2 s, one every 3 s, plus 2 long blinks of 0.8 s, over 40 s.
        let ex = AttentionFeatureExtractor()
        let frames = perceptionRun(ex, to: 40) { t in
            let phase = t.truncatingRemainder(dividingBy: 3)
            let k = Int(t / 3)
            let dur = (k == 4 || k == 9) ? 0.8 : 0.2
            return face(t, closure: (k < 12 && phase >= 1 && phase < 1 + dur) ? 0.95 : 0.05)
        }
        let last = frames.last!
        // Blinks complete at k = 0...11 (onsets 1, 4, ..., 34; all within 40 s).
        XCTAssertEqual(last.longBlinkCount, 2)
        let elapsed = Float(min(60, last.t + 0.1))
        XCTAssertEqual(last.blinkRate, 12 / elapsed * 60, accuracy: 0.01)
        let expectedMean: Float = (10 * 0.2 + 2 * 0.8) / 12
        XCTAssertEqual(last.blinkMeanDuration, expectedMean, accuracy: 0.03)   // 30 Hz quantisation
    }

    func testBlinksLeaveTrailingWindow() {
        let ex = AttentionFeatureExtractor()
        // Blinks only in the first 10 s, then open eyes until t = 80.
        let frames = perceptionRun(ex, to: 80) { t in
            let phase = t.truncatingRemainder(dividingBy: 2)
            return face(t, closure: (t < 10 && phase >= 1 && phase < 1.2) ? 0.95 : 0.05)
        }
        let at30 = frames.first { abs($0.t - 30) < 0.05 }!
        XCTAssertEqual(at30.blinkRate, 5 / Float(30.1) * 60, accuracy: 0.05)
        XCTAssertEqual(frames.last!.blinkRate, 0)
        XCTAssertEqual(frames.last!.blinkMeanDuration, 0)
    }

    func testPerclos() {
        let ex = AttentionFeatureExtractor()
        // Eyes closed (0.9) for the 2nd half of each 4 s cycle -> PERCLOS ~ 0.5.
        let frames = perceptionRun(ex, to: 70) { t in face(t, closure: t.truncatingRemainder(dividingBy: 4) >= 2 ? 0.9 : 0.1) }
        XCTAssertEqual(frames.last!.perclos, 0.5, accuracy: 0.02)
        // Closure of 0.75 (heavy lids) does not count.
        let ex2 = AttentionFeatureExtractor()
        let f2 = perceptionRun(ex2, to: 10) { t in face(t, closure: 0.75) }
        XCTAssertEqual(f2.last!.perclos, 0)
        // Fully closed: 1.
        let ex3 = AttentionFeatureExtractor()
        XCTAssertEqual(perceptionRun(ex3, to: 5) { t in face(t, closure: 0.95) }.last!.perclos, 1)
    }
}

final class PerceptionEventTests: XCTestCase {
    func testYawn() {
        let ex2 = AttentionFeatureExtractor()
        let f2 = perceptionRun(ex2, to: 25) { t in
            var jaw: Float = 0.03
            if t >= 5 && t < 7 { jaw = 0.8 }          // 2 s sustained: yawn
            if t >= 15 && t < 16 { jaw = 0.9 }        // 1 s: not a yawn
            return face(t, jaw: jaw)
        }
        XCTAssertEqual(f2.last!.yawnCount, 1)
        XCTAssertEqual(f2.first { abs($0.t - 6) < 0.05 }!.yawnActive, 1)
        XCTAssertEqual(f2.first { abs($0.t - 6.4) < 0.05 }!.yawnCount, 0, "not confirmed before 1.5 s")
        XCTAssertEqual(f2.first { abs($0.t - 6.6) < 0.05 }!.yawnCount, 1)
        XCTAssertEqual(f2.last!.yawnActive, 0)
    }

    func testNod() {
        let ex = AttentionFeatureExtractor()
        // Synthetic-generator nod: 0.35 rad half-sine over 1.6 s at t = 10 (counts).
        // Slow head-down at t = 20 for 5 s (looking at desk: doesn't count).
        // Shallow 0.15 rad dip at t = 40 (doesn't count).
        let frames = perceptionRun(ex, to: 50) { t in
            var p: Float = 0
            if t >= 10 && t < 11.6 { p = -0.35 * Float(sin(Double.pi * (t - 10) / 1.6)) }
            if t >= 20 && t < 25 { p = -0.4 }
            if t >= 40 && t < 41 { p = -0.15 * Float(sin(Double.pi * (t - 40))) }
            return face(t, pitch: p)
        }
        XCTAssertEqual(frames.first { abs($0.t - 15) < 0.05 }!.nodCount, 1)
        XCTAssertEqual(frames.last!.nodCount, 1)
    }

    func testNodDetectorDirect() {
        var d = PerceptionNodDetector()
        var events: [Double] = []
        for i in 0..<300 {
            let t = Double(i) / 30
            var p: Float = 0.05
            if t >= 3 && t < 4 { p = 0.05 - 0.3 * Float(sin(Double.pi * (t - 3))) }
            if let e = d.update(pitch: p, t: t) { events.append(e) }
        }
        XCTAssertEqual(events.count, 1)
        XCTAssertGreaterThan(events[0], 3.5); XCTAssertLessThan(events[0], 4.1)
    }

    func testBrowTensionAndHeadMotion() {
        let ex = AttentionFeatureExtractor()
        let frames = perceptionRun(ex, to: 10) { t in face(t, yaw: t < 5 ? 0 : Float(t - 5) * 0.5, brow: t < 5 ? 0.1 : 0.6) }
        let at49 = frames.first { abs($0.t - 4.9) < 0.05 }!
        XCTAssertEqual(at49.browTension, 0.1, accuracy: 1e-4)
        XCTAssertEqual(at49.headMotion, 0, accuracy: 1e-4)
        let at6 = frames.first { abs($0.t - 6.0) < 0.05 }!
        XCTAssertEqual(at6.browTension, 0.35, accuracy: 0.03, "2 s moving average halfway through the step")
        XCTAssertEqual(frames.last!.browTension, 0.6, accuracy: 1e-4)
        XCTAssertEqual(frames.last!.headMotion, 0.5, accuracy: 0.01)
    }

    func testHeartRateAndMotionPassThrough() {
        let ex = AttentionFeatureExtractor()
        ex.ingest(face: face(0))
        _ = ex.frame(at: 0)
        ex.ingest(heartRate: HeartRateSample(t: 0.05, bpm: 80, quality: 0.7, snrDB: 6, motionFlagged: false,
                                             baselineBPM: 66, waveform: []))
        ex.ingest(motion: MotionSample(t: 0.05, pickedUp: true, tilt: 0.6, accelerationMagnitude: 0.2))
        let f = ex.frame(at: 0.1)!
        XCTAssertEqual(f.heartRate, 80); XCTAssertEqual(f.heartRateQuality, 0.7)
        XCTAssertEqual(f.heartRateDelta, 14, accuracy: 1e-5)
        XCTAssertEqual(f.phonePickup, 1)
        XCTAssertEqual(f.motionMagnitude, 0.2, accuracy: 1e-5)
        // Stale HR (> 5 s) -> 0.
        var g: FeatureFrame?
        for k in 2...80 { g = ex.frame(at: Double(k) / 10) ?? g }
        XCTAssertEqual(g!.heartRate, 0); XCTAssertEqual(g!.heartRateDelta, 0)
        // No baseline -> delta 0.
        ex.ingest(heartRate: HeartRateSample(t: 8, bpm: 70, quality: 0.5, snrDB: 3, motionFlagged: false,
                                             baselineBPM: nil, waveform: []))
        let h = ex.frame(at: 8.1)!
        XCTAssertEqual(h.heartRate, 70); XCTAssertEqual(h.heartRateDelta, 0)
    }
}

final class PerceptionWorkZoneTests: XCTestCase {
    func testCalibrationMedianAndMargin() {
        func fix(_ y: Float, _ p: Float) -> [(yaw: Float, pitch: Float)] {
            // 20 samples with a little jitter and two wild outliers (blink / glance).
            var s = (0..<20).map { i in (yaw: y + Float(i % 3 - 1) * 0.005, pitch: p + Float(i % 2) * 0.004) }
            s.append((yaw: y + 1.0, pitch: p - 1.0)); s.append((yaw: .nan, pitch: p))
            return s
        }
        let z = WorkZoneMath.calibrate(center: fix(0.1, -0.05),
                                       corners: [fix(0.4, 0.15), fix(-0.2, 0.15), fix(0.4, -0.25), fix(-0.2, -0.25)],
                                       isPhoneScreen: false)
        XCTAssertEqual(z.centerYaw, 0.1, accuracy: 0.006)
        XCTAssertEqual(z.centerPitch, -0.05, accuracy: 0.006)
        // extent 0.3 yaw, 0.2 pitch -> x1.15 + 0.05
        XCTAssertEqual(z.halfWidthYaw, 0.3 * 1.15 + 0.05, accuracy: 0.01)
        XCTAssertEqual(z.halfWidthPitch, 0.2 * 1.15 + 0.05, accuracy: 0.01)
        XCTAssertFalse(z.isPhoneScreen)
        for c in [(0.4, 0.15), (-0.2, 0.15), (0.4, -0.25), (-0.2, -0.25)] as [(Float, Float)] {
            XCTAssertTrue(WorkZoneMath.isInside(yaw: c.0, pitch: c.1, zone: z))
        }
    }

    func testCalibrationClampsAndFallbacks() {
        let tiny = WorkZoneMath.calibrate(center: [(0, 0)], corners: [[(0.001, 0.001)]], isPhoneScreen: true)
        XCTAssertEqual(tiny.halfWidthYaw, WorkZoneMath.minHalfWidth)
        XCTAssertTrue(tiny.isPhoneScreen)
        let empty = WorkZoneMath.calibrate(center: [], corners: [], isPhoneScreen: false)
        XCTAssertEqual(empty.halfWidthYaw, WorkZone.default.halfWidthYaw)
        XCTAssertEqual(empty.centerYaw, 0)
        let huge = WorkZoneMath.calibrate(center: [(0, 0)], corners: [[(3, 3)]], isPhoneScreen: false)
        XCTAssertEqual(huge.halfWidthPitch, WorkZoneMath.maxHalfWidth)
    }

    func testDeviation() {
        let z = WorkZone(centerYaw: 0, centerPitch: 0, halfWidthYaw: 0.44, halfWidthPitch: 0.31)
        XCTAssertEqual(WorkZoneMath.deviation(yaw: 0.2, pitch: -0.2, zone: z), 0)
        XCTAssertEqual(WorkZoneMath.deviation(yaw: 0.54, pitch: 0, zone: z), 0.1, accuracy: 1e-6)
        XCTAssertEqual(WorkZoneMath.deviation(yaw: -0.74, pitch: -0.71, zone: z), 0.5, accuracy: 1e-6) // 3-4-5
        // Same formula as ml/synth/make_replay.py: combined = gaze + 0.3 * head.
        let c = WorkZoneMath.combinedDirection(gazeYaw: 0.75, gazePitch: -0.25, headYaw: 0.375, headPitch: 0)
        XCTAssertEqual(c.yaw, 0.8625, accuracy: 1e-6); XCTAssertEqual(c.pitch, -0.25, accuracy: 1e-6)
    }

    func testGazeFromLookAtPoint() {
        // lookAtPoint to the face's own left (+x) -> + yaw (user's left); up -> + pitch.
        let g = WorkZoneMath.gazeDirection(lookAtPoint: SIMD3(0.5, 0, 0.5), headYaw: 0.1, headPitch: 0)
        XCTAssertEqual(g.yaw, 0.1 + .pi / 4, accuracy: 1e-5)
        let u = WorkZoneMath.gazeDirection(lookAtPoint: SIMD3(0, 0.5, 0.5), headYaw: 0, headPitch: 0)
        XCTAssertEqual(u.pitch, .pi / 4, accuracy: 1e-5)
        XCTAssertNil(WorkZoneMath.combinedDirection(for: FaceSample(t: 0, faceDetected: false)))
    }

    func testOffZoneDwellAndInWorkZone() {
        let ex = AttentionFeatureExtractor(workZone: .default)
        // Inside 0-5 s, look away 5-8 s, back 8-10 s, face lost 10-11 s.
        let frames = perceptionRun(ex, to: 12) { t in
            if t >= 10 && t < 11 { return face(t, detected: false) }
            let away = t >= 5 && t < 8
            return face(t, yaw: away ? 0.45 : 0, look: away ? SIMD3(0.5 * tan(0.45), 0, 0.5) : SIMD3(0, 0, 0.5))
        }
        let at4 = frames.first { abs($0.t - 4) < 0.05 }!
        XCTAssertEqual(at4.inWorkZone, 1); XCTAssertEqual(at4.offZoneDwell, 0); XCTAssertEqual(at4.gazeDeviation, 0)
        let at7 = frames.first { abs($0.t - 7) < 0.05 }!
        XCTAssertEqual(at7.inWorkZone, 0)
        XCTAssertEqual(at7.offZoneDwell, 2.1, accuracy: 0.05)   // first outside frame counts 0.1 s, as in the generator
        // combined yaw = 0.9 + 0.3 * 0.45 = 1.035 -> deviation 1.035 - 0.44
        XCTAssertEqual(at7.gazeDeviation, 1.035 - 0.44, accuracy: 1e-3)
        XCTAssertEqual(at7.gazeYaw, 0.9, accuracy: 1e-4)
        let at9 = frames.first { abs($0.t - 9) < 0.05 }!
        XCTAssertEqual(at9.offZoneDwell, 0); XCTAssertEqual(at9.inWorkZone, 1)
        // Face lost: stale after 0.5 s -> outside zone, dwell accumulates.
        let at108 = frames.last { $0.t < 10.85 }!
        XCTAssertEqual(at108.faceDetected, 0); XCTAssertEqual(at108.inWorkZone, 0)
        XCTAssertGreaterThan(at108.offZoneDwell, 0.7)
        XCTAssertEqual(frames.last!.inWorkZone, 1)
    }

    func testRecalibratedZoneIsUsed() {
        let ex = AttentionFeatureExtractor()
        ex.workZone = WorkZone(centerYaw: 0.9, centerPitch: 0, halfWidthYaw: 0.1, halfWidthPitch: 0.1)
        let f = perceptionRun(ex, to: 1) { t in face(t) }
        XCTAssertEqual(f.last!.inWorkZone, 0)
        XCTAssertEqual(f.last!.gazeDeviation, 0.8, accuracy: 1e-5)
    }
}

final class PerceptionPickupTests: XCTestCase {
    private let flat = SIMD3<Float>(0, 0, -1)
    private func tilted(_ deg: Float) -> SIMD3<Float> {
        let r = deg * .pi / 180
        return SIMD3(0, -sin(r), -cos(r))
    }

    func testNoPickupBeforeRestLearned() {
        var d = PerceptionPickupDetector()
        let s = d.update(t: 0, gravity: tilted(60), userAcceleration: .zero)
        XCTAssertFalse(s.pickedUp); XCTAssertNil(d.restGravity)
    }

    func testTiltPickupWithHysteresis() {
        var d = PerceptionPickupDetector()
        var t = 0.0
        func step(_ g: SIMD3<Float>, _ a: SIMD3<Float> = .zero, seconds: Double) -> MotionSample {
            var last: MotionSample!
            let n = Int(seconds * 30)
            for _ in 0..<n { last = d.update(t: t, gravity: g, userAcceleration: a); t += 1.0 / 30 }
            return last
        }
        _ = step(flat, seconds: 2.5)
        XCTAssertNotNil(d.restGravity)
        XCTAssertFalse(step(tilted(20), SIMD3(0.05, 0, 0), seconds: 0.5).pickedUp, "20 deg is below pickup threshold")
        let up = step(tilted(30), SIMD3(0.05, 0, 0), seconds: 0.2)
        XCTAssertTrue(up.pickedUp)
        XCTAssertEqual(up.tilt, 30 * .pi / 180, accuracy: 0.01)
        // Back to 20 deg: inside the hysteresis band -> still picked up.
        XCTAssertTrue(step(tilted(20), SIMD3(0.05, 0, 0), seconds: 2).pickedUp)
        // Below release tilt but not yet held for 1 s.
        XCTAssertTrue(step(tilted(10), SIMD3(0.04, 0, 0), seconds: 0.5).pickedUp)
        XCTAssertFalse(step(tilted(10), SIMD3(0.04, 0, 0), seconds: 0.7).pickedUp)
    }

    func testAccelerationSpikeAndRestRelearn() {
        var d = PerceptionPickupDetector()
        var t = 0.0
        func step(_ g: SIMD3<Float>, _ a: SIMD3<Float> = .zero, seconds: Double) -> MotionSample {
            var last: MotionSample!
            for _ in 0..<Int(seconds * 30) { last = d.update(t: t, gravity: g, userAcceleration: a); t += 1.0 / 30 }
            return last
        }
        _ = step(flat, seconds: 2.5)
        XCTAssertTrue(step(flat, SIMD3(0.5, 0, 0), seconds: 0.1).pickedUp, "spike counts as pickup")
        // Phone set down at a new 40 deg pose and left perfectly still: re-learned rest -> released.
        XCTAssertTrue(step(tilted(40), seconds: 3).pickedUp)
        let s = step(tilted(40), seconds: 3)
        XCTAssertFalse(s.pickedUp)
        XCTAssertEqual(s.tilt, 0, accuracy: 0.01)
    }
}

final class PerceptionCadenceTests: XCTestCase {
    func testTenHzCadence() {
        let ex = AttentionFeatureExtractor()
        var frames: [FeatureFrame] = []
        // 60 Hz camera with jitter; frames requested on every sample.
        for i in 0..<600 {
            let t = 5.0 + Double(i) / 60 + (i % 7 == 3 ? 0.004 : 0)
            ex.ingest(face: face(t))
            if let f = ex.frame(at: t) { frames.append(f) }
        }
        XCTAssertEqual(frames.count, 100)
        XCTAssertEqual(frames.first!.t, 5.0, accuracy: 1e-9)
        for (a, b) in zip(frames, frames.dropFirst()) { XCTAssertEqual(b.t - a.t, 0.1, accuracy: 1e-6) }
        XCTAssertNil(ex.frame(at: frames.last!.t + 0.05))
    }

    func testCadenceSkipsMissedTicksWithoutBackfill() {
        let ex = AttentionFeatureExtractor()
        XCTAssertNotNil(ex.frame(at: 0))
        let f = ex.frame(at: 0.73)
        XCTAssertEqual(f!.t, 0.7, accuracy: 1e-9)
        XCTAssertNil(ex.frame(at: 0.75))
        XCTAssertEqual(ex.frame(at: 0.8)!.t, 0.8, accuracy: 1e-9)
        ex.reset()
        XCTAssertEqual(ex.frame(at: 3.33)!.t, 3.33, accuracy: 1e-9)
    }

    func testFrameVectorShape() {
        let ex = AttentionFeatureExtractor()
        ex.ingest(face: face(0))
        XCTAssertEqual(ex.frame(at: 0)!.vector.count, FeatureFrame.featureNames.count)
    }
}

final class PerceptionSourceTests: XCTestCase {
    func testMockFaceSourceScriptIsDeterministicAndSynthetic() {
        let m = MockFaceSource(script: [.init(.focused(blinkEvery: 2), duration: 4), .init(.noFace, duration: 1)])
        XCTAssertTrue(m.isSynthetic); XCTAssertTrue(m.isAvailable); XCTAssertEqual(MockFaceSource.label, "SYNTHETIC")
        let a = m.render(to: 5), b = m.render(to: 5)
        XCTAssertEqual(a.count, 150)
        XCTAssertEqual(a.map(\.faceDetected), b.map(\.faceDetected))
        XCTAssertFalse(a.last!.faceDetected)
        let ex = AttentionFeatureExtractor()
        var frames: [FeatureFrame] = []
        for s in a { ex.ingest(face: s); if let f = ex.frame(at: s.t) { frames.append(f) } }
        // Blink at 1.8-2.0 completes; the one at 3.8 is cut off by face loss at 4.0.
        XCTAssertEqual(frames.first { abs($0.t - 3.9) < 0.05 }!.blinkRate, 1 / Float(4.0) * 60, accuracy: 0.01)
    }

    func testMockDemoDrivesAllEvents() {
        let m = MockFaceSource.demo()
        let ex = AttentionFeatureExtractor()
        var last: FeatureFrame?
        var sawOffZone = false
        for s in m.render(to: 62.6) {
            ex.ingest(face: s)
            if let f = ex.frame(at: s.t) { last = f; if f.offZoneDwell > 3 { sawOffZone = true } }
        }
        XCTAssertTrue(sawOffZone)
        XCTAssertEqual(last!.yawnCount, 1)
        XCTAssertEqual(last!.nodCount, 1)
        XCTAssertGreaterThan(last!.longBlinkCount, 2)
    }

    func testMockFaceSourceStreamsInRealTime() async throws {
        let m = MockFaceSource(rateHz: 60) { t in FaceSample(t: t, faceDetected: true) }
        try m.start()
        var n = 0
        for await _ in m.samples { n += 1; if n >= 5 { break } }
        m.stop()
        XCTAssertEqual(n, 5)
    }

    func testARKitSourceUnavailableInSimulator() {
        #if targetEnvironment(simulator)
        let s = ARKitFaceSource()
        XCTAssertFalse(s.isAvailable)
        XCTAssertThrowsError(try s.start())
        #endif
    }

    func testCoreMotionSimulatorMockNeverPicksUp() async {
        #if targetEnvironment(simulator)
        let s = CoreMotionSource()
        s.start()
        var n = 0
        for await m in s.samples { XCTAssertFalse(m.pickedUp); n += 1; if n >= 3 { break } }
        s.stop()
        XCTAssertFalse(s.isHardware)
        #endif
    }

    func testHeadPoseMath() {
        XCTAssertEqual(PerceptionPoseMath.headPose(faceInView: matrix_identity_float4x4).yaw, 0, accuracy: 1e-6)
        // Turn the head toward the user's left (+x) by 0.3 rad about +y.
        let yawM = simd_float4x4(simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0)))
        XCTAssertEqual(PerceptionPoseMath.headPose(faceInView: yawM).yaw, 0.3, accuracy: 1e-5)
        // Tilt head up: forward rotates toward +y (rotation about -x).
        let pitchM = simd_float4x4(simd_quatf(angle: 0.2, axis: SIMD3(-1, 0, 0)))
        XCTAssertEqual(PerceptionPoseMath.headPose(faceInView: pitchM).pitch, 0.2, accuracy: 1e-5)
        let rollM = simd_float4x4(simd_quatf(angle: 0.25, axis: SIMD3(0, 0, 1)))
        XCTAssertEqual(PerceptionPoseMath.headPose(faceInView: rollM).roll, 0.25, accuracy: 1e-5)
    }
}
