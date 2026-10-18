import XCTest
@testable import FocusLoop

final class BrainContentTests: XCTestCase {

    // MARK: Bundled content

    private func bundledBrainJSONs() -> [URL] {
        (Bundle.main.urls(forResourcesWithExtension: "json", subdirectory: nil) ?? [])
            .filter { $0.lastPathComponent.hasSuffix(".brain.json") }
    }

    func testEveryBundledBrainJSONDecodes() throws {
        let urls = bundledBrainJSONs()
        XCTAssertGreaterThanOrEqual(urls.count, 4, "3 lectures + sample_lecture")
        for url in urls {
            let tl = try JSONDecoder().decode(BrainTimeline.self, from: Data(contentsOf: url))
            XCTAssertEqual(Set(tl.networks.keys), Set(BrainNetwork.allCases.map(\.key)), url.lastPathComponent)
            XCTAssertEqual(tl.hz, 1)
            XCTAssertEqual(tl.mock, true, "MOCK model output must be flagged: \(url.lastPathComponent)")
            let n = tl.predictedEngagement.count
            XCTAssertGreaterThan(n, 0)
            XCTAssertTrue(tl.networks.values.allSatisfy { $0.count == n })
            XCTAssertEqual(tl.predictedDriftRisk.count, n)
            XCTAssertTrue(tl.predictedEngagement.allSatisfy { (0...1).contains($0) })
            for v in tl.valleys ?? [] { XCTAssertGreaterThanOrEqual(v.end - v.start, 6) }
            XCTAssertEqual(tl.activations(at: 3).count, 7)
        }
    }

    func testCatalogLoadsWithTimelinesAndVideos() throws {
        let lib = LectureLibrary()
        XCTAssertGreaterThanOrEqual(lib.lectures.count, 4)
        let sample = try XCTUnwrap(lib.lecture(id: "sample_lecture"))
        XCTAssertNil(sample.videoFile)
        XCTAssertNil(lib.videoURL(for: "sample_lecture"))
        for lec in lib.lectures {
            let tl = try XCTUnwrap(lib.timeline(for: lec.id), lec.id)
            XCTAssertTrue(tl.isMock)
            XCTAssertEqual(tl.lectureID, lec.id)
            XCTAssertFalse(lec.recaps.isEmpty)
            XCTAssertFalse(lec.recallQuestions.isEmpty)
            if lec.videoFile != nil {
                let url = try XCTUnwrap(lib.videoURL(for: lec.id), lec.id)
                XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
                XCTAssertEqual(tl.duration, floor(lec.durationSeconds), accuracy: 1)
            }
            let f = tl.features(at: 30)
            XCTAssertEqual(f.vector.count, ContentFeatures.featureNames.count)
        }
        XCTAssertTrue(lib.timeline(for: "black-holes-01") === lib.timeline(for: "black-holes-01"), "cached")
    }

    func testVertexPlaybackFile() throws {
        let lib = LectureLibrary()
        let withVerts = lib.lectures.filter { lib.hasVertexPlayback(for: $0.id) }
        XCTAssertFalse(withVerts.isEmpty, "at least one lecture ships a .vertices.f16")
        for lec in withVerts {
            let frames = try XCTUnwrap(lib.vertexFrames(for: lec.id))
            let tl = try XCTUnwrap(lib.timeline(for: lec.id))
            XCTAssertEqual(frames.count, tl.timeline.predictedEngagement.count)
            XCTAssertTrue(frames.allSatisfy { $0.count == BrainMesh.expectedVertexCount })
            let sample = frames[frames.count / 2]
            XCTAssertTrue(sample.allSatisfy { $0 >= 0 && $0 <= 1 })
            XCTAssertGreaterThan(sample.max() ?? 0, 0)
        }
        XCTAssertNil(lib.vertexFrames(for: "sample_lecture"))
    }

    func testHalfFloatDecoding() {
        let values: [(UInt16, Float)] = [(0x0000, 0), (0x3C00, 1), (0x3800, 0.5), (0xC000, -2), (0x7BFF, 65504),
                                         (0x0001, 5.960464477539063e-08), (0x3555, 0.33325195)]
        for (bits, f) in values { XCTAssertEqual(LectureLibrary.halfToFloat(bits), f, accuracy: 1e-6) }
        XCTAssertTrue(LectureLibrary.halfToFloat(0x7C00).isInfinite)
        var d = Data()
        for b: UInt16 in [0x3C00, 0x3800, 0x0000, 0x3400] { withUnsafeBytes(of: b.littleEndian) { d.append(contentsOf: $0) } }
        XCTAssertEqual(LectureLibrary.decodeVertexFrames(d, vertexCount: 2), [[1, 0.5], [0, 0.25]])
        XCTAssertNil(LectureLibrary.decodeVertexFrames(d, vertexCount: 3))
    }

    func testRecapAndRecallLookup() throws {
        let lib = LectureLibrary()
        let lec = try XCTUnwrap(lib.lecture(id: "black-holes-01"))
        let first = try XCTUnwrap(lib.recap(for: lec.id, at: 1))
        XCTAssertEqual(first, lec.recaps.first)
        XCTAssertNil(lib.recallQuestion(for: lec.id, at: 1))
        let q = try XCTUnwrap(lib.recallQuestion(for: lec.id, at: 120))
        XCTAssertLessThanOrEqual(q.question.after, 120)
        let next = lib.recallQuestion(for: lec.id, at: 120, excluding: [q.index])
        XCTAssertNotEqual(next?.index, q.index)
    }

    // MARK: Nudge priors

    func testBundledNudgePriors() {
        let r = NudgePriorsLoader.load()
        XCTAssertTrue(r.loadedFromFile)
        XCTAssertTrue(r.isMock)
        XCTAssertEqual(r.priors.map(\.arm), NudgeArm.allCases)
        for p in r.priors {
            XCTAssertGreaterThan(p.alpha, 0)
            XCTAssertGreaterThan(p.beta, 0)
            XCTAssertLessThanOrEqual(p.alpha + p.beta, 6)
        }
        XCTAssertEqual(r.prior(for: .noNudge), NudgePrior(arm: .noNudge, alpha: 1, beta: 1))
        XCTAssertEqual(r.prior(for: .lightPulse).alpha, r.prior(for: .lightPulse).beta, "experimental arm is neutral")
    }

    func testNudgePriorsDecodingAndFallback() throws {
        let bare = #"[{"arm": "chime", "alpha": 3, "beta": 2}, {"arm": "bogus", "alpha": 1, "beta": 1}, {"arm": "haptic", "alpha": -1, "beta": 2}]"#
        let r = try NudgePriorsLoader.decode(Data(bare.utf8))
        XCTAssertFalse(r.isMock)
        XCTAssertEqual(r.priors.count, NudgeArm.allCases.count)
        XCTAssertEqual(r.prior(for: .chime), NudgePrior(arm: .chime, alpha: 3, beta: 2))
        XCTAssertEqual(r.prior(for: .haptic), NudgePrior(arm: .haptic, alpha: 1, beta: 1))
        XCTAssertThrowsError(try NudgePriorsLoader.decode(Data("nope".utf8)))
        let missing = NudgePriorsLoader.load(bundle: .main, name: "does_not_exist")
        XCTAssertFalse(missing.loadedFromFile)
        XCTAssertEqual(missing.priors, NudgePriorsLoader.fallback)
    }

    // MARK: Valley tracking

    func testValleyQueries() {
        let tr = ValleyTracker(valleys: [(start: 40, end: 50), (start: 10, end: 20)], approachHorizon: 5)
        XCTAssertEqual(tr.valleys.map(\.start), [10, 40], "sorted")
        XCTAssertEqual(tr.timeToNextValley(at: 0), 10)
        XCTAssertEqual(tr.timeToNextValley(at: 15), 0)
        XCTAssertEqual(tr.timeToNextValley(at: 20), 20, "end is exclusive")
        XCTAssertNil(tr.timeToNextValley(at: 55))
        XCTAssertTrue(tr.isValleyApproaching(at: 36, within: 5))
        XCTAssertFalse(tr.isValleyApproaching(at: 30, within: 5))
        XCTAssertFalse(tr.isValleyApproaching(at: 45, within: 100), "inside a valley")
        XCTAssertEqual(tr.valley(at: 12), ValleyTracker.Valley(start: 10, end: 20))
    }

    func testValleyEventsWhilePlaying() {
        let tr = ValleyTracker(valleys: [(start: 10, end: 20), (start: 40, end: 50)], approachHorizon: 5)
        var log: [(Double, ValleyTracker.Event)] = []
        for i in 0...120 {
            let t = Double(i) * 0.5
            for e in tr.update(playhead: t) { log.append((t, e)) }
        }
        let v0 = ValleyTracker.Valley(start: 10, end: 20), v1 = ValleyTracker.Valley(start: 40, end: 50)
        XCTAssertEqual(log.map(\.1), [.approaching(v0, secondsUntil: 5), .entered(v0), .left(v0),
                                      .approaching(v1, secondsUntil: 5), .entered(v1), .left(v1)])
        XCTAssertEqual(log.map(\.0), [5, 10, 20, 35, 40, 50])
    }

    func testValleyEventsOnSeek() {
        let tr = ValleyTracker(valleys: [(start: 10, end: 20), (start: 40, end: 50)], approachHorizon: 5)
        XCTAssertEqual(tr.update(playhead: 0), [])
        let v0 = ValleyTracker.Valley(start: 10, end: 20), v1 = ValleyTracker.Valley(start: 40, end: 50)
        XCTAssertEqual(tr.update(playhead: 12), [.entered(v0)], "jumping straight in: no approach event")
        XCTAssertEqual(tr.update(playhead: 37), [.left(v0), .approaching(v1, secondsUntil: 3)])
        XCTAssertEqual(tr.update(playhead: 38), [])
        XCTAssertEqual(tr.update(playhead: 7), [.approaching(v0, secondsUntil: 3)], "seek back re-arms")
        tr.reset()
        XCTAssertEqual(tr.update(playhead: 45), [.entered(v1)])
    }

    func testTrackerMatchesBundledTimeline() throws {
        let lib = LectureLibrary()
        let tl = try XCTUnwrap(lib.timeline(for: "black-holes-01"))
        let tracker = try XCTUnwrap(lib.valleyTracker(for: "black-holes-01", approachHorizon: 10))
        XCTAssertFalse(tracker.valleys.isEmpty)
        XCTAssertEqual(tracker.valleys.map(\.start), tl.valleys().map(\.start))
        var entered = 0, approached = 0
        for s in 0..<Int(tl.duration) {
            let t = Double(s)
            for e in tracker.update(playhead: t) {
                switch e {
                case .entered: entered += 1
                case .approaching(let v, let until):
                    approached += 1
                    XCTAssertLessThanOrEqual(until, 10)
                    // The tracker announces the next valley even while inside the current one.
                    XCTAssertEqual(tl.valleys().first { $0.start > t }?.start, v.start)
                case .left: break
                }
            }
            XCTAssertEqual(tl.features(at: t).inValley, tracker.isInValley(at: t))
        }
        XCTAssertEqual(entered, tracker.valleys.count)
        XCTAssertGreaterThan(approached, 0)
    }
}
