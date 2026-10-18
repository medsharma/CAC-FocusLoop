import Foundation

/// Loads `<lecture>.brain.json` (written offline by `ml/brain/precompute_content.py`) and
/// answers "what does the model predict the brain is doing at playhead time t?".
///
/// JSON schema (1 Hz, already shifted for hemodynamic lag, z-scored within the lecture):
/// {
///   "lecture_id": "intro-bio-01", "hz": 1, "mock": false,
///   "networks": { "Vis": [..], "SomMot": [..], "DorsAttn": [..], "SalVentAttn": [..],
///                 "Limbic": [..], "Cont": [..], "Default": [..] },
///   "predicted_engagement": [..],   // 0...1
///   "predicted_drift_risk": [..],   // z-score, higher = content invites mind-wandering
///   "valleys": [ { "start": 312, "end": 341 } ]
/// }
public struct BrainTimeline: Decodable {
    public struct Valley: Decodable { public let start: Double; public let end: Double }

    public let lectureID: String
    public let hz: Double
    /// True when produced by the MOCK brain model. Show a "MOCK" badge in the UI when set.
    public let mock: Bool?
    public let networks: [String: [Float]]
    public let predictedEngagement: [Float]
    public let predictedDriftRisk: [Float]
    public let valleys: [Valley]?

    enum CodingKeys: String, CodingKey {
        case lectureID = "lecture_id", hz, mock, networks, valleys
        case predictedEngagement = "predicted_engagement"
        case predictedDriftRisk = "predicted_drift_risk"
    }

    public static func load(_ name: String, bundle: Bundle = .main) throws -> BrainTimeline {
        guard let url = bundle.url(forResource: name, withExtension: "brain.json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode(BrainTimeline.self, from: Data(contentsOf: url))
    }

    public var duration: Double {
        Double(networks.values.map(\.count).max() ?? 0) / hz
    }

    /// Display activations (0...1) for `BrainCortexView`. Only positive deviations light up:
    /// z = 0 (lecture average) is rest, z >= 2.5 is fully active.
    public func activations(at t: Double) -> [Float] {
        BrainNetwork.allCases.map { n in
            let z = Self.sample(networks[n.key] ?? [], at: t * hz)
            return max(0, min(1, z / 2.5))
        }
    }

    public func engagement(at t: Double) -> Float { Self.sample(predictedEngagement, at: t * hz) }
    public func driftRisk(at t: Double) -> Float { Self.sample(predictedDriftRisk, at: t * hz) }

    /// Next predicted low-engagement segment starting within `horizon` seconds.
    public func upcomingValley(after t: Double, within horizon: Double = 30) -> Valley? {
        (valleys ?? []).first { $0.start >= t && $0.start - t <= horizon }
    }

    private static func sample(_ s: [Float], at x: Double) -> Float {
        guard !s.isEmpty else { return 0 }
        let c = max(0, min(Double(s.count - 1), x))
        let i = Int(c), j = min(i + 1, s.count - 1), f = Float(c - Double(i))
        return s[i] * (1 - f) + s[j] * f
    }
}
