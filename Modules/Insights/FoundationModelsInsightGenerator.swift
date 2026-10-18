import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

/// On-device AI coach using Apple's Foundation Models framework (iOS 26+, Apple Intelligence devices).
///
/// The prompt contains ONLY the aggregated fact list from `InsightFacts.factList(for:)`. The output is
/// post-filtered by `InsightGuardrails` (banned phrases, invented numbers, 2-4 items) and any failure,
/// error, unavailability or timeout falls back to `TemplateInsightGenerator`.
public final class FoundationModelsInsightGenerator: InsightGenerator {

    /// Produces raw insight strings from (instructions, prompt). Injected in tests; the default is the
    /// system language model.
    public typealias Producer = @Sendable (_ instructions: String, _ prompt: String) async throws -> [String]

    public enum FallbackReason: Equatable, Sendable {
        case unavailable
        case timedOut
        case error(String)
        case guardrail([String])
        case unsupportedNumbers([Double])
        case badCount(Int)
        case unsupportedClaim(String)
    }

    public let timeout: TimeInterval
    private let fallback: TemplateInsightGenerator
    private let producer: Producer?

    public init(timeout: TimeInterval = 15, fallback: TemplateInsightGenerator = TemplateInsightGenerator()) {
        self.timeout = timeout
        self.fallback = fallback
        self.producer = nil
    }

    /// Test seam: use `producer` instead of the system model (treated as available).
    init(timeout: TimeInterval, fallback: TemplateInsightGenerator = TemplateInsightGenerator(), producer: @escaping Producer) {
        self.timeout = timeout
        self.fallback = fallback
        self.producer = producer
    }

    /// True when the system model can run now (iOS 26+, eligible device, Apple Intelligence on, model ready).
    public static var isSystemModelAvailable: Bool {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, *) {
            if case .available = SystemLanguageModel.default.availability { return true }
        }
        #endif
        return false
    }

    public var usesFoundationModels: Bool { producer != nil || Self.isSystemModelAvailable }

    public func insights(for summary: AggregateSummary) async -> [Insight] {
        await generate(for: summary).insights
    }

    /// Insights plus the reason for falling back to the template (nil = model output was used).
    public func generate(for summary: AggregateSummary) async -> (insights: [Insight], fallbackReason: FallbackReason?) {
        func templated(_ reason: FallbackReason) -> ([Insight], FallbackReason?) {
            (fallback.texts(for: summary).map { Insight(text: $0, source: .template) }, reason)
        }
        guard let run = activeProducer() else { return templated(.unavailable) }
        let facts = InsightFacts.factList(for: summary)
        let prompt = Self.prompt(facts: facts.text)
        let instructions = Self.instructions

        let raw: [String]
        do {
            raw = try await InsightsTimeout.run(seconds: timeout) { try await run(instructions, prompt) }
        } catch is InsightsTimeout.TimedOut {
            return templated(.timedOut)
        } catch {
            return templated(.error(String(describing: error)))
        }

        let cleaned = Self.dedupeSentences(raw.map { Self.firstSentences(Self.normalise($0), max: 2) })
        guard (2...6).contains(cleaned.count) else { return templated(.badCount(cleaned.count)) }
        let texts = Array(cleaned.prefix(4)).map { text -> String in
            let aboutContent = Self.mentionsContent(text)
            return InsightFacts.qualified(text, mock: summary.contentIsMock && aboutContent, replay: summary.isReplay)
        }
        let violations = texts.flatMap(InsightGuardrails.violations(in:))
        guard violations.isEmpty else { return templated(.guardrail(violations)) }
        let invented = texts.flatMap { InsightGuardrails.unsupportedNumbers(in: $0, allowed: facts.numbers) }
        guard invented.isEmpty else { return templated(.unsupportedNumbers(invented)) }
        if let claim = texts.first(where: { Self.hasUnsupportedStateClaim($0, summary: summary) }) {
            return templated(.unsupportedClaim(claim))
        }
        return (texts.map { Insight(text: $0, source: .foundationModels) }, nil)
    }

    /// The fact list never links a specific state to a time, and only mentions states by their share
    /// of time. Reject sentences that tie a non-focused state to a minute/hour, or that name a state the
    /// student spent under 5% of the time in (orchestrator guardrail added at integration).
    static func hasUnsupportedStateClaim(_ text: String, summary: AggregateSummary) -> Bool {
        let lower = text.lowercased()
        // A nudge may only be recommended when the trial says it helps (verdict "helps").
        let recommends = ["try using", "consider using", "might want to try", "might want to consider", "nudge like"]
        if recommends.contains(where: lower.contains) {
            let helping = InsightFacts.reportableEffects(summary).filter { InsightArmVerdict.of($0) == .helps }
            if !helping.contains(where: { lower.contains($0.arm.displayName.lowercased()) }) { return true }
        }
        if lower.contains("minute 0") { return true }
        let words: [(AttentionState, [String])] = [(.distracted, ["distract"]), (.drowsy, ["drows", "tired", "sleepy"]),
                                                   (.stressed, ["stress", "tense"])]
        let mentionsTime = lower.contains("minute") || lower.contains(" hour") || lower.contains(" am") || lower.contains(" pm")
        for (state, stems) in words where stems.contains(where: lower.contains) {
            if mentionsTime { return true }
            let frac = state.index < summary.stateFractions.count ? summary.stateFractions[state.index] : 0
            if frac < 0.05 { return true }
        }
        return false
    }

    // MARK: - Prompt

    static let bannedWordsForPrompt = [
        "diagnose", "diagnosis", "ADHD", "disorder", "treat", "treatment", "cure", "heal", "clinical",
        "clinically proven", "proven", "first", "reads your brain", "symptom", "disease", "therapy",
        "medication", "doctor", "anxiety", "depression", "patient", "healthy", "good sign", "your brain",
    ]

    static var instructions: String {
        """
        You are a friendly study coach inside a wellness app that supports focus. \
        Write 2 to 4 short insights for a student, each one or two sentences, in second person. \
        Use ONLY the facts provided. Do not invent numbers, causes or events. Copy numbers exactly as given. \
        Topics, when facts exist: when focus drops, which nudges work, best hours, heart-rate trend, and how drift \
        lined up with the predicted brain response to the lecture (average brain). \
        Only say a nudge helps if its verdict is "helps"; if the verdict is "still learning", say the app is still learning. \
        This is wellness, not medicine: give no medical advice, no diagnosis, and no health judgements about heart rate; \
        describe heart rate neutrally (for example "steady", "trending down"), never as good or bad. \
        "Still learning" refers to the app's nudge trial, not to the student. \
        Never use these words: \(bannedWordsForPrompt.joined(separator: ", ")).\(InsightFacts.inlineDataLabels
            ? " If the facts say SYNTHETIC or MOCK, keep that label in the sentences it applies to." : "")
        """
    }

    static func prompt(facts: String) -> String {
        "Facts (aggregated numbers only):\n\(facts)\n\nWrite the insights now."
    }

    // MARK: - Post-processing

    /// Trims whitespace and leading list markers ("- ", "* ", "1. ", "2) ").
    static func normalise(_ s: String) -> String {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let r = t.range(of: #"^(?:[-*•]|\d{1,2}[.)])\s+"#, options: .regularExpression) else { return t }
        return String(t[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Keeps at most `max` sentences (a terminator followed by whitespace or end of text ends a sentence).
    static func firstSentences(_ text: String, max: Int) -> String {
        var count = 0
        var idx = text.startIndex
        while idx < text.endIndex {
            let next = text.index(after: idx)
            if ".!?".contains(text[idx]), next == text.endIndex || text[next].isWhitespace {
                count += 1
                if count == max { return String(text[..<next]) }
            }
            idx = next
        }
        return text
    }

    static func sentences(_ text: String) -> [String] {
        var out: [String] = []
        var start = text.startIndex
        var idx = text.startIndex
        while idx < text.endIndex {
            let next = text.index(after: idx)
            if ".!?".contains(text[idx]), next == text.endIndex || text[next].isWhitespace {
                out.append(text[start..<next].trimmingCharacters(in: .whitespaces))
                start = next
            }
            idx = next
        }
        let tail = text[start...].trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty { out.append(tail) }
        return out
    }

    /// Removes sentences already used by an earlier item (small models tend to repeat themselves);
    /// drops items that end up empty.
    static func dedupeSentences(_ items: [String]) -> [String] {
        var seen = Set<String>()
        return items.compactMap { item in
            let kept = sentences(item).filter { seen.insert($0.lowercased()).inserted }
            let joined = kept.joined(separator: " ")
            return joined.isEmpty ? nil : joined
        }
    }

    static func mentionsContent(_ text: String) -> Bool {
        let l = text.lowercased()
        return ["lecture", "content", "valley", "predicted", "brain"].contains { l.contains($0) }
    }

    private func activeProducer() -> Producer? {
        if let producer { return producer }
        guard Self.isSystemModelAvailable else { return nil }
        #if canImport(FoundationModels)
        if #available(iOS 26.0, *) {
            return { instructions, prompt in
                try await InsightsFMBackend.generate(instructions: instructions, prompt: prompt)
            }
        }
        #endif
        return nil
    }
}

#if canImport(FoundationModels)
/// Guided-generation schema for the coach.
@available(iOS 26.0, *)
@Generable
struct InsightsFMOutput {
    @Guide(description: "Two to four short coaching insights, each one or two sentences, using only the given facts.",
           .count(2...4))
    var insights: [String]
}

@available(iOS 26.0, *)
enum InsightsFMBackend {
    static func generate(instructions: String, prompt: String) async throws -> [String] {
        let session = LanguageModelSession(instructions: instructions)
        let options = GenerationOptions(sampling: .greedy, maximumResponseTokens: 400)
        let response = try await session.respond(to: prompt, generating: InsightsFMOutput.self, options: options)
        return response.content.insights
    }
}
#endif

/// Races an async operation against a deadline without waiting for a non-cooperative operation to finish.
enum InsightsTimeout {
    struct TimedOut: Error {}

    private final class Gate: @unchecked Sendable {
        private let lock = NSLock()
        private var done = false
        func claim() -> Bool {
            lock.lock(); defer { lock.unlock() }
            if done { return false }
            done = true
            return true
        }
    }

    static func run(seconds: TimeInterval, _ op: @escaping @Sendable () async throws -> [String]) async throws -> [String] {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<[String], Error>) in
            let gate = Gate()
            let work = Task.detached {
                do {
                    let value = try await op()
                    if gate.claim() { cont.resume(returning: value) }
                } catch {
                    if gate.claim() { cont.resume(throwing: error) }
                }
            }
            // The deadline runs on GCD, not the Swift cooperative pool, so it still fires if the
            // model call blocks cooperative threads.
            DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + max(0, seconds)) {
                if gate.claim() {
                    work.cancel()
                    cont.resume(throwing: TimedOut())
                }
            }
        }
    }
}
