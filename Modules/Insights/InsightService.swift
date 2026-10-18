import Foundation

/// Entry point for the app: uses the Foundation Models coach when the on-device model is available,
/// otherwise the template coach. Results for a session are cached by session ID.
///
/// Typical use (session report):
/// ```swift
/// let insights = await insightService.insights(for: record, armEffects: policy.effectEstimates())
/// ```
/// Weekly profile: `await insightService.insights(for: profileData.summary)`.
public final class InsightService: InsightGenerator {

    private let foundationModels: InsightGenerator?
    private let template: TemplateInsightGenerator
    private let calendar: Calendar
    private let lock = NSLock()
    private var cache: [UUID: [Insight]] = [:]

    /// - Parameter preferFoundationModels: set false to always use the deterministic template coach.
    public init(preferFoundationModels: Bool = true, timeout: TimeInterval = 15, calendar: Calendar = .current) {
        let template = TemplateInsightGenerator()
        self.template = template
        self.foundationModels = preferFoundationModels
            ? FoundationModelsInsightGenerator(timeout: timeout, fallback: template) : nil
        self.calendar = calendar
    }

    /// Test seam: inject the model-backed generator.
    init(foundationModels: InsightGenerator?, calendar: Calendar = .current) {
        self.template = TemplateInsightGenerator()
        self.foundationModels = foundationModels
        self.calendar = calendar
    }

    /// True when insights will come from Apple's on-device Foundation Models (checked live).
    public var usesFoundationModels: Bool { foundationModels?.usesFoundationModels ?? false }

    public func insights(for summary: AggregateSummary) async -> [Insight] {
        if let fm = foundationModels, fm.usesFoundationModels {
            let out = await fm.insights(for: summary)
            if (2...4).contains(out.count), InsightGuardrails.isClean(out.map(\.text)) { return out }
        }
        return await template.insights(for: summary)
    }

    /// Insights for one session's report, cached by `session.id`. If the record already stores
    /// insights (persisted earlier) and `useStored` is true, those are returned as-is.
    public func insights(for session: SessionRecord, armEffects: [ArmEffect] = [],
                         useStored: Bool = true) async -> [Insight] {
        if useStored, !session.insights.isEmpty { return store(session.insights, for: session.id) }
        if let hit = cachedInsights(for: session.id) { return hit }
        let summary = AttentionProfileBuilder.summary(for: session, armEffects: armEffects, calendar: calendar)
        return store(await insights(for: summary), for: session.id)
    }

    public func cachedInsights(for sessionID: UUID) -> [Insight]? {
        lock.lock(); defer { lock.unlock() }
        return cache[sessionID]
    }

    public func invalidate(sessionID: UUID) {
        lock.lock(); defer { lock.unlock() }
        cache[sessionID] = nil
    }

    public func clearCache() {
        lock.lock(); defer { lock.unlock() }
        cache.removeAll()
    }

    private func store(_ value: [Insight], for id: UUID) -> [Insight] {
        lock.lock(); defer { lock.unlock() }
        if let existing = cache[id] { return existing }   // first writer wins under concurrent calls
        cache[id] = value
        return value
    }
}
