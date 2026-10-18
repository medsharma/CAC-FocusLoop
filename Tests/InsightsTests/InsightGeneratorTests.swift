import XCTest
@testable import FocusLoop

final class TemplateInsightGeneratorTests: XCTestCase {
    // These tests check the inline data-label path, which the app turns off (the report shows a footnote instead).
    override func setUp() { super.setUp(); InsightFacts.inlineDataLabels = true }
    override func tearDown() { InsightFacts.inlineDataLabels = false; super.tearDown() }

    typealias F = InsightsFixtures
    let gen = TemplateInsightGenerator()

    private var variedSummaries: [AggregateSummary] {
        [
            F.summary(),
            F.summary(effects: F.hapticHelps),
            F.summary(sessions: 1, minutes: 25, focused: 0.55, drop: nil, refocus: nil, bestHours: [], hr: nil,
                      hrTrend: nil, corr: nil, effects: F.crossingZero),
            F.summary(sessions: 9, minutes: 410, focused: 0.81, drop: 18, bestHours: [20], hrTrend: 0.1, corr: -0.3,
                      mock: true, replay: true, effects: F.hapticHelps),
            F.summary(sessions: 2, minutes: 40, focused: 0.6, drop: 3, corr: 0.05, mock: true),
            F.summary(sessions: 3, minutes: 70, focused: 0.6, drop: 7, hrTrend: 1.4, corr: 0.2, replay: true),
            F.summary(sessions: 0, minutes: 0, focused: 0, drop: nil, refocus: nil, bestHours: [], hr: nil, hrTrend: nil, corr: nil),
        ]
    }

    func testCountsSentencesAndGuardrailsForManySummaries() async {
        for s in variedSummaries {
            let out = await gen.insights(for: s)
            XCTAssertTrue((2...4).contains(out.count), "count \(out.count) for \(s)")
            for i in out {
                XCTAssertEqual(i.source, .template)
                XCTAssertFalse(i.text.isEmpty)
                XCTAssertTrue((1...2).contains(F.sentenceCount(i.text)), "sentences: \(i.text)")
                XCTAssertEqual(InsightGuardrails.violations(in: i.text), [], i.text)
                XCTAssertFalse(i.text.lowercased().contains("reads your brain"))
            }
        }
    }

    func testDeterministic() {
        for s in variedSummaries { XCTAssertEqual(gen.texts(for: s), gen.texts(for: s)) }
        XCTAssertNotEqual(gen.texts(for: variedSummaries[0]), gen.texts(for: variedSummaries[3]))
    }

    func testReplayQualifierOnEveryInsight() {
        let out = gen.texts(for: F.summary(replay: true, effects: F.hapticHelps))
        XCTAssertFalse(out.isEmpty)
        for t in out { XCTAssertTrue(t.contains("(SYNTHETIC replay data)"), t) }
    }

    func testMockQualifierOnContentInsightOnly() {
        let out = gen.texts(for: F.summary(corr: 0.45, mock: true))
        let drift = out.filter { $0.contains("predicted brain response to") }
        XCTAssertEqual(drift.count, 1)
        XCTAssertTrue(drift[0].contains("(average brain)"))
        XCTAssertTrue(drift[0].contains("(based on MOCK content predictions)"))
        XCTAssertTrue(drift[0].contains("r = 0.45"))
        for t in out where !drift.contains(t) { XCTAssertFalse(t.contains("MOCK"), t) }
        // Not mock: no qualifier.
        XCTAssertFalse(gen.texts(for: F.summary(corr: 0.45)).joined().contains("MOCK"))
    }

    func testMockAndReplayTogether() {
        let out = gen.texts(for: F.summary(corr: 0.45, mock: true, replay: true))
        let drift = out.first { $0.contains("predicted brain response") }!
        XCTAssertTrue(drift.contains("MOCK"))
        XCTAssertTrue(drift.contains("SYNTHETIC"))
    }

    func testNoEffectClaimWhenIntervalCrossesZero() {
        let out = gen.texts(for: F.summary(effects: F.crossingZero)).joined(separator: " ")
        XCTAssertTrue(out.contains("still learning"), out)
        XCTAssertFalse(out.contains("has helped"))
        XCTAssertFalse(out.contains("working for you"))
        XCTAssertFalse(out.contains(NudgeArm.haptic.displayName))
    }

    func testEffectClaimWhenIntervalExcludesZero() {
        let out = gen.texts(for: F.summary(effects: F.hapticHelps)).joined(separator: " ")
        XCTAssertTrue(out.contains(NudgeArm.haptic.displayName), out)
        XCTAssertTrue(out.contains("+18"), out)
        XCTAssertTrue(out.contains("90% interval +5 to +31"), out)
        XCTAssertFalse(out.contains(NudgeArm.chime.displayName), "chime's interval crosses 0")
    }

    func testNegativeEffectIsReportedWithoutPraise() {
        let effects = [ArmEffect(arm: .chime, policy: .reactive, effect: -0.2, lower: -0.35, upper: -0.05, level: 0.9, n: 30)]
        let out = gen.texts(for: F.summary(effects: effects)).joined(separator: " ")
        XCTAssertTrue(out.contains("has not helped"), out)
    }

    func testExperimentalLightPulseNeverClaimed() {
        let effects = [ArmEffect(arm: .lightPulse, policy: .reactive, effect: 0.3, lower: 0.1, upper: 0.5, level: 0.9, n: 40)]
        let out = gen.texts(for: F.summary(effects: effects)).joined(separator: " ")
        XCTAssertFalse(out.contains(NudgeArm.lightPulse.displayName), out)
        XCTAssertTrue(out.contains("still learning"))
    }

    func testHeartRateWordingIsNeutral() {
        let down = gen.texts(for: F.summary(drop: nil, bestHours: [], corr: nil)).joined(separator: " ")
        XCTAssertTrue(down.contains("68 BPM"), down)
        XCTAssertTrue(down.contains("trending down by about 0.8 BPM per session"), down)
        let steady = gen.texts(for: F.summary(drop: nil, bestHours: [], hrTrend: 0.2, corr: nil)).joined(separator: " ")
        XCTAssertTrue(steady.contains("steady"), steady)
    }

    func testBestHoursAndDropMinuteMentioned() {
        let out = gen.texts(for: F.summary(refocus: nil, bestHours: [16, 9], hr: nil, hrTrend: nil, corr: nil)).joined(separator: " ")
        XCTAssertTrue(out.contains("minute 12"), out)
        XCTAssertTrue(out.contains("4 PM"), out)
        XCTAssertTrue(out.contains("9 AM"), out)
    }

    func testFactListContainsOnlyAggregates() {
        let facts = InsightFacts.factList(for: F.summary(mock: true, replay: true, effects: F.hapticHelps))
        XCTAssertTrue(facts.text.contains("time focused: 70%"))
        XCTAssertTrue(facts.text.contains("verdict: helps"))
        XCTAssertTrue(facts.text.contains("verdict: still learning"))
        XCTAssertTrue(facts.text.contains("SYNTHETIC"))
        XCTAssertTrue(facts.text.contains("MOCK"))
        XCTAssertTrue(facts.numbers.contains(68))
    }
}

final class InsightGuardrailsTests: XCTestCase {
    func testFlagsBannedPhrases() {
        let bad = [
            "This could help diagnose ADHD.",
            "You may have an attention disorder.",
            "Breathing will treat your stress.",
            "FocusLoop reads your brain while you study.",
            "This is the first app to do it.",
            "Clinically proven to boost focus.",
            "Your heart rate suggests anxiety; see a doctor.",
            "Consider medication for these symptoms.",
            "Seek medical attention if this continues.",
            "Signs of tachycardia were present.",
        ]
        for t in bad { XCTAssertFalse(InsightGuardrails.violations(in: t).isEmpty, t) }
        XCTAssertTrue(InsightGuardrails.violations(in: "A DIAGNOSIS").contains("diagnosis"))
        XCTAssertEqual(Set(InsightGuardrails.violations(in: "diagnose ADHD")), ["diagnosis", "ADHD"])
    }

    func testCleanWellnessTextPasses() {
        let ok = [
            "Your focus most often starts to slip around minute 12 of a session.",
            "Your data stays secure on this device, and the estimate is fairly accurate in good light.",
            "A short retreat from the screen may help you reset.",
            "Your heart rate while focused settled around 68 BPM. It is a general wellness reading from the camera, not a medical measurement.",
            "The predicted brain response to this lecture (average brain) dipped around minute 4.",
            "Treating yourself to a break",   // conservatively blocked (see assertion below)
        ]
        for t in ok.dropLast() { XCTAssertEqual(InsightGuardrails.violations(in: t), [], t) }
        XCTAssertFalse(InsightGuardrails.isClean(ok.last!), "treating is conservatively blocked")
    }

    func testUnsupportedNumbers() {
        XCTAssertEqual(InsightGuardrails.numbers(in: "r = -0.45 at minute 12, +18 points"), [-0.45, 12, 18])
        XCTAssertEqual(InsightGuardrails.unsupportedNumbers(in: "Focused 70% around minute 12.", allowed: [70, 12]), [])
        XCTAssertEqual(InsightGuardrails.unsupportedNumbers(in: "Focused 95% of the time.", allowed: [70, 12]), [95])
    }
}

final class FoundationModelsInsightGeneratorTests: XCTestCase {
    // These tests check the inline data-label path, which the app turns off (the report shows a footnote instead).
    override func setUp() { super.setUp(); InsightFacts.inlineDataLabels = true }
    override func tearDown() { InsightFacts.inlineDataLabels = false; super.tearDown() }

    typealias F = InsightsFixtures

    func testUsesModelOutputWhenCleanAndAddsQualifiers() async {
        let gen = FoundationModelsInsightGenerator(timeout: 5) { _, prompt in
            XCTAssertTrue(prompt.contains("time focused: 70%"))
            return ["- You were focused 70% of the time.",
                    "1. Your drift lined up with the lecture's predicted valleys (r = 0.42).",
                    "Your best hours start around 4 PM."]
        }
        let (out, reason) = await gen.generate(for: F.summary(mock: true, replay: true))
        XCTAssertNil(reason)
        XCTAssertEqual(out.count, 3)
        XCTAssertTrue(out.allSatisfy { $0.source == .foundationModels })
        XCTAssertEqual(out[0].text, "You were focused 70% of the time. (SYNTHETIC replay data)")
        XCTAssertTrue(out[1].text.contains("MOCK") && out[1].text.contains("SYNTHETIC"))
        XCTAssertFalse(out[2].text.contains("MOCK"))
    }

    func testGuardrailViolationFallsBackToTemplate() async {
        let gen = FoundationModelsInsightGenerator(timeout: 5) { _, _ in
            ["Your pattern suggests ADHD.", "You were focused 70% of the time."]
        }
        let (out, reason) = await gen.generate(for: F.summary())
        guard case .guardrail? = reason else { return XCTFail("expected guardrail fallback, got \(String(describing: reason))") }
        XCTAssertTrue(out.allSatisfy { $0.source == .template })
        XCTAssertEqual(out.map(\.text), TemplateInsightGenerator().texts(for: F.summary()))
    }

    func testInventedNumbersFallBack() async {
        let gen = FoundationModelsInsightGenerator(timeout: 5) { _, _ in
            ["You were focused 97% of the time.", "Your focus drops at minute 44."]
        }
        let (_, reason) = await gen.generate(for: F.summary())
        guard case .unsupportedNumbers? = reason else { return XCTFail("got \(String(describing: reason))") }
    }

    func testErrorFallsBack() async {
        struct Boom: Error {}
        let gen = FoundationModelsInsightGenerator(timeout: 5) { _, _ in throw Boom() }
        let (out, reason) = await gen.generate(for: F.summary())
        guard case .error? = reason else { return XCTFail("got \(String(describing: reason))") }
        XCTAssertTrue((2...4).contains(out.count))
    }

    func testTooFewItemsFallsBack() async {
        let gen = FoundationModelsInsightGenerator(timeout: 5) { _, _ in ["Only one."] }
        let (_, reason) = await gen.generate(for: F.summary())
        XCTAssertEqual(reason, .badCount(1))
    }

    func testTimeoutFallsBack() async {
        let gen = FoundationModelsInsightGenerator(timeout: 0.2) { _, _ in
            try await Task.sleep(nanoseconds: 30_000_000_000)
            return ["late", "late"]
        }
        let started = Date()
        let (out, reason) = await gen.generate(for: F.summary())
        XCTAssertEqual(reason, .timedOut)
        XCTAssertLessThan(Date().timeIntervalSince(started), 5)
        XCTAssertTrue(out.allSatisfy { $0.source == .template })
    }

    func testLongItemsAreTrimmedToTwoSentences() async {
        let gen = FoundationModelsInsightGenerator(timeout: 5) { _, _ in
            ["You were focused 70% of the time. That is solid! Keep it up.", "Your best hours start around 4 PM."]
        }
        let (out, reason) = await gen.generate(for: F.summary())
        XCTAssertNil(reason)
        XCTAssertEqual(out[0].text, "You were focused 70% of the time. That is solid!")
        XCTAssertEqual(FoundationModelsInsightGenerator.firstSentences("r = 0.42 here. Two. Three.", max: 2), "r = 0.42 here. Two.")
    }

    func testRepeatedSentencesAreRemoved() {
        let out = FoundationModelsInsightGenerator.dedupeSentences([
            "Focus dips at minute 12. You refocus in 8.4 s.",
            "Drowsy at minute 12. You refocus in 8.4 s.",
            "You refocus in 8.4 s.",
        ])
        XCTAssertEqual(out, ["Focus dips at minute 12. You refocus in 8.4 s.", "Drowsy at minute 12."])
    }

    func testPromptCarriesConstraints() {
        let i = FoundationModelsInsightGenerator.instructions
        XCTAssertTrue(i.contains("ONLY the facts"))
        XCTAssertTrue(i.contains("no medical advice"))
        XCTAssertTrue(i.contains("ADHD"))
        XCTAssertTrue(i.contains("SYNTHETIC or MOCK"))
    }

    /// Runs Apple's on-device model for real. Opt-in (set FL_RUN_SYSTEM_MODEL_TEST=1): it is slow and the model
    /// process can get the test runner killed, which would make the regular suite flaky.
    func testSystemGeneratorOnThisDevice() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["FL_RUN_SYSTEM_MODEL_TEST"] == "1",
                          "Set FL_RUN_SYSTEM_MODEL_TEST=1 to exercise the on-device Foundation Models coach.")
        let gen = FoundationModelsInsightGenerator(timeout: 20)
        XCTAssertEqual(gen.usesFoundationModels, FoundationModelsInsightGenerator.isSystemModelAvailable)
        let started = Date()
        let (out, reason) = await gen.generate(for: F.summary(replay: true, effects: F.crossingZero))
        print("[InsightsTests] system model available: \(FoundationModelsInsightGenerator.isSystemModelAvailable), "
              + "fallback: \(String(describing: reason)), seconds: \(Date().timeIntervalSince(started)), texts: \(out.map(\.text))")
        XCTAssertLessThan(Date().timeIntervalSince(started), 60)
        XCTAssertTrue((2...4).contains(out.count))
        XCTAssertTrue(InsightGuardrails.isClean(out.map(\.text)))
        XCTAssertTrue(out.allSatisfy { $0.text.contains("SYNTHETIC") })
        XCTAssertTrue(out.allSatisfy { (1...2).contains(F.sentenceCount($0.text)) })
        if !FoundationModelsInsightGenerator.isSystemModelAvailable {
            XCTAssertEqual(reason, .unavailable)
            XCTAssertTrue(out.allSatisfy { $0.source == .template })
        }
    }
}

final class InsightServiceTests: XCTestCase {
    typealias F = InsightsFixtures

    final class CountingGenerator: InsightGenerator {
        var calls = 0
        var texts: [String]
        init(texts: [String]) { self.texts = texts }
        var usesFoundationModels: Bool { true }
        func insights(for summary: AggregateSummary) async -> [Insight] {
            calls += 1
            return texts.map { Insight(text: $0, source: .foundationModels) }
        }
    }

    func testServiceOnSimulatorFallsBackOrIsClean() async {
        // The live on-device model only runs when opted in (see testSystemGeneratorOnThisDevice).
        let live = ProcessInfo.processInfo.environment["FL_RUN_SYSTEM_MODEL_TEST"] == "1"
        let service = InsightService(preferFoundationModels: live, timeout: 20, calendar: F.utc)
        let session = F.session([(.focused, 300), (.distracted, 40), (.focused, 200)], heartRate: 70,
                                events: [.refocused(t: 345, afterSeconds: 5)], replay: true)
        let out = await service.insights(for: session)
        XCTAssertFalse(out.isEmpty)
        XCTAssertTrue((2...4).contains(out.count))
        XCTAssertTrue(InsightGuardrails.isClean(out.map(\.text)))
        if !service.usesFoundationModels {
            XCTAssertTrue(out.allSatisfy { $0.source == .template })
        }
    }

    func testTemplateOnlyService() async {
        let service = InsightService(preferFoundationModels: false)
        XCTAssertFalse(service.usesFoundationModels)
        let out = await service.insights(for: F.summary())
        XCTAssertTrue(out.allSatisfy { $0.source == .template })
    }

    func testCachesPerSessionID() async {
        let fm = CountingGenerator(texts: ["You were focused most of the time.", "Keep going."])
        let service = InsightService(foundationModels: fm, calendar: F.utc)
        let session = F.session([(.focused, 120)])
        let a = await service.insights(for: session)
        let b = await service.insights(for: session)
        XCTAssertEqual(fm.calls, 1)
        XCTAssertEqual(a.map(\.id), b.map(\.id))
        XCTAssertEqual(service.cachedInsights(for: session.id), a)
        service.invalidate(sessionID: session.id)
        _ = await service.insights(for: session)
        XCTAssertEqual(fm.calls, 2)
        // Another session is a separate cache entry.
        _ = await service.insights(for: F.session([(.focused, 60)]))
        XCTAssertEqual(fm.calls, 3)
    }

    func testStoredInsightsAreReused() async {
        let fm = CountingGenerator(texts: ["a.", "b."])
        let service = InsightService(foundationModels: fm)
        var session = F.session([(.focused, 60)])
        session.insights = [Insight(text: "Stored insight.", source: .template)]
        let out = await service.insights(for: session)
        XCTAssertEqual(out, session.insights)
        XCTAssertEqual(fm.calls, 0)
    }

    func testServiceRejectsDirtyModelOutput() async {
        let fm = CountingGenerator(texts: ["This will cure your focus problems.", "Fine."])
        let service = InsightService(foundationModels: fm)
        let out = await service.insights(for: F.summary())
        XCTAssertTrue(out.allSatisfy { $0.source == .template })
        XCTAssertTrue(InsightGuardrails.isClean(out.map(\.text)))
    }
}

final class InsightClaimGuardrailTests: XCTestCase {
    private func summary(_ fr: [Double]) -> AggregateSummary {
        AggregateSummary(sessionCount: 1, totalMinutes: 20, focusedFraction: fr[0], stateFractions: fr,
                         typicalDropMinute: 12, medianRefocusSeconds: 4, bestHours: [], restingHeartRate: nil,
                         heartRateTrendPerSession: nil, contentDriftCorrelation: nil, contentIsMock: false,
                         isReplay: false, armEffects: [])
    }

    func testRejectsStateTiedToMinute() {
        let s = summary([0.7, 0.2, 0.05, 0.05])
        XCTAssertTrue(FoundationModelsInsightGenerator.hasUnsupportedStateClaim(
            "You're getting stressed around session minute 12.", summary: s))
    }

    func testRejectsRareState() {
        let s = summary([0.9, 0.08, 0.02, 0.0])
        XCTAssertTrue(FoundationModelsInsightGenerator.hasUnsupportedStateClaim("Feeling stressed is common for you.", summary: s))
        XCTAssertFalse(FoundationModelsInsightGenerator.hasUnsupportedStateClaim("You were distracted 8% of the time.", summary: s))
    }

    func testAllowsPlainFocusDropSentence() {
        let s = summary([0.7, 0.2, 0.05, 0.05])
        XCTAssertFalse(FoundationModelsInsightGenerator.hasUnsupportedStateClaim(
            "Your focus tends to dip around minute 12.", summary: s))
    }
}

final class InsightDefaultLabelTests: XCTestCase {
    func testDefaultInsightsCarryNoInlineDataLabels() {
        XCTAssertFalse(InsightFacts.inlineDataLabels)
        let out = TemplateInsightGenerator().texts(for: InsightsFixtures.summary(corr: 0.45, mock: true, replay: true))
        XCTAssertFalse(out.isEmpty)
        for t in out { XCTAssertFalse(t.contains("SYNTHETIC") || t.contains("MOCK"), t) }
    }
}
