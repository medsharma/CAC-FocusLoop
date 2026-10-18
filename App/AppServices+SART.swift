import Foundation

// Sustained-attention profile (every SART run is saved) and the "Share with your doctor" PDF.

extension AppServices {

    // MARK: SART history

    func sartHistory() -> [SARTRunRecord] { (try? store.sartRuns()) ?? [] }

    /// Saves a completed (non-practice) SART run. The taps are real on a phone and in the Simulator alike;
    /// `origin` records which, and `sensorsRecorded` whether the camera recorded features alongside.
    func saveSARTRun(_ result: SARTResult, sensorsRecorded: Bool) {
        guard !result.config.isPractice, !result.trials.isEmpty else { return }
        try? store.saveSARTRun(SARTRunRecord(result: result, origin: .current, sensorsRecorded: sensorsRecorded))
        // The CSV export (Debug menu) keeps reading the latest run's trials from this key.
        if let d = try? JSONEncoder().encode(result.trials) { try? store.saveModelWeights(d, key: Self.sartKey) }
    }

    static let legacySARTImportedKey = "focusloop.sart.legacyImported"

    /// Before every run was saved, only the last run's trials were kept (weights key "sartTrials"). Import that
    /// run once so it shows in the history (dated when it was saved).
    func importLegacySARTRunIfNeeded() {
        let defaults = UserDefaults.standard
        guard !defaults.bool(forKey: Self.legacySARTImportedKey) else { return }
        defaults.set(true, forKey: Self.legacySARTImportedKey)
        guard sartHistory().isEmpty, let d = try? store.modelWeights(key: Self.sartKey),
              let trials = try? JSONDecoder().decode([SARTTrial].self, from: d), !trials.isEmpty else { return }
        let date = (try? store.modelWeightsUpdatedAt(key: Self.sartKey)) ?? Date()
        var config = SARTConfig()
        config.trialCount = trials.count
        try? store.saveSARTRun(SARTRunRecord(date: date, origin: .current, sensorsRecorded: false, config: config, trials: trials))
    }

    // MARK: Clinician report

    func doctorReportCounts() -> DoctorReportCounts {
        let sessions = (try? store.sessions()) ?? []
        let demo = sessions.filter(DoctorReportBuilder.isDemoSession).count
        return DoctorReportCounts(realSessions: sessions.count - demo, demoSessions: demo,
                                  sartRuns: sartHistory().filter { $0.origin.isRealTaps }.count)
    }

    func makeDoctorReport(includeDemo: Bool) async throws -> URL {
        let input = DoctorReportBuilder.Input(sessions: (try? store.sessions()) ?? [],
                                              realArmEffects: real.policy.effectEstimates(),
                                              demoArmEffects: includeDemo ? replay.policy.effectEstimates() : [],
                                              sartRuns: sartHistory(), includeDemo: includeDemo)
        let content = DoctorReportBuilder.build(input)
        return try DoctorReportExporter.exportPDF(content)
    }

    // MARK: DEBUG sample runs (screenshots only)

    #if DEBUG
    /// `-seedSampleSART` (DEBUG builds only): adds 5 clearly labelled SAMPLE runs (generated, not a person's taps)
    /// so the sustained-attention screens can be screenshotted. They carry `origin = .sample`, show a SAMPLE tag,
    /// and stay out of the clinician report unless demo data is included (then marked as demo data).
    /// `-removeSampleSART` deletes them.
    func applySARTDebugLaunchArguments() {
        let args = ProcessInfo.processInfo.arguments
        if args.contains("-removeSampleSART") { _ = try? store.deleteSARTRuns(origin: .sample) }
        guard args.contains("-seedSampleSART"), !sartHistory().contains(where: { $0.origin == .sample }) else { return }
        for run in SARTSampleRuns.make() { try? store.saveSARTRun(run) }
    }
    #endif
}

#if DEBUG
/// Generated SAMPLE SART runs (DEBUG only). Deterministic, so screenshots are stable.
enum SARTSampleRuns {
    static func make(now: Date = Date()) -> [SARTRunRecord] {
        // (days ago, commission probability, mean RT s, RT SD s, omission probability)
        let plans: [(Int, Double, Double, Double, Double)] = [
            (13, 0.44, 0.335, 0.105, 0.03), (10, 0.40, 0.345, 0.095, 0.025), (7, 0.36, 0.352, 0.088, 0.02),
            (4, 0.32, 0.360, 0.080, 0.015), (1, 0.28, 0.368, 0.072, 0.012),
        ]
        return plans.enumerated().map { k, p in
            var rng = CalibrationSplitMix64(seed: UInt64(100 + k))
            let config = SARTConfig()
            let seq = SARTEngine.makeSequence(count: config.trialCount, using: &rng)
            var trials: [SARTTrial] = []
            for (i, s) in seq.enumerated() {
                let onset = Double(i) * config.trialDuration
                let noGo = s.digit == config.noGoDigit
                var tap: Double?
                let u = Double.random(in: 0..<1, using: &rng)
                if noGo ? u < p.1 : u >= p.4 {
                    // Box-Muller normal RT, clamped into the response window.
                    let a = Double.random(in: 0.0001..<1, using: &rng), b = Double.random(in: 0..<1, using: &rng)
                    let z = (-2 * log(a)).squareRoot() * cos(2 * .pi * b)
                    tap = onset + min(1.1, max(0.12, p.2 + p.3 * z))
                }
                trials.append(SARTTrial(index: i, digit: s.digit, fontSize: s.fontSize, isNoGo: noGo, onset: onset,
                                        maskOnset: onset + config.stimulusDuration, tapTime: tap,
                                        end: onset + config.trialDuration))
            }
            let date = Calendar.current.date(byAdding: .day, value: -p.0, to: now) ?? now
            return SARTRunRecord(date: date, origin: .sample, sensorsRecorded: false, config: config, trials: trials)
        }
    }
}
#endif
