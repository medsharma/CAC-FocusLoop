import XCTest

/// Walks the whole app the way a judge would, attaching a screenshot at every step (`.keepAlways`).
/// Runs in the Simulator, where sessions use the SYNTHETIC replay stream and calibration uses the
/// "Simulate calibration" path.
final class FocusLoopUITests: XCTestCase {
    private var app: XCUIApplication!
    private var step = 0

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    override func tearDownWithError() throws {
        app?.terminate()
    }

    // MARK: Helpers

    @discardableResult
    private func launch(_ args: [String]) -> XCUIApplication {
        let a = XCUIApplication()
        a.launchArguments = args
        a.launch()
        app = a
        return a
    }

    /// Attaches a screenshot named `NN-<test>-<name>`.
    private func snap(_ name: String) {
        step += 1
        let shot = XCUIScreen.main.screenshot()
        let att = XCTAttachment(screenshot: shot)
        let test = String(name: self.name)
        att.name = String(format: "%02d-%@-%@", step, test, name)
        att.lifetime = .keepAlways
        add(att)
    }

    private func wait(_ e: XCUIElement, _ timeout: TimeInterval = 10, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(e.waitForExistence(timeout: timeout), "Missing: \(e)", file: file, line: line)
    }

    private func pause(_ seconds: TimeInterval) {
        let exp = expectation(description: "pause")
        exp.isInverted = true
        wait(for: [exp], timeout: seconds)
    }

    private func tapButton(_ label: String, timeout: TimeInterval = 10, file: StaticString = #filePath, line: UInt = #line) {
        let b = app.buttons[label].firstMatch
        wait(b, timeout, file: file, line: line)
        b.tap()
    }

    private func back() {
        let b = app.navigationBars.buttons.element(boundBy: 0)
        if b.waitForExistence(timeout: 5) { b.tap() }
        pause(0.8)
    }

    /// Scrolls the main scroll view (or list) down `times` times, taking a screenshot after each.
    private func scrollAndSnap(_ name: String, times: Int, in element: XCUIElement? = nil) {
        let target = element ?? app.scrollViews.firstMatch
        for i in 1...times {
            target.swipeUp(velocity: .slow)
            pause(0.6)
            snap("\(name)-scroll\(i)")
        }
    }

    /// Flips a SwiftUI Toggle by tapping its switch (right edge of the row).
    private func flip(_ id: String) {
        let row = app.switches[id].firstMatch
        wait(row)
        if !row.isHittable { app.collectionViews.firstMatch.swipeUp(velocity: .slow) }
        let before = row.value as? String
        row.coordinate(withNormalizedOffset: CGVector(dx: 0.93, dy: 0.5)).tap()
        pause(0.6)
        XCTAssertNotEqual(row.value as? String, before, "Toggle \(id) did not change")
    }

    private func scrollTo(_ e: XCUIElement, in container: XCUIElement, maxSwipes: Int = 12) {
        var n = 0
        while !(e.exists && e.isHittable) && n < maxSwipes {
            container.swipeUp(velocity: .slow); n += 1
        }
    }

    private func acceptSystemAlertIfShown() {
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        for label in ["Allow", "OK", "Allow While Using App"] {
            let b = springboard.alerts.buttons[label]
            if b.waitForExistence(timeout: 3) { b.tap(); return }
        }
    }

    private func ensureSampleHistory() {
        app.buttons["home.settings"].tap()
        let list = app.collectionViews.firstMatch
        wait(list)
        let load = app.buttons["settings.loadSamples"]
        let remove = app.buttons["settings.removeSamples"]
        scrollTo(load.exists ? load : remove, in: list, maxSwipes: 3)
        if load.exists {
            load.tap()
            wait(remove, 120)
        }
        back()
    }

    // MARK: 1. Fresh install: onboarding

    func test01_FreshInstallOnboarding() throws {
        launch(["-resetAllData"])
        wait(app.buttons["Get started"], 15)
        snap("mission")
        app.buttons["Get started"].tap()

        wait(app.staticTexts["0 bytes of video stored or sent."])
        snap("privacy-promise")
        tapButton("Continue")

        wait(app.staticTexts["A few permissions"])
        snap("permissions")
        // Ask for each permission the Simulator offers.
        for _ in 0..<3 {
            let allow = app.buttons["Allow"].firstMatch
            guard allow.waitForExistence(timeout: 2) else { break }
            allow.tap()
            acceptSystemAlertIfShown()
            pause(1)
        }
        snap("permissions-after")
        tapButton("Continue")

        wait(app.staticTexts["Set your work zone"])
        snap("calibration-intro")
        tapButton("Simulate calibration")
        pause(3)
        snap("calibration-collecting")
        wait(app.staticTexts["Work zone saved"], 40)
        snap("calibration-saved")
        tapButton("Done")

        wait(app.staticTexts["Attention check"])
        snap("sart-intro")
        let practice = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Try a practice round'")).firstMatch
        wait(practice)
        practice.tap()
        // Tap along for a few digits of the practice block.
        for _ in 0..<6 {
            pause(1.1)
            app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        }
        snap("sart-practice-running")
        wait(app.staticTexts["Practice done"], 60)
        snap("sart-practice-done")
        // The full SART takes about 3 minutes; the tour skips it (the "Skip for now" path).
        tapButton("Skip for now")

        wait(app.staticTexts["You're set"])
        snap("done")
        tapButton("Start using FocusLoop")
        wait(app.buttons["home.startFocus"], 10)
        pause(1)
        snap("home")
    }

    // MARK: 2. Focus session with the Science view

    func test02_FocusSessionAndScience() throws {
        launch(["-screen", "home"])
        wait(app.buttons["home.startFocus"], 15)
        snap("home")
        app.swipeUp(velocity: .slow)
        snap("home-bottom")
        app.swipeDown(velocity: .slow)
        app.buttons["home.startFocus"].tap()

        wait(app.buttons["focus.end"], 10)
        snap("focus-start")
        for t in [10, 20, 30] {
            pause(10)
            snap("focus-\(t)s")
        }

        app.buttons["focus.science"].tap()
        wait(app.staticTexts["Live pipeline"], 10)
        pause(2)
        snap("science-top")
        let sv = app.scrollViews.firstMatch
        for i in 1...8 {
            sv.swipeUp(velocity: .slow)
            pause(0.8)
            snap("science-scroll\(i)")
        }
        XCTAssertTrue(app.staticTexts["Predicted brain response to this content (average brain)"].exists
                      || app.staticTexts["Predicted brain response to this content"].exists)
        tapButton("Done")

        pause(1)
        app.buttons["focus.end"].tap()
        snap("end-confirm")
        tapButton("End and see report")
        wait(app.navigationBars["Session report"], 20)
        pause(2)
        snap("report-top")
        scrollAndSnap("report", times: 5)
    }

    // MARK: 3. Study Mode

    func test03_StudyMode() throws {
        launch(["-screen", "home"])
        wait(app.buttons["home.study"], 15)
        app.buttons["home.study"].tap()
        wait(app.navigationBars["Study Mode"])
        pause(1)
        snap("lecture-list")

        let ids = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'lecture.'"))
        let count = ids.count
        XCTAssertGreaterThanOrEqual(count, 1)
        let lectureIDs = (0..<count).map { ids.element(boundBy: $0).identifier }
        for id in lectureIDs {
            let row = app.buttons[id]
            wait(row)
            row.tap()
            wait(app.buttons["study.close"], 15)
            pause(6)
            snap("\(id)-playing")
            XCTAssertTrue(app.otherElements["study.scrubber"].exists || app.descendants(matching: .any)["study.scrubber"].exists,
                          "No scrubber for \(id)")
            let cortex = app.descendants(matching: .any)["study.cortex"].firstMatch
            wait(cortex)
            cortex.tap()
            wait(app.staticTexts["Content brain map"], 10)
            pause(2)
            snap("\(id)-cortex-sheet")
            app.buttons["study.cortexSheet.close"].tap()
            pause(1)
            snap("\(id)-after-sheet")
            app.buttons["study.close"].tap()
            // A long enough session opens its report; go back to the lecture list.
            pause(2)
            if app.navigationBars["Session report"].waitForExistence(timeout: 4) {
                snap("\(id)-report")
                back()
            }
            wait(app.navigationBars["Study Mode"], 10)
        }
    }

    // MARK: 4. History, Attention Profile, Privacy, Settings, sample history, report export

    func test04_ProfilePrivacySettings() throws {
        launch(["-screen", "home"])
        wait(app.buttons["home.history"], 15)

        app.buttons["home.history"].tap()
        wait(app.navigationBars["Attention Profile"])
        pause(2)
        snap("profile-before-samples")
        back()

        app.buttons["home.privacy"].tap()
        wait(app.navigationBars["Privacy"])
        snap("privacy-top")
        scrollAndSnap("privacy", times: 4)
        back()

        app.buttons["home.settings"].tap()
        wait(app.navigationBars["Settings"])
        snap("settings-top")
        let list = app.collectionViews.firstMatch
        // Force Replay is locked on in the Simulator (no face tracking), so only flip it where it's enabled.
        for id in ["settings.forceReplay", "settings.experienceSampling", "settings.demoDirector"]
            where id != "settings.forceReplay" || app.switches[id].firstMatch.isEnabled {
            flip(id)
            snap("toggle-\(id)-flipped")
            flip(id)
        }
        let readiness = app.buttons["settings.readiness"]
        scrollTo(readiness, in: list)
        snap("settings-demo")
        readiness.tap()
        wait(app.navigationBars["Demo readiness"])
        pause(2)
        snap("demo-readiness")
        app.collectionViews.firstMatch.swipeUp(velocity: .slow)
        pause(0.5)
        snap("demo-readiness-bottom")
        back()

        let load = app.buttons["settings.loadSamples"]
        scrollTo(load, in: list, maxSwipes: 3)
        if load.exists {
            load.tap()
            pause(1)
            snap("samples-loading")
        }
        wait(app.buttons["settings.removeSamples"], 120)
        snap("samples-loaded")

        scrollTo(app.switches["settings.recordStreams"], in: list)
        flip("settings.recordStreams")
        snap("record-streams-on")
        flip("settings.recordStreams")
        list.swipeUp(velocity: .slow)
        pause(0.5)
        snap("settings-bottom")

        // Calibration from Settings (phone screen), then close.
        let phoneCal = app.buttons["Recalibrate for phone screen (Study Mode)"]
        list.swipeDown(velocity: .slow)
        scrollTo(phoneCal, in: list)
        phoneCal.tap()
        wait(app.staticTexts["Calibrate for this screen"])
        snap("phone-calibration-intro")
        tapButton("Close")
        pause(1)
        back()

        // Profile with the SYNTHETIC sample history.
        app.buttons["home.history"].tap()
        wait(app.navigationBars["Attention Profile"])
        pause(3)
        snap("profile-top")
        scrollAndSnap("profile", times: 6)
        let row = app.buttons["history.session"].firstMatch
        wait(row, 10)
        scrollTo(row, in: app.scrollViews.firstMatch)
        row.tap()
        wait(app.navigationBars["Session report"], 15)
        pause(2)
        snap("sample-report-top")
        scrollAndSnap("sample-report", times: 6)

        // Export the PDF; the share sheet appears.
        let export = app.buttons["report.export"]
        scrollTo(export, in: app.scrollViews.firstMatch)
        export.tap()
        // Export opens the share sheet by itself (one tap). Fall back to "Share PDF" if it was dismissed.
        let sheet = app.otherElements["ActivityListView"].firstMatch
        if !sheet.waitForExistence(timeout: 30) {
            let share = app.buttons["report.sharePDF"]
            wait(share, 10)
            share.tap()
        }
        snap("export-ready")
        XCTAssertTrue(sheet.waitForExistence(timeout: 15) || app.sheets.firstMatch.waitForExistence(timeout: 2),
                      "The share sheet did not appear")
        pause(1.5)
        snap("share-sheet")
        let close = app.buttons["Close"].firstMatch
        if close.exists { close.tap() } else { app.swipeDown() }
        pause(1)
        back()
        back()

        // Remove the sample history.
        app.buttons["home.settings"].tap()
        let remove = app.buttons["settings.removeSamples"]
        scrollTo(remove, in: app.collectionViews.firstMatch)
        remove.tap()
        snap("remove-samples-confirm")
        let confirm = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Remove '")).firstMatch
        wait(confirm)
        confirm.tap()
        wait(app.buttons["settings.loadSamples"], 10)
        snap("samples-removed")
    }

    // MARK: 5. Scripted demo end to end

    func test05_ScriptedDemo() throws {
        launch(["-screen", "home", "-scriptedDemo"])
        wait(app.buttons["study.close"], 20)
        pause(3)
        snap("demo-start")
        let report = app.navigationBars["Session report"]
        var t = 0
        while !report.exists && t < 260 {
            pause(15); t += 15
            snap("demo-\(t)s")
            if t == 45 {
                let cortex = app.descendants(matching: .any)["study.cortex"].firstMatch
                if cortex.exists && cortex.isHittable {
                    cortex.tap()
                    if app.staticTexts["Content brain map"].waitForExistence(timeout: 5) {
                        pause(1.5); snap("demo-cortex-sheet")
                        app.buttons["study.cortexSheet.close"].tap()
                    }
                }
            }
        }
        wait(report, 60)
        pause(2)
        snap("demo-report-top")
        scrollAndSnap("demo-report", times: 6)
    }

    // MARK: 6. Delete all data

    func test06_DeleteAllData() throws {
        launch(["-screen", "home"])
        wait(app.buttons["home.settings"], 15)
        ensureSampleHistory()
        app.buttons["home.history"].tap()
        wait(app.buttons["history.session"].firstMatch, 15)
        snap("profile-with-data")
        back()

        app.buttons["home.privacy"].tap()
        let del = app.buttons["privacy.deleteAll"]
        scrollTo(del, in: app.scrollViews.firstMatch)
        snap("privacy-delete-button")
        del.tap()
        snap("delete-confirm")
        tapButton("Delete everything")
        pause(2)
        snap("after-delete")
        // Deleting resets onboarding too.
        XCTAssertTrue(app.buttons["Get started"].waitForExistence(timeout: 10), "Onboarding should return after delete")

        app.terminate()
        launch(["-screen", "history"])
        wait(app.navigationBars["Attention Profile"], 15)
        wait(app.staticTexts["history.empty"], 15)
        XCTAssertFalse(app.buttons["history.session"].exists, "History should be empty after Delete all data")
        snap("history-empty")
    }

    // MARK: 7. Demo tour for a screen recording
    //
    // Paced walk through every feature, for `xcrun simctl io booted recordVideo`. Run test07a first (loads the
    // sample history), start recording, then run test07b. Every optional step is guarded so the recording never
    // stops halfway.

    func test07a_DemoTourPrepare() throws {
        launch(["-screen", "home"])
        wait(app.buttons["home.settings"], 15)
        app.buttons["home.settings"].tap()
        wait(app.navigationBars["Settings"])
        let load = app.buttons["settings.loadSamples"], remove = app.buttons["settings.removeSamples"]
        var n = 0
        while !((load.exists && load.isHittable) || (remove.exists && remove.isHittable)) && n < 8 { app.swipeUp(velocity: 250); n += 1 }
        if load.exists { load.tap(); wait(remove, 120) }
        // Sessions should analyse the recorded face video (Settings → Debug).
        let rec = app.switches["settings.recordedVideo.toggle"].firstMatch
        n = 0
        while !(rec.exists && rec.isHittable) && n < 12 { app.swipeUp(velocity: 250); n += 1 }
        if rec.exists, (rec.value as? String) == "0" {
            rec.coordinate(withNormalizedOffset: CGVector(dx: 0.93, dy: 0.5)).tap()
            pause(1)
        }
        snap("tour-ready")
    }

    func test07b_DemoTour() throws {
        launch(["-screen", "home"])
        wait(app.buttons["home.startFocus"], 15)
        pause(4)

        // Focus session: the orb with the recorded face.
        app.buttons["home.startFocus"].tap()
        wait(app.buttons["focus.end"], 10)
        pause(10)

        // Science view: sense, understand, predict, intervene, learn.
        app.buttons["focus.science"].tap()
        wait(app.staticTexts["Live pipeline"], 10)
        pause(6)
        let sv = app.scrollViews.firstMatch
        for _ in 1...9 {
            sv.swipeUp(velocity: 250)
            pause(3.5)
        }
        tapButton("Done")
        pause(4)

        // Session report.
        app.buttons["focus.end"].tap()
        tapButton("End and see report")
        wait(app.navigationBars["Session report"], 20)
        pause(3)
        for _ in 1...4 {
            app.scrollViews.firstMatch.swipeUp(velocity: 250)
            pause(2.5)
        }
        back()
        pause(1.5)

        // Study Mode: the leaf lecture with the brain model.
        if app.buttons["home.study"].waitForExistence(timeout: 10) {
            app.buttons["home.study"].tap()
            pause(2.5)
            let leaf = app.buttons["lecture.leaf-photosynthesis-01"]
            if leaf.waitForExistence(timeout: 10) {
                leaf.tap()
                if app.buttons["study.close"].waitForExistence(timeout: 15) {
                    // A "Quick check" card can appear before a predicted dip; read it, then continue.
                    let answerQuickCheck = {
                        let c = self.app.buttons["Continue"].firstMatch
                        if c.exists && c.isHittable { self.pause(3); c.tap(); self.pause(1) }
                    }
                    pause(8)
                    answerQuickCheck()
                    let cortex = app.descendants(matching: .any)["study.cortex"].firstMatch
                    if cortex.exists && cortex.isHittable {
                        cortex.tap()
                        if app.staticTexts["Content brain map"].waitForExistence(timeout: 5) {
                            pause(6)
                            app.buttons["study.cortexSheet.close"].tap()
                        }
                    }
                    pause(6)
                    answerQuickCheck()
                    let close = app.buttons["study.close"]
                    if close.exists && close.isHittable { close.tap() }
                    pause(2)
                }
            }
        }
        // Back to Home from wherever Study Mode left us (report, lecture list, or the player).
        for _ in 0..<5 where !(app.buttons["home.history"].exists && app.buttons["home.history"].isHittable) {
            if app.navigationBars["Session report"].exists || app.navigationBars["Study Mode"].exists {
                back()
            } else if app.buttons["study.close"].exists {
                let c = app.buttons["Continue"].firstMatch
                if c.exists && c.isHittable { c.tap(); pause(1) }
                app.buttons["study.close"].tap(); pause(2)
            } else {
                back()
            }
        }

        // Attention Profile, sustained attention (SART), doctor report.
        if app.buttons["home.history"].waitForExistence(timeout: 10) {
            app.buttons["home.history"].tap()
            wait(app.navigationBars["Attention Profile"])
            pause(3.5)
            let profile = app.scrollViews.firstMatch
            for _ in 1...3 {
                profile.swipeUp(velocity: 250)
                pause(2.5)
            }
            // Find a button by scrolling up first (it may be above us), then down.
            let reveal = { (e: XCUIElement) in
                var n = 0
                while !(e.exists && e.isHittable) && n < 8 { profile.swipeDown(velocity: 400); n += 1 }
                n = 0
                while !(e.exists && e.isHittable) && n < 10 { profile.swipeUp(velocity: 250); n += 1 }
            }
            let sart = app.buttons["history.sart.details"].firstMatch
            reveal(sart)
            if sart.exists && sart.isHittable {
                pause(1.5)
                sart.tap()
                pause(4)
                app.scrollViews.firstMatch.swipeUp(velocity: 250)
                pause(3.5)
                back()
            }
            let doctor = app.buttons["history.doctorReport"]
            reveal(doctor)
            if doctor.exists && doctor.isHittable {
                doctor.tap()
                pause(5)
                let done = app.buttons["doctor.done"]
                if done.waitForExistence(timeout: 5) { done.tap() } else { app.swipeDown() }
                pause(1.5)
            }
            back()
        }

        // Privacy, then home.
        if app.buttons["home.privacy"].waitForExistence(timeout: 10) {
            app.buttons["home.privacy"].tap()
            pause(4)
            app.scrollViews.firstMatch.swipeUp(velocity: 250)
            pause(3)
            back()
        }
        pause(4)
    }
}

private extension String {
    /// "-[FocusLoopUITests test01_FreshInstallOnboarding]" → "test01".
    init(name: String) {
        let raw = name.split(separator: " ").last.map(String.init) ?? name
        self = String(raw.prefix(6))
    }
}
