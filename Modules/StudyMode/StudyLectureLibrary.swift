import Foundation

/// Lecture catalog for Study Mode: bundled `lectures.json` plus a built-in sample lecture
/// (no video, timer-driven) that pairs with the MOCK `sample_lecture.brain.json`.
public enum StudyLectureLibrary {
    public static let sampleLectureID = "sample_lecture"
    /// "Travel deep inside a leaf": the default lecture for debug entry points.
    public static let leafLectureID = "leaf-photosynthesis-01"

    /// Lectures to LIST in the UI: the bundled catalog without the `sample_lecture` test fixture
    /// (its brain timeline is a placeholder preview, so it is never offered to a student).
    public static func lectures(bundle: Bundle = .main) -> [LectureInfo] {
        LectureInfo.catalog(bundle: bundle).filter { $0.id != sampleLectureID }
    }

    /// Every lecture including the `sample_lecture` fixture (appended unless the catalog already has it).
    /// For lookups by id (tests, debug roots); not for listing.
    public static func allLectures(bundle: Bundle = .main) -> [LectureInfo] {
        var list = LectureInfo.catalog(bundle: bundle)
        if !list.contains(where: { $0.id == sampleLectureID }) { list.append(sampleLecture) }
        return list
    }

    /// Looks a lecture up by id, including the hidden `sample_lecture` fixture.
    public static func lecture(id: String, bundle: Bundle = .main) -> LectureInfo? {
        allLectures(bundle: bundle).first { $0.id == id }
    }

    /// Loads `<lecture.id>.brain.json` from the bundle root, or nil if it is missing/invalid.
    public static func timeline(for lecture: LectureInfo, bundle: Bundle = .main) -> ContentTimeline? {
        try? BrainContentTimeline(named: lecture.id, bundle: bundle)
    }

    /// URL of the lecture video at the bundle root, or nil (=> timer-driven "no video" playback).
    public static func videoURL(for lecture: LectureInfo, bundle: Bundle = .main) -> URL? {
        guard let file = lecture.videoFile, !file.isEmpty else { return nil }
        let ns = file as NSString
        let ext = ns.pathExtension
        return bundle.url(forResource: ext.isEmpty ? file : ns.deletingPathExtension,
                          withExtension: ext.isEmpty ? nil : ext)
    }

    /// Built-in sample (text written for FocusLoop, no video). Its brain timeline is MOCK.
    public static let sampleLecture = LectureInfo(
        id: sampleLectureID,
        title: "How Photosynthesis Works",
        subject: "Biology · Built-in sample",
        source: "FocusLoop built-in sample",
        sourceURL: "",
        license: "CC0-1.0",
        attribution: "Sample narration written for FocusLoop (no video). Brain timeline is a placeholder preview, not real model output.",
        videoFile: nil,
        durationSeconds: 180,
        recaps: [
            .init(start: 0, end: 30,
                  text: "Plants turn light into chemical energy. Photosynthesis happens in chloroplasts, mostly in leaf cells."),
            .init(start: 30, end: 60,
                  text: "Chlorophyll absorbs red and blue light and reflects green, which is why leaves look green."),
            .init(start: 60, end: 90,
                  text: "The light reactions split water, release oxygen, and store energy in ATP and NADPH."),
            .init(start: 90, end: 120,
                  text: "The Calvin cycle uses ATP and NADPH to fix carbon dioxide into sugar in the stroma."),
            .init(start: 120, end: 150,
                  text: "Overall: carbon dioxide + water + light energy become glucose + oxygen."),
            .init(start: 150, end: 180,
                  text: "Light intensity, CO\u{2082} level and temperature all limit how fast photosynthesis runs."),
        ],
        recallQuestions: [
            .init(after: 30, question: "Where in the cell does photosynthesis happen?",
                  choices: ["Mitochondria", "Chloroplasts", "Nucleus"], correctIndex: 1),
            .init(after: 60, question: "Why do most leaves look green?",
                  choices: ["They absorb green light", "They reflect green light", "They emit green light"],
                  correctIndex: 1),
            .init(after: 120, question: "Which stage fixes CO\u{2082} into sugar?",
                  choices: ["Light reactions", "Calvin cycle", "Glycolysis"], correctIndex: 1),
        ])
}
