import SwiftUI

/// DEBUG ONLY root for the `-debugStudyMode` / `-debugStudyPicker` launch arguments.
/// Drives the model with `StudyModeDemoDriver` (SYNTHETIC values, preview-model content).
/// `-studySeek <s>` starts at that playhead; `-studyNoAutoplay` starts paused.
struct StudyModeDebugRoot: View {
    let showPicker: Bool
    @State private var model = LiveSessionModel()
    @State private var driver: StudyModeDemoDriver?

    /// The leaf lecture (a real, listed lecture); the hidden `sample_lecture` fixture only as a fallback.
    private let lecture = StudyLectureLibrary.lecture(id: StudyLectureLibrary.leafLectureID)
        ?? StudyLectureLibrary.lectures().first
        ?? StudyLectureLibrary.sampleLecture

    var body: some View {
        Group {
            if showPicker {
                LecturePickerView(model: model)
            } else if let driver {
                StudyModeView(model: model, lecture: lecture, timeline: driver.timeline,
                              autoplay: !Self.args.contains("-studyNoAutoplay"),
                              initialPlayhead: Self.seek, onClose: {})
            } else {
                Color.black
            }
        }
        .onAppear {
            // Install the (SYNTHETIC) engine stand-in before the player appears.
            if driver == nil, let timeline = StudyLectureLibrary.timeline(for: lecture) {
                driver = StudyModeDemoDriver(model: model, lecture: lecture, timeline: timeline)
            }
        }
    }

    private static var args: [String] { ProcessInfo.processInfo.arguments }
    private static var seek: Double {
        guard let i = args.firstIndex(of: "-studySeek"), i + 1 < args.count else { return 0 }
        return Double(args[i + 1]) ?? 0
    }
}
