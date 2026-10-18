import Foundation

/// Pure recap / recall content source for Study Mode (no UI, no engine state).
///
/// The session engine owns *when* a nudge fires; this type answers *what* to show:
///  * `currentRecap(at:)`: the recap whose range contains the playhead, else the most recent one
///    that started before it (nil before the first recap starts).
///  * `nextRecallQuestion(at:)`: the earliest recall question whose `after` <= playhead that has
///    not been asked yet. `markAsked` / `recallCard(at:)` record it so it is never repeated.
public struct StudyContentProvider: Equatable, Sendable {
    public let lecture: LectureInfo
    /// Indices into `lecture.recallQuestions` that have been asked.
    public private(set) var askedRecallIndices: Set<Int> = []

    public init(lecture: LectureInfo) { self.lecture = lecture }

    // MARK: Recap

    public func currentRecap(at playhead: Double) -> LectureInfo.TimedText? {
        let recaps = lecture.recaps
        if let containing = recaps.last(where: { playhead >= $0.start && playhead < $0.end }) {
            return containing
        }
        return recaps.filter { $0.start <= playhead }.max { $0.start < $1.start }
    }

    // MARK: Recall

    /// Index of the earliest eligible, not-yet-asked recall question.
    public func nextRecallIndex(at playhead: Double) -> Int? {
        lecture.recallQuestions.indices
            .filter { !askedRecallIndices.contains($0) && lecture.recallQuestions[$0].after <= playhead }
            .min { lecture.recallQuestions[$0].after < lecture.recallQuestions[$1].after }
    }

    /// Eligible recall question (does not mark it asked).
    public func nextRecallQuestion(at playhead: Double) -> LectureInfo.RecallQuestion? {
        nextRecallIndex(at: playhead).map { lecture.recallQuestions[$0] }
    }

    public mutating func markAsked(index: Int) {
        guard lecture.recallQuestions.indices.contains(index) else { return }
        askedRecallIndices.insert(index)
    }

    public mutating func markAsked(_ question: LectureInfo.RecallQuestion) {
        if let i = lecture.recallQuestions.firstIndex(of: question) { markAsked(index: i) }
    }

    public var remainingRecallCount: Int { lecture.recallQuestions.count - askedRecallIndices.count }

    public mutating func reset() { askedRecallIndices = [] }

    // MARK: Ready-made cards for the engine (`model.activePresentation = .card(...)`)

    public func recapCard(at playhead: Double) -> NudgeCard? {
        guard let recap = currentRecap(at: playhead) else { return nil }
        return NudgeCard(kind: .recap, title: "Quick recap", body: recap.text)
    }

    /// Returns the next eligible recall question as a card and marks it asked.
    public mutating func recallCard(at playhead: Double) -> NudgeCard? {
        guard let i = nextRecallIndex(at: playhead) else { return nil }
        markAsked(index: i)
        let q = lecture.recallQuestions[i]
        return NudgeCard(kind: .recall, title: "Quick check", body: q.question,
                         choices: q.choices, correctIndex: q.correctIndex)
    }
}
