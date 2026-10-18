import Charts
import CoreGraphics
import SwiftUI
import UIKit

/// Lays out `DoctorReportContent` as a print-friendly US Letter PDF: black on white, multi-page, with a running
/// head from page 2 and a footer on every page ("Page x of y" + the not-a-diagnosis statement).
///
/// Each block is rendered separately with `ImageRenderer` (vector text and charts), measured, and placed on the
/// page where it fits; section headings stay with the block that follows them and long tables are split with
/// their header row repeated. Files go to the temp dir as `FocusLoop-Attention-Summary-...pdf`, so Delete all
/// data (and `ReportExporter.removeExports`) sweeps them. Nothing is sent anywhere.
@MainActor
public enum DoctorReportExporter {
    public static let pageSize = CGSize(width: 612, height: 792)
    static let marginX: CGFloat = 54
    static let contentTop: CGFloat = 58
    static let footerTop: CGFloat = 792 - 66
    static var contentWidth: CGFloat { pageSize.width - 2 * marginX }
    static var contentBottom: CGFloat { footerTop - 14 }
    static let tableChunk = 10

    public enum ExportError: Error { case renderFailed }

    /// Writes the PDF and returns its URL.
    public static func exportPDF(_ content: DoctorReportContent, now: Date = Date(),
                                 directory: URL = FileManager.default.temporaryDirectory) throws -> URL {
        let url = directory.appendingPathComponent(fileName(content, now: now))
        let units = makeUnits(content)
        let sizes = units.map { measure($0.view) }
        let pages = paginate(units: units, heights: sizes.map(\.height))

        var media = CGRect(origin: .zero, size: pageSize)
        let info = [kCGPDFContextTitle as String: DoctorReportContent.title + (content.includesDemo ? " (includes demo data)" : ""),
                    kCGPDFContextCreator as String: "FocusLoop"] as CFDictionary
        guard let pdf = CGContext(url as CFURL, mediaBox: &media, info) else { throw ExportError.renderFailed }
        for (p, page) in pages.enumerated() {
            pdf.beginPDFPage(nil)
            pdf.setFillColor(UIColor.white.cgColor)
            pdf.fill(media)
            if p > 0 { draw(AnyView(RunningHead(range: content.dateRange)), at: 26, in: pdf) }
            var y = contentTop
            for i in page {
                draw(units[i].view, at: y, in: pdf)
                y += sizes[i].height
            }
            draw(AnyView(PageFooter(page: p + 1, of: pages.count)), at: footerTop, in: pdf)
            pdf.endPDFPage()
        }
        pdf.closePDF()
        return url
    }

    /// `FocusLoop-Attention-Summary-2026-10-25-1530[-with-demo-data].pdf`
    public static func fileName(_ content: DoctorReportContent, now: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd-HHmm"
        return "FocusLoop-Attention-Summary-\(f.string(from: now))\(content.includesDemo ? "-with-demo-data" : "").pdf"
    }

    // MARK: Layout units

    struct Unit {
        var view: AnyView
        /// Keep on the same page as the next unit (section headings).
        var keepWithNext = false
        /// Start a new page before this unit if less than this much room is left (points).
        var minRoom: CGFloat = 0
    }

    static func makeUnits(_ c: DoctorReportContent) -> [Unit] {
        var u: [Unit] = [Unit(view: AnyView(TitleBlock(content: c)))]
        for s in c.sections {
            u.append(Unit(view: AnyView(SectionHeading(section: s)), keepWithNext: true, minRoom: 120))
            for (k, b) in s.blocks.enumerated() {
                let nextIsTable: Bool = {
                    guard case .paragraph = b, k + 1 < s.blocks.count else { return false }
                    switch s.blocks[k + 1] { case .table, .bullets: return true; default: return false }
                }()
                switch b {
                case let .table(t):
                    let chunks = stride(from: 0, to: max(1, t.rows.count), by: tableChunk).map {
                        Array(t.rows[min($0, t.rows.count)..<min($0 + tableChunk, t.rows.count)])
                    }
                    for rows in chunks {
                        u.append(Unit(view: AnyView(TableBlock(table: .init(headers: t.headers, rows: rows, weights: t.weights)))))
                    }
                case let .bullets(items):
                    for item in items { u.append(Unit(view: AnyView(BulletBlock(text: item)))) }
                default:
                    // A lead-in line stays with the table or list it introduces.
                    u.append(Unit(view: AnyView(BlockView(block: b)), keepWithNext: nextIsTable))
                }
            }
        }
        return u
    }

    /// Greedy pagination: indices of units per page.
    static func paginate(units: [Unit], heights: [CGFloat]) -> [[Int]] {
        let available = contentBottom - contentTop
        var pages: [[Int]] = [[]]
        var used: CGFloat = 0
        var i = 0
        while i < units.count {
            var need = heights[i]
            if units[i].keepWithNext, i + 1 < units.count { need += heights[i + 1] }
            need = max(need, min(units[i].minRoom, available))
            if used > 0, used + need > available {
                pages.append([])
                used = 0
            }
            pages[pages.count - 1].append(i)
            used += heights[i]
            i += 1
        }
        return pages.filter { !$0.isEmpty }.isEmpty ? [[]] : pages.filter { !$0.isEmpty }
    }

    static func renderer(_ view: AnyView) -> ImageRenderer<some View> {
        let r = ImageRenderer(content: view.frame(width: contentWidth).environment(\.colorScheme, .light))
        r.proposedSize = ProposedViewSize(width: contentWidth, height: nil)
        return r
    }

    static func measure(_ view: AnyView) -> CGSize {
        var size = CGSize.zero
        renderer(view).render { s, _ in size = s }
        return size
    }

    /// Draws `view` with its top `top` points below the top of the page.
    static func draw(_ view: AnyView, at top: CGFloat, in ctx: CGContext) {
        renderer(view).render { size, render in
            ctx.saveGState()
            // PDF space is y-up and the renderer draws the view's top at y = size.height.
            ctx.translateBy(x: marginX, y: pageSize.height - top - size.height)
            render(ctx)
            ctx.restoreGState()
        }
    }
}

// MARK: - Print styles

private enum Ink {
    static let text = Color(white: 0.08)
    static let muted = Color(white: 0.38)
    static let rule = Color(white: 0.82)
    static let faint = Color(white: 0.95)
    static let line = Color(white: 0.15)
}

private struct TitleBlock: View {
    let content: DoctorReportContent
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("TO SHARE WITH A CLINICIAN").font(.system(size: 8, weight: .semibold)).tracking(1.6).foregroundStyle(Ink.muted)
            Text(DoctorReportContent.title).font(.system(size: 24, weight: .semibold)).tracking(-0.4)
                .padding(.top, 6)
            Text(content.dateRange).font(.system(size: 11)).padding(.top, 6)
            Text(content.generated).font(.system(size: 8.5)).foregroundStyle(Ink.muted).padding(.top, 2)
            HStack(alignment: .top, spacing: 10) {
                Rectangle().fill(Ink.line).frame(width: 2)
                Text(DoctorReportContent.statement).font(.system(size: 10, weight: .medium))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Ink.faint)
            .padding(.top, 16)
            Text(content.dataScope).font(.system(size: 9)).foregroundStyle(Ink.muted)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 10)
            if content.includesDemo {
                Text("Sections marked \u{201C}DEMO DATA\u{201D} include demo sessions that were not measured from the student.")
                    .font(.system(size: 9, weight: .semibold)).padding(.top, 4)
            }
            Rectangle().fill(Ink.rule).frame(height: 0.5).padding(.top, 14)
        }
        .foregroundStyle(Ink.text)
        .padding(.bottom, 4)
    }
}

private struct SectionHeading: View {
    let section: DoctorReportContent.Section
    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("\(section.number)").font(.system(size: 13, weight: .regular).monospacedDigit()).foregroundStyle(Ink.muted)
            Text(section.title).font(.system(size: 13, weight: .semibold))
            Spacer(minLength: 8)
            if section.isDemoData {
                Text(DoctorReportContent.demoMark.uppercased()).font(.system(size: 7.5, weight: .bold)).tracking(1)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(Ink.text, lineWidth: 0.75))
            }
        }
        .foregroundStyle(Ink.text)
        .padding(.top, 16)
        .padding(.bottom, 8)
        .overlay(alignment: .bottom) { Rectangle().fill(Ink.text).frame(height: 0.75).padding(.bottom, 4) }
        .padding(.bottom, 4)
    }
}

private struct BlockView: View {
    let block: DoctorReportContent.Block
    var body: some View {
        Group {
            switch block {
            case let .paragraph(t):
                Text(t).font(.system(size: 9.5)).lineSpacing(2).fixedSize(horizontal: false, vertical: true)
            case let .note(t):
                Text(t).font(.system(size: 7.8)).foregroundStyle(Ink.muted).lineSpacing(1.5)
                    .fixedSize(horizontal: false, vertical: true)
            case let .keyValues(kv):
                VStack(spacing: 0) {
                    ForEach(Array(kv.enumerated()), id: \.offset) { _, e in
                        HStack(alignment: .firstTextBaseline) {
                            Text(e.key).foregroundStyle(Ink.muted)
                            Spacer(minLength: 12)
                            Text(e.value).monospacedDigit().multilineTextAlignment(.trailing)
                        }
                        .font(.system(size: 9.5))
                        .padding(.vertical, 4)
                        .overlay(alignment: .bottom) { Rectangle().fill(Ink.rule).frame(height: 0.5) }
                    }
                }
            case let .chart(c):
                ReportChart(series: c)
            case .table, .bullets:
                EmptyView()   // laid out as their own units
            }
        }
        .foregroundStyle(Ink.text)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.bottom, 9)
    }
}

private struct TableBlock: View {
    let table: DoctorReportContent.Table
    var body: some View {
        let total = table.weights.reduce(0, +)
        let width = DoctorReportExporter.contentWidth
        VStack(spacing: 0) {
            row(table.headers, header: true, total: total, width: width)
            ForEach(Array(table.rows.enumerated()), id: \.offset) { _, r in row(r, header: false, total: total, width: width) }
        }
        .foregroundStyle(Ink.text)
        .padding(.bottom, 9)
    }

    private func row(_ cells: [String], header: Bool, total: Double, width: CGFloat) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 0) {
            ForEach(Array(cells.enumerated()), id: \.offset) { i, c in
                Text(c)
                    .font(.system(size: header ? 7.5 : 8.5, weight: header ? .semibold : .regular))
                    .foregroundStyle(header ? Ink.muted : Ink.text)
                    .monospacedDigit()
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(width: width * CGFloat(table.weights[min(i, table.weights.count - 1)] / total) - 4, alignment: .leading)
                    .padding(.trailing, 4)
            }
        }
        .padding(.vertical, header ? 4 : 3.5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(alignment: .bottom) { Rectangle().fill(header ? Ink.text : Ink.rule).frame(height: header ? 0.75 : 0.5) }
    }
}

private struct BulletBlock: View {
    let text: String
    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("\u{2022}").font(.system(size: 9.5))
            Text(text).font(.system(size: 9.5)).lineSpacing(2).fixedSize(horizontal: false, vertical: true)
        }
        .foregroundStyle(Ink.text)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.bottom, 6)
    }
}

private struct ReportChart: View {
    let series: DoctorReportContent.ChartSeries
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(series.title).font(.system(size: 8.5, weight: .semibold))
            Chart {
                if let band = series.band, let first = series.points.first?.date, let last = series.points.last?.date {
                    RectangleMark(xStart: .value("From", first), xEnd: .value("To", last),
                                  yStart: .value("Low", band.lowerBound), yEnd: .value("High", band.upperBound))
                        .foregroundStyle(Ink.faint)
                }
                ForEach(Array(series.points.enumerated()), id: \.offset) { _, p in
                    LineMark(x: .value("Date", p.date), y: .value("Value", p.value))
                        .foregroundStyle(Ink.line)
                        .lineStyle(StrokeStyle(lineWidth: 1.2))
                    PointMark(x: .value("Date", p.date), y: .value("Value", p.value))
                        .foregroundStyle(Ink.line).symbolSize(14)
                }
            }
            .chartYScale(domain: yDomain)
            .chartXScale(domain: xDomain)
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 5)) { v in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.4)).foregroundStyle(Ink.rule)
                    AxisValueLabel(format: .dateTime.month(.abbreviated).day()).font(.system(size: 7)).foregroundStyle(Ink.muted)
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { v in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.4)).foregroundStyle(Ink.rule)
                    AxisValueLabel {
                        if let d = v.as(Double.self) { Text(format(d)).font(.system(size: 7)).foregroundStyle(Ink.muted) }
                    }
                }
            }
            .frame(height: 110)
            if series.band != nil {
                Text("Shaded band: published reference range (see the sources listed in this section).")
                    .font(.system(size: 7)).foregroundStyle(Ink.muted)
            }
        }
    }

    /// The run/session span with a margin on both sides, so edge points and date labels are not clipped.
    private var xDomain: ClosedRange<Date> {
        let a = series.points.first?.date ?? Date(), b = series.points.last?.date ?? a
        let pad = max(3600 * 6, b.timeIntervalSince(a) * 0.06)
        return a.addingTimeInterval(-pad)...b.addingTimeInterval(pad)
    }

    /// Rates and CV start at zero (and include any reference band); heart rate is zoomed to its own range.
    private var yDomain: ClosedRange<Double> {
        var values = series.points.map(\.value)
        if let b = series.band { values += [b.lowerBound, b.upperBound] }
        let lo = values.min() ?? 0, hi = values.max() ?? 1
        switch series.format {
        case .bpm: return (lo - 5).rounded(.down)...(hi + 5).rounded(.up)
        case .percent: return 0...max(0.1, min(1, hi * 1.1))
        case .ratio: return 0...max(0.05, hi * 1.15)
        }
    }

    private func format(_ v: Double) -> String {
        switch series.format {
        case .percent: "\(Int((v * 100).rounded()))%"
        case .bpm: "\(Int(v.rounded()))"
        case .ratio: String(format: "%.2f", v)
        }
    }
}

private struct RunningHead: View {
    let range: String
    var body: some View {
        HStack {
            Text(DoctorReportContent.title.uppercased()).font(.system(size: 7, weight: .semibold)).tracking(1.2)
            Spacer()
            Text(range).font(.system(size: 7))
        }
        .foregroundStyle(Ink.muted)
    }
}

private struct PageFooter: View {
    let page: Int
    let of: Int
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Rectangle().fill(Ink.rule).frame(height: 0.5)
            HStack(alignment: .top, spacing: 16) {
                Text(DoctorReportContent.statement).font(.system(size: 6.8)).lineSpacing(1)
                    .fixedSize(horizontal: false, vertical: true)
                Text("Page \(page) of \(of)").font(.system(size: 7.5, weight: .medium).monospacedDigit())
                    .fixedSize()
            }
        }
        .foregroundStyle(Ink.muted)
    }
}
