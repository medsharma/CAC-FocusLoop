import CoreGraphics
import SwiftUI
import UIKit

/// Exports a SwiftUI report view to PDF (US Letter, multi-page) or PNG in the temporary directory,
/// for `ShareLink`. Files are only ever written when the user explicitly asks to export, are named
/// `FocusLoop-...` (so `SwiftDataSessionStore.deleteAll()` and `removeExports()` sweep them), and
/// carry `-SYNTHETIC` / `-MOCK` in the file name when the report's data is synthetic or mock.
///
/// Note: `ImageRenderer` cannot capture Metal views, so a live `BrainCortexView` renders blank in
/// exports. Pass the export variant of the report a still from `CortexSnapshotter` (as `Image(uiImage:)`).
@MainActor
public enum ReportExporter {
    /// US Letter in points.
    public static let pageSize = CGSize(width: 612, height: 792)
    public static let margin: CGFloat = 36
    public static var contentWidth: CGFloat { pageSize.width - 2 * margin }

    public enum ExportError: Error { case renderFailed, pngEncodingFailed }

    /// Accepts any view, including `AnyView`.
    /// Renders `view` at the page content width (height = the view's ideal height) and paginates it.
    /// - Returns: the PDF file URL in the temp dir.
    public static func exportPDF<V: View>(_ view: V, report: SessionReport? = nil, colorScheme: ColorScheme = .dark,
                                          directory: URL = FileManager.default.temporaryDirectory) throws -> URL {
        let url = directory.appendingPathComponent(fileName(report: report, ext: "pdf"))
        let renderer = ImageRenderer(content: styled(view, width: contentWidth, colorScheme: colorScheme))
        renderer.proposedSize = ProposedViewSize(width: contentWidth, height: nil)

        var ok = false
        renderer.render { size, draw in
            var media = CGRect(origin: .zero, size: pageSize)
            guard size.width > 0, size.height > 0,
                  let pdf = CGContext(url as CFURL, mediaBox: &media, pdfInfo(report: report)) else { return }
            let scale = min(1, contentWidth / size.width)
            let pageContentHeight = (pageSize.height - 2 * margin) / scale   // in view points
            let pages = max(1, Int((size.height / pageContentHeight).rounded(.up)))
            for page in 0..<pages {
                pdf.beginPDFPage(nil)
                if colorScheme == .dark {   // full-bleed dark page behind dark content
                    pdf.setFillColor(UIColor(white: 0.06, alpha: 1).cgColor)
                    pdf.fill(media)
                }
                pdf.saveGState()
                pdf.clip(to: CGRect(x: margin, y: margin, width: contentWidth, height: pageSize.height - 2 * margin))
                // PDF space is y-up and the renderer draws the view's top at y = height. Shift so the
                // slice [page * h, (page + 1) * h) from the top lands in the page's content box.
                pdf.translateBy(x: margin, y: margin)
                pdf.scaleBy(x: scale, y: scale)
                pdf.translateBy(x: 0, y: -(size.height - CGFloat(page + 1) * pageContentHeight))
                draw(pdf)
                pdf.restoreGState()
                pdf.endPDFPage()
            }
            pdf.closePDF()
            ok = true
        }
        guard ok else { throw ExportError.renderFailed }
        return url
    }

    /// Renders `view` at `width` points (ideal height) to a PNG at `scale`.
    public static func exportPNG<V: View>(_ view: V, report: SessionReport? = nil, width: CGFloat = 430,
                                          scale: CGFloat = 2, colorScheme: ColorScheme = .dark,
                                          directory: URL = FileManager.default.temporaryDirectory) throws -> URL {
        let renderer = ImageRenderer(content: styled(view, width: width, colorScheme: colorScheme)
            .background(colorScheme == .dark ? Color(white: 0.06) : Color.white))
        renderer.proposedSize = ProposedViewSize(width: width, height: nil)
        renderer.scale = scale
        guard let image = renderer.uiImage else { throw ExportError.renderFailed }
        guard let png = image.pngData() else { throw ExportError.pngEncodingFailed }
        let url = directory.appendingPathComponent(fileName(report: report, ext: "png"))
        try png.write(to: url, options: [.atomic, .completeFileProtection])
        return url
    }

    /// Deletes every FocusLoop export in the temp dir (call after the share sheet closes).
    public static func removeExports(directory: URL = FileManager.default.temporaryDirectory) {
        SwiftDataSessionStore.deleteTempFiles(in: directory)
    }

    /// `FocusLoop-Report-2026-10-25-1530[-SYNTHETIC][-MOCK].pdf`
    public static func fileName(report: SessionReport?, ext: String) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd-HHmm"
        var name = "FocusLoop-Report-\(f.string(from: report?.startedAt ?? Date()))"
        if report?.isReplay == true { name += "-SYNTHETIC" }
        if report?.contentIsMock == true { name += "-MOCK" }
        return name + "." + ext
    }

    private static func styled<V: View>(_ view: V, width: CGFloat, colorScheme: ColorScheme) -> some View {
        view.frame(width: width).environment(\.colorScheme, colorScheme)
    }

    private static func pdfInfo(report: SessionReport?) -> CFDictionary {
        var title = "FocusLoop session report"
        if report?.isReplay == true { title += " (SYNTHETIC data)" }
        if report?.contentIsMock == true { title += " (MOCK brain timeline)" }
        return [kCGPDFContextTitle as String: title, kCGPDFContextCreator as String: "FocusLoop"] as CFDictionary
    }
}
