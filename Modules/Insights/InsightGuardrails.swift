import Foundation

/// Post-filter for coach text (spec §1.9). Wellness framing only: no diagnosis, no treatment
/// claims, no medical terms, never "first", "clinically proven" or "reads your brain".
public enum InsightGuardrails {

    /// (label, case-insensitive regex). Word boundaries keep "secure", "retreat", "accurate" clean.
    public static let bannedPatterns: [(label: String, pattern: String)] = [
        ("diagnosis", #"diagnos"#),
        ("ADHD", #"\badhd\b"#),
        ("disorder", #"disorder"#),
        ("treat", #"\btreat(s|ed|ing|ment|ments|able)?\b"#),
        ("cure", #"\bcur(e|es|ed|ing)\b"#),
        ("heal", #"\bheal(s|ed|ing)?\b"#),
        ("clinical", #"clinical"#),
        ("proven", #"\bproven\b"#),
        ("first", #"\bfirst\b"#),
        ("reads your brain", #"\bread(s|ing)?\s+(your|the|a)\s+(brain|mind)"#),
        ("brain reading", #"\b(brain|mind)[\s-]+read(ing|er|s)?\b"#),
        ("disease", #"\bdisease"#),
        ("symptom", #"\bsymptom"#),
        ("syndrome", #"\bsyndrome"#),
        ("medication", #"\bmedicat"#),
        ("medicine", #"\bmedicine"#),
        ("prescription", #"\bprescri"#),
        ("therapy", #"\btherap"#),
        ("doctor", #"\bdoctor"#),
        ("physician", #"\bphysician"#),
        ("pathology", #"patholog"#),
        ("arrhythmia", #"arrhythm"#),
        ("tachycardia", #"tachycard"#),
        ("bradycardia", #"bradycard"#),
        ("hypertension", #"hypertens"#),
        ("anxiety", #"\banxi(ety|ous)\b"#),
        ("depression", #"\bdepress(ion|ed|ive)\b"#),
        ("autism", #"\bautis"#),
        ("dementia", #"\bdementia"#),
        ("insomnia", #"\binsomnia"#),
        ("medical advice/condition", #"\bmedical\s+(advice|attention|condition|help|issue|problem|care)"#),
        ("health problem", #"\bhealth\s+(problem|issue|risk|condition)"#),
        ("patient", #"\bpatient"#),
        // Heart rate is described neutrally, never judged.
        ("health judgement", #"\b(un)?healthy\b|\b(good|bad)\s+sign\b"#),
        // The cortex is the content's predicted response for an average brain, never the user's brain.
        ("your brain", #"\byour\s+(predicted\s+)?(brain|mind)\b"#),
    ]

    private static let compiled: [(String, NSRegularExpression)] = bannedPatterns.compactMap { item in
        guard let re = try? NSRegularExpression(pattern: item.pattern, options: [.caseInsensitive]) else { return nil }
        return (item.label, re)
    }

    /// Labels of every banned phrase found in `text` (empty = clean).
    public static func violations(in text: String) -> [String] {
        let range = NSRange(text.startIndex..., in: text)
        return compiled.compactMap { label, re in
            re.firstMatch(in: text, options: [], range: range) != nil ? label : nil
        }
    }

    public static func isClean(_ text: String) -> Bool { violations(in: text).isEmpty }
    public static func isClean(_ texts: [String]) -> Bool { texts.allSatisfy(isClean) }

    // MARK: - Fact grounding

    private static let numberRegex = try! NSRegularExpression(pattern: #"[-+−]?\d+(?:[.,]\d+)?"#)

    /// Every number written in `text` (signs kept, "0.45", "12", "-3").
    public static func numbers(in text: String) -> [Double] {
        let ns = text as NSString
        return numberRegex.matches(in: text, range: NSRange(location: 0, length: ns.length)).compactMap { m in
            let raw = ns.substring(with: m.range)
                .replacingOccurrences(of: "\u{2212}", with: "-")
                .replacingOccurrences(of: "+", with: "")
                .replacingOccurrences(of: ",", with: ".")
            return Double(raw)
        }
    }

    /// Numbers in `text` that do not match (within rounding) any number in `allowed`, sign-insensitive.
    /// Used to reject model output that invents figures not present in the fact list.
    public static func unsupportedNumbers(in text: String, allowed: [Double]) -> [Double] {
        numbers(in: text).filter { x in
            !allowed.contains { a in
                let tol = max(0.51, abs(a) * 0.02)
                return abs(abs(x) - abs(a)) <= tol
            }
        }
    }
}
