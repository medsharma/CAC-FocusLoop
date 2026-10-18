import SwiftUI

/// Yeo et al. (2011) 7 functional networks, in the order used by `BrainMesh.bin`,
/// `BrainAtlas.json`, `<lecture>.brain.json` and the shader's activation vector.
public enum BrainNetwork: Int, CaseIterable, Identifiable {
    case visual, somatomotor, dorsalAttention, ventralAttention, limbic, control, defaultMode

    public var id: Int { rawValue }

    /// Key used in `<lecture>.brain.json` (Schaefer short names).
    public var key: String {
        ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"][rawValue]
    }

    public var displayName: String {
        ["Visual", "Somatomotor", "Dorsal Attention", "Salience / Ventral Attention",
         "Limbic", "Frontoparietal Control", "Default Mode"][rawValue]
    }

    /// Canonical Yeo-7 atlas colors (for legends and lanes; the cortex itself uses the heat ramp).
    public var color: Color {
        let rgb: [(Double, Double, Double)] = [(120, 18, 134), (70, 130, 180), (0, 118, 14),
                                               (196, 58, 250), (220, 248, 164), (230, 148, 34),
                                               (205, 62, 78)]
        let c = rgb[rawValue]
        return Color(red: c.0 / 255, green: c.1 / 255, blue: c.2 / 255)
    }
}
