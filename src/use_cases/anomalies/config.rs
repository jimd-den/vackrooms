//! User-facing controls for Level 0 anomaly planning and mutation.

use crate::domain::entities::anomaly::AnomalyKind;

/// Anomaly controls are separate from ordinary office-fabric tuning: changing
/// wall or light density must never silently change encounter frequency.
/// Multipliers are centered on `1.0`; distances are world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnomalyTuning {
    /// Overall organic candidate frequency. Debug forcing remains independent.
    pub frequency: f32,
    /// Macro-footprint scale for region-spanning instances.
    pub size: f32,
    pub pillar_expanses: f32,
    pub blackouts: f32,
    pub red_rooms: f32,
    pub pit_lattices: f32,
    /// Strength/probability of epoch-dependent mutable infill.
    pub remap_intensity: f32,
    /// Minimum distance behind a crossed threshold before infill may differ.
    pub remap_distance: f32,
    /// Radius around the player protected from any resident transition.
    pub safe_radius: f32,
    /// Width of the deterministic single Red Room escape. Zero keeps the loop
    /// sealed; one opens the widest authored breach.
    pub red_escape_bias: f32,
    /// Stable archway anchor-room frequency multiplier.
    pub archways: f32,
    /// Fraction of blackout glimmers placed off the recovery skeleton.
    pub blackout_decoys: f32,
    /// Debug-only forced family at the protected point.
    pub forced_kind: Option<AnomalyKind>,
}

impl Default for AnomalyTuning {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            size: 1.0,
            pillar_expanses: 1.0,
            blackouts: 1.0,
            red_rooms: 1.0,
            pit_lattices: 1.0,
            remap_intensity: 1.0,
            remap_distance: 12.0,
            safe_radius: 8.0,
            red_escape_bias: 0.12,
            archways: 1.0,
            // Most glimmers lie: with the recovery skeleton buried under
            // this many decoys, walking out of a blackout is a feat.
            blackout_decoys: 0.85,
            forced_kind: None,
        }
    }
}
