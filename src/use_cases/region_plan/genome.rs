//! Architect genomes: the designers a region inherits.
//!
//! `salt` 0 = dominant, 1 = renovator, 2 = intruder. The circulation style
//! comes from low-frequency noise so neighboring regions tend to share a
//! culture; every other property is hash-derived, but *not independently*:
//! a real designer's decisions form a chain, not a set of parallel dice
//! rolls. `structural_system` and `renovation_history` are decided first;
//! everything downstream of them (room proportions, ceiling language,
//! threshold language, furnishing density, tolerance for symmetry) biases
//! toward what that earlier choice implies, with its own hash key still
//! supplying the roll *within* the biased range. A `CoreAndShell` designer
//! who chose that system to achieve big clear-span rooms should get big
//! rooms and a ceiling that doesn't pretend to hide beams it doesn't have;
//! a `DeepSpansWithBeams` designer's ceiling should show the beams that
//! justify the system's name.

use crate::domain::entities::architecture::*;
use crate::entities::models::Position;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::world_topology::hash01;

use super::{REGION_SIZE, pick_index};

/// Derive one designer for a region. `salt` 0 = dominant, 1 = renovator,
/// 2 = intruder. The circulation style comes from low-frequency noise so
/// neighboring regions tend to share a culture; everything else hashes.
pub fn derive_genome(
    seed: u32,
    rx: i64,
    rz: i64,
    salt: i64,
    noise: &dyn NoiseProvider,
) -> ArchitectGenome {
    let h = |k: i64| hash01(seed, &[0x6E0 + salt, rx, rz, k]);
    // ~130 u wavelength: cultures span a few regions.
    let culture = noise.evaluate_2d(
        seed ^ 0xA11C_0DE ^ (salt as u32),
        Position::new(
            (rx as f32 + 0.5) * REGION_SIZE * 0.15,
            (rz as f32 + 0.5) * REGION_SIZE * 0.15,
        ),
    );
    let circulation = match ((culture * 0.5 + 0.5).clamp(0.0, 0.999) * 4.0) as u32 {
        0 => CirculationStyle::StraightSpine,
        1 => CirculationStyle::MeanderingSpine,
        2 => CirculationStyle::PairedBranches,
        _ => CirculationStyle::TreeWithCulDeSacs,
    };
    // ---- site strategy: decided first, everything else answers to it ----
    let structural_system = *[
        StructuralSystem::RegularGrid,
        StructuralSystem::DeepSpansWithBeams,
        StructuralSystem::OffsetGrid,
        StructuralSystem::CoreAndShell,
    ]
    .get(pick_index(h(1), 4))
    .unwrap();
    let renovation_history = *[
        RenovationStyle::Untouched,
        RenovationStyle::PartialRefit,
        RenovationStyle::LayeredRefits,
    ]
    .get(pick_index(h(5), 3))
    .unwrap();

    // ---- consequences of the structural system --------------------------
    // CoreAndShell and DeepSpansWithBeams are chosen specifically to permit
    // large clear-span rooms (see `suites::on_column`, which gives
    // CoreAndShell zero interior columns); RegularGrid/OffsetGrid bound
    // their rooms to a small structural module instead. Room size is a
    // consequence of that choice, not a second independent roll.
    let spans_wide = matches!(
        structural_system,
        StructuralSystem::CoreAndShell | StructuralSystem::DeepSpansWithBeams
    );
    let (min_lo, min_span, max_lo, max_span) = if spans_wide {
        (16.0, 6.0, 24.0, 10.0)
    } else {
        (12.0, 4.0, 18.0, 6.0)
    };

    // A coffered or exposed-soffit ceiling is a description of what a deep
    // beam grid looks like from underneath, not an unrelated stylistic
    // choice — so it should follow from having beams to show. CoreAndShell
    // has no beam grid to express at all; flat tiles are the honest default
    // for it. RegularGrid/OffsetGrid are unbiased: either reads as fine for
    // an ordinary column grid.
    let ceiling_language = match structural_system {
        StructuralSystem::DeepSpansWithBeams => match h(3) {
            v if v < 0.55 => CeilingLanguage::Coffered,
            v if v < 0.85 => CeilingLanguage::ExposedSoffit,
            _ => CeilingLanguage::FlatTiles,
        },
        StructuralSystem::CoreAndShell => match h(3) {
            v if v < 0.70 => CeilingLanguage::FlatTiles,
            v if v < 0.90 => CeilingLanguage::ExposedSoffit,
            _ => CeilingLanguage::Coffered,
        },
        StructuralSystem::RegularGrid | StructuralSystem::OffsetGrid => *[
            CeilingLanguage::FlatTiles,
            CeilingLanguage::Coffered,
            CeilingLanguage::ExposedSoffit,
        ]
        .get(pick_index(h(3), 3))
        .unwrap(),
    };

    // A grid that never breaks its own module (Regular, CoreAndShell) reads
    // as more deliberately symmetric than one that already contradicts
    // itself every other row (OffsetGrid).
    let symmetry_bias = match structural_system {
        StructuralSystem::RegularGrid | StructuralSystem::CoreAndShell => 0.25,
        StructuralSystem::OffsetGrid => -0.20,
        StructuralSystem::DeepSpansWithBeams => 0.0,
    };

    // ---- consequences of the renovation history --------------------------
    // A framed office door is now an anomaly. The chosen language is only a
    // bias; each assembly still picks its own threshold so a whole region
    // never turns into a row of identical openings. A designer who left
    // layered refits behind also left wider, less consistent thresholds and
    // denser leftover furnishing than a plan nobody ever touched.
    let refit_bias = match renovation_history {
        RenovationStyle::Untouched => 0.0,
        RenovationStyle::PartialRefit => 0.12,
        RenovationStyle::LayeredRefits => 0.25,
    };
    let threshold_language = match (h(2) + refit_bias).min(0.999) {
        v if v < 0.10 => ThresholdLanguage::DoorWithLintel,
        v if v < 0.55 => ThresholdLanguage::OpenPortal,
        _ => ThresholdLanguage::WidePortal,
    };

    let lighting_language = *[
        LightingLanguage::GridPanels,
        LightingLanguage::GridPanels,
        LightingLanguage::StripsAlongCirculation,
        LightingLanguage::SparsePendants,
    ]
    .get(pick_index(h(4), 4))
    .unwrap();

    ArchitectGenome {
        circulation,
        structural_system,
        room_proportions: ProportionRules {
            // Level 0 reads as broken retail-backroom masses, not a cubicle
            // tiling. A suite frontage normally survives 12--24 u before a
            // meaningful interruption, wider still where the structure
            // permits it.
            min_side: min_lo + min_span * h(6),
            max_side: max_lo + max_span * h(7),
            elongation: h(8),
        },
        threshold_language,
        ceiling_language,
        lighting_language,
        furnishing_density: (0.3 + 0.7 * h(9) + refit_bias).min(1.0),
        renovation_history,
        tolerance_for_symmetry: (h(10) + symmetry_bias).clamp(0.0, 1.0),
    }
}
