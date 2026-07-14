//! Architect genomes: the designers a region inherits.
//!
//! `salt` 0 = dominant, 1 = renovator, 2 = intruder. The circulation style
//! comes from low-frequency noise so neighboring regions tend to share a
//! culture; everything else hashes.

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
    let structural_system = *[
        StructuralSystem::RegularGrid,
        StructuralSystem::DeepSpansWithBeams,
        StructuralSystem::OffsetGrid,
        StructuralSystem::CoreAndShell,
    ]
    .get(pick_index(h(1), 4))
    .unwrap();
    // A framed office door is now an anomaly. The chosen language is only a
    // bias; each assembly still picks its own threshold so a whole region
    // never turns into a row of identical openings.
    let threshold_language = match h(2) {
        v if v < 0.10 => ThresholdLanguage::DoorWithLintel,
        v if v < 0.55 => ThresholdLanguage::OpenPortal,
        _ => ThresholdLanguage::WidePortal,
    };
    let ceiling_language = *[
        CeilingLanguage::FlatTiles,
        CeilingLanguage::Coffered,
        CeilingLanguage::ExposedSoffit,
    ]
    .get(pick_index(h(3), 3))
    .unwrap();
    let lighting_language = *[
        LightingLanguage::GridPanels,
        LightingLanguage::GridPanels,
        LightingLanguage::StripsAlongCirculation,
        LightingLanguage::SparsePendants,
    ]
    .get(pick_index(h(4), 4))
    .unwrap();
    let renovation_history = *[
        RenovationStyle::Untouched,
        RenovationStyle::PartialRefit,
        RenovationStyle::LayeredRefits,
    ]
    .get(pick_index(h(5), 3))
    .unwrap();
    ArchitectGenome {
        circulation,
        structural_system,
        room_proportions: ProportionRules {
            // Level 0 reads as broken retail-backroom masses, not a cubicle
            // tiling. A suite frontage normally survives 12--24 u before a
            // meaningful interruption.
            min_side: 12.0 + 4.0 * h(6),
            max_side: 20.0 + 8.0 * h(7),
            elongation: h(8),
        },
        threshold_language,
        ceiling_language,
        lighting_language,
        furnishing_density: 0.3 + 0.7 * h(9),
        renovation_history,
        tolerance_for_symmetry: h(10),
    }
}
