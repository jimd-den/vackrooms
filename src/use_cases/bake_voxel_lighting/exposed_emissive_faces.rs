use crate::domain::entities::voxel_grid::{
    VOXEL_AIR, VOXEL_GLIMMER, VOXEL_LIGHT, VOXEL_RED_LIGHT, VoxelGrid,
};

use super::voxel_neighborhood::{GridDimensions, VoxelCoord, for_each_voxel};

/// One spectral emission class in the compact 0--15 bake domain.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EmissionProfile {
    pub(crate) voxel_type: u8,
    pub(crate) rgb: [f32; 3],
}

pub(crate) const EMISSION_PROFILES: [EmissionProfile; 3] = [
    EmissionProfile {
        voxel_type: VOXEL_LIGHT,
        rgb: [15.0, 14.0, 11.0],
    },
    EmissionProfile {
        voxel_type: VOXEL_RED_LIGHT,
        rgb: [15.0, 3.0, 2.0],
    },
    EmissionProfile {
        voxel_type: VOXEL_GLIMMER,
        rgb: [3.0, 5.0, 7.0],
    },
];

pub(crate) fn emission_rgb(voxel_type: u8) -> Option<[f32; 3]> {
    EMISSION_PROFILES
        .iter()
        .find(|profile| profile.voxel_type == voxel_type)
        .map(|profile| profile.rgb)
}

/// Returns the air-cell centers immediately below exposed fixture undersides.
///
/// The voxel materials in [`EMISSION_PROFILES`] describe ceiling-style
/// fixtures: ordinary fluorescent panels, red panels, and dim emergency
/// glimmers. They all emit through their downward face. A material placed on
/// the floor (`y == 0`), buried below a solid, or exposed only on a side is
/// still visible geometry, but it does not seed the diffuse-fill field. This
/// is the same one-sided source contract used by analytic scene lights.
pub(crate) fn exposed_downward_air_cells(
    grid: &VoxelGrid,
    dimensions: GridDimensions,
    profile: EmissionProfile,
) -> Vec<VoxelCoord> {
    let mut exposed = Vec::new();
    for_each_voxel(dimensions, |coord| {
        if let Some(air_cell) = exposed_air_cell_below(grid, coord, profile.voxel_type) {
            exposed.push(air_cell);
        }
    });
    exposed
}

/// Identifies the single face through which a ceiling fixture may seed the
/// approximate diffuse-fill field. Keeping this predicate separate makes the
/// orientation and boundary behavior explicit and independently testable.
fn exposed_air_cell_below(
    grid: &VoxelGrid,
    emitter: VoxelCoord,
    expected_material: u8,
) -> Option<VoxelCoord> {
    if emitter.y == 0 || grid.get(emitter.x, emitter.y, emitter.z) != expected_material {
        return None;
    }

    let air_cell = VoxelCoord {
        y: emitter.y - 1,
        ..emitter
    };
    (grid.get(air_cell.x, air_cell.y, air_cell.z) == VOXEL_AIR).then_some(air_cell)
}
