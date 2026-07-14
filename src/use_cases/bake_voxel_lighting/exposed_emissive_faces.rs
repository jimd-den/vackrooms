use crate::domain::entities::voxel_grid::{
    VOXEL_AIR, VOXEL_GLIMMER, VOXEL_LIGHT, VOXEL_RED_LIGHT, VoxelGrid,
};

use super::voxel_neighborhood::{
    GridDimensions, VoxelCoord, for_each_face_neighbor, for_each_voxel,
};

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

/// Returns the air cells touching at least one face of this emissive class.
/// An emissive voxel buried in geometry deliberately contributes no seeds.
pub(crate) fn exposed_air_cells(
    grid: &VoxelGrid,
    dimensions: GridDimensions,
    profile: EmissionProfile,
) -> Vec<VoxelCoord> {
    let mut exposed = Vec::new();
    for_each_voxel(dimensions, |coord| {
        if grid.get(coord.x, coord.y, coord.z) != profile.voxel_type {
            return;
        }
        for_each_face_neighbor(coord, dimensions, |neighbor| {
            if grid.get(neighbor.x, neighbor.y, neighbor.z) == VOXEL_AIR {
                exposed.push(neighbor);
            }
        });
    });
    exposed
}
