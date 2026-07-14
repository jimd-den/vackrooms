use crate::domain::entities::voxel_grid::{VOXEL_AIR, VoxelGrid};

use super::exposed_emissive_faces::emission_rgb;
use super::propagate_through_air::{AirPropagation, UNREACHABLE};
use super::settings::VoxelLightingSettings;
use super::voxel_neighborhood::{for_each_face_neighbor, for_each_voxel};
use super::world_space_attenuation::{accumulate_attenuated, pack_emission, pack_rgb};

/// Copies incident air irradiance onto the first occupied voxel at a surface.
/// Receivers are never fed back into propagation, which is the invariant that
/// prevents a lit wall from leaking energy into the room behind it.
pub(crate) fn project_onto_solid_receivers(
    grid: &mut VoxelGrid,
    propagated: &AirPropagation,
    settings: VoxelLightingSettings,
) {
    for_each_voxel(propagated.dimensions, |coord| {
        let voxel_type = grid.get(coord.x, coord.y, coord.z);
        if voxel_type == VOXEL_AIR {
            return;
        }

        if let Some(rgb) = emission_rgb(voxel_type) {
            grid.set_light_rgb(coord.x, coord.y, coord.z, pack_emission(rgb));
            return;
        }

        let mut irradiance = [0.0; 3];
        for emission in &propagated.emissions {
            let mut nearest_air_steps = UNREACHABLE;
            for_each_face_neighbor(coord, propagated.dimensions, |neighbor| {
                if grid.get(neighbor.x, neighbor.y, neighbor.z) != VOXEL_AIR {
                    return;
                }
                nearest_air_steps = nearest_air_steps
                    .min(emission.distance_steps[propagated.dimensions.index(neighbor)]);
            });
            if nearest_air_steps != UNREACHABLE {
                accumulate_attenuated(
                    &mut irradiance,
                    emission.profile,
                    nearest_air_steps,
                    1.0,
                    settings,
                );
            }
        }
        grid.set_light_rgb(coord.x, coord.y, coord.z, pack_rgb(irradiance));
    });
}
