use crate::domain::entities::voxel_grid::{VOXEL_AIR, VoxelGrid};

use super::propagate_through_air::{AirPropagation, UNREACHABLE};
use super::settings::VoxelLightingSettings;
use super::voxel_neighborhood::for_each_voxel;
use super::world_space_attenuation::{accumulate_attenuated, pack_rgb};

pub(crate) fn write_air_irradiance(
    grid: &mut VoxelGrid,
    propagated: &AirPropagation,
    settings: VoxelLightingSettings,
) {
    for_each_voxel(propagated.dimensions, |coord| {
        if grid.get(coord.x, coord.y, coord.z) != VOXEL_AIR {
            return;
        }
        let index = propagated.dimensions.index(coord);
        let mut irradiance = [0.0; 3];
        for emission in &propagated.emissions {
            let distance_steps = emission.distance_steps[index];
            if distance_steps != UNREACHABLE {
                accumulate_attenuated(
                    &mut irradiance,
                    emission.profile,
                    distance_steps,
                    0.5,
                    settings,
                );
            }
        }
        grid.set_light_rgb(coord.x, coord.y, coord.z, pack_rgb(irradiance));
    });
}
