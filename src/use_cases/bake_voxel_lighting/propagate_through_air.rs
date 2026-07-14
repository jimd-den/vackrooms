use std::collections::VecDeque;

use crate::domain::entities::voxel_grid::{VOXEL_AIR, VoxelGrid};

use super::exposed_emissive_faces::{EMISSION_PROFILES, EmissionProfile, exposed_air_cells};
use super::settings::VoxelLightingSettings;
use super::voxel_neighborhood::{GridDimensions, for_each_face_neighbor};

pub(crate) const UNREACHABLE: u32 = u32::MAX;

pub(crate) struct PropagatedEmission {
    pub(crate) profile: EmissionProfile,
    /// Shortest count of air-cell center transitions from an exposed seed.
    pub(crate) distance_steps: Vec<u32>,
}

pub(crate) struct AirPropagation {
    pub(crate) dimensions: GridDimensions,
    pub(crate) emissions: Vec<PropagatedEmission>,
}

/// Finds shortest face-connected paths through air for every emission class.
///
/// A class uses one multi-source breadth-first search. This means equally
/// colored panels form a nearest-emitter envelope instead of multiplying
/// energy merely because a large panel occupies several source voxels.
pub(crate) fn propagate_through_air(
    grid: &VoxelGrid,
    settings: VoxelLightingSettings,
) -> AirPropagation {
    let dimensions = GridDimensions::of(grid);
    let mut emissions = Vec::with_capacity(EMISSION_PROFILES.len());

    for profile in EMISSION_PROFILES {
        let mut distance_steps = vec![UNREACHABLE; dimensions.volume()];
        let mut queue = VecDeque::new();

        // An exposed air center is half a voxel from the emitting face. Do
        // not inject a sample that is already beyond the requested range.
        if settings.voxel_size_world_units() * 0.5 < settings.max_range_world_units() {
            for seed in exposed_air_cells(grid, dimensions, profile) {
                let index = dimensions.index(seed);
                if distance_steps[index] == UNREACHABLE {
                    distance_steps[index] = 0;
                    queue.push_back(seed);
                }
            }
        }

        while let Some(coord) = queue.pop_front() {
            let current_steps = distance_steps[dimensions.index(coord)];
            let next_steps = current_steps + 1;
            let next_center_distance =
                (next_steps as f32 + 0.5) * settings.voxel_size_world_units();
            if next_center_distance >= settings.max_range_world_units() {
                continue;
            }

            for_each_face_neighbor(coord, dimensions, |neighbor| {
                if grid.get(neighbor.x, neighbor.y, neighbor.z) != VOXEL_AIR {
                    return;
                }
                let neighbor_index = dimensions.index(neighbor);
                if next_steps < distance_steps[neighbor_index] {
                    distance_steps[neighbor_index] = next_steps;
                    queue.push_back(neighbor);
                }
            });
        }

        emissions.push(PropagatedEmission {
            profile,
            distance_steps,
        });
    }

    AirPropagation {
        dimensions,
        emissions,
    }
}
