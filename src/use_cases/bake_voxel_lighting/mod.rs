//! Bake finite-range voxel irradiance from the scene's emissive geometry.
//!
//! This use case deliberately separates four physically meaningful stages:
//!
//! 1. erase the previous bake;
//! 2. inject emission through emissive faces that actually border air;
//! 3. find shortest unobstructed paths through air in world-space units;
//! 4. project that air irradiance onto the first solid surface it reaches.
//!
//! Solids are receivers, never propagation nodes. A wall can therefore be lit
//! on its near side without becoming a light source for the room behind it.
//! The working calculation is floating point; quantization happens only when
//! the result is written back to the grid's established RGB 0--15 contract.

mod clear_previous_bake;
mod exposed_emissive_faces;
mod project_onto_solid_receivers;
mod propagate_through_air;
mod settings;
mod voxel_neighborhood;
mod world_space_attenuation;
mod write_air_irradiance;

use crate::domain::entities::voxel_grid::VoxelGrid;

pub use settings::{
    DEFAULT_MAX_LIGHT_RANGE_WORLD_UNITS, InvalidVoxelLightingSettings, VoxelLightingSettings,
};

/// Replaces every prior light sample in `grid` with a deterministic bake.
///
/// Attenuation is symmetric in all six cardinal directions. Its independent
/// variable is shortest unobstructed path length in world units, so changing
/// voxel resolution does not silently change a fixture's physical reach.
pub fn bake_voxel_lighting(grid: &mut VoxelGrid, settings: VoxelLightingSettings) {
    clear_previous_bake::clear_previous_bake(grid);

    let propagated = propagate_through_air::propagate_through_air(grid, settings);
    write_air_irradiance::write_air_irradiance(grid, &propagated, settings);
    project_onto_solid_receivers::project_onto_solid_receivers(grid, &propagated, settings);
}

#[cfg(test)]
mod tests;
