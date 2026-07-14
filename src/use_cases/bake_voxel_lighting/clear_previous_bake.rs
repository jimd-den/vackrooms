use crate::domain::entities::voxel_grid::VoxelGrid;

use super::voxel_neighborhood::{GridDimensions, for_each_voxel};

pub(crate) fn clear_previous_bake(grid: &mut VoxelGrid) {
    let dimensions = GridDimensions::of(grid);
    for_each_voxel(dimensions, |coord| {
        grid.set_light_rgb(coord.x, coord.y, coord.z, [0; 3]);
    });
}
