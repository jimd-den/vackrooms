use crate::domain::entities::grid::Grid;
use crate::domain::entities::voxel_grid::{
    VoxelGrid, VOXEL_AIR, VOXEL_WALL, VOXEL_FLOOR, VOXEL_CEILING, VOXEL_LIGHT, VOXEL_RED_WALL
};

/// Strategy Pattern for generating dense VoxelGrids from abstract room Grids.
/// This fulfills the "engine should be extensible for other backrooms levels" rule.
pub trait LevelBuilder {
    fn build_voxels(&self, grid: &Grid) -> VoxelGrid;
}

/// Concrete implementation for Backrooms Level 0.
/// Uses smaller voxels (e.g. 8x8x4 per room) to allow realistic light sources.
pub struct Level0Builder {
    pub room_size: usize,
    pub wall_height: usize,
}

impl LevelBuilder for Level0Builder {
    fn build_voxels(&self, grid: &Grid) -> VoxelGrid {
        let rs = self.room_size;
        let wh = self.wall_height;
        
        // Voxel grid dimensions
        let vw = grid.width() * rs;
        let vh = wh + 3; // Floor (1) + Walls (wh) + Ceiling (1) + Recessed Lights (1)
        let vd = grid.depth() * rs;
        
        let mut voxels = VoxelGrid::new(vw, vh, vd);

        for gy in 0..grid.depth() {
            for gx in 0..grid.width() {
                let cell = grid.get(gx, gy).unwrap();
                let base_x = gx * rs;
                let base_z = gy * rs;

                for lx in 0..rs {
                    for lz in 0..rs {
                        let vx = base_x + lx;
                        let vz = base_z + lz;
                        
                        // Floor
                        voxels.set(vx, 0, vz, VOXEL_FLOOR);

                        // Walls
                        let is_north_wall = lz == 0 && cell.walls[0];
                        let is_east_wall = lx == rs - 1 && cell.walls[1];
                        let is_south_wall = lz == rs - 1 && cell.walls[2];
                        let is_west_wall = lx == 0 && cell.walls[3];

                        let wall_type = if cell.is_red { VOXEL_RED_WALL } else { VOXEL_WALL };

                        if is_north_wall || is_east_wall || is_south_wall || is_west_wall {
                            for h in 1..=wh {
                                voxels.set(vx, h, vz, wall_type);
                            }
                        }

                        // Checkerboard fluorescent lights (Realistic light sources)
                        // With 16x16 rooms, we can model actual recessed 2x6 fluorescent fixtures.
                        let mut is_light = false;
                        if !cell.is_dark {
                            // Two fixtures per room
                            let fixture1 = lx >= 4 && lx < 6 && lz >= 5 && lz < 11;
                            let fixture2 = lx >= 10 && lx < 12 && lz >= 5 && lz < 11;
                            if fixture1 || fixture2 {
                                is_light = true;
                            }
                        }

                        if is_light {
                            // Recessed light: hole in the ceiling, light block above it
                            voxels.set(vx, vh - 2, vz, VOXEL_AIR); // hole
                            voxels.set(vx, vh - 1, vz, VOXEL_LIGHT); // light emitting block above
                        } else {
                            voxels.set(vx, vh - 2, vz, VOXEL_CEILING);
                        }
                    }
                }
            }
        }
        
        voxels
    }
}
