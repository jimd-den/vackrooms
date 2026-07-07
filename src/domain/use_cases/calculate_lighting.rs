use crate::domain::entities::voxel_grid::{VoxelGrid, VOXEL_AIR, VOXEL_LIGHT};
use std::collections::VecDeque;

/// Calculates 3D voxel lighting across the dense VoxelGrid using a BFS flood-fill algorithm.
/// Time Complexity: O(n) where n is the number of voxels.
/// Achieves pure voxel cast lighting.
pub fn calculate_voxel_lighting(grid: &mut VoxelGrid) {
    let w = grid.width();
    let h = grid.height();
    let d = grid.depth();
    
    let max_light: u8 = 15;
    let falloff: u8 = 1;

    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

    // Pass 1: Find all light sources
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                if grid.get(x, y, z) == VOXEL_LIGHT {
                    grid.set_light(x, y, z, max_light);
                    queue.push_back((x, y, z));
                }
            }
        }
    }

    // Pass 2: BFS flood-fill 3D
    while let Some((cx, cy, cz)) = queue.pop_front() {
        let current_light = grid.get_light(cx, cy, cz);
        if current_light <= falloff {
            continue;
        }

        let next_light = current_light - falloff;

        // Check 6 directions
        let neighbors: [(isize, isize, isize); 6] = [
            (cx as isize + 1, cy as isize, cz as isize),
            (cx as isize - 1, cy as isize, cz as isize),
            (cx as isize, cy as isize + 1, cz as isize),
            (cx as isize, cy as isize - 1, cz as isize),
            (cx as isize, cy as isize, cz as isize + 1),
            (cx as isize, cy as isize, cz as isize - 1),
        ];

        for &(nx, ny, nz) in &neighbors {
            if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize && nz >= 0 && nz < d as isize {
                let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                
                // Light only travels through Air (or maybe we allow it to light up solid faces)
                // We want to light up the faces of solid blocks! So we update the light level of ANY block.
                // But light only PROPAGATES through AIR!
                
                let n_type = grid.get(nx, ny, nz);
                let n_light = grid.get_light(nx, ny, nz);
                
                if n_light < next_light {
                    grid.set_light(nx, ny, nz, next_light);
                    // Only continue propagating if it's air
                    if n_type == VOXEL_AIR {
                        queue.push_back((nx, ny, nz));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_propagation() {
        let mut grid = VoxelGrid::new(3, 3, 3);
        
        // Setup:
        // A single light in the middle
        grid.set(1, 1, 1, VOXEL_LIGHT);
        // A wall blocking light
        grid.set(2, 1, 1, crate::domain::entities::voxel_grid::VOXEL_WALL);
        
        calculate_voxel_lighting(&mut grid);
        
        assert_eq!(grid.get_light(1, 1, 1), 15);
        assert_eq!(grid.get_light(1, 2, 1), 14); // Above
        assert_eq!(grid.get_light(2, 1, 1), 14); // Wall itself lights up
        assert_eq!(grid.get_light(3, 1, 1), 0);  // Behind the wall is out of bounds, so it returns 0 anyway
    }
}
