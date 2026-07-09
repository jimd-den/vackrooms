use crate::domain::entities::voxel_grid::{VOXEL_AIR, VOXEL_LIGHT, VOXEL_RED_LIGHT, VoxelGrid};
use std::collections::VecDeque;

/// Warm fluorescent tube color for standard lights (0-15 per channel).
const LIGHT_WARM: [u8; 3] = [15, 14, 11];
/// Deep red exit/alarm light color.
const LIGHT_RED: [u8; 3] = [15, 3, 2];

/// Calculates colored 3D voxel lighting across the dense VoxelGrid using a
/// BFS flood-fill, Rethinking-Voxels style: three independent light channels
/// (RGB, 0-15 each) propagate simultaneously, so a red exit light and a warm
/// ceiling tube blend smoothly where their falloffs overlap instead of
/// snapping at a binary boundary.
/// Time complexity stays O(n) in the number of voxels.
pub fn calculate_voxel_lighting(grid: &mut VoxelGrid) {
    let w = grid.width();
    let h = grid.height();
    let d = grid.depth();

    let falloff: u8 = 1;

    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

    // Pass 1: seed all light sources with their emission color.
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let v = grid.get(x, y, z);
                if v == VOXEL_LIGHT || v == VOXEL_RED_LIGHT {
                    let rgb = if v == VOXEL_RED_LIGHT {
                        LIGHT_RED
                    } else {
                        LIGHT_WARM
                    };
                    grid.set_light_rgb(x, y, z, rgb);
                    queue.push_back((x, y, z));
                }
            }
        }
    }

    // Pass 2: BFS flood-fill, all three channels at once.
    while let Some((cx, cy, cz)) = queue.pop_front() {
        let current = grid.get_light_rgb(cx, cy, cz);
        if current.iter().all(|&c| c <= falloff) {
            continue;
        }

        let neighbors: [(isize, isize, isize); 6] = [
            (cx as isize + 1, cy as isize, cz as isize),
            (cx as isize - 1, cy as isize, cz as isize),
            (cx as isize, cy as isize + 1, cz as isize),
            (cx as isize, cy as isize - 1, cz as isize),
            (cx as isize, cy as isize, cz as isize + 1),
            (cx as isize, cy as isize, cz as isize - 1),
        ];

        for &(nx, ny, nz) in &neighbors {
            if nx >= 0
                && nx < w as isize
                && ny >= 0
                && ny < h as isize
                && nz >= 0
                && nz < d as isize
            {
                let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);

                // Downward propagation has 0 falloff, creating vertical
                // light columns.
                let is_downward = ny < cy;
                let actual_falloff = if is_downward { 0 } else { falloff };
                let next = current.map(|c| c.saturating_sub(actual_falloff));

                if next == [0; 3] {
                    continue;
                }

                let n_light = grid.get_light_rgb(nx, ny, nz);
                // Component-wise max: overlapping colored lights blend.
                let merged = [
                    n_light[0].max(next[0]),
                    n_light[1].max(next[1]),
                    n_light[2].max(next[2]),
                ];

                if merged != n_light {
                    grid.set_light_rgb(nx, ny, nz, merged);
                    // Only continue propagating through air.
                    if grid.get(nx, ny, nz) == VOXEL_AIR {
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
        assert_eq!(grid.get_light(3, 1, 1), 0); // Behind the wall is out of bounds, so it returns 0 anyway
    }

    #[test]
    fn warm_light_is_tinted_not_white() {
        let mut grid = VoxelGrid::new(3, 3, 3);
        grid.set(1, 1, 1, VOXEL_LIGHT);
        calculate_voxel_lighting(&mut grid);

        let [r, g, b] = grid.get_light_rgb(1, 1, 1);
        assert_eq!([r, g, b], LIGHT_WARM);
        // Propagation preserves relative warmth.
        let [nr, _, nb] = grid.get_light_rgb(0, 1, 1);
        assert!(nr > nb, "red channel must stay above blue: {nr} vs {nb}");
    }

    #[test]
    fn overlapping_colored_lights_blend_component_wise() {
        let mut grid = VoxelGrid::new(5, 1, 1);
        grid.set(0, 0, 0, VOXEL_LIGHT);
        grid.set(4, 0, 0, VOXEL_RED_LIGHT);
        calculate_voxel_lighting(&mut grid);

        // The midpoint sees red strongly from the red light (distance 2:
        // 15-2=13) and green/blue mostly from the warm light.
        let [r, g, b] = grid.get_light_rgb(2, 0, 0);
        assert_eq!(r, 13, "red channel from the nearer-in-red source");
        assert_eq!(g, 12, "green channel from the warm source (14-2)");
        assert_eq!(b, 9, "blue channel from the warm source (11-2)");
        assert!(grid.get_light(2, 0, 0) == 13, "scalar is the max channel");
    }
}
