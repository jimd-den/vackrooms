//! Level 34-B — "The Grassland".
//!
//! A safe, flat, endless grass plane you can noclip into from Level 0:
//! * grass ground everywhere, with smooth-noise **lakes** of shallow water,
//! * scattered **trees** (solid trunk, leaf canopy) on a hashed world grid,
//! * a glowing **sky layer** at the top of the grid plus dim ground glow
//!   sources hidden inside the turf, so the plane reads as daylight even
//!   though lighting is the same BFS flood used indoors.
//!
//! Like every [`LevelGenerator`], all placement is a pure function of
//! world-space coordinates so chunks tile seamlessly.

use crate::domain::entities::voxel_grid::{
    VOXEL_GRASS, VOXEL_LIGHT, VOXEL_TREE, VOXEL_WATER, VoxelGrid,
};
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::level_generator::LevelGenerator;
use crate::use_cases::ports::NoiseProvider;

/// Grid height in world units; the sky glow layer sits at the top.
const GRID_HEIGHT_UNITS: f32 = 5.8;
/// Trees sit on a world grid of this period (units); most cells are empty.
const TREE_PERIOD: f32 = 9.0;
/// Ground glow sources ("sun-lit turf") every N units.
const GLOW_PERIOD: f32 = 3.2;

pub struct GrasslandLevel;

impl GrasslandLevel {
    /// Uniform per-cell hash in [0, 1): samples the value noise exactly on
    /// its lattice (provider frequency 0.05 → inputs that are multiples of
    /// 20 hit lattice points, where the noise is a raw uniform hash).
    fn hash(noise: &dyn NoiseProvider, seed: u32, salt: u32, cx: i64, cz: i64) -> f32 {
        let v = noise.evaluate_2d(
            seed ^ salt,
            Position::new(cx as f32 * 20.0, cz as f32 * 20.0),
        );
        (v * 0.5 + 0.5).clamp(0.0, 0.999)
    }
}

impl LevelGenerator for GrasslandLevel {
    fn generate(
        &self,
        chunk_pos: Position,
        seed: u32,
        config: GeneratorConfig,
        noise: &dyn NoiseProvider,
    ) -> VoxelGrid {
        let s = config.voxel_scale;
        let width = (config.chunk_size / s).round() as usize;
        let depth = (config.chunk_size / s).round() as usize;
        let height = (GRID_HEIGHT_UNITS / s) as usize;
        let mut grid = VoxelGrid::new(width, height, depth);
        let sky_y = height - 1;

        for z in 0..depth {
            for x in 0..width {
                let wx = chunk_pos.x + (x as f32 + 0.5) * s;
                let wz = chunk_pos.z + (z as f32 + 0.5) * s;

                // Lakes: smooth blobs of the world-space noise field.
                let lake = noise.evaluate_2d(seed ^ 0x1A5E, Position::new(wx * 0.7, wz * 0.7));
                let ground = if lake > 0.62 {
                    VOXEL_WATER
                } else {
                    VOXEL_GRASS
                };
                grid.set(x, 0, z, ground);

                // Sky glow: an unbroken luminous layer overhead.
                grid.set(x, sky_y, z, VOXEL_LIGHT);

                // Ground glow grid keeps the turf bright at eye level (the
                // BFS light from the sky layer alone fades before the floor).
                let gx = wx.rem_euclid(GLOW_PERIOD);
                let gz = wz.rem_euclid(GLOW_PERIOD);
                if ground == VOXEL_GRASS && gx < s && gz < s {
                    grid.set(x, 0, z, VOXEL_LIGHT);
                }

                // Trees: hashed world cells, never in water.
                let cell_x = (wx / TREE_PERIOD).floor() as i64;
                let cell_z = (wz / TREE_PERIOD).floor() as i64;
                if Self::hash(noise, seed, 0x7EE5, cell_x, cell_z) > 0.6 && ground == VOXEL_GRASS {
                    // Trunk position jittered inside the cell.
                    let jx =
                        Self::hash(noise, seed, 0x7EE6, cell_x, cell_z) * (TREE_PERIOD - 2.0) + 1.0;
                    let jz =
                        Self::hash(noise, seed, 0x7EE7, cell_x, cell_z) * (TREE_PERIOD - 2.0) + 1.0;
                    let tx = cell_x as f32 * TREE_PERIOD + jx;
                    let tz = cell_z as f32 * TREE_PERIOD + jz;
                    let dx = wx - tx;
                    let dz = wz - tz;
                    let d2 = dx * dx + dz * dz;

                    let trunk_top = (2.2 / s) as usize;
                    let canopy_top = (3.6 / s) as usize;
                    if d2 < (0.25f32).powi(2) {
                        for y in 1..trunk_top.min(height - 2) {
                            grid.set(x, y, z, VOXEL_TREE);
                        }
                    }
                    if d2 < (1.6f32).powi(2) {
                        for y in trunk_top..canopy_top.min(height - 2) {
                            grid.set(x, y, z, VOXEL_GRASS);
                        }
                    }
                }
            }
        }

        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frameworks_drivers::simple_noise::SimpleNoiseProvider;

    fn generate(ox: f32, oz: f32) -> VoxelGrid {
        GrasslandLevel.generate(
            Position::new(ox, oz),
            42,
            GeneratorConfig::low_spec(),
            &SimpleNoiseProvider::new(),
        )
    }

    #[test]
    fn grassland_has_grass_sky_and_open_walkway() {
        let grid = generate(0.0, 0.0);
        let sky_y = grid.height() - 1;
        let mut grass = 0;
        let mut open_at_eye = 0;
        for z in 0..grid.depth() {
            for x in 0..grid.width() {
                assert_eq!(grid.get(x, sky_y, z), VOXEL_LIGHT, "sky layer must glow");
                let g = grid.get(x, 0, z);
                if g == VOXEL_GRASS || g == VOXEL_LIGHT {
                    grass += 1;
                }
                if grid.get(x, 8, z) == 0 {
                    open_at_eye += 1; // eye height 1.7 u = voxel y 8
                }
            }
        }
        let total = grid.width() * grid.depth();
        assert!(grass > total / 2, "mostly grass, got {grass}/{total}");
        assert!(
            open_at_eye > total * 9 / 10,
            "plane must be open at eye height"
        );
    }

    #[test]
    fn grassland_scatters_trees_deterministically() {
        let a = generate(0.0, 0.0);
        let b = generate(0.0, 0.0);
        let mut trees = 0;
        for z in 0..a.depth() {
            for x in 0..a.width() {
                assert_eq!(a.get(x, 3, z), b.get(x, 3, z), "must be deterministic");
                if a.get(x, 3, z) == VOXEL_TREE {
                    trees += 1;
                }
            }
        }
        // Trees are sparse but should exist across a few chunks.
        let mut total_trees = trees;
        for (ox, oz) in [
            (10.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (-10.0, 0.0),
            (0.0, -10.0),
            (-10.0, -10.0),
            (20.0, 0.0),
            (0.0, 20.0),
        ] {
            let g = generate(ox, oz);
            for z in 0..g.depth() {
                for x in 0..g.width() {
                    if g.get(x, 3, z) == VOXEL_TREE {
                        total_trees += 1;
                    }
                }
            }
        }
        assert!(
            total_trees > 0,
            "expected at least one tree trunk in 4 chunks"
        );
    }
}
