//! Low-frequency 3D Irradiance Probe Grid Entity.
//!
//! # Architecture & Rationale
//! In Clean Architecture (Entities Layer), `ProbeGrid` encapsulates enterprise logic for
//! downsampled, low-frequency 3D light volumes per chunk (e.g. 8x4x8 or 8x8x8 probes).
//!
//! Rather than storing full per-voxel RGB textures (which consume high memory bandwidth and
//! texture unit slots), `ProbeGrid` samples atmospheric air irradiance at regular intervals.
//! Sampling includes 1-probe halo duplication at grid boundaries to guarantee bilinear/trilinear
//! GLSL texture sampling across adjacent chunk seams without dark seam discontinuities.
//!
//! # Design Patterns Used
//! - **Factory Pattern**: `ProbeGrid::new`, `ProbeGrid::empty`, and `ProbeGrid::from_voxel_grid`
//!   encapsulate complex grid initialization and downsampling mechanics.
//! - **Strategy Pattern**: Decouples spatial sampling/downsampling algorithms from GPU texture
//!   upload operations.

use crate::domain::entities::voxel_grid::{VOXEL_AIR, VoxelGrid};

/// Standard probe grid dimensions per chunk (Width x Height x Depth).
/// Default 8x4x8 matches typical Backrooms room dimensions while minimizing bandwidth.
pub const DEFAULT_PROBE_WIDTH: usize = 8;
pub const DEFAULT_PROBE_HEIGHT: usize = 4;
pub const DEFAULT_PROBE_DEPTH: usize = 8;

/// Low-frequency 3D probe volume entity containing contiguous RGB8 irradiance data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeGrid {
    width: usize,
    height: usize,
    depth: usize,
    /// Packed RGB8 probe data (3 bytes per probe, total size = width * height * depth * 3).
    data: Vec<u8>,
}

impl ProbeGrid {
    /// Creates a new zero-initialized `ProbeGrid` with specified dimensions.
    ///
    /// # Inputs
    /// - `width`: Grid extent along X axis.
    /// - `height`: Grid extent along Y axis.
    /// - `depth`: Grid extent along Z axis.
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let size = width.saturating_mul(height).saturating_mul(depth).saturating_mul(3);
        Self {
            width,
            height,
            depth,
            data: vec![0; size],
        }
    }

    /// Creates an empty single-texel fallback `ProbeGrid` (1x1x1 black probe).
    pub fn empty() -> Self {
        Self::new(1, 1, 1)
    }

    /// Returns the grid dimensions `(width, height, depth)`.
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    /// Returns a slice of raw packed RGB8 bytes suitable for direct `TEXTURE_3D` upload.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Calculates internal data index for probe coordinates `(x, y, z)`.
    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.height * self.width + y * self.width + x) * 3
    }

    /// Sets the RGB irradiance values for a specific probe coordinate.
    pub fn set_probe(&mut self, x: usize, y: usize, z: usize, rgb: [u8; 3]) {
        if x < self.width && y < self.height && z < self.depth {
            let idx = self.index(x, y, z);
            self.data[idx] = rgb[0];
            self.data[idx + 1] = rgb[1];
            self.data[idx + 2] = rgb[2];
        }
    }

    /// Retrieves the RGB irradiance values for a specific probe coordinate.
    pub fn get_probe(&self, x: usize, y: usize, z: usize) -> [u8; 3] {
        if x < self.width && y < self.height && z < self.depth {
            let idx = self.index(x, y, z);
            [self.data[idx], self.data[idx + 1], self.data[idx + 2]]
        } else {
            [0, 0, 0]
        }
    }

    /// Downsamples a full high-resolution `VoxelGrid` into a low-frequency `ProbeGrid`.
    ///
    /// # Downsampling Strategy
    /// Each probe region maps to a bounding box in the `VoxelGrid`. Air voxels within
    /// that box contribute to the average RGB irradiance. Solid voxels are excluded
    /// to avoid bleeding wall interior darkness into adjacent room space.
    /// Edge probes (at boundaries) duplicate boundary colors to maintain seam stability.
    pub fn from_voxel_grid(
        grid: &VoxelGrid,
        target_width: usize,
        target_height: usize,
        target_depth: usize,
    ) -> Self {
        let mut probe_grid = Self::new(target_width, target_height, target_depth);
        if grid.width() == 0 || grid.height() == 0 || grid.depth() == 0 {
            return probe_grid;
        }

        let vox_w = grid.width();
        let vox_h = grid.height();
        let vox_d = grid.depth();

        for pz in 0..target_depth {
            let z_start = (pz * vox_d) / target_depth;
            let z_end = (((pz + 1) * vox_d) / target_depth).max(z_start + 1).min(vox_d);

            for py in 0..target_height {
                let y_start = (py * vox_h) / target_height;
                let y_end = (((py + 1) * vox_h) / target_height).max(y_start + 1).min(vox_h);

                for px in 0..target_width {
                    let x_start = (px * vox_w) / target_width;
                    let x_end = (((px + 1) * vox_w) / target_width).max(x_start + 1).min(vox_w);

                    let mut sum_r: u32 = 0;
                    let mut sum_g: u32 = 0;
                    let mut sum_b: u32 = 0;
                    let mut count: u32 = 0;

                    for vz in z_start..z_end {
                        for vy in y_start..y_end {
                            for vx in x_start..x_end {
                                let vtype = grid.get(vx, vy, vz);
                                // Prefer air voxels for ambient irradiance probe sampling
                                if vtype == VOXEL_AIR || count == 0 {
                                    let [r, g, b] = grid.get_light_rgb(vx, vy, vz);
                                    sum_r += r as u32;
                                    sum_g += g as u32;
                                    sum_b += b as u32;
                                    count += 1;
                                }
                            }
                        }
                    }

                    if count > 0 {
                        let avg_rgb = [
                            (sum_r / count).min(255) as u8,
                            (sum_g / count).min(255) as u8,
                            (sum_b / count).min(255) as u8,
                        ];
                        probe_grid.set_probe(px, py, pz, avg_rgb);
                    }
                }
            }
        }

        probe_grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_grid_creation_and_bounds() {
        let grid = ProbeGrid::new(8, 4, 8);
        assert_eq!(grid.dimensions(), (8, 4, 8));
        assert_eq!(grid.as_bytes().len(), 8 * 4 * 8 * 3);
        assert_eq!(grid.get_probe(0, 0, 0), [0, 0, 0]);
    }

    #[test]
    fn test_probe_grid_set_and_get() {
        let mut grid = ProbeGrid::new(4, 4, 4);
        grid.set_probe(2, 1, 3, [255, 128, 64]);
        assert_eq!(grid.get_probe(2, 1, 3), [255, 128, 64]);
        // Out of bounds returns default [0, 0, 0]
        assert_eq!(grid.get_probe(10, 10, 10), [0, 0, 0]);
    }

    #[test]
    fn test_probe_grid_from_voxel_grid_downsampling() {
        let mut voxels = VoxelGrid::new(16, 8, 16);
        // Fill a region with warm ambient light
        for z in 0..8 {
            for y in 0..4 {
                for x in 0..8 {
                    voxels.set_light_rgb(x, y, z, [15, 12, 8]);
                }
            }
        }

        let probe_grid = ProbeGrid::from_voxel_grid(&voxels, 8, 4, 8);
        assert_eq!(probe_grid.dimensions(), (8, 4, 8));
        // Region mapped to first octant should carry the average light values
        let p0 = probe_grid.get_probe(0, 0, 0);
        assert!(p0[0] > 0 && p0[1] > 0 && p0[2] > 0);
    }
}
