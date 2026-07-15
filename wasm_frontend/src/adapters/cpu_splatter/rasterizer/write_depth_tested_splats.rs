//! Write square splats into a reusable, depth-tested CPU framebuffer.
//!
//! This module knows only pixels. Traversal decides *what* to draw and owns
//! frame policy; the target clips a square splat, performs fine depth tests,
//! enforces the caller's remaining write budget exactly, and maintains the
//! coarse occlusion summary when requested.

use super::super::surface_geometry::ProjectedSurfaceDepth;

/// Coarse hierarchical-Z tile edge in pixels.
const COARSE_TILE: usize = 8;

/// One depth-tested square request in target coordinates.
pub(super) struct SquareSplat {
    center: [f32; 2],
    half: f32,
    depth: ProjectedSurfaceDepth,
    color: [u8; 3],
    /// Overlapping square reconstruction kernels can cover the same physical
    /// coplanar face. An emitter owns that exact material tie, regardless of
    /// traversal order; ordinary and genuinely farther samples never do.
    equal_depth_wins: bool,
}

impl SquareSplat {
    pub(super) fn new(cx: f32, cy: f32, half: f32, depth: f32, color: [u8; 3]) -> Self {
        Self {
            center: [cx, cy],
            half,
            depth: ProjectedSurfaceDepth::constant(depth),
            color,
            equal_depth_wins: false,
        }
    }

    pub(super) fn with_surface_depth(mut self, depth: ProjectedSurfaceDepth) -> Self {
        self.depth = depth;
        self
    }

    pub(super) fn with_equal_depth_priority(mut self) -> Self {
        self.equal_depth_wins = true;
        self
    }
}

/// Result of one square-splat write.
pub(super) struct SplatWrite {
    /// Fine-depth-tested pixels whose color and depth changed.
    pub(super) pixel_writes: usize,
    /// A further depth-passing pixel existed after the write budget ran out.
    pub(super) budget_exhausted: bool,
}

/// Reusable CPU render target. Reciprocal depth is affine under perspective,
/// so zero means uncovered and larger finite values are nearer. A zero coarse
/// value likewise means that the tile is not yet fully covered.
pub(super) struct DepthTestedSplatTarget {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
    reciprocal_depth: Vec<f32>,
    /// Smallest (farthest) reciprocal fine depth in each fully covered tile.
    coarse_reciprocal_depth: Vec<f32>,
    /// Number of pixels that have transitioned from uncovered to covered.
    coarse_filled: Vec<usize>,
    coarse_width: usize,
    coarse_height: usize,
}

impl DepthTestedSplatTarget {
    pub(super) fn new(width: usize, height: usize) -> Self {
        let mut target = Self {
            width: 0,
            height: 0,
            rgba: Vec::new(),
            reciprocal_depth: Vec::new(),
            coarse_reciprocal_depth: Vec::new(),
            coarse_filled: Vec::new(),
            coarse_width: 0,
            coarse_height: 0,
        };
        target.resize(width, height);
        target
    }

    pub(super) fn resize(&mut self, width: usize, height: usize) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.rgba = vec![0; self.width * self.height * 4];
        self.reciprocal_depth = vec![0.0; self.width * self.height];
        self.coarse_width = self.width.div_ceil(COARSE_TILE);
        self.coarse_height = self.height.div_ceil(COARSE_TILE);
        self.coarse_reciprocal_depth = vec![0.0; self.coarse_width * self.coarse_height];
        self.coarse_filled = vec![0; self.coarse_width * self.coarse_height];
    }

    pub(super) fn width(&self) -> usize {
        self.width
    }

    pub(super) fn height(&self) -> usize {
        self.height
    }

    pub(super) fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    pub(super) fn clear(&mut self, background: [u8; 3]) {
        for pixel in self.rgba.chunks_exact_mut(4) {
            pixel[0] = background[0];
            pixel[1] = background[1];
            pixel[2] = background[2];
            pixel[3] = 255;
        }
        self.reciprocal_depth.fill(0.0);
        self.coarse_reciprocal_depth.fill(0.0);
        self.coarse_filled.fill(0);
    }

    /// Exact fine-depth query for an unshaded square splat.
    ///
    /// Traversal uses this immediately before fixture, hero, and flashlight
    /// evaluation. A `false` result proves that `write_splat` would reject
    /// every pixel, so skipping those radiance calculations cannot change the
    /// framebuffer. A likely-visible splat returns at its first passing pixel.
    #[cfg(test)]
    pub(super) fn splat_may_contribute(&self, center: [f32; 2], half: f32, depth: f32) -> bool {
        self.request_may_contribute(&SquareSplat::new(center[0], center[1], half, depth, [0; 3]))
    }

    /// Fine-depth query for a projected physical surface. The exact same
    /// per-pixel depth and tie policy is consumed by [`Self::write_splat`].
    pub(super) fn surface_splat_may_contribute(&self, splat: &SquareSplat) -> bool {
        self.request_may_contribute(splat)
    }

    fn request_may_contribute(&self, splat: &SquareSplat) -> bool {
        let [x0, x1, y0, y1] = self.clipped_square_bounds(splat.center, splat.half);
        for y in y0..y1 {
            let row = y * self.width;
            for x in x0..x1 {
                let Some(reciprocal_depth) = splat
                    .depth
                    .reciprocal_at_pixel([x as f32 + 0.5, y as f32 + 0.5])
                else {
                    continue;
                };
                let stored = self.reciprocal_depth[row + x];
                if reciprocal_depth > stored
                    || (splat.equal_depth_wins && reciprocal_depth == stored)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Fills a depth-tested square. `half` is its half-extent in pixels.
    ///
    /// The budget check lives inside the depth-passing branch: rejected
    /// pixels cost no writes, while the first additional visible pixel after
    /// the budget is exhausted stops the splat and reports exhaustion.
    pub(super) fn write_splat(
        &mut self,
        splat: SquareSplat,
        write_budget: usize,
        maintain_coarse: bool,
    ) -> SplatWrite {
        // Traversal normally subdivides large footprints. Camera-plane and
        // budget fallbacks remain legal here; clipping plus `write_budget`
        // bounds their work without changing their requested coverage.
        let [x0, x1, y0, y1] = self.clipped_square_bounds(splat.center, splat.half);

        let mut pixel_writes = 0;
        let mut budget_exhausted = false;
        'pixels: for y in y0..y1 {
            let row = y * self.width;
            for x in x0..x1 {
                let idx = row + x;
                let Some(reciprocal_depth) = splat
                    .depth
                    .reciprocal_at_pixel([x as f32 + 0.5, y as f32 + 0.5])
                else {
                    continue;
                };
                let stored = self.reciprocal_depth[idx];
                if reciprocal_depth > stored
                    || (splat.equal_depth_wins && reciprocal_depth == stored)
                {
                    if pixel_writes >= write_budget {
                        budget_exhausted = true;
                        break 'pixels;
                    }
                    let was_uncovered = stored == 0.0;
                    self.reciprocal_depth[idx] = reciprocal_depth;
                    let rgba_index = idx * 4;
                    self.rgba[rgba_index] = splat.color[0];
                    self.rgba[rgba_index + 1] = splat.color[1];
                    self.rgba[rgba_index + 2] = splat.color[2];
                    pixel_writes += 1;
                    if maintain_coarse && was_uncovered {
                        self.record_first_coverage(x, y);
                    }
                }
            }
        }

        SplatWrite {
            pixel_writes,
            budget_exhausted,
        }
    }

    fn clipped_square_bounds(&self, center: [f32; 2], half: f32) -> [usize; 4] {
        // Traversal normally subdivides to the configured maximum radius,
        // but camera-plane intersections and explicit budget fallbacks can
        // legitimately cover more than 32 pixels.  The framebuffer is
        // already a finite clip bound and the caller supplies an exact write
        // budget, so silently shrinking the requested footprint would only
        // create holes.  A malformed infinity conservatively covers the
        // finite target instead of entering float-to-integer edge cases.
        let half = if half.is_finite() {
            half.max(0.0)
        } else {
            self.width.max(self.height) as f32
        };
        [
            (center[0] - half).floor().max(0.0) as usize,
            ((center[0] + half).ceil() as usize).min(self.width),
            (center[1] - half).floor().max(0.0) as usize,
            ((center[1] + half).ceil() as usize).min(self.height),
        ]
    }

    /// True only when every coarse tile under the screen rect is fully
    /// covered and its farthest stored pixel is nearer than `z_near`.
    #[cfg(test)]
    pub(super) fn coarse_occludes(&self, px: f32, py: f32, proj_radius: f32, z_near: f32) -> bool {
        self.coarse_rect_occludes([px, py], [proj_radius, proj_radius], z_near)
    }

    /// Rectangular form used by the exact projected AABB footprint. Keeping
    /// X/Y extents separate avoids turning a wide wall or shallow floor into
    /// an unnecessarily large square HZ query.
    pub(super) fn coarse_rect_occludes(
        &self,
        center: [f32; 2],
        half_extent: [f32; 2],
        z_near: f32,
    ) -> bool {
        let lower_tile = |value: f32, tiles: usize| {
            ((value.max(0.0) as usize) / COARSE_TILE).min(tiles.saturating_sub(1))
        };
        // Upper bounds are exclusive after `ceil`, matching the fine
        // `x0..x1`/`y0..y1` loops. Dividing that exclusive coordinate
        // directly includes the next tile at every exact 8-pixel boundary.
        let upper_tile = |value: f32, tiles: usize| {
            ((value.ceil().max(0.0) as usize).saturating_sub(1) / COARSE_TILE)
                .min(tiles.saturating_sub(1))
        };
        let x0 = lower_tile((center[0] - half_extent[0]).floor(), self.coarse_width);
        let x1 = upper_tile(center[0] + half_extent[0], self.coarse_width);
        let y0 = lower_tile((center[1] - half_extent[1]).floor(), self.coarse_height);
        let y1 = upper_tile(center[1] + half_extent[1], self.coarse_height);
        if !z_near.is_finite() || z_near <= 0.0 {
            return false;
        }
        let reciprocal_near = z_near.recip();

        for tile_y in y0..=y1 {
            for tile_x in x0..=x1 {
                if reciprocal_near
                    >= self.coarse_reciprocal_depth[tile_y * self.coarse_width + tile_x]
                {
                    return false;
                }
            }
        }
        true
    }

    /// Records an infinity-to-finite transition. A tile is scanned exactly
    /// once, when its final uncovered pixel becomes covered. Later depth
    /// writes only decrease values, so the stored maximum becomes a stale
    /// upper bound: conservative for rejection and free to maintain.
    fn record_first_coverage(&mut self, x: usize, y: usize) {
        let tile_x = x / COARSE_TILE;
        let tile_y = y / COARSE_TILE;
        let tile_index = tile_y * self.coarse_width + tile_x;
        self.coarse_filled[tile_index] += 1;
        if self.coarse_filled[tile_index] == self.tile_pixel_count(tile_x, tile_y) {
            self.coarse_reciprocal_depth[tile_index] =
                self.scan_tile_farthest_reciprocal_depth(tile_x, tile_y);
        }
    }

    fn tile_pixel_count(&self, tile_x: usize, tile_y: usize) -> usize {
        let width = ((tile_x + 1) * COARSE_TILE).min(self.width) - tile_x * COARSE_TILE;
        let height = ((tile_y + 1) * COARSE_TILE).min(self.height) - tile_y * COARSE_TILE;
        width * height
    }

    fn scan_tile_farthest_reciprocal_depth(&self, tile_x: usize, tile_y: usize) -> f32 {
        let x0 = tile_x * COARSE_TILE;
        let x1 = (x0 + COARSE_TILE).min(self.width);
        let y0 = tile_y * COARSE_TILE;
        let y1 = (y0 + COARSE_TILE).min(self.height);
        let mut farthest = f32::INFINITY;
        for y in y0..y1 {
            for x in x0..x1 {
                farthest = farthest.min(self.reciprocal_depth[y * self.width + x]);
            }
        }
        farthest
    }
}

#[cfg(test)]
mod tests {
    use super::{DepthTestedSplatTarget, SquareSplat};

    #[test]
    fn a_partially_covered_tile_never_occludes_a_subtree() {
        let mut target = DepthTestedSplatTarget::new(8, 8);
        target.clear([0; 3]);
        target.write_splat(
            SquareSplat::new(1.0, 1.0, 0.4, 1.0, [255; 3]),
            usize::MAX,
            true,
        );

        assert!(
            !target.coarse_occludes(4.0, 4.0, 4.0, 2.0),
            "one near pixel cannot stand in for a fully covered 8x8 tile"
        );
    }

    #[test]
    fn a_fully_covered_near_tile_occludes_farther_geometry() {
        let mut target = DepthTestedSplatTarget::new(8, 8);
        target.clear([0; 3]);
        target.write_splat(
            SquareSplat::new(4.0, 4.0, 4.0, 1.0, [255; 3]),
            usize::MAX,
            true,
        );

        assert!(target.coarse_occludes(4.0, 4.0, 4.0, 2.0));
    }

    #[test]
    fn edge_tile_uses_its_actual_pixel_count() {
        let mut target = DepthTestedSplatTarget::new(10, 10);
        target.clear([0; 3]);
        // The bottom-right coarse tile is only 2x2, not 8x8.
        target.write_splat(
            SquareSplat::new(9.0, 9.0, 1.0, 1.0, [255; 3]),
            usize::MAX,
            true,
        );

        assert!(target.coarse_occludes(9.0, 9.0, 1.0, 2.0));
    }

    #[test]
    fn later_nearer_writes_keep_a_conservative_upper_bound() {
        let mut target = DepthTestedSplatTarget::new(8, 8);
        target.clear([0; 3]);
        target.write_splat(
            SquareSplat::new(4.0, 4.0, 4.0, 5.0, [100; 3]),
            usize::MAX,
            true,
        );
        target.write_splat(
            SquareSplat::new(4.0, 4.0, 4.0, 1.0, [255; 3]),
            usize::MAX,
            true,
        );

        // The stored 5.0 maximum is intentionally stale. It can miss this
        // legal cull, but can never reject geometry that the fine buffer
        // would reveal.
        assert!(!target.coarse_occludes(4.0, 4.0, 4.0, 3.0));
        assert!(target.coarse_occludes(4.0, 4.0, 4.0, 6.0));
    }

    #[test]
    fn rectangular_hz_query_uses_the_exact_projected_extents() {
        let mut target = DepthTestedSplatTarget::new(16, 8);
        target.clear([0; 3]);
        target.write_splat(
            SquareSplat::new(4.0, 4.0, 4.0, 1.0, [255; 3]),
            usize::MAX,
            true,
        );

        assert!(target.coarse_rect_occludes([4.0, 4.0], [4.0, 1.0], 2.0));
        assert!(!target.coarse_rect_occludes([8.0, 4.0], [8.0, 1.0], 2.0));
    }

    #[test]
    fn pixel_write_budget_stops_inside_the_splat_loop() {
        let mut target = DepthTestedSplatTarget::new(8, 8);
        target.clear([0; 3]);
        let write = target.write_splat(SquareSplat::new(4.0, 4.0, 4.0, 1.0, [255; 3]), 7, true);

        assert_eq!(write.pixel_writes, 7);
        assert!(write.budget_exhausted);
        assert_eq!(
            target
                .rgba()
                .chunks_exact(4)
                .filter(|p| p[0] == 255)
                .count(),
            7
        );
    }

    #[test]
    fn fine_depth_query_matches_whether_a_splat_would_write() {
        let mut target = DepthTestedSplatTarget::new(8, 8);
        target.clear([0; 3]);
        target.write_splat(
            SquareSplat::new(4.0, 4.0, 4.0, 2.0, [20; 3]),
            usize::MAX,
            true,
        );

        assert!(!target.splat_may_contribute([4.0, 4.0], 4.0, 3.0));
        assert!(target.splat_may_contribute([4.0, 4.0], 4.0, 1.0));
        assert!(target.splat_may_contribute([7.5, 7.5], 1.0, 1.0));
        assert!(!target.splat_may_contribute([-5.0, -5.0], 1.0, 1.0));
    }

    #[test]
    fn large_conservative_footprints_are_not_silently_clamped() {
        let mut target = DepthTestedSplatTarget::new(96, 8);
        target.clear([0; 3]);

        let write = target.write_splat(
            SquareSplat::new(48.0, 4.0, 48.0, 1.0, [255; 3]),
            usize::MAX,
            false,
        );

        assert_eq!(write.pixel_writes, 96 * 8);
    }

    #[test]
    fn emissive_priority_changes_only_an_exact_coplanar_tie() {
        let mut target = DepthTestedSplatTarget::new(2, 2);
        target.clear([0; 3]);
        let ordinary = |depth, color| SquareSplat::new(1.0, 1.0, 1.0, depth, color);
        let emitter = |depth, color| ordinary(depth, color).with_equal_depth_priority();

        assert_eq!(
            target
                .write_splat(ordinary(2.0, [20, 0, 0]), usize::MAX, false)
                .pixel_writes,
            4
        );
        assert_eq!(
            target
                .write_splat(ordinary(2.0, [0, 20, 0]), usize::MAX, false)
                .pixel_writes,
            0,
            "ordinary equal-depth reconstruction keeps the existing material"
        );
        assert_eq!(
            target
                .write_splat(emitter(2.0, [255, 240, 200]), usize::MAX, false)
                .pixel_writes,
            4,
            "an emitter owns its exact coplanar material tie"
        );
        assert_eq!(
            target
                .write_splat(emitter(3.0, [0, 0, 255]), usize::MAX, false)
                .pixel_writes,
            0,
            "priority never lets farther geometry pass"
        );
        assert_eq!(
            target
                .write_splat(ordinary(1.0, [255, 0, 0]), usize::MAX, false)
                .pixel_writes,
            4,
            "ordinary nearer geometry still wins"
        );
    }
}
