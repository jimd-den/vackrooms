//! Framebuffer, depth buffer, and conservative hierarchical-Z storage.
//!
//! This module knows only pixels. Traversal decides *what* to draw and owns
//! frame policy; the target clips a square splat, performs fine depth tests,
//! enforces the caller's remaining write budget exactly, and maintains the
//! coarse occlusion summary when requested.

/// Coarse hierarchical-Z tile edge in pixels.
const COARSE_TILE: usize = 8;

/// One depth-tested square request in target coordinates.
pub(super) struct SquareSplat {
    center: [f32; 2],
    half: f32,
    depth: f32,
    color: [u8; 3],
}

impl SquareSplat {
    pub(super) fn new(cx: f32, cy: f32, half: f32, depth: f32, color: [u8; 3]) -> Self {
        Self {
            center: [cx, cy],
            half,
            depth,
            color,
        }
    }
}

/// Result of one square-splat write.
pub(super) struct SplatWrite {
    /// Fine-depth-tested pixels whose color and depth changed.
    pub(super) pixel_writes: usize,
    /// A further depth-passing pixel existed after the write budget ran out.
    pub(super) budget_exhausted: bool,
}

/// Reusable CPU render target. Infinity in `depth` means an uncovered pixel;
/// infinity in `coarse_depth` therefore means the tile is not fully covered.
pub(super) struct RenderTarget {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
    depth: Vec<f32>,
    /// Farthest fine depth in each fully covered tile.
    coarse_depth: Vec<f32>,
    /// Number of pixels that have transitioned from uncovered to covered.
    coarse_filled: Vec<usize>,
    coarse_width: usize,
    coarse_height: usize,
}

impl RenderTarget {
    pub(super) fn new(width: usize, height: usize) -> Self {
        let mut target = Self {
            width: 0,
            height: 0,
            rgba: Vec::new(),
            depth: Vec::new(),
            coarse_depth: Vec::new(),
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
        self.depth = vec![f32::INFINITY; self.width * self.height];
        self.coarse_width = self.width.div_ceil(COARSE_TILE);
        self.coarse_height = self.height.div_ceil(COARSE_TILE);
        self.coarse_depth = vec![f32::INFINITY; self.coarse_width * self.coarse_height];
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
        self.depth.fill(f32::INFINITY);
        self.coarse_depth.fill(f32::INFINITY);
        self.coarse_filled.fill(0);
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
        // Anything larger should have been subdivided; the cap also prevents
        // malformed input from filling an unbounded target area in one call.
        let half = splat.half.min(32.0);
        let x0 = (splat.center[0] - half).floor().max(0.0) as usize;
        let x1 = ((splat.center[0] + half).ceil() as usize).min(self.width);
        let y0 = (splat.center[1] - half).floor().max(0.0) as usize;
        let y1 = ((splat.center[1] + half).ceil() as usize).min(self.height);

        let mut pixel_writes = 0;
        let mut budget_exhausted = false;
        'pixels: for y in y0..y1 {
            let row = y * self.width;
            for x in x0..x1 {
                let idx = row + x;
                if splat.depth < self.depth[idx] {
                    if pixel_writes >= write_budget {
                        budget_exhausted = true;
                        break 'pixels;
                    }
                    let was_uncovered = self.depth[idx] == f32::INFINITY;
                    self.depth[idx] = splat.depth;
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

    /// True only when every coarse tile under the screen rect is fully
    /// covered and its farthest stored pixel is nearer than `z_near`.
    pub(super) fn coarse_occludes(&self, px: f32, py: f32, proj_radius: f32, z_near: f32) -> bool {
        let clamp_tile = |value: f32, tiles: usize| {
            ((value.max(0.0) as usize) / COARSE_TILE).min(tiles.saturating_sub(1))
        };
        let x0 = clamp_tile((px - proj_radius).floor(), self.coarse_width);
        let x1 = clamp_tile((px + proj_radius).ceil(), self.coarse_width);
        let y0 = clamp_tile((py - proj_radius).floor(), self.coarse_height);
        let y1 = clamp_tile((py + proj_radius).ceil(), self.coarse_height);

        for tile_y in y0..=y1 {
            for tile_x in x0..=x1 {
                if z_near <= self.coarse_depth[tile_y * self.coarse_width + tile_x] {
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
            self.coarse_depth[tile_index] = self.scan_tile_farthest(tile_x, tile_y);
        }
    }

    fn tile_pixel_count(&self, tile_x: usize, tile_y: usize) -> usize {
        let width = ((tile_x + 1) * COARSE_TILE).min(self.width) - tile_x * COARSE_TILE;
        let height = ((tile_y + 1) * COARSE_TILE).min(self.height) - tile_y * COARSE_TILE;
        width * height
    }

    fn scan_tile_farthest(&self, tile_x: usize, tile_y: usize) -> f32 {
        let x0 = tile_x * COARSE_TILE;
        let x1 = (x0 + COARSE_TILE).min(self.width);
        let y0 = tile_y * COARSE_TILE;
        let y1 = (y0 + COARSE_TILE).min(self.height);
        let mut farthest = 0.0f32;
        for y in y0..y1 {
            for x in x0..x1 {
                farthest = farthest.max(self.depth[y * self.width + x]);
            }
        }
        farthest
    }
}

#[cfg(test)]
mod tests {
    use super::{RenderTarget, SquareSplat};

    #[test]
    fn a_partially_covered_tile_never_occludes_a_subtree() {
        let mut target = RenderTarget::new(8, 8);
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
        let mut target = RenderTarget::new(8, 8);
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
        let mut target = RenderTarget::new(10, 10);
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
        let mut target = RenderTarget::new(8, 8);
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
    fn pixel_write_budget_stops_inside_the_splat_loop() {
        let mut target = RenderTarget::new(8, 8);
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
}
