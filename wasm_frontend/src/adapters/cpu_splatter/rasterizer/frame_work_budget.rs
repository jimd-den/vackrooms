//! Derive one bounded frame-work envelope from the actual render target.
//!
//! A fixed global cap makes a quality preset self-defeating: increasing the
//! backing size or asking for smaller splats spends the same allowance over
//! more pixels and can therefore reveal *less* of the scene.  This policy
//! instead assigns work per target pixel, then raises that density as the
//! validated geometry settings request more detail.
//!
//! The limits are safety valves, not quality targets.  Ordinary frames finish
//! well below them.  Absolute ceilings remain so a corrupt target size or
//! pathological atlas cannot monopolize the browser indefinitely.

use super::super::settings::CpuRenderSettings;

/// Hard stop for malformed/pathological traversal, independent of target
/// dimensions.  The 1920x1080 Maximum regression remains below this ceiling.
pub(super) const ABSOLUTE_NODE_VISIT_CEILING: usize = 8_000_000;
/// Hard stop for fine-depth writes.  Large enough for several layers over a
/// 1920x1080 Maximum target, but finite even if dimensions are corrupted.
pub(super) const ABSOLUTE_PIXEL_WRITE_CEILING: usize = 16_000_000;

const MINIMUM_NODE_VISITS: usize = 16_384;
const MINIMUM_PIXEL_WRITES: usize = 4_096;
const FOCUS_RESERVE_FRACTION: f64 = 0.90;

/// Immutable work limits for one framebuffer/settings snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FrameWorkBudget {
    node_visit_limit: usize,
    soft_node_visit_limit: usize,
    pixel_write_limit: usize,
}

impl FrameWorkBudget {
    /// Builds a target-aware envelope from validated renderer settings.
    ///
    /// The density functions deliberately use only existing, user-visible
    /// quality controls.  A custom profile therefore reproduces its budget
    /// from its URL/settings values without another hidden preset identity.
    pub(super) fn for_target(settings: &CpuRenderSettings, width: usize, height: usize) -> Self {
        let settings = settings.validated();
        let target_pixels = width.saturating_mul(height).max(1);

        let node_visit_limit = scaled_limit(
            target_pixels,
            node_visits_per_pixel(&settings),
            MINIMUM_NODE_VISITS,
            ABSOLUTE_NODE_VISIT_CEILING,
        );
        let soft_node_visit_limit = ((node_visit_limit as f64 * FOCUS_RESERVE_FRACTION).floor()
            as usize)
            .max(1)
            .min(node_visit_limit);

        // At supported target sizes the fine-write allowance can always
        // cover the framebuffer at least once.  Successful depth rewrites,
        // rather than rejected fragments, consume the remaining allowance.
        let one_full_coverage = target_pixels.min(ABSOLUTE_PIXEL_WRITE_CEILING);
        let pixel_write_limit = scaled_limit(
            target_pixels,
            pixel_writes_per_pixel(&settings),
            MINIMUM_PIXEL_WRITES.max(one_full_coverage),
            ABSOLUTE_PIXEL_WRITE_CEILING,
        );

        Self {
            node_visit_limit,
            soft_node_visit_limit,
            pixel_write_limit,
        }
    }

    pub(super) const fn node_visit_limit(self) -> usize {
        self.node_visit_limit
    }

    pub(super) const fn soft_node_visit_limit(self) -> usize {
        self.soft_node_visit_limit
    }

    pub(super) const fn pixel_write_limit(self) -> usize {
        self.pixel_write_limit
    }

    pub(super) const fn nodes_exhausted(self, visited_nodes: usize) -> bool {
        visited_nodes >= self.node_visit_limit
    }

    pub(super) const fn outside_focus_reserve(self, visited_nodes: usize) -> bool {
        visited_nodes >= self.soft_node_visit_limit
    }

    pub(super) const fn writes_exhausted(self, pixel_writes: usize) -> bool {
        pixel_writes >= self.pixel_write_limit
    }

    pub(super) const fn remaining_pixel_writes(self, pixel_writes: usize) -> usize {
        self.pixel_write_limit.saturating_sub(pixel_writes)
    }
}

impl Default for FrameWorkBudget {
    fn default() -> Self {
        Self::for_target(&CpuRenderSettings::default(), 1, 1)
    }
}

fn scaled_limit(pixels: usize, density: f64, minimum: usize, ceiling: usize) -> usize {
    let requested = (pixels as f64 * density).ceil();
    if !requested.is_finite() || requested >= ceiling as f64 {
        return ceiling;
    }
    (requested as usize).max(minimum).min(ceiling)
}

/// Traversal density rises monotonically with virtual depth, smaller maximum
/// splats, and a smaller MIP cutoff.  Coefficients keep Balanced close to one
/// node allowance per target pixel while giving Maximum room for its much
/// finer geometry without restoring an effectively unbounded walk.
fn node_visits_per_pixel(settings: &CpuRenderSettings) -> f64 {
    0.35 + f64::from(settings.max_virtual_depth) * 0.08
        + 0.4 / f64::from(settings.max_splat_radius_px)
        + 0.15 / f64::from(settings.lod_cutoff_px)
}

/// Fine-write density is always greater than one complete coverage and rises
/// with detail.  This budget counts only successful depth writes; conservative
/// footprint scans and rejected pixels do not consume it.
fn pixel_writes_per_pixel(settings: &CpuRenderSettings) -> f64 {
    1.0 + f64::from(settings.max_virtual_depth) * 0.12
        + 1.0 / f64::from(settings.max_splat_radius_px).sqrt()
        + 0.2 / f64::from(settings.lod_cutoff_px)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::cpu_splatter::settings::CpuQualityPreset;

    #[test]
    fn preset_work_density_never_decreases() {
        let budgets = [
            CpuQualityPreset::Performance,
            CpuQualityPreset::Balanced,
            CpuQualityPreset::Quality,
            CpuQualityPreset::Maximum,
        ]
        .map(|preset| FrameWorkBudget::for_target(&preset.settings(), 960, 540));

        for pair in budgets.windows(2) {
            assert!(pair[0].node_visit_limit() <= pair[1].node_visit_limit());
            assert!(pair[0].pixel_write_limit() <= pair[1].pixel_write_limit());
        }
    }

    #[test]
    fn maximum_1920_by_1080_can_cover_the_target_before_exhaustion() {
        let pixels = 1920 * 1080;
        let budget = FrameWorkBudget::for_target(&CpuQualityPreset::Maximum.settings(), 1920, 1080);

        assert!(budget.pixel_write_limit() >= pixels);
        assert!(budget.pixel_write_limit() < ABSOLUTE_PIXEL_WRITE_CEILING);
        assert!(budget.node_visit_limit() > 150_000);
        assert!(budget.node_visit_limit() < ABSOLUTE_NODE_VISIT_CEILING);
    }

    #[test]
    fn custom_finer_geometry_receives_more_work_per_pixel() {
        let coarse = CpuQualityPreset::Balanced.settings();
        let mut fine = coarse;
        fine.max_splat_radius_px = 2.0;
        fine.lod_cutoff_px = 0.35;
        fine.max_virtual_depth = 8;

        let coarse = FrameWorkBudget::for_target(&coarse, 640, 360);
        let fine = FrameWorkBudget::for_target(&fine, 640, 360);
        assert!(fine.node_visit_limit() > coarse.node_visit_limit());
        assert!(fine.pixel_write_limit() > coarse.pixel_write_limit());
    }

    #[test]
    fn absurd_target_dimensions_stop_at_absolute_ceilings() {
        let budget = FrameWorkBudget::for_target(
            &CpuQualityPreset::Maximum.settings(),
            usize::MAX,
            usize::MAX,
        );

        assert_eq!(budget.node_visit_limit(), ABSOLUTE_NODE_VISIT_CEILING);
        assert_eq!(budget.pixel_write_limit(), ABSOLUTE_PIXEL_WRITE_CEILING);
        assert!(budget.soft_node_visit_limit() < budget.node_visit_limit());
    }
}
