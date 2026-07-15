//! Named CPU quality profiles.
//!
//! Presets only choose scalar quality and workload budgets. They never hide
//! or mutate the renderer optimization switchboard, so an optimization can
//! still be disabled independently while diagnosing an image.

use crate::application::render_settings::RenderToggles;

use super::{CpuRenderSettings, CpuShadowMode};

/// Stable IDs cross the WASM boundary; names remain presentation concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CpuQualityPreset {
    Performance = 0,
    Balanced = 1,
    Quality = 2,
    Maximum = 3,
}

impl CpuQualityPreset {
    pub const fn from_id(id: u32) -> Self {
        match id {
            0 => Self::Performance,
            2 => Self::Quality,
            3 => Self::Maximum,
            _ => Self::Balanced,
        }
    }

    /// Complete renderer settings for this profile.
    pub fn settings(self) -> CpuRenderSettings {
        let (
            internal_scale,
            lod_cutoff_px,
            max_splat_radius_px,
            max_virtual_depth,
            max_draw_distance,
            min_mip_occupancy,
            shadows,
        ) = match self {
            Self::Performance => (0.5, 1.5, 14.0, 4, 72.0, 0.35, CpuShadowMode::Off),
            Self::Balanced => (1.0, 1.0, 8.0, 5, 96.0, 0.25, CpuShadowMode::Off),
            Self::Quality => (1.5, 0.65, 4.0, 6, 128.0, 0.12, CpuShadowMode::Hero),
            Self::Maximum => (2.0, 0.35, 2.0, 8, 160.0, 0.05, CpuShadowMode::Full),
        };

        CpuRenderSettings {
            internal_scale,
            lod_cutoff_px,
            max_splat_radius_px,
            max_virtual_depth,
            min_split_size: 0.005,
            max_draw_distance,
            min_mip_occupancy,
            shadows,
            fov_tan: 0.767,
            toggles: RenderToggles::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stronger_presets_increase_resolution_and_geometric_detail() {
        let presets = [
            CpuQualityPreset::Performance.settings(),
            CpuQualityPreset::Balanced.settings(),
            CpuQualityPreset::Quality.settings(),
            CpuQualityPreset::Maximum.settings(),
        ];

        for pair in presets.windows(2) {
            assert!(pair[0].internal_scale < pair[1].internal_scale);
            assert!(pair[0].lod_cutoff_px > pair[1].lod_cutoff_px);
            assert!(pair[0].max_splat_radius_px > pair[1].max_splat_radius_px);
            assert!(pair[0].max_virtual_depth < pair[1].max_virtual_depth);
            assert!(pair[0].max_draw_distance < pair[1].max_draw_distance);
        }
    }

    #[test]
    fn preset_ids_are_stable_and_unknown_ids_are_safe() {
        assert_eq!(CpuQualityPreset::from_id(0), CpuQualityPreset::Performance);
        assert_eq!(CpuQualityPreset::from_id(2), CpuQualityPreset::Quality);
        assert_eq!(CpuQualityPreset::from_id(3), CpuQualityPreset::Maximum);
        assert_eq!(CpuQualityPreset::from_id(99), CpuQualityPreset::Balanced);
    }

    #[test]
    fn maximum_is_the_expensive_all_fixture_visibility_reference() {
        assert_eq!(
            CpuQualityPreset::Maximum.settings().shadows,
            CpuShadowMode::Full
        );
    }
}
