//! Validated configuration for the CPU splatter.
//!
//! The browser owns *selection* (preset or custom controls), while the
//! renderer receives one immutable [`CpuRenderSettings`] snapshot per frame.
//! No traversal or shading function reads DOM state or atomics directly.
//!
//! The neighboring files make the two policies explicit:
//!
//! * [`quality_preset`] is the small, typed product model shown in Settings.
//! * [`validation`] is the only place that accepts untrusted browser values.

mod quality_preset;
mod validation;

use crate::application::render_settings::RenderToggles;

pub use quality_preset::CpuQualityPreset;
pub use validation::CpuSettingsLimits;

/// Per-splat shadow tracing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuShadowMode {
    /// No fixture shadow rays. Direct diffuse lighting still renders.
    Off,
    /// Trace the selected hero fixture for each affordable splat.
    Hero,
    /// Correctness reference: trace every contributing point-light endpoint
    /// and every 2x2 rectangle quadrature endpoint through the SVO scene.
    Full,
}

/// Effective flashlight visibility policy. It is derived from the independent
/// `rt_beam_occlusion` switch instead of being silently coupled to a preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuFlashlightVisibility {
    /// Fast diagnostic path: the cone is evaluated without scene visibility.
    Unoccluded,
    /// Correctness path: trace the scene between the light and receiver.
    TraceScene,
}

impl CpuShadowMode {
    /// Stable browser-facing representation.
    pub const fn from_id(id: u32) -> Self {
        match id {
            1 => Self::Hero,
            2 => Self::Full,
            _ => Self::Off,
        }
    }

    pub const fn id(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Hero => 1,
            Self::Full => 2,
        }
    }
}

/// Quality controls for the software rasterizer.
///
/// Fields are public because native reference fixtures construct exact test
/// configurations. Browser values must enter through [`Self::validated`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuRenderSettings {
    /// Multiplier on the CPU renderer's quarter-viewport baseline.
    /// `1.0` is 25% of the GPU backing dimensions; `2.0` is 50%.
    pub internal_scale: f32,
    /// Projected-radius threshold below which a subtree becomes one MIP
    /// splat. Smaller values preserve finer distant geometry.
    pub lod_cutoff_px: f32,
    /// Largest allowed projected splat *radius*. Smaller values virtually
    /// subdivide large leaves and reduce block artifacts.
    pub max_splat_radius_px: f32,
    /// Recursion cap for virtual subdivision of a solid leaf.
    pub max_virtual_depth: u8,
    /// World-space floor for virtual subdivision.
    pub min_split_size: f32,
    /// Chunk/node rejection distance in world units.
    pub max_draw_distance: f32,
    /// Minimum occupancy for one collapsed MIP splat.
    pub min_mip_occupancy: f32,
    /// Fixture-shadow quality. This is independent from flashlight
    /// occlusion, which remains an individually diagnosable optimization.
    pub shadows: CpuShadowMode,
    /// `tan(vertical_fov / 2)`; supplied by the camera composition root.
    pub fov_tan: f32,
    /// Independent optimization switches shared with other renderers.
    pub toggles: RenderToggles,
}

impl Default for CpuRenderSettings {
    fn default() -> Self {
        CpuQualityPreset::Balanced.settings()
    }
}

impl CpuRenderSettings {
    pub fn flashlight_visibility(self) -> CpuFlashlightVisibility {
        if self.toggles.flashlight_occlusion {
            CpuFlashlightVisibility::TraceScene
        } else {
            CpuFlashlightVisibility::Unoccluded
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_do_not_hide_the_flashlight_visibility_switch() {
        let mut settings = CpuQualityPreset::Maximum.settings();
        assert_eq!(
            settings.flashlight_visibility(),
            CpuFlashlightVisibility::TraceScene
        );
        settings.toggles.flashlight_occlusion = false;
        assert_eq!(
            settings.flashlight_visibility(),
            CpuFlashlightVisibility::Unoccluded
        );
    }
}
