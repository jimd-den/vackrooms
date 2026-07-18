//! Boundary validation for CPU render settings.
//!
//! Sliders, saved JSON, and query strings are untrusted input. Keeping every
//! numerical limit here prevents the UI, WASM exports, and renderer from
//! developing slightly different interpretations of the same setting.

use super::CpuRenderSettings;

pub struct CpuSettingsLimits;

impl CpuSettingsLimits {
    pub const INTERNAL_SCALE: (f32, f32) = (0.25, 2.0);
    pub const LOD_CUTOFF_PX: (f32, f32) = (0.25, 2.0);
    pub const MAX_SPLAT_RADIUS_PX: (f32, f32) = (1.0, 16.0);
    pub const MAX_VIRTUAL_DEPTH: (u8, u8) = (3, 8);
    pub const MIN_SPLIT_SIZE: (f32, f32) = (0.005, 1.0);
    pub const MAX_DRAW_DISTANCE: (f32, f32) = (16.0, 256.0);
    pub const MIN_MIP_OCCUPANCY: (f32, f32) = (0.0, 0.5);
    pub const FOV_TAN: (f32, f32) = (0.1, 4.0);
}

fn finite_clamp(value: f32, fallback: f32, limits: (f32, f32)) -> f32 {
    if value.is_finite() {
        value.clamp(limits.0, limits.1)
    } else {
        fallback
    }
}

impl CpuRenderSettings {
    /// Returns a finite, bounded snapshot safe for traversal and allocation.
    pub fn validated(mut self) -> Self {
        let fallback = Self::default();
        self.internal_scale = finite_clamp(
            self.internal_scale,
            fallback.internal_scale,
            CpuSettingsLimits::INTERNAL_SCALE,
        );
        self.lod_cutoff_px = finite_clamp(
            self.lod_cutoff_px,
            fallback.lod_cutoff_px,
            CpuSettingsLimits::LOD_CUTOFF_PX,
        );
        self.max_splat_radius_px = finite_clamp(
            self.max_splat_radius_px,
            fallback.max_splat_radius_px,
            CpuSettingsLimits::MAX_SPLAT_RADIUS_PX,
        );
        self.max_virtual_depth = self.max_virtual_depth.clamp(
            CpuSettingsLimits::MAX_VIRTUAL_DEPTH.0,
            CpuSettingsLimits::MAX_VIRTUAL_DEPTH.1,
        );
        self.min_split_size = finite_clamp(
            self.min_split_size,
            fallback.min_split_size,
            CpuSettingsLimits::MIN_SPLIT_SIZE,
        );
        self.max_draw_distance = finite_clamp(
            self.max_draw_distance,
            fallback.max_draw_distance,
            CpuSettingsLimits::MAX_DRAW_DISTANCE,
        );
        self.min_mip_occupancy = finite_clamp(
            self.min_mip_occupancy,
            fallback.min_mip_occupancy,
            CpuSettingsLimits::MIN_MIP_OCCUPANCY,
        );
        self.fov_tan = finite_clamp(self.fov_tan, fallback.fov_tan, CpuSettingsLimits::FOV_TAN);
        self
    }

    /// CPU backing-image size relative to the WebGPU presentation surface.
    /// The baseline is deliberately one quarter per axis; presets scale from
    /// 12.5% to 50%.
    pub fn canvas_resolution_factor(self) -> f64 {
        (self.validated().internal_scale * 0.25) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_clamps_every_untrusted_scalar() {
        let settings = CpuRenderSettings {
            internal_scale: f32::INFINITY,
            lod_cutoff_px: -5.0,
            max_splat_radius_px: 200.0,
            max_virtual_depth: u8::MAX,
            min_split_size: f32::NAN,
            max_draw_distance: 10_000.0,
            min_mip_occupancy: -2.0,
            fov_tan: 99.0,
            ..CpuRenderSettings::default()
        }
        .validated();

        assert_eq!(settings.internal_scale, 1.0);
        assert_eq!(settings.lod_cutoff_px, 0.25);
        assert_eq!(settings.max_splat_radius_px, 16.0);
        assert_eq!(settings.max_virtual_depth, 8);
        assert_eq!(settings.min_split_size, 0.005);
        assert_eq!(settings.max_draw_distance, 256.0);
        assert_eq!(settings.min_mip_occupancy, 0.0);
        assert_eq!(settings.fov_tan, 4.0);
    }

    #[test]
    fn cpu_resolution_scale_has_a_bounded_cost_envelope() {
        let mut settings = CpuRenderSettings::default();
        settings.internal_scale = 0.25;
        assert_eq!(settings.canvas_resolution_factor(), 0.0625);
        settings.internal_scale = 1.0;
        assert_eq!(settings.canvas_resolution_factor(), 0.25);
        settings.internal_scale = 2.0;
        assert_eq!(settings.canvas_resolution_factor(), 0.5);
    }
}
