//! Ambient visibility and the optional cached diffuse-fill contribution.
//! Neither term owns direct-light, flashlight, emission, or atmosphere math.

use crate::application::ports::Environment;
use crate::application::rendering::{
    CACHED_DIFFUSE_FILL_GAIN, INDOOR_AMBIENT_IRRADIANCE, LinearRgb, OUTDOOR_AMBIENT_IRRADIANCE,
};

use super::super::settings::CpuRenderSettings;
use super::SplatSurface;

pub(super) fn ambient_irradiance(environment: &Environment) -> LinearRgb {
    let base = if environment.outdoor {
        OUTDOOR_AMBIENT_IRRADIANCE
    } else {
        INDOOR_AMBIENT_IRRADIANCE
    };
    base.map(|channel| channel * environment.ambient_scale.max(0.0))
}

/// Crowding is only an ambient-visibility approximation. Analytic lights,
/// flashlight visibility, and emission are never multiplied by it.
pub(super) fn ambient_visibility(crowded_siblings: u32) -> f32 {
    let neighbors = crowded_siblings.saturating_sub(1).min(7) as f32;
    (1.0 - 0.055 * neighbors).clamp(0.62, 1.0)
}

pub(super) fn cached_fill_irradiance(
    surface: &SplatSurface,
    settings: &CpuRenderSettings,
) -> LinearRgb {
    if !settings.toggles.baked_lighting {
        return [0.0; 3];
    }
    surface
        .baked_irradiance
        .map(|channel| (channel / 15.0).clamp(0.0, 1.0) * CACHED_DIFFUSE_FILL_GAIN)
}
