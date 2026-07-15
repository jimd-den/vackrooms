//! Linear-radiance shading for one CPU splat.
//!
//! The file tree is the rendering sentence:
//!
//! 1. [`surface_reconstruction`] selects one physical receiver sample;
//! 2. [`compose_environment_irradiance`], [`evaluate_fixture_irradiance`],
//!    and [`evaluate_dynamic_irradiance`] build independent light terms;
//! 3. [`evaluate_emission`] handles exposed luminous panel undersides;
//! 4. [`shade_visible_splat`] composes reflection and flashlight energy;
//! 5. [`present_radiance`] applies atmosphere and display encoding once.
//!
//! Static fixtures in [`FrameLighting`] are authoritative. The quantized SVO
//! light level is an optional, restrained fill and cannot replace them.

mod compose_environment_irradiance;
mod evaluate_dynamic_irradiance;
mod evaluate_emission;
mod evaluate_fixture_irradiance;
mod present_radiance;
mod shade_visible_splat;
mod surface_reconstruction;
mod trace_direct_light_visibility;

use crate::application::ports::LightSource;

use super::surface_geometry::SurfaceExposure;

#[cfg(test)]
pub(crate) use evaluate_emission::emission_strength;
pub(crate) use shade_visible_splat::shade_predecoded_with_frame_lighting;
pub use shade_visible_splat::shade_with_frame_lighting;

/// Everything the lighting model needs to know about one splat.
pub struct SplatSurface {
    /// World-space center of the shaded box.
    pub center: [f32; 3],
    /// World-space side length of the box.
    pub world_size: f32,
    /// Camera-space depth used by projection/depth testing. Fog recomputes
    /// the Euclidean path length from `center`; depth is not a path length.
    pub dist: f32,
    /// Authored sRGB albedo, 0..255 per channel.
    pub base_color: [f32; 3],
    /// Quantized colored cached diffuse-fill channels, 0..15. Analytic
    /// fixtures remain authoritative; this optional field is never promoted
    /// to direct light.
    pub baked_irradiance: [f32; 3],
    /// Authored voxel material. Thin architectural materials use this to
    /// preserve their physical orientation instead of behaving like cubes.
    pub voxel_type: u32,
    pub is_emissive: bool,
    /// Prepared physical surface area open to air. Shading may choose among
    /// these faces but must never invent a face outside this contract.
    pub exposure: SurfaceExposure,
    /// Number of occupied siblings in the parent octant (ambient AO proxy).
    pub crowded_siblings: u32,
}

/// Visibility of one specific analytic fixture at the current receiver.
/// Keeping the light id beside the factor prevents an assumed-grid shadow
/// from accidentally darkening ambient or a different fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeroLightVisibility {
    pub light_id: u64,
    pub visibility: f32,
}

/// Immutable per-frame fixture inputs plus the one per-receiver hero query.
#[derive(Debug, Clone, Copy)]
pub struct FrameLighting<'a> {
    pub scene_lights: &'a [LightSource],
    pub hero_visibility: Option<HeroLightVisibility>,
}

impl<'a> FrameLighting<'a> {
    pub fn unshadowed(scene_lights: &'a [LightSource]) -> Self {
        Self {
            scene_lights,
            hero_visibility: None,
        }
    }

    pub fn with_hero_visibility(
        scene_lights: &'a [LightSource],
        light_id: u64,
        visibility: f32,
    ) -> Self {
        Self {
            scene_lights,
            hero_visibility: Some(HeroLightVisibility {
                light_id,
                visibility: visibility.clamp(0.0, 1.0),
            }),
        }
    }
}

#[cfg(test)]
mod tests;
