//! Facade for one complete visible-splat shading evaluation.

use crate::application::ports::{ChunkDraw, Environment};
use crate::application::rendering::{DIFFUSE_REFLECTANCE_INV_PI, LinearRgb};

use super::super::camera::Camera;
use super::super::flashlight;
use super::super::settings::{CpuRenderSettings, CpuShadowMode};
use super::compose_environment_irradiance::{
    ambient_irradiance, ambient_visibility, cached_fill_irradiance,
};
use super::evaluate_dynamic_irradiance::dynamic_irradiance;
use super::evaluate_emission::{emission_strength, emitted_radiance_with_strength};
use super::evaluate_fixture_irradiance::fixture_irradiance_with_visibility;
use super::present_radiance::{present_radiance, view_distance};
use super::surface_reconstruction::{
    decode_albedo, representative_normal, representative_surface_position,
};
use super::trace_direct_light_visibility::DirectLightVisibility;
use super::{FrameLighting, SplatSurface};

/// Correctness path with the complete analytic fixture contract.
pub fn shade_with_frame_lighting(
    cam: &Camera,
    environment: &Environment,
    settings: &CpuRenderSettings,
    atlas: &[u32],
    chunks: &[ChunkDraw],
    lighting: FrameLighting<'_>,
    surface: &SplatSurface,
) -> [u8; 3] {
    shade_predecoded_with_frame_lighting(
        cam,
        environment,
        settings,
        atlas,
        chunks,
        lighting,
        surface,
        decode_albedo(surface.base_color),
        surface
            .is_emissive
            .then(|| emission_strength(surface.base_color)),
    )
}

/// Hot production path. Atlas upload already decoded authored sRGB and MIP
/// filtering already averaged in linear space, so shading consumes the
/// prepared albedo without repeating transfer-function powers per splat.
pub(crate) fn shade_predecoded_with_frame_lighting(
    cam: &Camera,
    environment: &Environment,
    settings: &CpuRenderSettings,
    atlas: &[u32],
    chunks: &[ChunkDraw],
    lighting: FrameLighting<'_>,
    surface: &SplatSurface,
    albedo: LinearRgb,
    emission_strength: Option<f32>,
) -> [u8; 3] {
    let camera_to_surface = [
        surface.center[0] - cam.pos[0],
        surface.center[1] - cam.pos[1],
        surface.center[2] - cam.pos[2],
    ];
    let reconstructed_normal = representative_normal(surface, camera_to_surface);
    let normal = reconstructed_normal.unwrap_or([0.0, 1.0, 0.0]);
    let surface_position =
        representative_surface_position(surface.center, surface.world_size, normal);
    let distance = view_distance(cam, surface, surface_position);

    if let Some(emission) =
        emitted_radiance_with_strength(surface, albedo, normal, emission_strength)
    {
        return present_radiance(emission, distance, environment);
    }

    let ambient_visibility = if settings.toggles.ambient_occlusion {
        ambient_visibility(surface.crowded_siblings)
    } else {
        1.0
    };
    let mut irradiance =
        ambient_irradiance(environment).map(|channel| channel * ambient_visibility);
    // AO approximates only open environment visibility. The cached field
    // already represents transported diffuse fill and must not be darkened a
    // second time.
    let cached = cached_fill_irradiance(surface, settings);
    let direct_visibility = if settings.shadows == CpuShadowMode::Full {
        DirectLightVisibility::trace_scene(atlas, chunks, surface.world_size)
    } else {
        DirectLightVisibility::unoccluded()
    };
    let fixtures =
        fixture_irradiance_with_visibility(lighting, surface_position, normal, direct_visibility);
    let dynamics = dynamic_irradiance(cam, surface_position, normal);
    let flashlight = if cam.flashlight && reconstructed_normal.is_some() {
        flashlight::beam_contribution(
            cam,
            atlas,
            chunks,
            flashlight::BeamReceiver::surface_sample(
                surface.center,
                surface.world_size,
                surface_position,
                normal,
            ),
            settings.flashlight_visibility(),
        )
    } else {
        [0.0; 3]
    };

    for channel in 0..3 {
        irradiance[channel] +=
            cached[channel] + fixtures[channel] + dynamics[channel] + flashlight[channel];
    }
    let reflected = [
        albedo[0] * irradiance[0] * DIFFUSE_REFLECTANCE_INV_PI,
        albedo[1] * irradiance[1] * DIFFUSE_REFLECTANCE_INV_PI,
        albedo[2] * irradiance[2] * DIFFUSE_REFLECTANCE_INV_PI,
    ];
    present_radiance(reflected, distance, environment)
}
