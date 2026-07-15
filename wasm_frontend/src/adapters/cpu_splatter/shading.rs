//! Per-splat shading — the CPU path's lighting model, one small function
//! per light term. Because it runs per *splat* (not per pixel), it can
//! afford to be richer than the GPU paths:
//!
//! * quantized voxel diffuse fill (same atlas data as the surface GPU path),
//! * per-face directional response (tops bright, bottoms dark),
//! * contact AO from octant crowding,
//! * the flashlight cone ([`super::flashlight`]),
//! * dynamic point lights (dropped flares),
//! * exponential distance fog toward the level's palette.
//!
//! A splat has no true normal (it is an axis-aligned box seen from some
//! direction), so the normal is *approximated* from the squared components
//! of the camera→center direction: the axis you look at most contributes
//! most. This soft blend is deliberate — see the
//! `test_face_shading_discontinuity` regression test.

use crate::application::ports::{ChunkDraw, Environment};

use super::camera::{Camera, dot};
use super::flashlight;
use super::settings::CpuRenderSettings;

/// Everything the lighting model needs to know about one splat.
pub struct SplatSurface {
    /// World-space center of the shaded box.
    pub center: [f32; 3],
    /// World-space side length of the box.
    pub world_size: f32,
    /// Camera-space depth (used for fog).
    pub dist: f32,
    /// Albedo, 0..255 per channel.
    pub base_color: [f32; 3],
    /// Quantized diffuse-fill level, 0..15.
    pub light_level: f32,
    pub is_emissive: bool,
    /// Number of occupied siblings in the parent octant (contact AO).
    pub crowded_siblings: u32,
    /// Hero-light shadow factor, 1.0 = unshadowed.
    pub shadow_factor: f32,
}

/// Squared directional weights and outward signs of the camera→surface
/// vector; the shared basis for the approximate normal and face response.
struct AxisBlend {
    /// Normalized squared weights per axis (sum to 1).
    weights: [f32; 3],
    /// Which side of each axis faces the camera (+1/-1).
    signs: [f32; 3],
    valid: bool,
}

impl AxisBlend {
    fn from_to_camera(to_cam: [f32; 3]) -> Self {
        let sq = [
            to_cam[0] * to_cam[0],
            to_cam[1] * to_cam[1],
            to_cam[2] * to_cam[2],
        ];
        let sum = sq[0] + sq[1] + sq[2];
        if sum <= 1e-6 {
            return Self {
                weights: [0.0; 3],
                signs: [1.0; 3],
                valid: false,
            };
        }
        Self {
            weights: [sq[0] / sum, sq[1] / sum, sq[2] / sum],
            signs: [
                if to_cam[0] > 0.0 { -1.0 } else { 1.0 },
                if to_cam[1] > 0.0 { -1.0 } else { 1.0 },
                if to_cam[2] > 0.0 { -1.0 } else { 1.0 },
            ],
            valid: true,
        }
    }

    /// The blended approximate surface normal (unit-ish).
    fn normal(&self) -> [f32; 3] {
        let n = [
            self.signs[0] * self.weights[0],
            self.signs[1] * self.weights[1],
            self.signs[2] * self.weights[2],
        ];
        let len = dot(n, n).sqrt().max(0.001);
        [n[0] / len, n[1] / len, n[2] / len]
    }

    /// Signed vertical component of the approximate normal (for the
    /// hemisphere ambient mix).
    fn n_y(&self) -> f32 {
        self.signs[1] * self.weights[1]
    }
}

/// Directional face response: upward faces brightest (1.0), downward
/// darkest (0.35), walls in between with a subtle cardinal variation —
/// identical weights to the GPU surface shader.
fn face_response(blend: &AxisBlend) -> f32 {
    if !blend.valid {
        return 0.8;
    }
    let y_resp = if blend.signs[1] > 0.0 { 1.0 } else { 0.35 };
    let wall_resp = 0.65 + 0.1 * blend.signs[0] + 0.05 * blend.signs[2];
    blend.weights[1] * y_resp + (blend.weights[0] + blend.weights[2]) * wall_resp
}

/// Contact AO from octant crowding, clamped so corners never go pure black.
fn contact_ao(crowded_siblings: u32) -> f32 {
    let raw = 1.0 - 0.04 * crowded_siblings.saturating_sub(1) as f32;
    (raw * 1.3).clamp(0.22, 1.0)
}

/// Hemisphere ambient: a fluorescent yellow-green top against a dark olive
/// bottom indoors; a cool neutral daylight pair outdoors.
fn ambient_term(
    env: &Environment,
    light_norm: f32,
    blend: &AxisBlend,
    ao: f32,
    face: f32,
) -> [f32; 3] {
    let t_mix = blend.n_y() * 0.5 + 0.5;
    let (up, down) = if env.outdoor {
        ([0.98, 1.02, 1.08], [0.56, 0.58, 0.55])
    } else {
        ([1.05, 1.0, 0.7], [0.6, 0.55, 0.35])
    };
    let mut ambient = [0.0; 3];
    for c in 0..3 {
        let hemi = down[c] * (1.0 - t_mix) + up[c] * t_mix;
        ambient[c] = hemi * light_norm * ao * face;
    }
    ambient
}

/// Dynamic point lights (dropped flares): quadratic falloff, no occlusion
/// test — the CPU path has no ray budget for one per light, and the small
/// radius keeps light from reading through far walls.
fn dynamic_light_term(cam: &Camera, center: [f32; 3]) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    for light in &cam.dynamic_lights[..cam.dynamic_light_count] {
        let lx = light.position[0] - center[0];
        let ly = light.position[1] - center[1];
        let lz = light.position[2] - center[2];
        let d2 = lx * lx + ly * ly + lz * lz;
        let r2 = light.radius * light.radius;
        if d2 >= r2 {
            continue;
        }
        let atten = (1.0 - d2 / r2).powi(2) * light.intensity;
        sum[0] += light.color[0] * atten;
        sum[1] += light.color[1] * atten;
        sum[2] += light.color[2] * atten;
    }
    sum
}

/// Exponential fog toward the level palette's clear color. Returns final
/// display RGB.
fn fog_blend(
    color: [f32; 3],
    dist: f32,
    settings: &CpuRenderSettings,
    env: &Environment,
) -> [u8; 3] {
    let fog = (-settings.fog_density * (dist - settings.fog_start).max(0.0)).exp();
    // Indoors the haze is the hand-tuned sickly brown; outdoors it fades to
    // the environment's bright sky haze.
    let clear = if env.outdoor {
        [
            env.fog_color[0] * 255.0,
            env.fog_color[1] * 255.0,
            env.fog_color[2] * 255.0,
        ]
    } else {
        [0.15 * 255.0, 0.125 * 255.0, 0.055 * 255.0]
    };
    let mut out = [0u8; 3];
    for c in 0..3 {
        out[c] = (color[c] * fog + clear[c] * (1.0 - fog)).clamp(0.0, 255.0) as u8;
    }
    out
}

/// Full lighting for one splat: combines every term above and returns
/// display-ready RGB.
pub fn shade(
    cam: &Camera,
    env: &Environment,
    settings: &CpuRenderSettings,
    atlas: &[u32],
    chunks: &[ChunkDraw],
    surface: &SplatSurface,
) -> [u8; 3] {
    // Emissive fixtures skip lighting entirely: they *are* the light.
    if surface.is_emissive {
        let boosted = [
            surface.base_color[0] * 1.5,
            surface.base_color[1] * 1.5,
            surface.base_color[2] * 1.5,
        ];
        return fog_blend(boosted, surface.dist, settings, env);
    }

    let to_cam = [
        surface.center[0] - cam.pos[0],
        surface.center[1] - cam.pos[1],
        surface.center[2] - cam.pos[2],
    ];
    let blend = AxisBlend::from_to_camera(to_cam);

    let face = face_response(&blend);
    let ao = contact_ao(surface.crowded_siblings);
    let light_norm = (surface.light_level / 15.0) * surface.shadow_factor * env.ambient_scale;
    let ambient = ambient_term(env, light_norm, &blend, ao, face);

    let beam = if cam.flashlight && blend.valid {
        let normal = blend.normal();
        let receiver = flashlight::BeamReceiver::cube(surface.center, surface.world_size, normal);
        flashlight::beam_contribution(cam, atlas, chunks, receiver, &settings.toggles)
    } else {
        [0.0; 3]
    };

    let dynamic = dynamic_light_term(cam, surface.center);

    let mut lit = [0.0f32; 3];
    for c in 0..3 {
        lit[c] = surface.base_color[c] * (ambient[c] + beam[c] + dynamic[c]);
    }
    fog_blend(lit, surface.dist, settings, env)
}
