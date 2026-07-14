//! Per-frame light selection — one implementation for the surface and
//! splat renderers, which previously carried diverging copies.
//!
//! Inputs: fixture lights from every visible chunk (deduplicated by id —
//! a fixture near a chunk border is listed by several chunks) plus the
//! frame's dynamic lights (dropped flares). The four most important lights
//! win the shader's uniform slots; the most important *fixture* light also
//! wins the hero shadow map (a flare at ankle height would collapse the
//! top-down shadow frustum, so dynamic lights never own it).
//!
//! Importance = radius * sqrt(intensity) / (distance² + 0.1): bigger and
//! brighter lights reach farther, nearby lights dominate.

use crate::application::ports::{FrameParams, LightSource};

/// Maximum shader light slots (matches the GLSL uniform arrays).
pub const MAX_SHADER_LIGHTS: usize = 4;

/// The frame's selected lights, packed for `uniform[1i|3fv|4fv]` upload.
pub struct SelectedLights {
    pub count: usize,
    pub positions: [f32; MAX_SHADER_LIGHTS * 3],
    pub colors: [f32; MAX_SHADER_LIGHTS * 3],
    /// Per light: radius, intensity, half_size.x, half_size.y.
    pub params: [f32; MAX_SHADER_LIGHTS * 4],
    /// Slot index of the shadow-casting hero light, or -1.
    pub hero_slot: i32,
    /// Hero light's position/radius (meaningful when `hero_slot >= 0`).
    pub hero_position: [f32; 3],
    pub hero_radius: f32,
}

/// Wraps the frame's dynamic lights as fixture-shaped sources with sentinel
/// ids in the top range, so one selection pass handles both kinds.
pub fn dynamic_light_sources(frame: &FrameParams) -> Vec<LightSource> {
    frame
        .active_dynamic_lights()
        .iter()
        .enumerate()
        .map(|(i, d)| LightSource {
            id: u64::MAX - i as u64,
            position: d.position,
            half_size: [0.25, 0.25],
            color: d.color,
            radius: d.radius,
            intensity: d.intensity,
            flicker_mode: 0,
            enabled: true,
        })
        .collect()
}

fn is_dynamic(light: &LightSource) -> bool {
    light.id > u64::MAX - 16
}

/// Selects and packs the frame's lights. `fixtures` should already be
/// deduplicated by id (pass a `HashMap`'s values); `dynamic` comes from
/// [`dynamic_light_sources`].
pub fn select_lights<'a>(
    fixtures: impl Iterator<Item = &'a LightSource>,
    dynamic: &'a [LightSource],
    frame: &FrameParams,
) -> SelectedLights {
    let mut ranked: Vec<(&LightSource, f32)> = fixtures
        .chain(dynamic.iter())
        .map(|light| {
            let dx = light.position[0] - frame.camera_pos[0];
            let dy = light.position[1] - frame.camera_pos[1];
            let dz = light.position[2] - frame.camera_pos[2];
            let dist2 = dx * dx + dy * dy + dz * dz;
            (light, light.radius * light.intensity.sqrt() / (dist2 + 0.1))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let count = ranked.len().min(MAX_SHADER_LIGHTS);
    let mut selected = SelectedLights {
        count,
        positions: [0.0; MAX_SHADER_LIGHTS * 3],
        colors: [0.0; MAX_SHADER_LIGHTS * 3],
        params: [0.0; MAX_SHADER_LIGHTS * 4],
        hero_slot: -1,
        hero_position: [0.0; 3],
        hero_radius: 0.0,
    };
    for (i, (light, _)) in ranked[..count].iter().enumerate() {
        selected.positions[i * 3..i * 3 + 3].copy_from_slice(&light.position);
        selected.colors[i * 3..i * 3 + 3].copy_from_slice(&light.color);
        selected.params[i * 4] = light.radius;
        selected.params[i * 4 + 1] = light.intensity;
        selected.params[i * 4 + 2] = light.half_size[0];
        selected.params[i * 4 + 3] = light.half_size[1];
    }

    // Hero shadow: the strongest *fixture* light in the selected slots.
    if let Some(slot) = ranked[..count].iter().position(|(l, _)| !is_dynamic(l)) {
        selected.hero_slot = slot as i32;
        selected.hero_position = ranked[slot].0.position;
        selected.hero_radius = ranked[slot].0.radius;
    }
    selected
}

/// Flare-core uniforms (`uCoreCount`/`uCores`/`uCoreColors`), independent of
/// the merged light slots so a flare culled from the four shading lights
/// still shows its ember.
pub struct FlareCores {
    pub count: i32,
    /// xyz = position, w = pre-flickered intensity.
    pub pos_intensity: [f32; MAX_SHADER_LIGHTS * 4],
    pub colors: [f32; MAX_SHADER_LIGHTS * 3],
}

pub fn flare_cores(frame: &FrameParams) -> FlareCores {
    let cores = frame.active_dynamic_lights();
    let mut packed = FlareCores {
        count: cores.len() as i32,
        pos_intensity: [0.0; MAX_SHADER_LIGHTS * 4],
        colors: [0.0; MAX_SHADER_LIGHTS * 3],
    };
    for (i, c) in cores.iter().enumerate() {
        packed.pos_intensity[i * 4..i * 4 + 3].copy_from_slice(&c.position);
        packed.pos_intensity[i * 4 + 3] = c.intensity;
        packed.colors[i * 3..i * 3 + 3].copy_from_slice(&c.color);
    }
    packed
}
