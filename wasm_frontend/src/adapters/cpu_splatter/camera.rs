//! Pinhole camera for the splatter, matching the GPU shaders' yaw/pitch
//! basis exactly (same formulas as `drivers::webgl`'s per-frame uniforms),
//! so switching renderers never changes the framing.

use crate::application::ports::{DynamicLight, FrameParams, MAX_DYNAMIC_LIGHTS};

use super::settings::CpuRenderSettings;

pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Camera basis + per-frame light state, derived once per frame.
pub struct Camera {
    pub pos: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    /// Focal length in pixels: h / (2 * tan(fov/2)).
    pub focal_px: f32,
    pub half_w: f32,
    pub half_h: f32,
    pub flashlight: bool,
    /// Per-frame dynamic lights (flares), pre-flickered by the engine.
    pub dynamic_lights: [DynamicLight; MAX_DYNAMIC_LIGHTS],
    pub dynamic_light_count: usize,
}

impl Camera {
    pub fn new(
        frame: &FrameParams,
        width: usize,
        height: usize,
        settings: &CpuRenderSettings,
    ) -> Self {
        let (sy, cy) = frame.yaw.sin_cos();
        let (sp, cp) = frame.pitch.sin_cos();
        Self {
            pos: frame.camera_pos,
            right: [cy, 0.0, -sy],
            up: [sp * sy, cp, sp * cy],
            forward: [-cp * sy, sp, -cp * cy],
            focal_px: height as f32 / (2.0 * settings.fov_tan),
            half_w: width as f32 / 2.0,
            half_h: height as f32 / 2.0,
            flashlight: frame.flashlight,
            dynamic_lights: frame.dynamic_lights,
            dynamic_light_count: (frame.dynamic_light_count as usize).min(MAX_DYNAMIC_LIGHTS),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_dynamic_light_count_is_clamped_to_storage() {
        let frame = FrameParams {
            dynamic_light_count: u8::MAX,
            ..FrameParams::default()
        };
        let camera = Camera::new(&frame, 16, 16, &CpuRenderSettings::default());

        assert_eq!(frame.active_dynamic_lights().len(), MAX_DYNAMIC_LIGHTS);
        assert_eq!(camera.dynamic_light_count, MAX_DYNAMIC_LIGHTS);
    }
}
