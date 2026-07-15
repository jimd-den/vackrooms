use crate::adapters::cpu_splatter::settings::CpuRenderSettings;
use crate::application::ports::{Environment, FrameParams, LightSource};
use crate::application::render_settings::RenderToggles;
use crate::core::domain::room::CameraSpec;

#[derive(Debug, Clone, Copy)]
pub struct ReferenceRenderSettings {
    pub width: u32,
    pub height: u32,
    pub camera: CameraSpec,
    pub environment: Environment,
    pub toggles: RenderToggles,
    pub cpu: CpuRenderSettings,
}

impl ReferenceRenderSettings {
    /// Apply the fixture camera's vertical field of view to the CPU settings
    /// shared by both reference renderers.
    pub fn effective_cpu_settings(&self) -> CpuRenderSettings {
        let mut cpu = self.cpu;
        if self.camera.fov_degrees.is_finite() && (1.0..179.0).contains(&self.camera.fov_degrees) {
            cpu.fov_tan = (self.camera.fov_degrees.to_radians() * 0.5).tan();
        }
        cpu.toggles = self.toggles;
        cpu
    }

    pub fn frame_params(&self, scene_lights: &[LightSource]) -> FrameParams {
        FrameParams {
            camera_pos: self.camera.position,
            yaw: self.camera.yaw,
            pitch: self.camera.pitch,
            scene_lights: scene_lights.to_vec(),
            environment: self.environment,
            ..FrameParams::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSceneSnapshot {
    /// Row-padded RGBA32UI SVO texels (four `u32` values per node).
    pub atlas: Vec<u32>,
    pub chunks: Vec<crate::application::ports::ChunkDraw>,
    /// Analytic fixtures derived from the same emissive voxel surfaces as
    /// the atlas. Reference renderers must not silently fall back to the
    /// low-precision bake when production frames use physical lights.
    pub scene_lights: Vec<LightSource>,
}

pub trait ReferenceRendererPort {
    fn render(
        &mut self,
        scene: &RenderSceneSnapshot,
        settings: &ReferenceRenderSettings,
    ) -> RenderedImage;
}
