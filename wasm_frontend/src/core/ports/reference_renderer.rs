use crate::adapters::cpu_splatter::settings::CpuRenderSettings;
use crate::application::ports::Environment;
use crate::application::render_settings::RenderToggles;
use crate::core::domain::room::CameraSpec;

pub struct ReferenceRenderSettings {
    pub width: u32,
    pub height: u32,
    pub camera: CameraSpec,
    pub environment: Environment,
    pub fixed_frame: u64,
    pub toggles: RenderToggles,
    pub cpu: CpuRenderSettings,
}

pub struct RenderedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub struct RenderSceneSnapshot {
    // Need to define based on snapshot requirements (atlas, chunks, etc)
    pub atlas: Vec<u32>,
    pub chunks: Vec<crate::application::ports::ChunkDraw>,
    // we can expand this as needed
}

pub trait ReferenceRendererPort {
    fn render(
        &mut self,
        scene: &RenderSceneSnapshot,
        settings: &ReferenceRenderSettings,
    ) -> RenderedImage;
}
