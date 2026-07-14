use crate::adapters::cpu_splatter::camera::Camera;
use crate::adapters::cpu_splatter::rasterizer::SoftwareRasterizer;
use crate::adapters::cpu_splatter::settings::CpuRenderSettings;
use crate::application::ports::{ChunkDraw, FrameParams, RendererPort};
use crate::core::ports::reference_renderer::{
    ReferenceRenderSettings, ReferenceRendererPort, RenderSceneSnapshot, RenderedImage,
};

pub struct CpuReferenceRenderer {
    rasterizer: SoftwareRasterizer,
}

impl CpuReferenceRenderer {
    pub fn new() -> Self {
        Self {
            rasterizer: SoftwareRasterizer::new(160, 90),
        }
    }
}

impl ReferenceRendererPort for CpuReferenceRenderer {
    fn render(
        &mut self,
        scene: &RenderSceneSnapshot,
        settings: &ReferenceRenderSettings,
    ) -> RenderedImage {
        self.rasterizer.settings = settings.cpu.clone();
        self.rasterizer.settings.toggles = settings.toggles;
        self.rasterizer
            .resize(settings.width as usize, settings.height as usize);
        self.rasterizer.upload_atlas(&scene.atlas);

        let mut frame_params = FrameParams::default();
        frame_params.camera_pos = settings.camera.position;
        frame_params.yaw = settings.camera.yaw;
        frame_params.pitch = settings.camera.pitch;
        frame_params.environment = settings.environment;

        // Ensure deterministic shadow retracing by overriding frame index
        // Since frame_index is private and the shadow cache relies on it,
        // we might need to recreate the rasterizer or find a way to freeze it.
        // For test mode, the shadow_mode=Off or creating a new rasterizer per render handles this.

        self.rasterizer.draw(&frame_params, &scene.chunks);

        let rgba = self.rasterizer.framebuffer().to_vec();

        RenderedImage {
            width: settings.width,
            height: settings.height,
            rgba,
        }
    }
}
