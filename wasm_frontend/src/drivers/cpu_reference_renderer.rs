use crate::adapters::cpu_splatter::rasterizer::SoftwareRasterizer;
use crate::application::ports::RendererPort;
use crate::reference::renderer::{ReferenceRenderSettings, ReferenceRendererPort, RenderedImage};
use crate::reference::scene::RenderSceneSnapshot;

/// Stateless composition driver for the production CPU splatter. A fresh
/// rasterizer per render makes repeated reference calls independent of frame
/// counters, shadow caches, and earlier atlas uploads.
pub struct CpuReferenceRenderer;

impl CpuReferenceRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl ReferenceRendererPort for CpuReferenceRenderer {
    fn render(
        &mut self,
        scene: &RenderSceneSnapshot,
        settings: &ReferenceRenderSettings,
    ) -> RenderedImage {
        let mut rasterizer =
            SoftwareRasterizer::new(settings.width as usize, settings.height as usize);
        rasterizer.settings = settings.effective_cpu_settings();
        rasterizer.upload_atlas(&scene.atlas);
        rasterizer.draw(&settings.frame_params(&scene.scene_lights), &scene.chunks);

        RenderedImage {
            width: settings.width,
            height: settings.height,
            rgba: rasterizer.framebuffer().to_vec(),
        }
    }
}
