use crate::core::ports::reference_renderer::{
    ReferenceRenderSettings, ReferenceRendererPort, RenderSceneSnapshot, RenderedImage,
};

pub fn render_reference(
    scene: &RenderSceneSnapshot,
    settings: &ReferenceRenderSettings,
    mut renderer: impl ReferenceRendererPort,
) -> RenderedImage {
    renderer.render(scene, settings)
}
