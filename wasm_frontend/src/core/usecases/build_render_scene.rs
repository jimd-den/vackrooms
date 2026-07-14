use crate::core::domain::room::RoomScene;
use crate::core::ports::reference_renderer::RenderSceneSnapshot;

pub fn build_render_scene(scene: RoomScene) -> RenderSceneSnapshot {
    // TDD dummy implementation
    RenderSceneSnapshot {
        atlas: vec![],
        chunks: vec![],
    }
}
