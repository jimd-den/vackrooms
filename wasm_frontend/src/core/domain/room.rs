use crate::application::ports::Environment;

/// Test camera position and orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraSpec {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_degrees: f32,
}

pub struct RoomScene {
    // To be defined based on build_render_scene
}

pub trait RoomFixture {
    fn id(&self) -> &'static str;
    fn build(&self) -> RoomScene;
    fn camera(&self) -> CameraSpec;
    fn environment(&self) -> Environment;
}
