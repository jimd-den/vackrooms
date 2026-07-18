//! Deterministic reference-rendering composition for regression tests.
//!
//! This is deliberately an outer module: it composes application contracts,
//! renderer adapters, core voxel use cases, and filesystem-capable drivers.
//! Nothing under the inward `application` layer depends on it.

pub mod artifact_sink;
pub mod renderer;
pub mod room;
pub mod scene;

pub use artifact_sink::ArtifactSinkPort;
pub use renderer::{
    ReferenceRenderSettings, ReferenceRendererPort, RenderedImage, render_reference,
};
pub use room::{CameraSpec, RoomFixture, RoomScene};
pub use scene::{RenderSceneSnapshot, build_render_scene};
