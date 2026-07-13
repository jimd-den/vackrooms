//! CPU microvoxel splatting renderer — a software implementation of
//! [`crate::application::ports::RendererPort`] for machines with no usable
//! GPU (or `?renderer=cpu`).
//!
//! Instead of marching a ray per pixel, the CPU path inverts the loop: it
//! walks the SVO front-to-back and *splats* nodes onto a z-buffered
//! framebuffer, sized by their projected footprint. The module is split by
//! responsibility — each file owns exactly one concern:
//!
//! | module         | concern                                              |
//! |----------------|------------------------------------------------------|
//! | [`settings`]   | tuning knobs + the shared optimization switchboard   |
//! | [`atlas`]      | SVO texel decoding and per-upload MIP filtering      |
//! | [`raycast`]    | secondary rays (flashlight occlusion, shadow rays)   |
//! | [`camera`]     | yaw/pitch basis identical to the GPU shaders         |
//! | [`flashlight`] | the spotlight cone — the single spec of its shape    |
//! | [`shading`]    | per-splat lighting: ambient, AO, flares, fog         |
//! | [`rasterizer`] | facade; split target + traversal frame pipeline     |
//!
//! This module is platform-free (no web-sys): the browser driver only blits
//! the RGBA buffer (`drivers::cpu_canvas`). All geometry/shading logic is
//! natively unit-tested in [`tests`].

mod atlas;
mod camera;
mod flashlight;
mod rasterizer;
mod raycast;
mod settings;
mod shading;
#[cfg(test)]
mod tests;

pub use rasterizer::{SoftwareRasterizer, SoftwareRasterizerTelemetry};
pub use raycast::{RayHit, trace_svo};
pub use settings::{CpuRenderSettings, CpuShadowMode};
