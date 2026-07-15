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
//! | decode_srgb_albedo | authored color -> linear albedo at upload       |
//! | [`raycast`]    | secondary rays (flashlight occlusion, shadow rays)   |
//! | [`camera`]     | yaw/pitch basis identical to the GPU shaders         |
//! | [`flashlight`] | the spotlight cone — the single spec of its shape    |
//! | surface_geometry | exposed faces and material-oriented normals       |
//! | [`shading`]    | linear radiometry, analytic fixtures, flares, fog    |
//! | [`rasterizer`] | state facade + purpose-named frame and SVO stages    |
//!
//! This module is platform-free (no web-sys): the browser driver only blits
//! the RGBA buffer (`drivers::cpu_canvas`). All geometry/shading logic is
//! natively unit-tested in [`tests`].

pub mod atlas;
pub mod camera;
mod decode_srgb_albedo;
mod flashlight;
pub mod rasterizer;
mod raycast;
pub mod settings;
pub mod shading;
pub mod surface_geometry;
mod update_atlas_rows;
#[cfg(test)]
mod tests;

pub use rasterizer::{SoftwareRasterizer, SoftwareRasterizerTelemetry};
pub use raycast::{RayHit, trace_svo};
pub use settings::{
    CpuFlashlightVisibility, CpuQualityPreset, CpuRenderSettings, CpuSettingsLimits, CpuShadowMode,
};
