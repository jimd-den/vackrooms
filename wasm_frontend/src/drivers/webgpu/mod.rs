//! Portable WebGPU renderer infrastructure.
//!
//! Browser surface creation is a driver concern, while configuration and
//! render-pipeline construction stay target-independent so the same choices
//! can be exercised by native Vulkan tests.

pub mod config;
pub mod frame_resources;
pub mod gpu_types;
pub mod pipelines;
pub mod shader;

#[cfg(target_arch = "wasm32")]
pub mod browser_context;
#[cfg(target_arch = "wasm32")]
pub mod camera_state;
#[cfg(target_arch = "wasm32")]
pub mod renderer;
