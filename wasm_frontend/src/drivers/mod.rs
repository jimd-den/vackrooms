//! Frameworks & Drivers layer — the only code that touches the browser.
//! Compiled exclusively for wasm32; the inner layers never import from here.

pub mod browser;
pub mod console_telemetry;
pub mod cpu_canvas;
pub mod gl;
pub mod shaders;
pub mod splat_webgl;
pub mod surface_webgl;
pub mod webgl;
pub mod worker_source;
