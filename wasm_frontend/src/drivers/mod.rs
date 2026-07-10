//! Frameworks & Drivers layer — the only code that touches the browser.
//! Compiled exclusively for wasm32; the inner layers never import from here.

pub mod browser;
pub mod console_telemetry;
pub mod cpu_canvas;
pub mod shaders;
pub mod surface_webgl;
pub mod webgl;
