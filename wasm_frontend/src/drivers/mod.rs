//! Frameworks & Drivers layer — the only code that touches the browser.
//! Compiled exclusively for wasm32; the inner layers never import from here.

pub mod shaders;
pub mod webgl;
pub mod browser;
pub mod console_telemetry;
