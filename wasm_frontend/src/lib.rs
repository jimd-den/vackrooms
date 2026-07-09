//! # wasm_frontend — browser client for the vackrooms voxel engine
//!
//! This crate is the *outer rings* of the engine's Clean Architecture when it
//! runs in a browser. The dependency rule points strictly inward:
//!
//! ```text
//! drivers (web-sys, WebGL2, DOM)        <- wasm32 only
//!    |
//! adapters (input mapping, SVO atlas, local chunk source)
//!    |
//! application (player physics, streaming policy, frame orchestration, ports)
//!    |
//! vackrooms core (entities + use cases: VoxelGrid, SVO, generation, lighting)
//! ```
//!
//! * `application` and `adapters` are platform-agnostic and unit-tested with
//!   plain `cargo test` on the host.
//! * `drivers` is the only module that talks to the browser, and it is only
//!   compiled for `wasm32`.
//! * The `#[wasm_bindgen(start)]` entry point below is the composition root:
//!   it wires concrete drivers into the application's ports.

pub mod adapters;
pub mod application;

#[cfg(target_arch = "wasm32")]
pub mod drivers;

#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

#[cfg(target_arch = "wasm32")]
pub static DOOM_CONTROLS: AtomicBool = AtomicBool::new(false);

/// Mouse look inversion (Y axis).
#[cfg(target_arch = "wasm32")]
pub static INVERT_Y: AtomicBool = AtomicBool::new(false);

/// Mouse sensitivity multiplier, stored as f32 bits. 1.0 = default.
#[cfg(target_arch = "wasm32")]
pub static MOUSE_SENSITIVITY_BITS: AtomicU32 = AtomicU32::new(0x3F80_0000); // 1.0f32

/// Manual internal-resolution override as f32 bits; 0.0 = automatic
/// (adaptive governor).
#[cfg(target_arch = "wasm32")]
pub static RENDER_SCALE_BITS: AtomicU32 = AtomicU32::new(0);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_doom_controls(enabled: bool) {
    DOOM_CONTROLS.store(enabled, Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_invert_y(enabled: bool) {
    INVERT_Y.store(enabled, Ordering::Relaxed);
}

/// Mouse/touch look sensitivity multiplier (clamped to 0.1–5.0).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_mouse_sensitivity(multiplier: f32) {
    let m = if multiplier.is_finite() {
        multiplier.clamp(0.1, 5.0)
    } else {
        1.0
    };
    MOUSE_SENSITIVITY_BITS.store(m.to_bits(), Ordering::Relaxed);
}

/// Forces the internal render resolution scale (0.25–1.0), or restores the
/// adaptive governor when `scale` is 0.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_render_scale(scale: f32) {
    let s = if scale.is_finite() && scale > 0.0 {
        scale.clamp(0.25, 1.0)
    } else {
        0.0
    };
    RENDER_SCALE_BITS.store(s.to_bits(), Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
mod entry {
    use wasm_bindgen::prelude::*;

    /// Composition root. Runs automatically when the wasm module is
    /// instantiated by `static/index.html`.
    #[wasm_bindgen(start)]
    pub fn start() -> Result<(), JsValue> {
        console_error_panic_hook::set_once();
        crate::drivers::browser::boot()
    }
}
