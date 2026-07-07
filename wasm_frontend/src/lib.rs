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

pub mod application;
pub mod adapters;

#[cfg(target_arch = "wasm32")]
pub mod drivers;

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
