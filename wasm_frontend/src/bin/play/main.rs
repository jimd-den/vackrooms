//! Entry shim: the native viewer only exists off-wasm (winit/pollster are
//! native-only dependencies), but every file under `src/bin` is compiled for
//! each target, so the wasm32 build gets an empty main instead of the app.

#[cfg(not(target_arch = "wasm32"))]
mod app;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    app::run();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
