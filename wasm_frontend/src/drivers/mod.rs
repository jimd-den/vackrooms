#[cfg(target_arch = "wasm32")]
pub mod browser;
#[cfg(target_arch = "wasm32")]
pub mod console_telemetry;
#[cfg(target_arch = "wasm32")]
pub mod worker_source;

// Pure request bookkeeping for the browser generation-worker driver. Keeping
// it free of web-sys makes failure and stale-message behavior natively
// testable even though Worker itself only exists on wasm32.
#[cfg(any(target_arch = "wasm32", test))]
mod generation_worker_requests;

pub mod cpu_reference_renderer;
pub mod fs_artifact_sink;
pub mod raymarch_reference_renderer;
pub mod webgpu;
