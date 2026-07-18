//! Application (use-case) layer of the browser client.
//!
//! Everything in this module is pure Rust with no browser dependencies:
//! it can be exercised by native unit tests. Outer layers communicate with
//! it exclusively through the ports declared in [`ports`].

pub mod atlas;
pub mod body;
pub mod collision;
pub mod engine;
pub mod flares;
pub mod generation_worker_policy;
pub mod navigation;
pub mod perf_governor;
pub mod player;
pub mod ports;
pub mod prepare_frame_lighting;
pub mod quality;
pub mod render_settings;
pub mod rendering;
pub mod streaming;
mod survival_inventory;
pub mod thermal;
