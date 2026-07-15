//! Application (use-case) layer of the browser client.
//!
//! Everything in this module is pure Rust with no browser dependencies:
//! it can be exercised by native unit tests. Outer layers communicate with
//! it exclusively through the ports declared in [`ports`].

pub mod atlas;
pub mod collision;
pub mod engine;
pub mod generation_worker_policy;
pub mod player;
pub mod ports;
pub mod prepare_frame_lighting;
pub mod render_settings;
pub mod rendering;
pub mod streaming;
