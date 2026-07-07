//! Application (use-case) layer of the browser client.
//!
//! Everything in this module is pure Rust with no browser dependencies:
//! it can be exercised by native unit tests. Outer layers communicate with
//! it exclusively through the ports declared in [`ports`].

pub mod ports;
pub mod collision;
pub mod player;
pub mod streaming;
pub mod atlas;
pub mod engine;
