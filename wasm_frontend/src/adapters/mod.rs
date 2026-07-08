//! Interface adapters: translate between the application layer's ports and
//! the concrete outside world (browser input events, the procedural
//! generation core).

pub mod input;
pub mod local_chunk_source;
pub mod cpu_splatter;
pub mod query_config;
