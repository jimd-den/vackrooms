//! Interface adapters: translate between the application layer's ports and
//! the concrete outside world (browser input events, the procedural
//! generation core).

pub mod chunk_codec;
pub mod cpu_splatter;
pub mod face_instances;
pub mod input;
pub mod local_chunk_source;
pub mod query_config;
pub mod section_locator;
pub mod surface_mesh;
