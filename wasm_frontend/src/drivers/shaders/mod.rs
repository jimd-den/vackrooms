//! GLSL ES 3.00 programs for every GPU render path, one module per program:
//!
//! | module      | program                                                |
//! |-------------|--------------------------------------------------------|
//! | [`raymarch`]| fullscreen SVO raymarcher (`?renderer=raymarch`)       |
//! | [`surface`] | indexed greedy meshes (default renderer)               |
//! | [`splat`]   | instanced face splats (`?renderer=splat`)              |
//! | [`shadow`]  | depth-only hero-light shadow pass (surface + splat)    |
//! | [`chunks`]  | shared GLSL snippets spliced into the programs above   |
//!
//! Cross-cutting effects (the flashlight cone, the material palette, noise
//! helpers, tone mapping, flare cores) exist exactly once, in [`chunks`],
//! and are concatenated into each program's source at link time. The
//! flashlight cone's constants are the GLSL mirror of the CPU spec in
//! `adapters::cpu_splatter::flashlight` — change both together.
//!
//! The SVO node atlas texel encoding is documented in the core's
//! `OctreeGpuSerializer`; the compact vertex/instance formats in
//! `application::ports`.

pub mod chunks;
pub mod raymarch;
pub mod shadow;
pub mod splat;
pub mod surface;
