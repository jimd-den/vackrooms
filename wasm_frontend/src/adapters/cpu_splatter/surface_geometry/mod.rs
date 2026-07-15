//! Prepared surface geometry for CPU splats.
//!
//! A splat is a projection primitive, not permission to invent a surface.
//! Its lighting normal must therefore come from an authored material face
//! that is actually open to air. The types in this module keep that geometry
//! decision separate from radiometry:
//!
//! * [`SurfaceExposure`] stores directional exposed-area coverage;
//! * [`exposure_for_octant`] propagates real parent/sibling topology;
//! * [`select_surface_normal`] selects one visible/material face from it.
//!
//! These operations are fixed-size and allocation-free in traversal. No
//! per-splat scene rays are needed.

mod project_surface_depth;
mod propagate_exposure_to_octant;
mod select_surface_normal;
mod surface_exposure;

pub(super) use project_surface_depth::{
    ProjectedSurfaceDepth, project_surface_depth, representative_surface_position,
};
pub(super) use propagate_exposure_to_octant::exposure_for_octant;
pub(super) use select_surface_normal::select_surface_normal;
pub use surface_exposure::{SurfaceExposure, SurfaceFace};

#[cfg(test)]
mod tests;
