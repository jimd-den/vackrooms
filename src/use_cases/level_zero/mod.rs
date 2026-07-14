//! Level 0 — the Backrooms proper.
//!
//! Everything is a pure function of *world-space* coordinates, so chunks tile
//! seamlessly no matter which order they stream in. The goal is *believable
//! building architecture* with depth — not flat mazes:
//!
//! * **Ceiling regions** use broad smooth fields instead of a low default
//!   slab: regular dropped ceilings are 3.2--3.6 u, open expanses 3.8--4.4 u,
//!   vaults 4.5--5.4 u, and compression zones are rare deliberate contrast.
//! * **Halls**: one dominant cross-region route is wide enough to read as
//!   circulation through a vast building, with at most two interior branches.
//!   Its edge walls persist in long runs and open into large rooms or fabric,
//!   not a repeated doorway grid.
//! * **Fabric**: the default fill *is* the labyrinth — an irregular warren
//!   of rooms on a hidden 7.2 u lattice, chained through hashed doorways
//!   (binary-tree rule guarantees global connectivity chunk-locally) and
//!   merged by porosity-driven wall dropout so no grid is ever readable.
//!   Open expanses are the exception that punctuates it, never the default.
//!
//! The stages scream their intent, one module each:
//!
//! * [`fabric`] — the endless unplanned warren between planned systems.
//! * [`circulation_sampler`] — corridor ceilings and dissolving edge walls.
//! * [`assembly_sampler`] — planned program masses and their fixtures.
//! * [`compose_column`] — the one priority order that arbitrates a column.
//! * [`generate`] — voxelization, semantic exports, and runtime lights.

mod assembly_sampler;
mod circulation_sampler;
mod column_field;
mod column_plan;
mod compose_column;
mod fabric;
mod generate;
#[cfg(test)]
mod tests;
mod voxelize;

pub(crate) use column_field::ColumnField;
pub(crate) use column_plan::ColumnPlan;
pub(crate) use voxelize::voxelize_columns;

/// Ceiling height of the tallest (atrium) vaults, world units.
pub const MAX_CEILING_UNITS: f32 = 5.4;
/// Total grid height: headroom above the tallest vault.
pub const GRID_HEIGHT_UNITS: f32 = 5.8;

pub(crate) const DOOR_WIDTH: f32 = 1.2;
pub(crate) const DOOR_HEIGHT: f32 = 2.2;

/// Level 0 — the Backrooms proper. One zero-sized generator; every stage it
/// composes is a pure function of world-space coordinates and the immutable
/// reality snapshot, so chunks tile seamlessly in any streaming order.
pub struct BackroomsLevel;
