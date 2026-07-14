//! Level 0 generation stages.
//!
//! Planning decides *what exists* at world coordinates.  This module owns
//! the later, resolution-dependent stages that turn those decisions into an
//! output voxel area.  Keeping that boundary explicit lets the same infinite
//! world be requested as many small chunks or one larger, finer area.

mod column_field;
mod voxelize;

pub(crate) use column_field::ColumnField;
pub(crate) use voxelize::voxelize_columns;
