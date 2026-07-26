//! The shared column vocabulary every Level 0 sampler speaks.

use crate::domain::entities::voxel_grid::{VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_WALL};


/// What one (x, z) column of the level looks like.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ColumnPlan {
    /// Whether the walkable floor slab exists under this column.
    pub floor: bool,
    /// Raised solid floor above the base slab, world units (stair treads,
    /// landings, plinths). `0.0` is the ordinary one-voxel slab. Raised
    /// floor voxelizes as solid wall material so it collides — a tread the
    /// player ghosts through would be worse than one that blocks.
    pub floor_units: f32,
    /// Floor-to-ceiling solid (wall or pillar).
    pub solid: bool,
    /// Ceiling height in world units (top of the walkable space).
    pub ceiling_units: f32,
    /// Ceiling light above this column.
    pub light: bool,
    /// The light above this column burns red (red-room corruption). Only
    /// ever true inside an assembly whose whole room is the anomaly.
    pub red_light: bool,
    /// Solid band hanging from the ceiling down to this height (door
    /// lintels). The space below stays walkable.
    pub lintel_from_units: Option<f32>,
    /// A visible, non-colliding door leaf occupies the opening below its
    /// lintel. Gates and level exits remain separate semantic records.
    pub door_leaf: bool,
    /// Material used for solid columns and lintels (a wall-treatment voxel
    /// from the shared
    /// [`crate::domain::entities::environment::EnvironmentProfile`] semantics).
    pub wall_material: u8,
    /// Material of the floor slab (carpet depth/condition/fluid semantics).
    pub floor_material: u8,
    /// Material of the fixture voxel when `light` is set (warm fluorescent,
    /// red pressure, or cool glimmer).
    pub light_material: u8,
}

impl ColumnPlan {
    /// An open ordinary-fabric column with the default Level 0 materials.
    pub(crate) fn open(ceiling_units: f32) -> Self {
        Self {
            floor: true,
            floor_units: 0.0,
            solid: false,
            ceiling_units,
            light: false,
            red_light: false,
            lintel_from_units: None,
            door_leaf: false,
            wall_material: VOXEL_WALL,
            floor_material: VOXEL_FLOOR,
            light_material: VOXEL_LIGHT,
        }
    }
}
