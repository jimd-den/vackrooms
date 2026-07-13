//! Pure domain language for the infinite macro-scale world graph.
//!
//! The world is not a heap of independent regions: it is an unbounded,
//! lazily evaluated *graph* whose vertices are architectural places and whose
//! edges are portals and vertical links. This module holds only that
//! language — plain data with stable identities — so every layer above it
//! (macro planner, region planner, voxelizer) can agree on what the world
//! *is* without agreeing on how it is rendered.
//!
//! Scale hierarchy (each level constrains the one below it):
//!
//! ```text
//! world seed
//!   -> MacroFields        multi-octave parameter fields (this file: data)
//!     -> MacroCell        160 u graph cell: nodes, portals, vertical links
//!       -> RegionPlan     80 u architectural plan (entities::architecture)
//!         -> ColumnPlan   one voxel column (use_cases::backrooms_level)
//! ```
//!
//! Determinism contract: everything here is derived from
//! `(seed, lattice coordinate)` by the planner in
//! `use_cases::world_topology`; nothing stored in these types may depend on
//! query order, chunk size, or LOD. Neighboring macro cells must compute
//! *identical* shared portals from their own side of an edge.

use crate::entities::models::Position;

/// Side of one macro graph cell, world units. Exactly 2×2 regions and one
/// anomaly macro-lattice cell, so all three lattices stay phase-aligned.
pub const MACRO_CELL_SIZE: f32 = 160.0;

/// Regions per macro cell along each axis (`MACRO_CELL_SIZE / REGION_SIZE`).
pub const REGIONS_PER_MACRO_CELL: i64 = 2;

/// Address of one macro cell on the infinite lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MacroCellId {
    pub x: i64,
    pub z: i64,
}

impl MacroCellId {
    pub fn new(x: i64, z: i64) -> Self {
        Self { x, z }
    }

    /// The macro cell containing a world point.
    pub fn containing(world_x: f32, world_z: f32) -> Self {
        Self {
            x: (world_x / MACRO_CELL_SIZE).floor() as i64,
            z: (world_z / MACRO_CELL_SIZE).floor() as i64,
        }
    }

    /// The macro cell containing a region lattice coordinate.
    pub fn of_region(region_x: i64, region_z: i64) -> Self {
        Self {
            x: region_x.div_euclid(REGIONS_PER_MACRO_CELL),
            z: region_z.div_euclid(REGIONS_PER_MACRO_CELL),
        }
    }

    /// Lower-left world corner.
    pub fn origin(&self) -> Position {
        Position::new(
            self.x as f32 * MACRO_CELL_SIZE,
            self.z as f32 * MACRO_CELL_SIZE,
        )
    }

    /// World-space center (where the cell samples its parameter fields).
    pub fn center(&self) -> Position {
        Position::new(
            (self.x as f32 + 0.5) * MACRO_CELL_SIZE,
            (self.z as f32 + 0.5) * MACRO_CELL_SIZE,
        )
    }
}

// ---------------------------------------------------------------------------
// Parameter fields: what the fractal math is actually *for*.
// ---------------------------------------------------------------------------

/// Multi-octave deterministic parameters sampled at a world point.
///
/// Noise never places a wall. It selects *where the building's rules change*:
/// each field below biases a probability or a style choice in a planner, and
/// the planner still decides the actual corridors, rooms, stairs and
/// thresholds. All values are in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacroFields {
    /// 0 = dense office warren, 1 = large incomplete open floor plates.
    pub openness: f32,
    /// Probability weight for stairwells, shafts and multi-height volumes.
    pub vertical_pressure: f32,
    /// 0 = freshly fitted out, 1 = decayed: dead lights, remnants, refits.
    pub institution_age: f32,
    /// Chance multiplier for impossible-architecture macro anomalies.
    pub anomaly_pressure: f32,
    /// Red-room event clustering weight — rare, clustered, never uniform.
    pub redroom_pressure: f32,
    /// How much a neighboring architect genome leaks into this area.
    pub style_blend: f32,
}

// ---------------------------------------------------------------------------
// Graph vertices.
// ---------------------------------------------------------------------------

/// What a graph node *is*, architecturally. Region planners translate the
/// kind into concrete circulation and assemblies; the kind never carries
/// geometry itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaceKind {
    /// The default endless office labyrinth fabric.
    Warren,
    /// A broad incomplete floor plate: sparse columns, long sight lines.
    OpenPlate,
    /// A multi-height volume with exposed structure.
    Atrium,
    /// A region that hosts a planned vertical link (stairs / shaft).
    Stairwell,
    /// A region reserved for a red-room encounter assembly.
    RedRoomEncounter,
}

/// One vertex of the world graph: a region-sized place with a stable
/// identity, a kind, and a macro elevation. `elevation` is an integer story
/// index in macro space — vertical links connect nodes whose elevations
/// differ, they never invent terrain-style fractional heights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldNode {
    pub id: u64,
    /// Region lattice coordinate this node plans.
    pub region: (i64, i64),
    pub kind: PlaceKind,
    /// Story index. The streamed world currently voxelizes elevation 0;
    /// links to other elevations are contracts for the vertical roadmap.
    pub elevation: i32,
}

// ---------------------------------------------------------------------------
// Graph edges.
// ---------------------------------------------------------------------------

/// A shared crossing point on a region edge. Both regions adjacent to the
/// edge derive the identical portal from the *edge's* lattice coordinate —
/// this is the contract that lets the primary circulation chain across
/// regions forever.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Portal {
    pub id: u64,
    /// Vertical edge lattice coordinate: the edge plane is
    /// `x = edge_x * REGION_SIZE`.
    pub edge_x: i64,
    /// Region row the portal belongs to.
    pub row_z: i64,
    /// World point where the primary route crosses the edge.
    pub at: Position,
}

/// How a vertical link behaves. Ordinary stairs are honest; the endless
/// variants are anomalies whose far ends re-address the world instead of
/// meshing a literal infinite staircase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VerticalLinkKind {
    /// Connects `elevation n` to `n + 1` over a constrained footprint.
    OrdinaryStair,
    /// Rises into a higher ceiling regime that never resolves — the flight
    /// visually continues into a dark aperture.
    EndlessAscent,
    /// Descends past believable landings into a re-addressed reality.
    EndlessDescent,
    /// A rare utilitarian shortcut: ladders, platforms, mechanical rooms.
    ServiceShaft,
}

/// One planned vertical connection anchored inside a region. The anchor is a
/// *reservation*: the region planner must place the corresponding stair
/// assembly near it (or fail loudly), never silently relocate it into a
/// different region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerticalLink {
    pub id: u64,
    pub kind: VerticalLinkKind,
    /// Region lattice coordinate that owns the reservation.
    pub region: (i64, i64),
    /// World-space anchor the stair footprint should gravitate toward.
    pub anchor: Position,
    pub from_elevation: i32,
    pub to_elevation: i32,
}

/// A red-room encounter planned as a *graph event*, not a biome. Events are
/// separated by a minimum macro-cell radius (the cooldown), and each event
/// targets exactly one region, where the planner promotes one ordinary
/// assembly into the encounter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RedRoomEvent {
    pub id: u64,
    /// The single region this event targets.
    pub region: (i64, i64),
    /// Tournament score in `[0, 1)` — lower wins. Kept for debugging and
    /// snapshot stability; consumers never branch on it.
    pub score: f32,
}

// ---------------------------------------------------------------------------
// The cell.
// ---------------------------------------------------------------------------

/// Everything the world graph knows about one 160 u macro cell. A pure
/// function of `(seed, cell id)` — see `use_cases::world_topology`.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroCell {
    pub id: MacroCellId,
    /// Parameter fields sampled at the cell center.
    pub fields: MacroFields,
    /// One node per region in the cell (row-major, 2×2).
    pub nodes: Vec<WorldNode>,
    /// Primary-route portals on the region edges this cell can see. Portals
    /// on shared edges are duplicated identically by both neighbors.
    pub portals: Vec<Portal>,
    /// Stair / shaft reservations inside this cell.
    pub vertical_links: Vec<VerticalLink>,
    /// The red-room event this cell won, if any.
    pub red_room_event: Option<RedRoomEvent>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macro_cell_addresses_are_stable_across_negative_space() {
        assert_eq!(MacroCellId::containing(-0.1, 0.1), MacroCellId::new(-1, 0));
        assert_eq!(MacroCellId::containing(159.9, -160.0), MacroCellId::new(0, -1));
        assert_eq!(MacroCellId::of_region(-1, -2), MacroCellId::new(-1, -1));
        assert_eq!(MacroCellId::of_region(2, 3), MacroCellId::new(1, 1));
    }

    #[test]
    fn cell_center_sits_inside_its_own_bounds() {
        let cell = MacroCellId::new(-3, 2);
        let origin = cell.origin();
        let center = cell.center();
        assert!(center.x > origin.x && center.x < origin.x + MACRO_CELL_SIZE);
        assert!(center.z > origin.z && center.z < origin.z + MACRO_CELL_SIZE);
    }
}
