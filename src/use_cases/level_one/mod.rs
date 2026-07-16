//! Level 1 — the Habitable Zone.
//!
//! A sprawling warehouse/parking structure honoring the Backrooms wiki canon:
//! bare concrete walls and massive pillars, institutional tile floors,
//! exposed ceiling pipes, low-hanging fog (the renderer's atmosphere, not
//! voxels), and fluorescent fixtures dimmer and less reliable than Level 0's.
//! No entities are generated at this stage.
//!
//! The level is carved into functional **sectors** on a fixed world lattice,
//! each with its own `StructuralSystem` flavor:
//!
//! * **Aquila** — bland parking-lot expanses: concrete slab, huge square
//!   pillars on a deep-span grid, sparse lighting.
//! * **Gild** — dense storage architecture: concrete room fabric packed with
//!   supply crates (almond water, food) behind doorway'd walls.
//! * **Gothic** — curved masonry: circular pillars joined by real arches
//!   under vaulted ceilings.
//! * **Construction** — perpetually unfinished: half-height walls, rebar
//!   clusters, dead ends, chaotic ceiling heights.
//!
//! Every column is a pure function of world-space coordinates and the seed,
//! so chunks tile seamlessly in any streaming order. Consumable supplies are
//! exported beside the geometry; whether one still exists is reality state
//! (`RealitySnapshot::supply_consumed`), never grid state.

pub(crate) mod generate;
#[cfg(test)]
mod tests;

/// Ceiling height budget shared with Level 0's grid allocation.
pub const GRID_HEIGHT_UNITS: f32 = 5.8;

/// Side of one sector cell, world units. Sector *type* changes per cell but
/// every structural lattice below is phased to world coordinates, so walls
/// and pillar grids run continuously across sector borders.
pub const SECTOR_CELL: f32 = 96.0;

/// Level 1 — one zero-sized generator, like its siblings.
pub struct HabitableLevel;

/// The functional sub-regions of Level 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sector {
    Aquila,
    Gild,
    Gothic,
    Construction,
}

impl Sector {
    /// Deterministic sector of a world position (hash of its lattice cell).
    pub fn at(seed: u32, wx: f32, wz: f32) -> Self {
        let cx = (wx / SECTOR_CELL).floor() as i64;
        let cz = (wz / SECTOR_CELL).floor() as i64;
        let roll = crate::use_cases::anomalies::determinism::hash(
            seed,
            0x1EE1_5EC7_0A00_0001,
            cx,
            cz,
        ) % 100;
        // The arrival cell is always Aquila: the wanderer steps out of the
        // door into readable parking-lot openness, exactly the first-report
        // canon of Level 1.
        if cx == 0 && cz == 0 {
            return Self::Aquila;
        }
        match roll {
            0..=34 => Self::Aquila,
            35..=62 => Self::Gild,
            63..=81 => Self::Gothic,
            _ => Self::Construction,
        }
    }

    /// HUD name of this sector.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Aquila => "AQUILA SECTOR",
            Self::Gild => "GILD SECTOR",
            Self::Gothic => "GOTHIC SECTOR",
            Self::Construction => "CONSTRUCTION SECTOR",
        }
    }
}
