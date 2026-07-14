//! Immutable anomaly instances and family-specific authored parameters.
//!
//! Placement belongs to the planner. Once an instance exists, this module
//! provides only deterministic geometry queries derived from its stable data.

use crate::entities::models::Position;

use super::geometry::{OrientedFootprint, OrthoBasis, WorldBounds};
use super::mix64;
use super::reality::{AnomalyStateStamp, RealitySnapshot};
use super::traversal::{PitHazard, TraversalGate};

pub type AnomalyId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum AnomalyKind {
    PillarExpanse = 0,
    BlackoutExpanse = 1,
    PitLattice = 2,
    RedRoom = 3,
    /// Stable pale arch-room anchor. Never carries gates, epochs, or wake
    /// mutation: the fixed landmark the mutable world is measured against.
    ArchwayRoom = 4,
}

pub(super) fn decode_kind(value: u32) -> Option<AnomalyKind> {
    match value {
        0 => Some(AnomalyKind::PillarExpanse),
        1 => Some(AnomalyKind::BlackoutExpanse),
        2 => Some(AnomalyKind::PitLattice),
        3 => Some(AnomalyKind::RedRoom),
        4 => Some(AnomalyKind::ArchwayRoom),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PillarLattice {
    pub bay_x: f32,
    pub bay_z: f32,
    pub phase_x: f32,
    pub phase_z: f32,
    pub min_side: f32,
    pub max_side: f32,
    pub variation_seed: u64,
}

impl PillarLattice {
    /// Variable wallpapered piers, always snapped to the shared 0.4u plan
    /// lattice. Entry regularity comes from placement, not a fixed size.
    pub fn pillar_size(&self, cell_x: i64, cell_z: i64) -> f32 {
        let hash = mix64(
            self.variation_seed
                ^ (cell_x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (cell_z as u64).rotate_left(29),
        );
        let steps = (((self.max_side - self.min_side) / 0.4).round() as u64).max(1);
        let step = (hash % (steps + 1)) as f32;
        (self.min_side + 0.4 * step).clamp(self.min_side, self.max_side)
    }
}

/// The two authored arch-room layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ArchLayout {
    /// Circulation-to-anomaly airlock: arch rhythm on both long walls.
    Transition = 0,
    /// Dead-end composition: one entrance arch, an alcove band at the far
    /// end, blind arcade rhythm along the long walls.
    DeadEnd = 1,
}

/// What passing through this archway *means* in the world graph. An arch is
/// a topological connector, not decorative carving: crossing a `Transition`
/// room changes which spatial grammar you are inside, and the two halves of
/// the room disagree just enough to be read in hindsight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ArchBehavior {
    /// A plain stable anchor room: both sides share one grammar (the
    /// historical arch-room behavior, still the most common).
    Anchor = 0,
    /// The far side belongs to a different architectural culture: ceiling
    /// regime, floor finish, and fixture rhythm disagree with the near side,
    /// and the far arcade's lintels ride lower.
    CultureSeam = 1,
    /// The far side repeats the same grammar at a larger scale: the ceiling
    /// lifts, openings broaden, and the light rhythm stretches.
    ScaleBreach = 2,
}

/// Arch-room composition parameters, all snapped to the 0.4u plan lattice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchProfile {
    pub layout: ArchLayout,
    /// Graph meaning of the crossing. Always [`ArchBehavior::Anchor`] for
    /// [`ArchLayout::DeadEnd`] rooms — a dead end connects nothing.
    pub behavior: ArchBehavior,
    /// Center-to-center rhythm of arch openings along the long (local X) walls.
    pub bay: f32,
    /// Clear width of one arch opening.
    pub opening: f32,
    /// Every `blind_every`-th arch is blind (a shelter niche reads as a
    /// skipped opening in the same rhythm).
    pub blind_every: u8,
    /// Height at which the arch springs from its jambs, world units. The
    /// head rises another 0.6 u toward the opening's center, so every
    /// non-blind arch is walkable through its middle by construction.
    pub spring_units: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitLattice {
    pub spacing_x: f32,
    pub spacing_z: f32,
    pub phase_x: f32,
    pub phase_z: f32,
    pub side: f32,
    pub depth: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnomalyInstance {
    pub id: AnomalyId,
    pub kind: AnomalyKind,
    pub footprint: OrientedFootprint,
    pub basis: OrthoBasis,
    pub macro_anchor: (i64, i64),
    pub pillar_lattice: Option<PillarLattice>,
    pub pit_lattice: Option<PitLattice>,
    pub arch: Option<ArchProfile>,
    pub gates: Vec<TraversalGate>,
    /// Epoch-invariant internal route half-width.
    pub skeleton_half_width: f32,
    /// Boundary depth kept regular before corruption/remapping begins.
    pub entry_band: f32,
}

impl AnomalyInstance {
    pub fn contains(&self, x: f32, z: f32) -> bool {
        self.footprint.contains(x, z)
    }

    pub fn local_coords(&self, x: f32, z: f32) -> (f32, f32) {
        self.footprint.local_coords(x, z)
    }

    pub fn world_coords(&self, x: f32, z: f32) -> Position {
        self.footprint.world_coords(x, z)
    }

    pub fn boundary_distance(&self, x: f32, z: f32) -> f32 {
        self.footprint.boundary_distance(x, z)
    }

    pub fn normalized_depth(&self, x: f32, z: f32) -> f32 {
        self.footprint.normalized_depth(x, z)
    }

    pub fn state<'a>(&self, reality: &'a RealitySnapshot) -> Option<&'a AnomalyStateStamp> {
        reality.lookup(self.id)
    }

    pub fn traversal_gates(&self) -> &[TraversalGate] {
        &self.gates
    }

    pub fn pillar_size(&self, cell_x: i64, cell_z: i64) -> Option<f32> {
        self.pillar_lattice
            .map(|lattice| lattice.pillar_size(cell_x, cell_z))
    }

    pub fn pit_hazards_for_bounds(&self, bounds: WorldBounds) -> Vec<PitHazard> {
        let Some(lattice) = self.pit_lattice else {
            return Vec::new();
        };
        let corners = [
            self.local_coords(bounds.min_x, bounds.min_z),
            self.local_coords(bounds.min_x, bounds.max_z),
            self.local_coords(bounds.max_x, bounds.min_z),
            self.local_coords(bounds.max_x, bounds.max_z),
        ];
        let min_lx = corners
            .iter()
            .map(|point| point.0)
            .fold(f32::INFINITY, f32::min);
        let max_lx = corners
            .iter()
            .map(|point| point.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_lz = corners
            .iter()
            .map(|point| point.1)
            .fold(f32::INFINITY, f32::min);
        let max_lz = corners
            .iter()
            .map(|point| point.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let ix0 = ((min_lx - lattice.phase_x) / lattice.spacing_x).floor() as i64 - 1;
        let ix1 = ((max_lx - lattice.phase_x) / lattice.spacing_x).ceil() as i64 + 1;
        let iz0 = ((min_lz - lattice.phase_z) / lattice.spacing_z).floor() as i64 - 1;
        let iz1 = ((max_lz - lattice.phase_z) / lattice.spacing_z).ceil() as i64 + 1;

        let mut hazards = Vec::new();
        for iz in iz0..=iz1 {
            for ix in ix0..=ix1 {
                let lx = lattice.phase_x + ix as f32 * lattice.spacing_x;
                let lz = lattice.phase_z + iz as f32 * lattice.spacing_z;
                let center = self.world_coords(lx, lz);
                if !self.contains(center.x, center.z)
                    || self.boundary_distance(center.x, center.z) < lattice.side
                    || !bounds.expanded(lattice.side).contains(center.x, center.z)
                {
                    continue;
                }
                // The immutable center lane remains solid and navigable.
                if lz.abs() < self.skeleton_half_width + lattice.side {
                    continue;
                }
                // The center between four pits is deterministically clear even
                // when pit side and spacing vary between instances.
                let recovery =
                    self.world_coords(lx + lattice.spacing_x * 0.5, lz + lattice.spacing_z * 0.5);
                let id = mix64(self.id ^ (ix as u64).rotate_left(17) ^ (iz as u64).rotate_left(43));
                hazards.push(PitHazard {
                    id,
                    instance_id: self.id,
                    center,
                    half_side: lattice.side * 0.5,
                    depth: lattice.depth,
                    recovery,
                });
            }
        }
        hazards.sort_by_key(|hazard| hazard.id);
        hazards.dedup_by_key(|hazard| hazard.id);
        hazards
    }
}
