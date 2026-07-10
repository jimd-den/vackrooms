//! Pure architectural domain types for procedurally *planned* levels.
//!
//! These describe a region-scale building plan — who designed it
//! ([`ArchitectGenome`]), what each space is for ([`SpaceProgram`]), how
//! circulation feeds spaces ([`CirculationSpine`]), and how spaces are
//! assembled ([`AssemblyInstance`]) — before anything is voxelized. The
//! planner (`use_cases::region_plan`) fills these in; the level generator
//! samples them per voxel column. Nothing here knows about voxels, chunks,
//! rendering, or WASM.
//!
//! All types are small, plain data. Determinism comes from how they are
//! *derived* (seed + region coordinate), not from anything stored here.

use crate::entities::models::Position;

// ---------------------------------------------------------------------------
// Architect genomes: a "design culture" as a seed-derived parameter set.
// ---------------------------------------------------------------------------

/// How a designer routes people through a region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CirculationStyle {
    /// One straight main corridor wall-to-wall.
    StraightSpine,
    /// A main corridor that bends at interior waypoints.
    MeanderingSpine,
    /// A primary spine with two short, disconnected-looking branches.
    PairedBranches,
    /// A spine plus stub branches that dead-end (unease by design).
    TreeWithCulDeSacs,
}

/// How a designer holds the ceiling up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralSystem {
    /// Columns on a square grid.
    RegularGrid,
    /// Wider bays in one axis, expressed as ceiling beams.
    DeepSpansWithBeams,
    /// Alternate column rows shifted half a bay.
    OffsetGrid,
    /// Columns only around cores and the perimeter.
    CoreAndShell,
}

/// Preferred room shapes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProportionRules {
    /// Smallest room side a designer will draw, world units.
    pub min_side: f32,
    /// Largest room side, world units.
    pub max_side: f32,
    /// 0 = square rooms, 1 = strongly elongated rooms.
    pub elongation: f32,
}

/// How rooms open onto circulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdLanguage {
    /// Standard door-width opening under a solid lintel.
    DoorWithLintel,
    /// Full-height opening, no lintel.
    OpenPortal,
    /// Double-width opening under a lintel (suites, conference).
    WidePortal,
}

/// Ceiling articulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CeilingLanguage {
    /// Flat acoustic tile everywhere.
    FlatTiles,
    /// Beam grid with recessed panels.
    Coffered,
    /// Lowered continuous soffit (mechanical-heavy designers).
    ExposedSoffit,
}

/// Where the lights go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightingLanguage {
    /// Fluorescent modules on the ceiling grid.
    GridPanels,
    /// Continuous strips that follow circulation.
    StripsAlongCirculation,
    /// Sparse single fixtures (renovated or neglected areas).
    SparsePendants,
}

/// What later occupants did to the original plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenovationStyle {
    /// Plan left as designed.
    Untouched,
    /// One assembly re-partitioned in a second designer's language.
    PartialRefit,
    /// Renovation structure overlaid on the original (contradictory columns).
    LayeredRefits,
}

/// A design culture: every plan decision in a region is parameterized by one
/// of these. Derived deterministically from `(seed, region coordinate)`.
#[derive(Clone, Debug)]
pub struct ArchitectGenome {
    pub circulation: CirculationStyle,
    pub structural_system: StructuralSystem,
    pub room_proportions: ProportionRules,
    pub threshold_language: ThresholdLanguage,
    pub ceiling_language: CeilingLanguage,
    pub lighting_language: LightingLanguage,
    /// 0 = bare shells, 1 = densely partitioned interiors.
    pub furnishing_density: f32,
    pub renovation_history: RenovationStyle,
    /// 0 = happily asymmetric, 1 = mirrors and centers everything.
    pub tolerance_for_symmetry: f32,
}

// ---------------------------------------------------------------------------
// Program: what a space is *for*.
// ---------------------------------------------------------------------------

/// First-class space semantics (the old room stamps, promoted to meaning).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpaceProgram {
    Arrival,
    Reception,
    MainCorridor,
    SecondaryHall,
    OpenOffice,
    PrivateOffice,
    ConferenceRoom,
    WaitingArea,
    BreakRoom,
    Storage,
    ServerRoom,
    RestroomCore,
    Stair,
    Mechanical,
    Atrium,
    /// Planned, built, never occupied: lit shell or unlit void.
    AbandonedExpansion,
}

// ---------------------------------------------------------------------------
// Geometry helpers (deliberately minimal — rectilinear plans only).
// ---------------------------------------------------------------------------

/// A simple closed 2D polygon in world space. The planner only emits convex
/// rectilinear footprints, so containment is tested with the even-odd rule
/// and no general polygon library is needed.
#[derive(Clone, Debug)]
pub struct Polygon2 {
    /// (x, z) vertices in world space, wound consistently.
    pub vertices: Vec<(f32, f32)>,
}

impl Polygon2 {
    /// Axis-aligned rectangle from min corner and size.
    pub fn rect(x: f32, z: f32, w: f32, d: f32) -> Self {
        Self {
            vertices: vec![(x, z), (x + w, z), (x + w, z + d), (x, z + d)],
        }
    }

    /// Even-odd containment test.
    pub fn contains(&self, x: f32, z: f32) -> bool {
        let v = &self.vertices;
        let mut inside = false;
        let mut j = v.len() - 1;
        for i in 0..v.len() {
            let (xi, zi) = v[i];
            let (xj, zj) = v[j];
            if ((zi > z) != (zj > z)) && (x < (xj - xi) * (z - zi) / (zj - zi) + xi) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Axis-aligned bounds: (min_x, min_z, max_x, max_z).
    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        let mut b = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, z) in &self.vertices {
            b.0 = b.0.min(x);
            b.1 = b.1.min(z);
            b.2 = b.2.max(x);
            b.3 = b.3.max(z);
        }
        b
    }
}

// ---------------------------------------------------------------------------
// Assemblies: planned spaces with structure, ceilings and fixtures.
// ---------------------------------------------------------------------------

/// A doorway/portal on an assembly boundary, opening onto circulation.
#[derive(Clone, Copy, Debug)]
pub struct Opening {
    /// Center of the opening in world space (on the footprint boundary).
    pub center: Position,
    /// Clear width, world units (>= ~1.0 per walkability contract).
    pub width: f32,
    /// True if the opening runs through a wall parallel to the X axis
    /// (i.e. you walk through it along Z).
    pub through_x_wall: bool,
    /// Lintel underside height in units; `None` = full-height portal.
    pub lintel_units: Option<f32>,
}

/// A named subspace inside an assembly (a private office in a suite, a stall
/// block in a restroom core). v1 keeps these as tagged rectangles.
#[derive(Clone, Debug)]
pub struct Space {
    pub program: SpaceProgram,
    pub footprint: Polygon2,
}

/// A concrete structural layout for one assembly.
#[derive(Clone, Debug)]
pub struct StructuralSystemInstance {
    pub system: StructuralSystem,
    /// Column spacing along X / Z, world units.
    pub bay_x: f32,
    pub bay_z: f32,
    /// World-space phase of the grid (so neighboring assemblies don't align).
    pub phase: (f32, f32),
    /// Column side, world units.
    pub column_side: f32,
}

/// A ceiling treatment over a sub-area of an assembly.
#[derive(Clone, Debug)]
pub struct CeilingZone {
    pub area: Polygon2,
    pub language: CeilingLanguage,
    /// Finished ceiling height, world units.
    pub height_units: f32,
}

/// One light fixture, tied to a ceiling module (not a free grid point).
#[derive(Clone, Copy, Debug)]
pub struct Fixture {
    pub at: Position,
    /// Fixture half-extent along X / Z (a 2x0.6 strip, a 0.6x0.6 panel...).
    pub half_x: f32,
    pub half_z: f32,
    pub lit: bool,
}

/// A void reserved for building services (risers, plenums). v1: reserved and
/// rendered as sealed solids; kept in the model so later passes can route
/// through them.
#[derive(Clone, Debug)]
pub struct ServiceVoid {
    pub area: Polygon2,
}

/// Backrooms-specific distortions applied to one assembly.
#[derive(Clone, Copy, Debug, Default)]
pub struct CorruptionProfile {
    /// Duplicated from another assembly with this world-space offset error.
    pub misalignment: (f32, f32),
    /// Built but never occupied: no fixtures lit, no furnishing.
    pub abandoned: bool,
    /// A second genome's structural grid overlaid on the original.
    pub renovation_overlay: bool,
}

/// A planned, placed space: program + footprint + everything needed to
/// voxelize it believably.
#[derive(Clone, Debug)]
pub struct AssemblyInstance {
    pub id: u32,
    pub program: SpaceProgram,
    pub footprint: Polygon2,
    pub entrances: Vec<Opening>,
    pub spaces: Vec<Space>,
    pub structure: StructuralSystemInstance,
    pub ceiling_zones: Vec<CeilingZone>,
    pub fixtures: Vec<Fixture>,
    pub service_voids: Vec<ServiceVoid>,
    pub corruption: CorruptionProfile,
}

/// A corridor: a wide polyline with priority over everything it crosses.
#[derive(Clone, Debug)]
pub struct CirculationSpine {
    pub id: u32,
    /// `MainCorridor` or `SecondaryHall`.
    pub spine_kind: SpaceProgram,
    /// Axis-aligned polyline waypoints in world space.
    pub path: Vec<Position>,
    /// Clear width, world units.
    pub width: f32,
}

impl CirculationSpine {
    /// Distance from (x, z) to the spine centerline (axis-aligned segments).
    pub fn distance(&self, x: f32, z: f32) -> f32 {
        self.nearest(x, z).0
    }

    /// Nearest point on the centerline: `(distance, along, is_horizontal)`.
    /// `along` is the world coordinate *along* the nearest segment's axis, so
    /// modules (light strips, wall gaps) repeat in world space and stay
    /// continuous across chunk and region borders.
    pub fn nearest(&self, x: f32, z: f32) -> (f32, f32, bool) {
        let mut best = (f32::MAX, 0.0, true);
        for seg in self.path.windows(2) {
            let (a, b) = (seg[0], seg[1]);
            let (dx, dz) = (b.x - a.x, b.z - a.z);
            let len2 = dx * dx + dz * dz;
            let t = if len2 > 0.0 {
                (((x - a.x) * dx + (z - a.z) * dz) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (px, pz) = (a.x + t * dx, a.z + t * dz);
            let d2 = (x - px) * (x - px) + (z - pz) * (z - pz);
            if d2 < best.0 {
                let horizontal = dx.abs() >= dz.abs();
                best = (d2, if horizontal { px } else { pz }, horizontal);
            }
        }
        (best.0.sqrt(), best.1, best.2)
    }
}

// ---------------------------------------------------------------------------
// The region plan.
// ---------------------------------------------------------------------------

/// Everything planned for one square region of the world. Regions tile a
/// fixed world-space lattice; a chunk generator asks for the plan of every
/// region its chunk overlaps and samples them per column.
#[derive(Clone, Debug)]
pub struct RegionPlan {
    /// Lower-left corner in world space.
    pub origin_world: Position,
    /// Side length, world units.
    pub size_world: f32,
    /// Dominant designer first, then renovator, then optional intruder.
    pub architects: Vec<ArchitectGenome>,
    pub assemblies: Vec<AssemblyInstance>,
    pub corridors: Vec<CirculationSpine>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_polygon_contains_interior_not_exterior() {
        let p = Polygon2::rect(10.0, 20.0, 5.0, 8.0);
        assert!(p.contains(12.0, 24.0));
        assert!(!p.contains(9.9, 24.0));
        assert!(!p.contains(12.0, 28.5));
        assert_eq!(p.bounds(), (10.0, 20.0, 15.0, 28.0));
    }

    #[test]
    fn spine_distance_measures_to_nearest_segment() {
        let spine = CirculationSpine {
            id: 0,
            spine_kind: SpaceProgram::MainCorridor,
            path: vec![
                Position::new(0.0, 0.0),
                Position::new(10.0, 0.0),
                Position::new(10.0, 10.0),
            ],
            width: 2.0,
        };
        assert!((spine.distance(5.0, 3.0) - 3.0).abs() < 1e-5);
        assert!((spine.distance(13.0, 10.0) - 3.0).abs() < 1e-5);
        assert!((spine.distance(10.0, 5.0) - 0.0).abs() < 1e-5);
    }
}
