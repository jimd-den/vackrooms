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

use std::collections::HashSet;

use crate::domain::entities::anomaly::AnomalyInstance;
use crate::domain::entities::position::Position;

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

/// Stable identity of a wall-like element inside one assembly.
///
/// IDs are local to the assembly. This is enough to keep openings attached
/// while an assembly is translated, duplicated, or rewritten by a grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostId(pub u32);

/// Stable identity of an opening inside one assembly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpeningId(pub u32);

/// Architectural purpose of a wall host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostRole {
    /// The enclosing shell of an assembly.
    Shell,
    /// A non-load-bearing wall dividing spaces inside an assembly.
    Partition,
    /// A wall that also participates in the structural system.
    Structural,
}

/// A finite axis-aligned wall host in plan space.
///
/// Voxelization is intentionally absent here. Hosts are authored in world
/// units, openings reference them by identity, and a later sampler chooses
/// the representation appropriate for the requested voxel size.
#[derive(Clone, Copy, Debug)]
pub struct HostSegment {
    pub id: HostId,
    pub start: Position,
    pub end: Position,
    pub thickness: f32,
    pub base_units: f32,
    pub top_units: f32,
    pub role: HostRole,
}

impl HostSegment {
    pub fn is_horizontal(&self) -> bool {
        (self.start.z - self.end.z).abs() <= 1e-4
    }

    pub fn is_axis_aligned(&self) -> bool {
        self.is_horizontal() || (self.start.x - self.end.x).abs() <= 1e-4
    }

    pub fn length(&self) -> f32 {
        self.start.distance(&self.end)
    }

    /// Whether a plan sample falls in this host's finite wall band.
    pub fn contains_plan(&self, x: f32, z: f32) -> bool {
        let half = self.thickness * 0.5 + 1e-4;
        if self.is_horizontal() {
            let (lo, hi) = ordered(self.start.x, self.end.x);
            x >= lo - half && x <= hi + half && (z - self.start.z).abs() <= half
        } else {
            let (lo, hi) = ordered(self.start.z, self.end.z);
            z >= lo - half && z <= hi + half && (x - self.start.x).abs() <= half
        }
    }

    pub fn translate(&mut self, dx: f32, dz: f32) {
        self.start.x += dx;
        self.start.z += dz;
        self.end.x += dx;
        self.end.z += dz;
    }

    /// Four explicit shell hosts for an axis-aligned footprint. IDs are
    /// south, east, north, west in that order.
    pub fn rectangular_shell(footprint: &Polygon2, thickness: f32, top_units: f32) -> Vec<Self> {
        let (x0, z0, x1, z1) = footprint.bounds();
        [
            (Position::new(x0, z0), Position::new(x1, z0)),
            (Position::new(x1, z0), Position::new(x1, z1)),
            (Position::new(x1, z1), Position::new(x0, z1)),
            (Position::new(x0, z1), Position::new(x0, z0)),
        ]
        .into_iter()
        .enumerate()
        .map(|(id, (start, end))| Self {
            id: HostId(id as u32),
            start,
            end,
            thickness,
            base_units: 0.0,
            top_units,
            role: HostRole::Shell,
        })
        .collect()
    }
}

fn ordered(a: f32, b: f32) -> (f32, f32) {
    (a.min(b), a.max(b))
}

/// How an opening participates in circulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpeningRole {
    /// Connects an assembly to circulation or another assembly.
    Entrance,
    /// Connects two spaces inside one assembly.
    Interior,
    /// Reserved penetration for building services.
    Service,
}

/// A doorway/portal that cuts one explicit host.
#[derive(Clone, Copy, Debug)]
pub struct Opening {
    pub id: OpeningId,
    pub host: HostId,
    pub role: OpeningRole,
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

impl Opening {
    /// Whether a plan sample is inside the opening cut. Host thickness is
    /// supplied by the referenced host rather than duplicated on the opening.
    pub fn contains_plan(&self, host: &HostSegment, x: f32, z: f32) -> bool {
        let (along, across) = if host.is_horizontal() {
            ((x - self.center.x).abs(), (z - self.center.z).abs())
        } else {
            ((z - self.center.z).abs(), (x - self.center.x).abs())
        };
        along < self.width * 0.5 && across <= host.thickness * 0.5 + 0.05
    }
}

/// A failed relationship in the architectural model. Keeping violations
/// semantic makes them useful to generators, tests, and future repair passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchitectureViolation {
    InvalidHost(HostId),
    MissingOpeningHost(OpeningId),
    OpeningOffHost(OpeningId),
    OpeningOutsideHost(OpeningId),
    OverlappingOpenings(OpeningId, OpeningId),
    DuplicateHostId(HostId),
    DuplicateOpeningId(OpeningId),
    OverlappingHosts(HostId, HostId),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    CeilingPanel,
    Strip,
    Emergency,
}

#[derive(Clone, Debug)]
pub struct RuntimeLight {
    pub world_pos: [f32; 3],
    pub half_size: [f32; 2],
    pub rgb: [f32; 3],
    pub range: f32,
    pub intensity: f32,
    pub enabled: bool,
    pub kind: LightKind,
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
    /// Every fixture in this room burns red. The *room* is the anomaly —
    /// its geometry stays ordinary; only the light exposes it. Red never
    /// appears as a wall material anywhere in Level 0.
    pub red_room: bool,
}

/// A planned, placed space: program + footprint + everything needed to
/// voxelize it believably.
#[derive(Clone, Debug)]
pub struct AssemblyInstance {
    pub id: u32,
    pub program: SpaceProgram,
    pub footprint: Polygon2,
    pub hosts: Vec<HostSegment>,
    pub openings: Vec<Opening>,
    pub spaces: Vec<Space>,
    pub structure: StructuralSystemInstance,
    pub ceiling_zones: Vec<CeilingZone>,
    pub fixtures: Vec<Fixture>,
    pub service_voids: Vec<ServiceVoid>,
    pub corruption: CorruptionProfile,
}

impl AssemblyInstance {
    pub fn entrances(&self) -> impl Iterator<Item = &Opening> {
        self.openings
            .iter()
            .filter(|opening| opening.role == OpeningRole::Entrance)
    }

    pub fn primary_entrance(&self) -> Option<&Opening> {
        self.entrances().next()
    }

    pub fn host(&self, id: HostId) -> Option<&HostSegment> {
        self.hosts.iter().find(|host| host.id == id)
    }

    /// Move every owned architectural element as one composition. Keeping
    /// this transform centralized prevents duplicated rooms from leaving
    /// ceilings, grids, service zones, or hosted openings behind.
    pub fn translate(&mut self, dx: f32, dz: f32) {
        let translate_polygon = |polygon: &mut Polygon2| {
            for vertex in &mut polygon.vertices {
                vertex.0 += dx;
                vertex.1 += dz;
            }
        };
        translate_polygon(&mut self.footprint);
        for space in &mut self.spaces {
            translate_polygon(&mut space.footprint);
        }
        for zone in &mut self.ceiling_zones {
            translate_polygon(&mut zone.area);
        }
        for service_void in &mut self.service_voids {
            translate_polygon(&mut service_void.area);
        }
        for host in &mut self.hosts {
            host.translate(dx, dz);
        }
        for opening in &mut self.openings {
            opening.center.x += dx;
            opening.center.z += dz;
        }
        for fixture in &mut self.fixtures {
            fixture.at.x += dx;
            fixture.at.z += dz;
        }
        self.structure.phase.0 += dx;
        self.structure.phase.1 += dz;
    }

    /// Validate the host/opening relationships without voxelizing anything.
    pub fn validate_architecture(&self) -> Vec<ArchitectureViolation> {
        let mut violations = Vec::new();
        let mut host_ids = HashSet::new();
        for host in &self.hosts {
            if !host_ids.insert(host.id) {
                violations.push(ArchitectureViolation::DuplicateHostId(host.id));
            }
            if !host.is_axis_aligned()
                || host.length() <= 1e-4
                || host.thickness <= 0.0
                || host.base_units.abs() > 1e-4
                || host.top_units <= host.base_units
            {
                violations.push(ArchitectureViolation::InvalidHost(host.id));
            }
        }
        for (index, a) in self.hosts.iter().enumerate() {
            for b in self.hosts.iter().skip(index + 1) {
                if hosts_overlap(a, b) {
                    violations.push(ArchitectureViolation::OverlappingHosts(a.id, b.id));
                }
            }
        }
        let mut opening_ids = HashSet::new();
        for opening in &self.openings {
            if !opening_ids.insert(opening.id) {
                violations.push(ArchitectureViolation::DuplicateOpeningId(opening.id));
            }
            let Some(host) = self.host(opening.host) else {
                violations.push(ArchitectureViolation::MissingOpeningHost(opening.id));
                continue;
            };
            if host.is_horizontal() != opening.through_x_wall
                || !host.contains_plan(opening.center.x, opening.center.z)
            {
                violations.push(ArchitectureViolation::OpeningOffHost(opening.id));
                continue;
            }
            let (host_lo, host_hi, center) = if host.is_horizontal() {
                let (lo, hi) = ordered(host.start.x, host.end.x);
                (lo, hi, opening.center.x)
            } else {
                let (lo, hi) = ordered(host.start.z, host.end.z);
                (lo, hi, opening.center.z)
            };
            if opening.width <= 0.0
                || center - opening.width * 0.5 < host_lo - 1e-4
                || center + opening.width * 0.5 > host_hi + 1e-4
            {
                violations.push(ArchitectureViolation::OpeningOutsideHost(opening.id));
            }
        }
        for (index, a) in self.openings.iter().enumerate() {
            for b in self.openings.iter().skip(index + 1) {
                if a.host != b.host {
                    continue;
                }
                let host = self.host(a.host);
                let (a_center, b_center) = match host {
                    Some(host) if host.is_horizontal() => (a.center.x, b.center.x),
                    Some(_) => (a.center.z, b.center.z),
                    None => continue,
                };
                if (a_center - b_center).abs() < (a.width + b.width) * 0.5 {
                    violations.push(ArchitectureViolation::OverlappingOpenings(a.id, b.id));
                }
            }
        }
        violations
    }
}

fn hosts_overlap(a: &HostSegment, b: &HostSegment) -> bool {
    const EPSILON: f32 = 1e-4;
    if a.is_horizontal() && b.is_horizontal() {
        if (a.start.z - b.start.z).abs() > EPSILON {
            return false;
        }
        let (a0, a1) = ordered(a.start.x, a.end.x);
        let (b0, b1) = ordered(b.start.x, b.end.x);
        return a1.min(b1) - a0.max(b0) > EPSILON;
    }
    if !a.is_horizontal() && !b.is_horizontal() {
        if (a.start.x - b.start.x).abs() > EPSILON {
            return false;
        }
        let (a0, a1) = ordered(a.start.z, a.end.z);
        let (b0, b1) = ordered(b.start.z, b.end.z);
        return a1.min(b1) - a0.max(b0) > EPSILON;
    }
    false
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
    /// Region-overlapping anomaly instances. Macro instances deliberately
    /// repeat in every intersected region with the same stable identity;
    /// consumers deduplicate them by [`AnomalyInstance::id`].
    pub anomalies: Vec<AnomalyInstance>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hosted_room() -> AssemblyInstance {
        let footprint = Polygon2::rect(0.0, 0.0, 10.0, 8.0);
        AssemblyInstance {
            id: 1,
            program: SpaceProgram::OpenOffice,
            footprint: footprint.clone(),
            hosts: HostSegment::rectangular_shell(&footprint, 0.4, 3.4),
            openings: vec![Opening {
                id: OpeningId(0),
                host: HostId(0),
                role: OpeningRole::Entrance,
                center: Position::new(5.0, 0.0),
                width: 2.0,
                through_x_wall: true,
                lintel_units: Some(2.2),
            }],
            spaces: Vec::new(),
            structure: StructuralSystemInstance {
                system: StructuralSystem::CoreAndShell,
                bay_x: 4.0,
                bay_z: 4.0,
                phase: (0.0, 0.0),
                column_side: 0.4,
            },
            ceiling_zones: vec![CeilingZone {
                area: footprint,
                language: CeilingLanguage::FlatTiles,
                height_units: 3.4,
            }],
            fixtures: Vec::new(),
            service_voids: Vec::new(),
            corruption: CorruptionProfile::default(),
        }
    }

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

    #[test]
    fn hosted_opening_validation_rejects_broken_relationships() {
        let mut room = hosted_room();
        assert!(room.validate_architecture().is_empty());

        room.openings[0].host = HostId(99);
        assert_eq!(
            room.validate_architecture(),
            vec![ArchitectureViolation::MissingOpeningHost(OpeningId(0))]
        );
    }

    #[test]
    fn architecture_validation_rejects_ambiguous_element_identity() {
        let mut room = hosted_room();
        room.hosts.push(room.hosts[0]);
        room.openings.push(room.openings[0]);
        let violations = room.validate_architecture();
        assert!(violations.contains(&ArchitectureViolation::DuplicateHostId(HostId(0))));
        assert!(violations.contains(&ArchitectureViolation::DuplicateOpeningId(OpeningId(0))));
        assert!(violations.contains(&ArchitectureViolation::OverlappingHosts(
            HostId(0),
            HostId(0)
        )));
    }

    #[test]
    fn assembly_translation_moves_one_coherent_composition() {
        let mut room = hosted_room();
        room.translate(12.0, -4.0);
        assert_eq!(room.footprint.bounds(), (12.0, -4.0, 22.0, 4.0));
        assert_eq!(room.openings[0].center, Position::new(17.0, -4.0));
        assert_eq!(room.hosts[0].start, Position::new(12.0, -4.0));
        assert_eq!(room.structure.phase, (12.0, -4.0));
        assert!(room.validate_architecture().is_empty());
    }
}
