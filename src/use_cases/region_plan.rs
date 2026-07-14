//! Region-scale architectural planning for Backrooms Level 0.
//!
//! The world is tiled by fixed 80 u square regions on a world-space lattice.
//! [`generate_region_plan`] is a *pure* function of `(seed, region)`: it
//! derives the region's designers ([`ArchitectGenome`]), routes a dominant
//! circulation spine first (between edge portals shared with the neighboring
//! regions), attaches incomplete program masses ([`AssemblyInstance`]) beside
//! that circulation,
//! gives them structure / ceilings / fixtures, and finally applies a
//! Backrooms corruption pass (duplicated suites, abandoned expansions,
//! renovation overlays).
//!
//! Cross-region continuity: a corridor leaves a region only at a *portal* on
//! a shared edge, and the portal position is derived from the *edge's*
//! lattice coordinate — both neighbors compute the identical portal, so
//! corridors chain across regions forever (a main corridor that never ends
//! is the first and cheapest corruption).
//!
//! All coordinates are snapped to a 0.4 u lattice — the coarsest voxel size —
//! so plan geometry lands identically at every LOD.

use crate::domain::entities::anomaly::AnomalyKind;
use crate::domain::entities::architecture::*;
use crate::entities::models::Position;
use crate::use_cases::anomaly_plan::plan_anomalies_for_region;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::NoiseProvider;

/// Side of a region, world units. A multiple of both chunk sizes (10/20).
pub const REGION_SIZE: f32 = 80.0;
/// Plan wall thickness: one coarse voxel, so walls survive every LOD.
pub const PLAN_WALL_T: f32 = 0.4;
/// Snap lattice for all plan geometry (= coarsest voxel size).
const SNAP: f32 = 0.4;
/// Keep-out margin from region edges for assembly footprints.
const EDGE_MARGIN: f32 = 3.2;

/// Region lattice index of a world coordinate.
pub fn region_index(w: f32) -> i64 {
    (w / REGION_SIZE).floor() as i64
}

fn snap(v: f32) -> f32 {
    (v / SNAP).round() * SNAP
}

/// Corridor widths snap to 0.8 so *half*-widths stay on the 0.4 lattice and
/// suite front walls land flush against corridor edges.
fn snap_width(v: f32) -> f32 {
    ((v / (2.0 * SNAP)).round() * 2.0 * SNAP).max(2.0 * SNAP)
}

// ---------------------------------------------------------------------------
// Deterministic hashing and edge portals.
//
// All "randomness" flows through `world_topology::hash01`, and the portal a
// corridor uses to cross a region border is the macro graph's portal — one
// derivation shared by both neighbors of the edge and by the topology layer.
// ---------------------------------------------------------------------------

use crate::use_cases::world_topology::{hash01, primary_portal_z as v_edge_portal_z};

fn pick_index(h: f32, len: usize) -> usize {
    ((h * len as f32) as usize).min(len - 1)
}

/// World spawn: a point *on the main corridor centerline* of region (0, 0),
/// a few units east of its west portal. The first thing the player sees is
/// therefore the region's dominant circulation route — walls running away in
/// both directions — rather than unplanned fabric. Both the generator core
/// and the browser composition root derive spawn from this single function,
/// so the player and the voxelizer can never disagree about where "here" is.
pub fn spawn_point(seed: u32) -> Position {
    // The main spine's west leg always runs from (0, west_z) to at least
    // x = 0.35 * REGION_SIZE, so a few units in we are guaranteed to stand
    // on a straight, readable stretch of corridor.
    Position::new(6.0, v_edge_portal_z(seed, 0, 0))
}

// ---------------------------------------------------------------------------
// Genomes.
// ---------------------------------------------------------------------------

/// Derive one designer for a region. `salt` 0 = dominant, 1 = renovator,
/// 2 = intruder. The circulation style comes from low-frequency noise so
/// neighboring regions tend to share a culture; everything else hashes.
pub fn derive_genome(
    seed: u32,
    rx: i64,
    rz: i64,
    salt: i64,
    noise: &dyn NoiseProvider,
) -> ArchitectGenome {
    let h = |k: i64| hash01(seed, &[0x6E0 + salt, rx, rz, k]);
    // ~130 u wavelength: cultures span a few regions.
    let culture = noise.evaluate_2d(
        seed ^ 0xA11C_0DE ^ (salt as u32),
        Position::new(
            (rx as f32 + 0.5) * REGION_SIZE * 0.15,
            (rz as f32 + 0.5) * REGION_SIZE * 0.15,
        ),
    );
    let circulation = match ((culture * 0.5 + 0.5).clamp(0.0, 0.999) * 4.0) as u32 {
        0 => CirculationStyle::StraightSpine,
        1 => CirculationStyle::MeanderingSpine,
        2 => CirculationStyle::PairedBranches,
        _ => CirculationStyle::TreeWithCulDeSacs,
    };
    let structural_system = *[
        StructuralSystem::RegularGrid,
        StructuralSystem::DeepSpansWithBeams,
        StructuralSystem::OffsetGrid,
        StructuralSystem::CoreAndShell,
    ]
    .get(pick_index(h(1), 4))
    .unwrap();
    // A framed office door is now an anomaly. The chosen language is only a
    // bias; each assembly still picks its own threshold so a whole region
    // never turns into a row of identical openings.
    let threshold_language = match h(2) {
        v if v < 0.10 => ThresholdLanguage::DoorWithLintel,
        v if v < 0.55 => ThresholdLanguage::OpenPortal,
        _ => ThresholdLanguage::WidePortal,
    };
    let ceiling_language = *[
        CeilingLanguage::FlatTiles,
        CeilingLanguage::Coffered,
        CeilingLanguage::ExposedSoffit,
    ]
    .get(pick_index(h(3), 3))
    .unwrap();
    let lighting_language = *[
        LightingLanguage::GridPanels,
        LightingLanguage::GridPanels,
        LightingLanguage::StripsAlongCirculation,
        LightingLanguage::SparsePendants,
    ]
    .get(pick_index(h(4), 4))
    .unwrap();
    let renovation_history = *[
        RenovationStyle::Untouched,
        RenovationStyle::PartialRefit,
        RenovationStyle::LayeredRefits,
    ]
    .get(pick_index(h(5), 3))
    .unwrap();
    ArchitectGenome {
        circulation,
        structural_system,
        room_proportions: ProportionRules {
            // Level 0 reads as broken retail-backroom masses, not a cubicle
            // tiling. A suite frontage normally survives 12--24 u before a
            // meaningful interruption.
            min_side: 12.0 + 4.0 * h(6),
            max_side: 20.0 + 8.0 * h(7),
            elongation: h(8),
        },
        threshold_language,
        ceiling_language,
        lighting_language,
        furnishing_density: 0.3 + 0.7 * h(9),
        renovation_history,
        tolerance_for_symmetry: h(10),
    }
}

// ---------------------------------------------------------------------------
// Circulation.
// ---------------------------------------------------------------------------

/// Z coordinate of the main spine where it passes world x (assumes the spine
/// is a west-to-east Manhattan path).
fn spine_z_at(path: &[Position], x: f32) -> f32 {
    for seg in path.windows(2) {
        let (a, b) = (seg[0], seg[1]);
        if a.z == b.z && (x >= a.x.min(b.x)) && (x <= a.x.max(b.x)) {
            return a.z;
        }
    }
    path[0].z
}

fn build_corridors(
    seed: u32,
    rx: i64,
    rz: i64,
    genome: &ArchitectGenome,
    origin: Position,
    size: f32,
) -> Vec<CirculationSpine> {
    let h = |k: i64| hash01(seed, &[0xC0 + k, rx, rz]);
    let (x0, x1) = (origin.x, origin.x + size);
    let (z0, z1) = (origin.z, origin.z + size);
    let west_z = v_edge_portal_z(seed, rx, rz);
    let east_z = v_edge_portal_z(seed, rx + 1, rz);
    // The primary route is deliberately too broad for an ordinary office
    // corridor. It carries the infinite cross-region continuity; anything
    // else is an optional branch, never a competing street grid.
    let main_w = snap_width(4.8 + 2.4 * h(1));
    let sec_w = snap_width(3.6 + 1.4 * hash01(seed, &[0xC1, rx, rz]));

    let mut spines = Vec::new();
    let mut id = 0u32;
    let mut push = |kind: SpaceProgram, path: Vec<Position>, width: f32, id: &mut u32| {
        spines_push(&mut spines, kind, path, width, id);
    };

    // Main spine: west portal -> east portal, always. Its Manhattan bend
    // position is the designer's one free choice.
    let bend_x = snap(x0 + size * (0.35 + 0.30 * h(2)));
    let main_path: Vec<Position> = match genome.circulation {
        CirculationStyle::MeanderingSpine => {
            let mid_z = snap(z0 + size * (0.30 + 0.40 * h(3)));
            let bend2_x = snap(x0 + size * (0.60 + 0.25 * h(4)));
            vec![
                Position::new(x0, west_z),
                Position::new(bend_x, west_z),
                Position::new(bend_x, mid_z),
                Position::new(bend2_x, mid_z),
                Position::new(bend2_x, east_z),
                Position::new(x1, east_z),
            ]
        }
        _ => vec![
            Position::new(x0, west_z),
            Position::new(bend_x, west_z),
            Position::new(bend_x, east_z),
            Position::new(x1, east_z),
        ],
    };
    let main_z_at = main_path.clone();
    push(SpaceProgram::MainCorridor, main_path, main_w, &mut id);

    // Secondary routes are branches, not a second regional grid: roughly a
    // quarter of regions have none, most have one, and a few have two. The
    // former ring and north/south through-route are intentionally absent.
    let branch_count = match genome.circulation {
        CirculationStyle::StraightSpine if h(11) < 0.45 => 0,
        CirculationStyle::MeanderingSpine if h(11) < 0.20 => 0,
        _ if h(11) < 0.28 => 0,
        _ if h(11) < 0.82 => 1,
        _ => 2,
    };
    for k in 0..branch_count {
        let sx = snap(x0 + size * (0.18 + 0.64 * h(12 + k)));
        let sz = spine_z_at(&main_z_at, sx);
        let len = snap(12.0 + 14.0 * h(22 + k));
        let dir = if h(32 + k) < 0.5 { 1.0 } else { -1.0 };
        let end = (sz + dir * len).clamp(z0 + EDGE_MARGIN, z1 - EDGE_MARGIN);
        if (end - sz).abs() >= 8.0 {
            push(
                SpaceProgram::SecondaryHall,
                vec![Position::new(sx, sz), Position::new(sx, snap(end))],
                sec_w,
                &mut id,
            );
        }
    }
    spines
}

fn spines_push(
    spines: &mut Vec<CirculationSpine>,
    kind: SpaceProgram,
    path: Vec<Position>,
    width: f32,
    id: &mut u32,
) {
    spines.push(CirculationSpine {
        id: *id,
        spine_kind: kind,
        path,
        width,
    });
    *id += 1;
}

// ---------------------------------------------------------------------------
// Assemblies.
// ---------------------------------------------------------------------------

/// The suite program palette placed beside main corridors, roughly weighted.
/// Large, unfinished open-office masses dominate. Small private rooms remain
/// present only as occasional evidence that this once had an office program.
const SUITE_PROGRAMS: [SpaceProgram; 12] = [
    SpaceProgram::OpenOffice,
    SpaceProgram::OpenOffice,
    SpaceProgram::OpenOffice,
    SpaceProgram::OpenOffice,
    SpaceProgram::OpenOffice,
    SpaceProgram::ConferenceRoom,
    SpaceProgram::ConferenceRoom,
    SpaceProgram::BreakRoom,
    SpaceProgram::Storage,
    SpaceProgram::ServerRoom,
    SpaceProgram::WaitingArea,
    SpaceProgram::PrivateOffice,
];

fn ceiling_height_for(program: SpaceProgram, aseed: f32) -> f32 {
    match program {
        SpaceProgram::Atrium => 4.8 + 0.6 * aseed,
        // Compression is a deliberate contrast, never the default ceiling.
        SpaceProgram::ServerRoom | SpaceProgram::Mechanical => 2.6 + 0.2 * aseed,
        SpaceProgram::Storage | SpaceProgram::RestroomCore => 3.0 + 0.3 * aseed,
        SpaceProgram::OpenOffice | SpaceProgram::ConferenceRoom => 3.5 + 0.7 * aseed,
        _ => 3.2 + 0.5 * aseed,
    }
}

fn structure_for(genome: &ArchitectGenome, aseed: f32) -> StructuralSystemInstance {
    let bay = snap(4.4 + 1.6 * aseed);
    let (bay_x, bay_z) = match genome.structural_system {
        StructuralSystem::DeepSpansWithBeams => (bay * 1.5, bay),
        _ => (bay, bay),
    };
    StructuralSystemInstance {
        system: genome.structural_system,
        bay_x,
        bay_z,
        phase: (snap(aseed * 4.0), snap(aseed * 8.0 % 4.0)),
        column_side: 0.4,
    }
}

/// Fixtures for a rectangular footprint in the genome's lighting language.
fn fixtures_for(
    genome: &ArchitectGenome,
    footprint: &Polygon2,
    lit: bool,
    aseed: f32,
) -> Vec<Fixture> {
    let (x0, z0, x1, z1) = footprint.bounds();
    let mut out = Vec::new();
    match genome.lighting_language {
        LightingLanguage::GridPanels => {
            let step = 2.4;
            let mut z = z0 + 1.2;
            while z < z1 - 0.8 {
                let mut x = x0 + 1.2;
                while x < x1 - 0.8 {
                    out.push(Fixture {
                        at: Position::new(snap(x), snap(z)),
                        half_x: 0.4,
                        half_z: 0.4,
                        lit,
                    });
                    x += step;
                }
                z += step;
            }
        }
        LightingLanguage::StripsAlongCirculation => {
            // Strips run down the room's long axis.
            let long_x = (x1 - x0) >= (z1 - z0);
            let (cx, cz) = ((x0 + x1) * 0.5, (z0 + z1) * 0.5);
            let step = 3.2;
            if long_x {
                let mut x = x0 + 1.6;
                while x < x1 - 1.2 {
                    out.push(Fixture {
                        at: Position::new(snap(x), snap(cz)),
                        half_x: 1.0,
                        half_z: 0.25,
                        lit,
                    });
                    x += step;
                }
            } else {
                let mut z = z0 + 1.6;
                while z < z1 - 1.2 {
                    out.push(Fixture {
                        at: Position::new(snap(cx), snap(z)),
                        half_x: 0.25,
                        half_z: 1.0,
                        lit,
                    });
                    z += step;
                }
            }
        }
        LightingLanguage::SparsePendants => {
            let n = 1 + (aseed * 2.0) as i32;
            for k in 0..n {
                let fx = x0 + (x1 - x0) * (0.3 + 0.4 * ((k as f32 * 0.618 + aseed) % 1.0));
                let fz = z0 + (z1 - z0) * (0.3 + 0.4 * ((k as f32 * 0.382 + aseed * 2.0) % 1.0));
                out.push(Fixture {
                    at: Position::new(snap(fx), snap(fz)),
                    half_x: 0.3,
                    half_z: 0.3,
                    lit,
                });
            }
        }
    }
    out
}

/// Interior partitioning is deliberately sparse. A private-office suite may
/// split once, producing one long interruption rather than a cell grid.
fn spaces_for(
    program: SpaceProgram,
    footprint: &Polygon2,
    genome: &ArchitectGenome,
    aseed: f32,
) -> Vec<Space> {
    if program != SpaceProgram::PrivateOffice {
        return Vec::new();
    }
    let (x0, z0, x1, z1) = footprint.bounds();
    let long_x = (x1 - x0) >= (z1 - z0);
    let span = if long_x { x1 - x0 } else { z1 - z0 };
    let want = genome.room_proportions.min_side.max(12.0);
    let n = ((span / want) as usize).clamp(1, 2);
    let n = if aseed > 0.62 { n } else { 1 };
    if n <= 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for k in 0..n {
        let (a, b) = (
            snap(k as f32 / n as f32 * span),
            snap((k + 1) as f32 / n as f32 * span),
        );
        let rect = if long_x {
            Polygon2::rect(x0 + a, z0, b - a, z1 - z0)
        } else {
            Polygon2::rect(x0, z0 + a, x1 - x0, b - a)
        };
        out.push(Space {
            program: SpaceProgram::PrivateOffice,
            footprint: rect,
        });
    }
    out
}

fn aabb_overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32), gap: f32) -> bool {
    a.0 < b.2 + gap && b.0 < a.2 + gap && a.1 < b.3 + gap && b.1 < a.3 + gap
}

/// Places one suite beside a horizontal corridor segment. Returns `None` if
/// the footprint would leave the region, collide with a prior assembly, or
/// cross another corridor.
#[allow(clippy::too_many_arguments)]
fn place_suite(
    id: u32,
    program: SpaceProgram,
    genome: &ArchitectGenome,
    aseed: f32,
    threshold_seed: f32,
    cursor_x: f32,
    corridor_z: f32,
    corridor_half: f32,
    side: f32, // +1 = suite on +Z side of corridor, -1 = -Z side
    origin: Position,
    size: f32,
    taken: &[(f32, f32, f32, f32)],
    spines: &[CirculationSpine],
) -> Option<AssemblyInstance> {
    let p = &genome.room_proportions;
    let w = snap((p.min_side + (p.max_side - p.min_side) * aseed).clamp(12.0, 24.0));
    let d = snap((w * (0.72 - 0.28 * p.elongation)).clamp(8.0, 18.0));

    let x0 = snap(cursor_x);
    let z_front = snap(if side > 0.0 {
        corridor_z + corridor_half
    } else {
        corridor_z - corridor_half - d
    });
    let footprint = Polygon2::rect(x0, z_front, w, d);
    let b = footprint.bounds();

    // Stay inside the region with margin.
    if b.0 < origin.x + EDGE_MARGIN
        || b.1 < origin.z + EDGE_MARGIN
        || b.2 > origin.x + size - EDGE_MARGIN
        || b.3 > origin.z + size - EDGE_MARGIN
    {
        return None;
    }
    if taken.iter().any(|t| aabb_overlap(*t, b, 0.8)) {
        return None;
    }
    // Don't let a suite swallow a *different* corridor (its front corridor
    // touching the footprint edge is fine and expected).
    let center = ((b.0 + b.2) * 0.5, (b.1 + b.3) * 0.5);
    for s in spines {
        if s.distance(center.0, center.1) < s.width * 0.5 + d.min(w) * 0.25 {
            return None;
        }
    }

    // Entrance on the corridor-facing wall. The genome biases the language,
    // but a narrow framed door is only about 8--14% of thresholds globally.
    // Most fronts dissolve into a room through an unframed or broad portal.
    let threshold = match genome.threshold_language {
        ThresholdLanguage::DoorWithLintel if threshold_seed < 0.14 => {
            ThresholdLanguage::DoorWithLintel
        }
        ThresholdLanguage::OpenPortal if threshold_seed < 0.08 => ThresholdLanguage::DoorWithLintel,
        ThresholdLanguage::WidePortal if threshold_seed < 0.10 => ThresholdLanguage::DoorWithLintel,
        ThresholdLanguage::OpenPortal if threshold_seed < 0.62 => ThresholdLanguage::OpenPortal,
        ThresholdLanguage::WidePortal if threshold_seed < 0.50 => ThresholdLanguage::OpenPortal,
        ThresholdLanguage::DoorWithLintel if threshold_seed < 0.50 => ThresholdLanguage::OpenPortal,
        _ => ThresholdLanguage::WidePortal,
    };
    let front_z = if side > 0.0 { b.1 } else { b.3 };
    let (width, lintel) = match threshold {
        ThresholdLanguage::DoorWithLintel => (1.2, Some(2.2)),
        ThresholdLanguage::OpenPortal => (4.8, None),
        ThresholdLanguage::WidePortal => (3.6, Some(3.0)),
    };
    let edge = width * 0.5 + 0.8;
    let door_x = snap((b.0 + w * (0.25 + 0.5 * threshold_seed)).clamp(b.0 + edge, b.2 - edge));
    let entrances = vec![Opening {
        center: Position::new(door_x, front_z),
        width,
        through_x_wall: true,
        lintel_units: lintel,
    }];

    let ceiling = CeilingZone {
        area: footprint.clone(),
        language: genome.ceiling_language,
        height_units: ceiling_height_for(program, aseed),
    };
    Some(AssemblyInstance {
        id,
        program,
        spaces: spaces_for(program, &footprint, genome, aseed),
        structure: structure_for(genome, aseed),
        ceiling_zones: vec![ceiling],
        fixtures: fixtures_for(genome, &footprint, true, aseed),
        service_voids: Vec::new(),
        corruption: CorruptionProfile::default(),
        entrances,
        footprint,
    })
}

// ---------------------------------------------------------------------------
// The planner.
// ---------------------------------------------------------------------------

/// Pure, deterministic plan for the region whose lower-left corner is
/// `region_origin` (must lie on the region lattice).
pub fn generate_region_plan(
    seed: u32,
    region_origin: Position,
    region_size: f32,
    config: &GeneratorConfig,
    noise: &dyn NoiseProvider,
) -> RegionPlan {
    let rx = region_index(region_origin.x + 0.1);
    let rz = region_index(region_origin.z + 0.1);
    let h = |k: i64| hash01(seed, &[0xA0 + k, rx, rz]);

    let dominant = derive_genome(seed, rx, rz, 0, noise);
    let renovator = derive_genome(seed, rx, rz, 1, noise);
    let mut architects = vec![dominant.clone(), renovator];
    // A third designer intrudes where the style-blend field runs high: the
    // macro fields decide *where* cultures leak into each other; the hash
    // only decides whether this particular region exposes the seam.
    let fields = crate::use_cases::world_topology::sample_fields(
        noise,
        seed,
        (rx as f32 + 0.5) * REGION_SIZE,
        (rz as f32 + 0.5) * REGION_SIZE,
    );
    if h(0) < 0.10 + 0.55 * fields.style_blend {
        architects.push(derive_genome(seed, rx, rz, 2, noise));
    }

    let corridors = build_corridors(seed, rx, rz, &dominant, region_origin, region_size);

    // --- incomplete masses beside the dominant route -----------------------
    // Only the primary route receives masses. The secondary stubs remain
    // mostly exposed circulation, which prevents a region from resolving
    // into a tiled office floorplan.
    let mut assemblies: Vec<AssemblyInstance> = Vec::new();
    let mut taken: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut id = 0u32;
    let legs: Vec<(f32, f32, f32, f32)> = corridors
        .iter()
        .filter(|s| s.spine_kind == SpaceProgram::MainCorridor)
        .flat_map(|s| {
            let w = s.width;
            s.path
                .windows(2)
                .filter(|seg| seg[0].z == seg[1].z)
                .map(move |seg| (seg[0].x.min(seg[1].x), seg[0].x.max(seg[1].x), seg[0].z, w))
                .collect::<Vec<_>>()
        })
        .filter(|(a, b, _, _)| b - a >= 14.0)
        .collect();

    for (li, &(lx0, lx1, lz, lw)) in legs.iter().enumerate() {
        let mut cursor = lx0 + 4.0;
        let mut side = if h(40 + li as i64) < 0.5 { 1.0 } else { -1.0 };
        while cursor < lx1 - 12.0 {
            let aseed = hash01(seed, &[0x5EA, rx, rz, id as i64]);
            let threshold_seed = hash01(seed, &[0x7A11, rx, rz, id as i64]);
            // Leave substantial pieces of the route exposed. When a mass is
            // placed, the next candidate begins 8--16 u beyond its end.
            if hash01(seed, &[0x5E8, rx, rz, id as i64]) < 0.62 {
                let program = SUITE_PROGRAMS[pick_index(aseed, SUITE_PROGRAMS.len())];
                if let Some(a) = place_suite(
                    id,
                    program,
                    &dominant,
                    aseed,
                    threshold_seed,
                    cursor,
                    lz,
                    lw * 0.5 + PLAN_WALL_T,
                    side,
                    region_origin,
                    region_size,
                    &taken,
                    &corridors,
                ) {
                    let b = a.footprint.bounds();
                    cursor += (b.2 - b.0) + 8.0 + 8.0 * threshold_seed;
                    taken.push(b);
                    assemblies.push(a);
                } else {
                    cursor += 8.0 + 8.0 * threshold_seed;
                }
            } else {
                cursor += 8.0 + 8.0 * threshold_seed;
            }
            side = -side;
            id += 1;
        }
    }

    // --- corruption pass ----------------------------------------------------
    corrupt(
        seed,
        rx,
        rz,
        &dominant,
        config,
        noise,
        &mut assemblies,
        &mut taken,
        &corridors,
    );

    // --- vertical circulation ------------------------------------------------
    // The macro graph may have reserved a stairwell here. Upward links are
    // realized as a Stair assembly beside the main spine; downward links
    // stay graph reservations until the engine streams below elevation 0.
    // Placed after corruption so no pass can duplicate, abandon, or redden
    // a stair core.
    if let Some(link) = crate::use_cases::world_topology::vertical_link_for_region(seed, noise, rx, rz)
        && crate::use_cases::vertical_circulation::link_wants_geometry(&link)
        && let Some(stair) = crate::use_cases::vertical_circulation::place_stairwell(
            id + 2000,
            &link,
            &legs,
            region_origin,
            region_size,
            EDGE_MARGIN,
            PLAN_WALL_T,
            &taken,
            |cx, cz, clearance| {
                corridors
                    .iter()
                    .all(|s| s.distance(cx, cz) >= s.width * 0.5 + clearance)
            },
        )
    {
        taken.push(stair.footprint.bounds());
        assemblies.push(stair);
    }

    // Macro anomalies are derived after ordinary architecture so compact red
    // rooms can promote a real assembly, while region-spanning families keep
    // stable world-lattice identities independent of this region query.
    let anomalies = plan_anomalies_for_region(
        seed,
        region_origin,
        region_size,
        &assemblies,
        spawn_point(seed),
        config,
        noise,
    );

    RegionPlan {
        origin_world: region_origin,
        size_world: region_size,
        architects,
        assemblies,
        corridors,
        anomalies,
    }
}

/// Backrooms corruption: the plan was sane; the building is not.
#[allow(clippy::too_many_arguments)]
fn corrupt(
    seed: u32,
    rx: i64,
    rz: i64,
    dominant: &ArchitectGenome,
    config: &GeneratorConfig,
    noise: &dyn NoiseProvider,
    assemblies: &mut Vec<AssemblyInstance>,
    taken: &mut Vec<(f32, f32, f32, f32)>,
    spines: &[CirculationSpine],
) {
    if assemblies.is_empty() {
        return;
    }
    let h = |k: i64| hash01(seed, &[0xC0DE + k, rx, rz]);

    // 1. A suite repeats itself further down the corridor, slightly wrong.
    if h(1) < 0.6 {
        let src = pick_index(h(2), assemblies.len());
        let shift = snap(12.0 + 12.0 * h(3));
        // Keep the copied threshold flush with its source corridor. The
        // repetition is wrong in its longitudinal position, not sealed away
        // behind an accidental strip of wall.
        let skew = 0.0;
        let mut dup = assemblies[src].clone();
        let translate = |poly: &mut Polygon2| {
            for v in &mut poly.vertices {
                v.0 += shift;
                v.1 += skew;
            }
        };
        translate(&mut dup.footprint);
        for s in &mut dup.spaces {
            translate(&mut s.footprint);
        }
        for c in &mut dup.ceiling_zones {
            translate(&mut c.area);
        }
        for f in &mut dup.fixtures {
            f.at.x += shift;
            f.at.z += skew;
        }
        for e in &mut dup.entrances {
            e.center.x += shift;
            e.center.z += skew;
        }
        dup.id = assemblies.len() as u32 + 1000;
        dup.corruption.misalignment = (shift, skew);
        let b = dup.footprint.bounds();
        let inside = b.0 > rx as f32 * REGION_SIZE + EDGE_MARGIN
            && b.2 < (rx + 1) as f32 * REGION_SIZE - EDGE_MARGIN
            && b.1 > rz as f32 * REGION_SIZE + EDGE_MARGIN
            && b.3 < (rz + 1) as f32 * REGION_SIZE - EDGE_MARGIN;
        // The skewed entrance must still reach a corridor, or the copy would
        // be a sealed pocket.
        let reachable = dup.entrances.iter().any(|e| {
            spines
                .iter()
                .any(|s| s.distance(e.center.x, e.center.z) <= s.width * 0.5 + PLAN_WALL_T + 0.05)
        });
        if inside && reachable && !taken.iter().any(|t| aabb_overlap(*t, b, 0.4)) {
            taken.push(b);
            assemblies.push(dup);
        }
    }

    // 2. One assembly was built and then never occupied.
    if h(6) < 0.6 {
        let k = pick_index(h(7), assemblies.len());
        let a = &mut assemblies[k];
        a.program = SpaceProgram::AbandonedExpansion;
        a.corruption.abandoned = true;
        a.spaces.clear(); // an empty shell
        for f in &mut a.fixtures {
            f.lit = false;
        }
    }

    // 3. A renovation overlays the original structural grid.
    if dominant.renovation_history != RenovationStyle::Untouched {
        let k = pick_index(h(8), assemblies.len());
        assemblies[k].corruption.renovation_overlay = true;
    }

    // 4. Rarely, one occupied room's fixtures all burn red. The lighting is
    // the anomaly, and it belongs to a whole architectural space — never a
    // lone fixture, never red masonry. Applied last so it respects whatever
    // the earlier corruption passes decided (an abandoned shell stays dark).
    //
    // Where a red room happens is a *graph event*, not a per-region coin
    // flip: the macro topology plans separated events on its lattice (see
    // `world_topology::red_room_event_for_region`), and this pass merely
    // realizes an event that targets this region on one eligible assembly.
    let red_room_scale = config.anomalies.frequency * config.anomalies.red_rooms;
    let mut red_room_forced = false;
    let sp = spawn_point(seed);
    if config.anomalies.forced_kind == Some(AnomalyKind::RedRoom)
        && rx == region_index(sp.x)
        && rz == region_index(sp.z)
    {
        if let Some(a) = assemblies
            .iter_mut()
            .find(|a| !a.entrances.is_empty() && !a.corruption.abandoned)
        {
            a.corruption.red_room = true;
            red_room_forced = true;
        } else if let Some(spine) = spines
            .iter()
            .find(|s| s.spine_kind == SpaceProgram::MainCorridor)
        {
            if let Some(seg) = spine.path.windows(2).find(|seg| seg[0].z == seg[1].z) {
                let lx0 = seg[0].x.min(seg[1].x);
                let lz = seg[0].z;
                let origin = Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE);
                if let Some(mut a) = place_suite(
                    9999,
                    SpaceProgram::PrivateOffice,
                    dominant,
                    0.5,
                    0.5,
                    lx0 + 12.0,
                    lz,
                    spine.width * 0.5 + PLAN_WALL_T,
                    1.0,
                    origin,
                    REGION_SIZE,
                    taken,
                    spines,
                ) {
                    a.corruption.red_room = true;
                    taken.push(a.footprint.bounds());
                    assemblies.push(a);
                    red_room_forced = true;
                }
            }
        }
    }

    if !red_room_forced
        && let Some(event) = crate::use_cases::world_topology::red_room_event_for_region(
            seed,
            noise,
            rx,
            rz,
            red_room_scale.clamp(0.0, 4.0),
        )
    {
        // Realize the event on one occupied, reachable assembly, rotating
        // from a stable start so the choice replays for every query of this
        // region. If no assembly qualifies the event stays unrealized — a
        // planned encounter never overwrites navigable circulation.
        let count = assemblies.len();
        let start = (event.id % count as u64) as usize;
        for offset in 0..count {
            let a = &mut assemblies[(start + offset) % count];
            if !a.corruption.abandoned && !a.entrances.is_empty() {
                a.corruption.red_room = true;
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Debug hook.
// ---------------------------------------------------------------------------

/// ASCII plan view at `step` world units per character. Corridors `=`, macro
/// anomalies `P/B/H` (pillar/blackout/pits), red loops `R`, assembly interiors
/// by program initial, entrances `+`, empty fabric `.`.
pub fn debug_region_ascii(plan: &RegionPlan, step: f32) -> String {
    let n = (plan.size_world / step) as usize;
    let mut out = String::with_capacity((n + 1) * n);
    for iz in 0..n {
        for ix in 0..n {
            let x = plan.origin_world.x + (ix as f32 + 0.5) * step;
            let z = plan.origin_world.z + (iz as f32 + 0.5) * step;
            let mut ch = '.';
            for a in &plan.assemblies {
                if a.footprint.contains(x, z) {
                    ch = match a.program {
                        SpaceProgram::OpenOffice => 'o',
                        SpaceProgram::PrivateOffice => 'p',
                        SpaceProgram::ConferenceRoom => 'c',
                        SpaceProgram::BreakRoom => 'b',
                        SpaceProgram::Storage => 's',
                        SpaceProgram::ServerRoom => 'v',
                        SpaceProgram::WaitingArea => 'w',
                        SpaceProgram::AbandonedExpansion => 'x',
                        SpaceProgram::Atrium => 'A',
                        SpaceProgram::Stair => 'S',
                        _ => 'r',
                    };
                }
            }
            for anomaly in &plan.anomalies {
                if anomaly.contains(x, z) {
                    ch = match anomaly.kind {
                        AnomalyKind::PillarExpanse => 'P',
                        AnomalyKind::BlackoutExpanse => 'B',
                        AnomalyKind::PitLattice => 'H',
                        AnomalyKind::RedRoom => 'R',
                        AnomalyKind::ArchwayRoom => 'M',
                    };
                }
            }
            for s in &plan.corridors {
                if s.distance(x, z) <= s.width * 0.5 {
                    ch = '=';
                }
            }
            for a in &plan.assemblies {
                for e in &a.entrances {
                    if (x - e.center.x).abs() < e.width * 0.5 && (z - e.center.z).abs() < step {
                        ch = '+';
                    }
                }
            }
            out.push(ch);
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frameworks_drivers::simple_noise::SimpleNoiseProvider;

    fn plan(rx: i64, rz: i64) -> RegionPlan {
        generate_region_plan(
            42,
            Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
            REGION_SIZE,
            &GeneratorConfig::low_spec(),
            &SimpleNoiseProvider::new(),
        )
    }

    #[test]
    fn region_plan_is_deterministic() {
        let a = debug_region_ascii(&plan(0, 0), 1.0);
        let b = debug_region_ascii(&plan(0, 0), 1.0);
        assert_eq!(a, b);
        assert!(!plan(0, 0).assemblies.is_empty(), "plan places suites");
    }

    #[test]
    fn corridors_meet_neighbors_at_shared_portals() {
        // Region (0,0)'s east edge is region (1,0)'s west edge: the main
        // spines must terminate at the identical portal point.
        let a = plan(0, 0);
        let b = plan(1, 0);
        let edge_x = REGION_SIZE;
        let end_a = a.corridors[0]
            .path
            .iter()
            .find(|p| (p.x - edge_x).abs() < 1e-3)
            .expect("main spine reaches east edge");
        let end_b = b.corridors[0]
            .path
            .iter()
            .find(|p| (p.x - edge_x).abs() < 1e-3)
            .expect("neighbor main spine reaches west edge");
        assert!(
            (end_a.z - end_b.z).abs() < 1e-3,
            "portal mismatch: {} vs {}",
            end_a.z,
            end_b.z
        );
    }

    #[test]
    fn primary_circulation_is_wide_and_secondary_routes_are_sparse() {
        for rx in -3..=3 {
            for rz in -3..=3 {
                let p = plan(rx, rz);
                let main: Vec<_> = p
                    .corridors
                    .iter()
                    .filter(|s| s.spine_kind == SpaceProgram::MainCorridor)
                    .collect();
                let secondary: Vec<_> = p
                    .corridors
                    .iter()
                    .filter(|s| s.spine_kind == SpaceProgram::SecondaryHall)
                    .collect();
                assert_eq!(main.len(), 1, "region ({rx},{rz}) lacks one dominant spine");
                assert!(
                    ((4.8 - 0.01)..=(7.2 + 0.01)).contains(&main[0].width),
                    "main width {} outside Level 0 target in ({rx},{rz})",
                    main[0].width
                );
                assert!(
                    secondary.len() <= 2,
                    "region ({rx},{rz}) made {} secondary routes",
                    secondary.len()
                );
                for branch in secondary {
                    assert!(
                        (3.5..=5.1).contains(&branch.width),
                        "secondary width {} outside target in ({rx},{rz})",
                        branch.width
                    );
                }
            }
        }
    }

    #[test]
    fn assemblies_are_large_and_narrow_doors_are_anomalies() {
        let mut entrances = 0usize;
        let mut narrow_doors = 0usize;
        let mut broad_or_unframed = 0usize;
        for rx in -6..=6 {
            for rz in -6..=6 {
                let p = plan(rx, rz);
                for a in &p.assemblies {
                    let b = a.footprint.bounds();
                    // Stair cores are deliberately compact circulation, not
                    // program suites; every other mass keeps suite scale.
                    if a.program != SpaceProgram::Stair {
                        assert!(
                            ((12.0 - 0.01)..=(24.0 + 0.01)).contains(&(b.2 - b.0)),
                            "assembly {} has {} u frontage",
                            a.id,
                            b.2 - b.0
                        );
                    }
                    for e in &a.entrances {
                        entrances += 1;
                        if e.width <= 1.3 {
                            narrow_doors += 1;
                        } else {
                            broad_or_unframed += 1;
                        }
                    }
                }
            }
        }
        let narrow_ratio = narrow_doors as f32 / entrances as f32;
        assert!(
            (0.06..=0.16).contains(&narrow_ratio),
            "narrow doors are {:.1}% of {entrances} openings",
            narrow_ratio * 100.0
        );
        assert!(
            broad_or_unframed * 100 >= entrances * 84,
            "broad or unframed openings were only {broad_or_unframed}/{entrances}"
        );
    }

    #[test]
    fn every_assembly_entrance_opens_onto_a_corridor() {
        for (rx, rz) in [(0i64, 0i64), (1, 0), (-1, 2), (3, -4)] {
            let p = plan(rx, rz);
            for a in &p.assemblies {
                assert!(!a.entrances.is_empty(), "assembly {} has no door", a.id);
                for e in &a.entrances {
                    let near = p.corridors.iter().any(|s| {
                        s.distance(e.center.x, e.center.z) <= s.width * 0.5 + PLAN_WALL_T + 0.05
                    });
                    assert!(
                        near,
                        "assembly {} ({:?}) door at ({}, {}) reaches no corridor in region ({rx},{rz})",
                        a.id, a.program, e.center.x, e.center.z
                    );
                }
            }
        }
    }

    #[test]
    fn assemblies_stay_inside_their_region_and_apart() {
        for (rx, rz) in [(0i64, 0i64), (2, 1)] {
            let p = plan(rx, rz);
            let (ox, oz) = (p.origin_world.x, p.origin_world.z);
            for (i, a) in p.assemblies.iter().enumerate() {
                let b = a.footprint.bounds();
                assert!(b.0 >= ox && b.2 <= ox + REGION_SIZE, "x escape in {i}");
                assert!(b.1 >= oz && b.3 <= oz + REGION_SIZE, "z escape in {i}");
                for other in p.assemblies.iter().skip(i + 1) {
                    assert!(
                        !aabb_overlap(b, other.footprint.bounds(), -0.05),
                        "assemblies {} and {} overlap",
                        a.id,
                        other.id
                    );
                }
            }
        }
    }

    /// Red rooms are graph events: a region shows one only when the macro
    /// topology targeted it, and committed encounters keep the cooldown
    /// distance from each other (no clustering into a red biome).
    #[test]
    fn red_rooms_realize_only_macro_graph_events_and_keep_their_distance() {
        use crate::use_cases::world_topology::red_room_event_for_region;
        let noise = SimpleNoiseProvider::new();
        let mut realized: Vec<(i64, i64)> = Vec::new();
        for rx in -8..=8 {
            for rz in -8..=8 {
                let p = plan(rx, rz);
                let has_red = p.assemblies.iter().any(|a| a.corruption.red_room);
                let event = red_room_event_for_region(42, &noise, rx, rz, 1.0);
                if event.is_none() {
                    assert!(
                        !has_red,
                        "region ({rx},{rz}) has a red room without a graph event"
                    );
                }
                if has_red {
                    realized.push((rx, rz));
                }
            }
        }
        assert!(!realized.is_empty(), "no red-room event realized in 289 regions");
        for (i, a) in realized.iter().enumerate() {
            for b in realized.iter().skip(i + 1) {
                let chebyshev = (a.0 - b.0).abs().max((a.1 - b.1).abs());
                assert!(
                    chebyshev >= 3,
                    "red rooms at {a:?} and {b:?} violate the macro cooldown"
                );
            }
        }
    }

    /// Where the macro graph reserved an upward vertical link and placement
    /// succeeded, the region owns exactly one Stair assembly with a
    /// corridor-facing entrance near the link's anchor.
    #[test]
    fn stairwells_realize_upward_vertical_links() {
        use crate::use_cases::vertical_circulation::link_wants_geometry;
        use crate::use_cases::world_topology::vertical_link_for_region;
        let noise = SimpleNoiseProvider::new();
        let mut realized = 0usize;
        for rx in -8..=8 {
            for rz in -8..=8 {
                let p = plan(rx, rz);
                let stairs: Vec<_> = p
                    .assemblies
                    .iter()
                    .filter(|a| a.program == SpaceProgram::Stair)
                    .collect();
                let link = vertical_link_for_region(42, &noise, rx, rz);
                match link {
                    Some(link) if link_wants_geometry(&link) => {
                        assert!(stairs.len() <= 1, "region ({rx},{rz}) built extra stairs");
                        if let Some(stair) = stairs.first() {
                            realized += 1;
                            assert!(!stair.entrances.is_empty(), "stair core has no entrance");
                            let b = stair.footprint.bounds();
                            let (cx, cz) = ((b.0 + b.2) * 0.5, (b.1 + b.3) * 0.5);
                            let d = ((cx - link.anchor.x).powi(2)
                                + (cz - link.anchor.z).powi(2))
                            .sqrt();
                            assert!(
                                d < REGION_SIZE,
                                "stair strayed {d} u from its reservation anchor"
                            );
                        }
                    }
                    _ => assert!(
                        stairs.is_empty(),
                        "region ({rx},{rz}) built a stair without a reservation"
                    ),
                }
            }
        }
        assert!(realized >= 3, "only {realized} stairwells in 289 regions");
    }

    #[test]
    fn corruption_appears_somewhere() {
        // Over a handful of regions the corruption pass must fire: at least
        // one abandoned expansion and one duplicated (misaligned) suite.
        let mut abandoned = 0;
        let mut duplicated = 0;
        for rx in -3..3 {
            for rz in -3..3 {
                let p = plan(rx, rz);
                abandoned += p
                    .assemblies
                    .iter()
                    .filter(|a| a.corruption.abandoned)
                    .count();
                duplicated += p
                    .assemblies
                    .iter()
                    .filter(|a| a.corruption.misalignment != (0.0, 0.0))
                    .count();
            }
        }
        assert!(abandoned > 0, "no abandoned expansions in 36 regions");
        assert!(duplicated > 0, "no duplicated suites in 36 regions");
    }
}
