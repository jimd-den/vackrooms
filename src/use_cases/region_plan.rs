//! Region-scale architectural planning for Backrooms Level 0.
//!
//! The world is tiled by fixed 80 u square regions on a world-space lattice.
//! [`generate_region_plan`] is a *pure* function of `(seed, region)`: it
//! derives the region's designers ([`ArchitectGenome`]), routes circulation
//! first (corridor spines between edge portals shared with the neighboring
//! regions), attaches program spaces ([`AssemblyInstance`]) to the corridors,
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

use crate::domain::entities::architecture::*;
use crate::entities::models::Position;
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
// Deterministic hashing (all "randomness" flows through here).
// ---------------------------------------------------------------------------

fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}

/// Hash of a seed plus any number of integer keys, uniform in [0, 1).
fn hash01(seed: u32, keys: &[i64]) -> f32 {
    let mut h = mix(seed ^ 0x9E37_79B9);
    for &k in keys {
        h = mix(h ^ (k as u32)).wrapping_add(mix((k >> 32) as u32));
    }
    (mix(h) >> 8) as f32 / (1u32 << 24) as f32
}

fn pick_index(h: f32, len: usize) -> usize {
    ((h * len as f32) as usize).min(len - 1)
}

// ---------------------------------------------------------------------------
// Edge portals: where corridors cross region borders.
// ---------------------------------------------------------------------------

/// Portal position on the vertical edge `x = ex * REGION_SIZE` of region row
/// `rz`. Both regions sharing the edge derive the same value.
fn v_edge_portal_z(seed: u32, ex: i64, rz: i64) -> f32 {
    let f = 0.30 + 0.40 * hash01(seed, &[0x0E1, ex, rz]);
    snap((rz as f32 + f) * REGION_SIZE)
}

/// Portal position on the horizontal edge `z = ez * REGION_SIZE` of region
/// column `rx`.
fn h_edge_portal_x(seed: u32, rx: i64, ez: i64) -> f32 {
    let f = 0.30 + 0.40 * hash01(seed, &[0x0E2, rx, ez]);
    snap((rx as f32 + f) * REGION_SIZE)
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
        2 => CirculationStyle::LoopingRing,
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
    let threshold_language = *[
        ThresholdLanguage::DoorWithLintel,
        ThresholdLanguage::DoorWithLintel, // doors are the office default
        ThresholdLanguage::OpenPortal,
        ThresholdLanguage::WidePortal,
    ]
    .get(pick_index(h(2), 4))
    .unwrap();
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
            min_side: 4.8 + 2.4 * h(6),
            max_side: 12.0 + 8.0 * h(7),
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
    let north_x = h_edge_portal_x(seed, rx, rz);
    let south_x = h_edge_portal_x(seed, rx, rz + 1);

    let main_w = snap_width(2.8 + 0.8 * genome.tolerance_for_symmetry);
    let sec_w = snap_width(2.0 + 0.4 * hash01(seed, &[0xC1, rx, rz]));

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

    // Secondary hall: north portal -> south portal, routed through a point
    // on the main spine so the two families always connect.
    let join_z = spine_z_at(&main_z_at, north_x);
    push(
        SpaceProgram::SecondaryHall,
        vec![
            Position::new(north_x, z0),
            Position::new(north_x, join_z),
            Position::new(south_x, join_z),
            Position::new(south_x, z1),
        ],
        sec_w,
        &mut id,
    );

    match genome.circulation {
        CirculationStyle::LoopingRing => {
            // A rectangular ring hung off the main spine.
            let m = snap(size * 0.22);
            let (rx0, rz0, rx1, rz1) = (x0 + m, z0 + m, x1 - m, z1 - m);
            push(
                SpaceProgram::SecondaryHall,
                vec![
                    Position::new(rx0, rz0),
                    Position::new(rx1, rz0),
                    Position::new(rx1, rz1),
                    Position::new(rx0, rz1),
                    Position::new(rx0, rz0),
                ],
                sec_w,
                &mut id,
            );
            // Connector from the ring up to the main spine.
            let cz = spine_z_at(&main_z_at, rx0);
            push(
                SpaceProgram::SecondaryHall,
                vec![Position::new(rx0, rz0), Position::new(rx0, cz)],
                sec_w,
                &mut id,
            );
        }
        CirculationStyle::TreeWithCulDeSacs => {
            // Dead-end stubs off the main spine: unease by design.
            for k in 0..2 {
                let sx = snap(x0 + size * (0.25 + 0.5 * h(10 + k)));
                let sz = spine_z_at(&main_z_at, sx);
                let len = snap(8.0 + 8.0 * h(20 + k));
                let dir = if h(30 + k) < 0.5 { 1.0 } else { -1.0 };
                let end = (sz + dir * len).clamp(z0 + EDGE_MARGIN, z1 - EDGE_MARGIN);
                push(
                    SpaceProgram::SecondaryHall,
                    vec![Position::new(sx, sz), Position::new(sx, snap(end))],
                    sec_w,
                    &mut id,
                );
            }
        }
        _ => {}
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

/// The suite program palette placed along main corridors, roughly weighted.
const SUITE_PROGRAMS: [SpaceProgram; 8] = [
    SpaceProgram::OpenOffice,
    SpaceProgram::OpenOffice,
    SpaceProgram::PrivateOffice,
    SpaceProgram::ConferenceRoom,
    SpaceProgram::BreakRoom,
    SpaceProgram::Storage,
    SpaceProgram::ServerRoom,
    SpaceProgram::WaitingArea,
];

fn ceiling_height_for(program: SpaceProgram) -> f32 {
    match program {
        SpaceProgram::Atrium => 5.2,
        SpaceProgram::ServerRoom | SpaceProgram::Mechanical => 2.6,
        SpaceProgram::Storage | SpaceProgram::RestroomCore => 2.5,
        _ => 2.8,
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

/// Interior partitioning: private-office suites get sliced into rooms along
/// their long axis; each slice is a `Space` (the voxelizer draws partition
/// walls with doorways between adjacent slices).
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
    let want = (genome.room_proportions.min_side * 0.9).max(2.8);
    let n = ((span / want) as usize).clamp(1, 5);
    let n = 1 + ((n - 1) as f32 * (0.5 + 0.5 * aseed)) as usize;
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
    let w = snap((p.min_side + (p.max_side - p.min_side) * aseed).clamp(4.0, 20.0));
    let d = snap((w * (1.0 - 0.5 * p.elongation)).clamp(4.0, 16.0));

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

    // Entrance on the corridor-facing wall.
    let front_z = if side > 0.0 { b.1 } else { b.3 };
    let (width, lintel) = match genome.threshold_language {
        ThresholdLanguage::DoorWithLintel => (1.2, Some(2.2)),
        ThresholdLanguage::OpenPortal => (2.0, None),
        ThresholdLanguage::WidePortal => (2.4, Some(2.2)),
    };
    let door_x = snap((b.0 + w * (0.3 + 0.4 * aseed)).clamp(b.0 + 1.2, b.2 - 1.2));
    let entrances = vec![Opening {
        center: Position::new(door_x, front_z),
        width,
        through_x_wall: true,
        lintel_units: lintel,
    }];

    let ceiling = CeilingZone {
        area: footprint.clone(),
        language: genome.ceiling_language,
        height_units: ceiling_height_for(program),
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
    _config: &GeneratorConfig,
    noise: &dyn NoiseProvider,
) -> RegionPlan {
    let rx = region_index(region_origin.x + 0.1);
    let rz = region_index(region_origin.z + 0.1);
    let h = |k: i64| hash01(seed, &[0xA0 + k, rx, rz]);

    let dominant = derive_genome(seed, rx, rz, 0, noise);
    let renovator = derive_genome(seed, rx, rz, 1, noise);
    let mut architects = vec![dominant.clone(), renovator];
    if h(0) < 0.3 {
        architects.push(derive_genome(seed, rx, rz, 2, noise));
    }

    let corridors = build_corridors(seed, rx, rz, &dominant, region_origin, region_size);

    // --- suites along every long horizontal corridor leg -------------------
    let mut assemblies: Vec<AssemblyInstance> = Vec::new();
    let mut taken: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut id = 0u32;
    let legs: Vec<(f32, f32, f32, f32)> = corridors
        .iter()
        .flat_map(|s| {
            let w = s.width;
            s.path
                .windows(2)
                .filter(|seg| seg[0].z == seg[1].z)
                .map(move |seg| {
                    (
                        seg[0].x.min(seg[1].x),
                        seg[0].x.max(seg[1].x),
                        seg[0].z,
                        w,
                    )
                })
                .collect::<Vec<_>>()
        })
        .filter(|(a, b, _, _)| b - a >= 14.0)
        .collect();

    for (li, &(lx0, lx1, lz, lw)) in legs.iter().enumerate() {
        let mut cursor = lx0 + 2.4;
        let mut side = if h(40 + li as i64) < 0.5 { 1.0 } else { -1.0 };
        while cursor < lx1 - 8.0 {
            let aseed = hash01(seed, &[0x5EA, rx, rz, id as i64]);
            let program = SUITE_PROGRAMS[pick_index(aseed, SUITE_PROGRAMS.len())];
            if let Some(a) = place_suite(
                id,
                program,
                &dominant,
                aseed,
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
                // Generous spacing: the gap between suites is real space,
                // not a slit between parallel walls.
                cursor += (b.2 - b.0) + 3.2;
                taken.push(b);
                assemblies.push(a);
            } else {
                cursor += 4.0;
            }
            side = -side;
            id += 1;
        }
    }

    // --- corruption pass ----------------------------------------------------
    corrupt(seed, rx, rz, &dominant, &mut assemblies, &mut taken, &corridors);

    RegionPlan {
        origin_world: region_origin,
        size_world: region_size,
        architects,
        assemblies,
        corridors,
    }
}

/// Backrooms corruption: the plan was sane; the building is not.
fn corrupt(
    seed: u32,
    rx: i64,
    rz: i64,
    dominant: &ArchitectGenome,
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
        let shift = snap(6.0 + 10.0 * h(3));
        let skew = snap(0.4 + 0.4 * h(4)) * if h(5) < 0.5 { 1.0 } else { -1.0 };
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
}

// ---------------------------------------------------------------------------
// Debug hook.
// ---------------------------------------------------------------------------

/// ASCII plan view at `step` world units per character. Corridors `=`,
/// assembly interiors by program initial, entrances `+`, empty fabric `.`.
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
                        _ => 'r',
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

    #[test]
    fn corruption_appears_somewhere() {
        // Over a handful of regions the corruption pass must fire: at least
        // one abandoned expansion and one duplicated (misaligned) suite.
        let mut abandoned = 0;
        let mut duplicated = 0;
        for rx in -3..3 {
            for rz in -3..3 {
                let p = plan(rx, rz);
                abandoned += p.assemblies.iter().filter(|a| a.corruption.abandoned).count();
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
