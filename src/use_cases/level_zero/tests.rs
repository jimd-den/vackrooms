use super::fabric::{FABRIC_CELL, FabricCeilingBand};
use super::*;
use crate::domain::entities::anomaly::{
    AnomalyInstance, AnomalyKind, RealitySnapshot, WorldBounds,
};
use crate::domain::entities::architecture::SpaceProgram;
use crate::domain::entities::voxel_grid::{VOXEL_AIR, VOXEL_STICKY_CARPET, VOXEL_WALL, VoxelGrid};
use crate::entities::models::Position;
use crate::use_cases::anomalies::geometry::sample_anomaly;
use crate::use_cases::generate_chunk::{GeneratorConfig, LevelTuning};
use crate::use_cases::level_generator::LevelGenerator;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::red_rooms::geometry::sample_red_room;
use crate::use_cases::region_plan::{PLAN_WALL_T, REGION_SIZE, region_index};

struct TestNoise;

impl TestNoise {
    fn hash2d(seed: u32, x: i32, y: i32) -> f32 {
        let mut h = seed
            .wrapping_add(x as u32 ^ 0x9E3779B9)
            .wrapping_add(y as u32 ^ 0x85EBCA6B);
        h ^= h >> 16;
        h = h.wrapping_mul(0x85EBCA6B);
        h ^= h >> 13;
        h = h.wrapping_mul(0xC2B2AE35);
        h ^= h >> 16;
        ((h as f32) / (std::u32::MAX as f32)) * 2.0 - 1.0
    }

    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }

    fn smoothstep(t: f32) -> f32 {
        t * t * (3.0 - 2.0 * t)
    }
}

impl NoiseProvider for TestNoise {
    fn evaluate_2d(&self, seed: u32, position: Position) -> f32 {
        let freq = 0.05;
        let x = position.x * freq;
        let y = position.z * freq;
        let x0 = x.floor() as i32;
        let x1 = x0 + 1;
        let y0 = y.floor() as i32;
        let y1 = y0 + 1;
        let tx = x - (x0 as f32);
        let ty = y - (y0 as f32);
        let u = Self::smoothstep(tx);
        let v = Self::smoothstep(ty);
        let v00 = Self::hash2d(seed, x0, y0);
        let v10 = Self::hash2d(seed, x1, y0);
        let v01 = Self::hash2d(seed, x0, y1);
        let v11 = Self::hash2d(seed, x1, y1);
        let nx0 = Self::lerp(v00, v10, u);
        let nx1 = Self::lerp(v01, v11, u);
        Self::lerp(nx0, nx1, v)
    }
}


fn generate(ox: f32, oz: f32) -> VoxelGrid {
    BackroomsLevel.generate(
        Position::new(ox, oz),
        42,
        GeneratorConfig::low_spec(),
        &TestNoise,
    )
}

fn is_open(grid: &VoxelGrid, x: usize, z: usize) -> bool {
    grid.get(x, 1, z) == VOXEL_AIR
}

/// Walkable-plane connectivity: nearly all open floor must be mutually
/// reachable (walls always leave doorways, pillars never seal a region).
/// We exclude a 2-voxel border to avoid edge-of-chunk artifacts where
/// walls at boundaries form isolated strips that would be connected by
/// the adjacent chunk at runtime.
#[test]
fn walkable_plane_is_connected() {
    let grid = generate(0.0, 0.0);
    let (w, d) = (grid.width(), grid.depth());
    let margin = 2usize; // exclude edge voxels

    let open: Vec<(usize, usize)> = (margin..w - margin)
        .flat_map(|x| (margin..d - margin).map(move |z| (x, z)))
        .filter(|&(x, z)| is_open(&grid, x, z))
        .collect();
    assert!(
        open.len() > (w - 2 * margin) * (d - 2 * margin) / 2,
        "backrooms must be mostly open space"
    );

    // BFS from the first open interior voxel.
    let &(sx, sz) = open.first().expect("some open floor exists");
    let mut visited = vec![false; w * d];
    let mut queue = std::collections::VecDeque::from([(sx, sz)]);
    visited[sx * d + sz] = true;
    let mut reached = 0usize;
    while let Some((x, z)) = queue.pop_front() {
        reached += 1;
        for (dx, dz) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
            let (nx, nz) = (x as i64 + dx, z as i64 + dz);
            if nx < margin as i64
                || nz < margin as i64
                || nx >= (w - margin) as i64
                || nz >= (d - margin) as i64
            {
                continue;
            }
            let (nx, nz) = (nx as usize, nz as usize);
            if !visited[nx * d + nz] && is_open(&grid, nx, nz) {
                visited[nx * d + nz] = true;
                queue.push_back((nx, nz));
            }
        }
    }
    let ratio = reached as f32 / open.len() as f32;
    assert!(
        ratio > 0.95,
        "only {:.0}% of open floor is reachable from spawn",
        ratio * 100.0
    );
}

/// Adjacent chunks must agree at their shared border: the column at
/// world position g is the same whether it came from chunk A's last
/// column or chunk B's first.
#[test]
fn chunks_tile_seamlessly() {
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let a = generate(0.0, 0.0);
    let b = generate(10.0, 0.0);
    let w = a.width();

    let plans =
        BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 20.0, 42, &config, &noise);
    for z in 0..a.depth() {
        for (grid, lx, gx) in [(&a, w - 1, w - 1), (&b, 0usize, w)] {
            let wx = (gx as f32 + 0.5) * config.voxel_scale;
            let wz = (z as f32 + 0.5) * config.voxel_scale;
            let key = (
                crate::use_cases::region_plan::region_index(wx),
                crate::use_cases::region_plan::region_index(wz),
            );
            let plan = plans
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, p)| p)
                .unwrap();
            let expect =
                BackroomsLevel::plan_column(plan, &noise, 42, &LevelTuning::default(), wx, wz);
            let got_solid = grid.get(lx, 1, z) == VOXEL_WALL;
            // Raised floor (stair treads) also writes wall material at
            // the walkable layer, so it counts as expected solid here.
            let expect_solid =
                expect.solid || (expect.floor_units / config.voxel_scale).round() >= 1.0;
            assert_eq!(
                got_solid, expect_solid,
                "column mismatch at world x={gx} z={z}"
            );
        }
    }
}

#[test]
fn recursive_level_zero_is_independent_of_output_partition() {
    use crate::domain::entities::anomaly::{
        AnomalyStateStamp, Axis2, AxisDirection, RedRoomPhase,
    };

    // A committed encounter selects a deterministic recursive Level 0
    // address.  The ID is intentionally wider than f32 can represent so
    // this also protects the integer-first branch derivation.
    let reality = RealitySnapshot::new(vec![AnomalyStateStamp::new(
        0xDEAD_BEEF_1234_5678,
        1,
        4.8,
        Axis2::X,
        AxisDirection::Positive,
        RedRoomPhase::Sealed,
        0,
        0xCAFE,
    )]);
    let noise = TestNoise;
    let small_config = GeneratorConfig::low_spec();
    let mut large_config = small_config;
    large_config.chunk_size = 20.0;

    let large = BackroomsLevel.generate_with_reality(
        Position::new(0.0, 0.0),
        42,
        large_config,
        &noise,
        &reality,
    );
    let chunks = [
        BackroomsLevel.generate_with_reality(
            Position::new(0.0, 0.0),
            42,
            small_config,
            &noise,
            &reality,
        ),
        BackroomsLevel.generate_with_reality(
            Position::new(10.0, 0.0),
            42,
            small_config,
            &noise,
            &reality,
        ),
        BackroomsLevel.generate_with_reality(
            Position::new(0.0, 10.0),
            42,
            small_config,
            &noise,
            &reality,
        ),
        BackroomsLevel.generate_with_reality(
            Position::new(10.0, 10.0),
            42,
            small_config,
            &noise,
            &reality,
        ),
    ];

    let tile = chunks[0].width();
    assert_eq!(large.width(), tile * 2);
    assert_eq!(large.depth(), tile * 2);
    for z in 0..large.depth() {
        for x in 0..large.width() {
            let tile_index = usize::from(x >= tile) + 2 * usize::from(z >= tile);
            let local_x = x % tile;
            let local_z = z % tile;
            for y in 0..large.height() {
                assert_eq!(
                    large.get(x, y, z),
                    chunks[tile_index].get(local_x, y, local_z),
                    "recursive Level 0 changed at ({x}, {y}, {z}) when the output was tiled"
                );
            }
        }
    }
}

/// The plan is authoritative: corridor centerlines must be carved open
/// in the voxelized chunks they cross.
#[test]
fn corridors_from_the_plan_are_carved_open() {
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let plans =
        BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 80.0, 42, &config, &noise);
    let plan = &plans.iter().find(|(k, _)| *k == (0, 0)).unwrap().1;
    let spine = &plan.corridors[0];

    let mut checked = 0;
    for seg in spine.path.windows(2) {
        let (p0, p1) = (seg[0], seg[1]);
        let steps = 8;
        for k in 1..steps {
            let t = k as f32 / steps as f32;
            let (wx, wz) = (p0.x + (p1.x - p0.x) * t, p0.z + (p1.z - p0.z) * t);
            // Stay inside region (0,0) and off chunk edges.
            if !(1.0..79.0).contains(&wx) || !(1.0..79.0).contains(&wz) {
                continue;
            }
            let (cx, cz) = ((wx / 10.0).floor() * 10.0, (wz / 10.0).floor() * 10.0);
            let grid = generate(cx, cz);
            let (lx, lz) = (
                ((wx - cx) / config.voxel_scale) as usize,
                ((wz - cz) / config.voxel_scale) as usize,
            );
            assert!(
                is_open(&grid, lx.min(grid.width() - 1), lz.min(grid.depth() - 1)),
                "main corridor blocked at world ({wx:.1}, {wz:.1})"
            );
            checked += 1;
        }
    }
    assert!(checked > 5, "spine barely sampled ({checked} points)");
}

#[test]
fn circulation_uses_the_raised_ceiling_hierarchy() {
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let mut secondary_checked = false;
    for rx in -3i64..=3 {
        for rz in -3i64..=3 {
            let plans = BackroomsLevel::region_plans_for(
                Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                1.0,
                42,
                &config,
                &noise,
            );
            let plan = &plans
                .iter()
                .find(|(key, _)| *key == (rx, rz))
                .expect("requested region plan")
                .1;
            for spine in &plan.corridors {
                let segment = spine.path.windows(2).next().expect("spine segment");
                let sample = Position::new(
                    segment[0].x * 0.45 + segment[1].x * 0.55,
                    segment[0].z * 0.45 + segment[1].z * 0.55,
                );
                let ceiling =
                    BackroomsLevel::corridor_ceiling(spine, &noise, 42, sample.x, sample.z);
                let expected = match spine.spine_kind {
                    SpaceProgram::MainCorridor => 3.4..=4.2,
                    SpaceProgram::SecondaryHall => {
                        secondary_checked = true;
                        3.0..=3.6
                    }
                    _ => unreachable!("non-circulation spine"),
                };
                assert!(
                    expected.contains(&ceiling),
                    "{:?} ceiling {} at ({}, {})",
                    spine.spine_kind,
                    ceiling,
                    sample.x,
                    sample.z
                );
            }
        }
    }
    assert!(secondary_checked, "sample contained no secondary branch");
}

/// Assemblies voxelize as walled rooms whose planned entrance is open
/// (with a lintel when the designer's threshold language wants one).
#[test]
fn assemblies_have_walls_and_open_entrances() {
    let noise = TestNoise;
    let mut config = GeneratorConfig::low_spec();
    config.anomalies.frequency = 0.0;
    let tuning = LevelTuning::default();
    let plans =
        BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 80.0, 42, &config, &noise);
    let plan = &plans.iter().find(|(k, _)| *k == (0, 0)).unwrap().1;
    assert!(!plan.assemblies.is_empty());

    for a in &plan.assemblies {
        let e = &a.entrances[0];
        // The entrance column itself: open (possibly under a lintel).
        let door =
            BackroomsLevel::plan_column(plan, &noise, 42, &tuning, e.center.x, e.center.z);
        assert!(!door.solid, "assembly {} door is walled shut", a.id);
        // Somewhere along the same front wall, clear of the door span,
        // there must be solid wall. Probe the middle of the wall band
        // (the entrance center sits exactly on the footprint boundary,
        // where containment is ambiguous).
        let b = a.footprint.bounds();
        let band = |c: f32, lo: f32, hi: f32| {
            if (c - lo).abs() < (c - hi).abs() {
                lo - PLAN_WALL_T * 0.5
            } else {
                hi + PLAN_WALL_T * 0.5
            }
        };
        let (lo, hi, door_along) = if e.through_x_wall {
            (b.0, b.2, e.center.x)
        } else {
            (b.1, b.3, e.center.z)
        };
        let mut solid_found = false;
        let mut along = lo + 0.3;
        while along < hi - 0.2 {
            if (along - door_along).abs() > e.width * 0.5 + 0.4 {
                let (wx, wz) = if e.through_x_wall {
                    (along, band(e.center.z, b.1, b.3))
                } else {
                    (band(e.center.x, b.0, b.2), along)
                };
                if BackroomsLevel::plan_column(plan, &noise, 42, &tuning, wx, wz).solid {
                    solid_found = true;
                    break;
                }
            }
            along += 0.2;
        }
        assert!(
            solid_found,
            "assembly {} has no solid front wall anywhere",
            a.id
        );
    }
}

/// An abandoned expansion is a dark shell: it keeps its walls but none of
/// its fixtures are lit.
#[test]
fn abandoned_expansions_are_unlit() {
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let tuning = LevelTuning::default();
    let mut found = false;
    for rx in -3i64..3 {
        for rz in -3i64..3 {
            let plans = BackroomsLevel::region_plans_for(
                Position::new(rx as f32 * 80.0, rz as f32 * 80.0),
                1.0,
                42,
                &config,
                &noise,
            );
            let plan = &plans[0].1;
            for a in &plan.assemblies {
                if !a.corruption.abandoned {
                    continue;
                }
                found = true;
                let (x0, z0, x1, z1) = a.footprint.bounds();
                // No interior column may carry a lit fixture.
                let mut probe_z = z0 + 0.6;
                while probe_z < z1 - 0.4 {
                    let mut probe_x = x0 + 0.6;
                    while probe_x < x1 - 0.4 {
                        // A corridor clipping the footprint may still run
                        // its own lit strip through the shell — that is
                        // canon ("unreachable but still lit"). Only the
                        // room's fixtures must be dark.
                        let in_corridor = plan
                            .corridors
                            .iter()
                            .any(|s| s.distance(probe_x, probe_z) <= s.width * 0.5);
                        if !in_corridor {
                            let c = BackroomsLevel::plan_column(
                                plan, &noise, 42, &tuning, probe_x, probe_z,
                            );
                            assert!(!c.light, "abandoned assembly {} is lit", a.id);
                        }
                        probe_x += 0.8;
                    }
                    probe_z += 0.8;
                }
            }
        }
    }
    assert!(found, "no abandoned expansion within 36 regions");
}

/// A planned stairwell samples through the full column pipeline as a
/// walkable flight: flat at the door, rising monotonically along the
/// walk axis, reaching its landing with headroom intact.
#[test]
fn stairwells_sample_as_rising_flights() {
    use crate::use_cases::vertical_circulation::link_wants_geometry;
    use crate::use_cases::world_topology::vertical_link_for_region;
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let tuning = LevelTuning::default();

    let mut checked = 0usize;
    for rz in -8i64..=8 {
        for rx in -8i64..=8 {
            let Some(link) = vertical_link_for_region(42, &noise, rx, rz) else {
                continue;
            };
            if !link_wants_geometry(&link) {
                continue;
            }
            let plans = BackroomsLevel::region_plans_for(
                Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                1.0,
                42,
                &config,
                &noise,
            );
            let plan = &plans.iter().find(|(k, _)| *k == (rx, rz)).unwrap().1;
            let Some(stair) = plan
                .assemblies
                .iter()
                .find(|a| a.program == SpaceProgram::Stair)
            else {
                continue;
            };
            let door = stair.entrances[0].center;
            let b = stair.footprint.bounds();
            let inward = if (door.z - b.1).abs() < (door.z - b.3).abs() {
                1.0
            } else {
                -1.0
            };

            let at_door =
                BackroomsLevel::plan_column(plan, &noise, 42, &tuning, door.x, door.z);
            assert!(!at_door.solid, "stair door is walled shut");
            assert_eq!(at_door.floor_units, 0.0, "stair door is not flat");

            let mut previous = 0.0f32;
            let mut peak = 0.0f32;
            let mut depth = 0.7;
            while depth < (b.3 - b.1) - 0.6 {
                let wz = door.z + inward * depth;
                let c = BackroomsLevel::plan_column(plan, &noise, 42, &tuning, door.x, wz);
                if !c.solid {
                    assert!(
                        c.floor_units >= previous - 1e-6,
                        "flight descends inside stair at region ({rx},{rz})"
                    );
                    assert!(
                        c.ceiling_units - c.floor_units >= 2.2 - 1e-6,
                        "flight headroom pinched at region ({rx},{rz})"
                    );
                    previous = c.floor_units;
                    peak = peak.max(c.floor_units);
                }
                depth += 0.2;
            }
            assert!(
                peak >= 1.6 - 1e-6,
                "flight in ({rx},{rz}) peaked at {peak} u"
            );
            checked += 1;
        }
    }
    assert!(checked >= 3, "only {checked} stairwells sampled");
}

/// The baseline is broad regular dropped ceiling, with enough expansive
/// and vaulted territory to prevent Level 0 from reading as a low maze.
#[test]
fn ceilings_are_vast_and_varied() {
    let noise = TestNoise;
    let mut counts = [0usize; 4];
    let mut lowest = f32::MAX;
    let mut tallest = 0.0f32;
    for z in (-600..=600).step_by(8) {
        for x in (-600..=600).step_by(8) {
            let (wx, wz) = (x as f32 + 0.5, z as f32 + 0.5);
            let band = BackroomsLevel::fabric_ceiling_band(&noise, 42, wx, wz);
            let ceiling = BackroomsLevel::fabric_ceiling_height(&noise, 42, wx, wz, band);
            let index = match band {
                FabricCeilingBand::Compression => 0,
                FabricCeilingBand::Regular => 1,
                FabricCeilingBand::Expanse => 2,
                FabricCeilingBand::Vault => 3,
            };
            counts[index] += 1;
            lowest = lowest.min(ceiling);
            tallest = tallest.max(ceiling);
        }
    }
    let total = counts.iter().sum::<usize>() as f32;
    let ratio = |index| counts[index] as f32 / total;
    // The labyrinth fabric is the default; open volumes punctuate it.
    assert!(
        (0.60..=0.85).contains(&ratio(1)),
        "ceiling territories: compression {:.1}%, regular {:.1}%, expanse {:.1}%, vault {:.1}%",
        ratio(0) * 100.0,
        ratio(1) * 100.0,
        ratio(2) * 100.0,
        ratio(3) * 100.0
    );
    assert!(
        (0.10..=0.28).contains(&ratio(2)),
        "open expanse territory was {:.1}%",
        ratio(2) * 100.0
    );
    assert!(
        (0.03..=0.15).contains(&ratio(3)),
        "vault territory was {:.1}%",
        ratio(3) * 100.0
    );
    assert!(
        (0.01..=0.10).contains(&ratio(0)),
        "compression territory was {:.1}%",
        ratio(0) * 100.0
    );
    assert!(
        lowest <= 2.8 && tallest >= 4.5,
        "ceiling range was only {lowest:.1}--{tallest:.1} u"
    );
}

/// Framed doorways still exist, but only as a rare architectural anomaly.
#[test]
fn rare_doorways_still_have_lintels() {
    let noise = TestNoise;
    let mut config = GeneratorConfig::low_spec();
    config.anomalies.frequency = 0.0;
    let tuning = LevelTuning::default();
    let mut found = false;
    for rx in -6i64..=6 {
        for rz in -6i64..=6 {
            let plans = BackroomsLevel::region_plans_for(
                Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                1.0,
                42,
                &config,
                &noise,
            );
            let plan = &plans
                .iter()
                .find(|(key, _)| *key == (rx, rz))
                .expect("requested region plan")
                .1;
            for a in &plan.assemblies {
                for e in &a.entrances {
                    if e.width <= DOOR_WIDTH + 0.01 {
                        let column = BackroomsLevel::plan_column(
                            plan, &noise, 42, &tuning, e.center.x, e.center.z,
                        );
                        assert!(!column.solid, "narrow doorway is blocked");
                        assert_eq!(column.lintel_from_units, Some(DOOR_HEIGHT));
                        found = true;
                    }
                }
            }
        }
    }
    assert!(found, "no rare doorway-with-lintel found in 169 regions");
}

/// The generation knobs actually steer the output: zeroing pillars and
/// walls empties the world of solids; cranking them fills it back up.
#[test]
fn tuning_knobs_control_density() {
    let noise = TestNoise;
    // Aggregate over chunks in different fabric regimes (walled rooms
    // and open expanses) so both knobs have something to steer.
    let count_solids = |tuning: LevelTuning| -> usize {
        let mut n = 0;
        for (ox, oz) in [(10.0, 10.0), (30.0, 10.0), (50.0, 30.0), (10.0, 50.0)] {
            let grid = BackroomsLevel.generate(
                Position::new(ox, oz),
                42,
                GeneratorConfig::low_spec().with_tuning(tuning),
                &noise,
            );
            for z in 0..grid.depth() {
                for x in 0..grid.width() {
                    if grid.get(x, 1, z) != VOXEL_AIR {
                        n += 1;
                    }
                }
            }
        }
        n
    };

    // Provisions (bottles, doors) are content, not architecture: zero them
    // so the knob comparison sees only pillar/wall mass.
    let no_provisions = LevelTuning {
        almond_water: 0.0,
        rations: 0.0,
        level_doors: 0.0,
        ..Default::default()
    };
    let none = count_solids(LevelTuning {
        pillars: 0.0,
        walls: 0.0,
        ..no_provisions
    });
    let sparse = count_solids(LevelTuning {
        pillars: 0.3,
        walls: 0.3,
        ..no_provisions
    });
    let default = count_solids(no_provisions);
    let dense = count_solids(LevelTuning {
        pillars: 2.0,
        walls: 2.0,
        ..no_provisions
    });

    assert_eq!(none, 0, "pillars=0 walls=0 must produce an empty plane");
    assert!(
        sparse < default,
        "sparse ({sparse}) must be < default ({default})"
    );
    assert!(
        default < dense,
        "default ({default}) must be < dense ({dense})"
    );
}

/// The guaranteed Level 1 door exists at its authored spot, exports its
/// LevelExit from exactly one chunk, and the `level_doors` knob at zero
/// removes it entirely.
#[test]
fn spawn_door_exports_a_level_exit_and_knob_zero_removes_it() {
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    // Chunk (170, -120)..(180, -110) contains the authored door (172, -116).
    let chunk = Position::new(170.0, -120.0);
    let grid = BackroomsLevel.generate(chunk, 42, config, &noise);
    assert_eq!(grid.level_exits.len(), 1, "authored door must export");
    let exit = grid.level_exits[0];
    assert_eq!(exit.target_level, 1);
    assert!(exit.contains(172.0, -116.0));

    let mut doorless = config;
    doorless.tuning.level_doors = 0.0;
    let grid = BackroomsLevel.generate(chunk, 42, doorless, &noise);
    assert!(grid.level_exits.is_empty(), "level_doors=0 removes doors");

    let mut dry = config;
    dry.tuning.almond_water = 0.0;
    dry.tuning.rations = 0.0;
    let grid = BackroomsLevel.generate(chunk, 42, dry, &noise);
    assert!(grid.supply_items.is_empty(), "provision knobs at 0 strip supplies");
}

/// Every LOD of a chunk must voxelize the same plan: coarse walls stay
/// within one fine voxel of fine walls (the streaming engine swaps LODs
/// of a chunk in place, so they must be faithful proxies).
#[test]
fn lods_of_the_same_chunk_correspond() {
    let noise = TestNoise;
    let fine = BackroomsLevel.generate(
        Position::new(10.0, 10.0),
        42,
        GeneratorConfig::low_spec(),
        &noise,
    );
    let coarse = BackroomsLevel.generate(
        Position::new(10.0, 10.0),
        42,
        GeneratorConfig::low_spec().at_lod(1),
        &noise,
    );
    let solid = |g: &VoxelGrid, x: usize, z: usize| g.get(x, 1, z) == VOXEL_WALL;
    let (mut matches, mut total) = (0usize, 0usize);
    for z in 0..coarse.depth() {
        for x in 0..coarse.width() {
            if !solid(&coarse, x, z) {
                continue;
            }
            total += 1;
            let mut near = false;
            for dz in -1i32..=2 {
                for dx in -1i32..=2 {
                    let (fx, fz) = (x as i32 * 2 + dx, z as i32 * 2 + dz);
                    if fx >= 0
                        && fz >= 0
                        && (fx as usize) < fine.width()
                        && (fz as usize) < fine.depth()
                        && solid(&fine, fx as usize, fz as usize)
                    {
                        near = true;
                    }
                }
            }
            if near {
                matches += 1;
            }
        }
    }
    assert!(total > 0, "coarse chunk has no walls at all");
    assert!(
        matches * 10 >= total * 9,
        "coarse walls stray from fine walls: {matches}/{total}"
    );
}

/// The player spawns *on the main corridor*: open floor along the
/// centerline, a lit fixture within a light-strip period, and — for the
/// readable opening sequence — solid edge walls on both sides.
#[test]
fn spawn_is_a_readable_walled_corridor() {
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let tuning = LevelTuning::default();
    let sp = crate::use_cases::region_plan::spawn_point(42);

    let plans =
        BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 80.0, 42, &config, &noise);
    let plan = &plans.iter().find(|(k, _)| *k == (0, 0)).unwrap().1;
    let spine = &plan.corridors[0];
    assert!(
        spine.distance(sp.x, sp.z) < 0.3,
        "spawn ({}, {}) is not on the main corridor centerline",
        sp.x,
        sp.z
    );

    // The corridor around spawn is open along the centerline, lit, and
    // predominantly walled on both sides. Framed entrances and branch
    // tees may pierce the run, but the edge must *read* as a wall — no
    // dissolution into open fabric inside the readable radius.
    let half = spine.width * 0.5;
    let mut lit = false;
    let mut solid_edges = [0usize; 2];
    const STEPS: usize = 16;
    for step in 0..STEPS {
        let wx = sp.x + step as f32;
        let c = BackroomsLevel::plan_column(plan, &noise, 42, &tuning, wx, sp.z);
        assert!(!c.solid, "main corridor blocked at ({wx}, {})", sp.z);
        lit |= c.light;
        for (i, side) in [-1.0, 1.0f32].iter().enumerate() {
            let wz = sp.z + side * (half + PLAN_WALL_T * 0.5);
            if BackroomsLevel::plan_column(plan, &noise, 42, &tuning, wx, wz).solid {
                solid_edges[i] += 1;
            }
        }
    }
    assert!(lit, "no lit fixture along the first {STEPS} u of corridor");
    for (i, solid) in solid_edges.iter().enumerate() {
        assert!(
            solid * 10 >= STEPS * 6,
            "start corridor side {i} is mostly open ({solid}/{STEPS}): \
             the opening sequence must read as a walled corridor"
        );
    }
}

/// The fabric connectivity invariant: every warren cell opens through
/// its west or its north wall (doorway, dropped wall, or an override by
/// corridor/assembly/expanse), so the labyrinth is globally connected
/// by induction — no chunk ever needs to see its neighbors to prove it.
#[test]
fn every_fabric_cell_opens_west_or_north() {
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let tuning = LevelTuning::default();
    let mut cells_checked = 0usize;
    for cell_x in -40i64..40 {
        for cell_z in -40i64..40 {
            let x0 = cell_x as f32 * FABRIC_CELL;
            let z0 = cell_z as f32 * FABRIC_CELL;
            let plans = BackroomsLevel::region_plans_for(
                Position::new(x0, z0),
                FABRIC_CELL,
                42,
                &config,
                &noise,
            );
            let plan_for = |wx: f32, wz: f32| {
                let key = (region_index(wx), region_index(wz));
                &plans.iter().find(|(k, _)| *k == key).unwrap().1
            };
            // The invariant belongs to *pure* fabric. Cells clipped by a
            // corridor edge or an assembly take their connectivity from
            // those systems instead (tested separately), so skip them.
            let (cx0, cz0) = (x0 - PLAN_WALL_T, z0 - PLAN_WALL_T);
            let (cx1, cz1) = (x0 + FABRIC_CELL, z0 + FABRIC_CELL);
            let clipped = plans.iter().any(|(_, p)| {
                p.corridors.iter().any(|s| {
                    let m = s.width * 0.5 + PLAN_WALL_T + 0.1;
                    [
                        (cx0, cz0),
                        (cx1, cz0),
                        (cx0, cz1),
                        (cx1, cz1),
                        ((cx0 + cx1) * 0.5, (cz0 + cz1) * 0.5),
                    ]
                    .iter()
                    .any(|&(px, pz)| s.distance(px, pz) <= m + FABRIC_CELL)
                }) || p.assemblies.iter().any(|a| {
                    let b = a.footprint.bounds();
                    cx0 < b.2 + PLAN_WALL_T
                        && b.0 - PLAN_WALL_T < cx1
                        && cz0 < b.3 + PLAN_WALL_T
                        && b.1 - PLAN_WALL_T < cz1
                }) || p.anomalies.iter().any(|an| {
                    // Anomaly interiors own their connectivity rules
                    // (entrances, skeleton lanes, arch openings) and are
                    // tested by their own family invariants.
                    an.footprint
                        .bounds()
                        .expanded(PLAN_WALL_T + 0.1)
                        .intersects(WorldBounds::new(cx0, cz0, cx1, cz1))
                })
            });
            if clipped {
                continue;
            }
            let mut open = false;
            let mut probe = |wx: f32, wz: f32| {
                let c =
                    BackroomsLevel::plan_column(plan_for(wx, wz), &noise, 42, &tuning, wx, wz);
                if !c.solid {
                    open = true;
                }
            };
            // Sample along the west and north wall bands of the cell.
            let mut a = PLAN_WALL_T + 0.1;
            while a < FABRIC_CELL - PLAN_WALL_T {
                probe(x0 + 0.2, z0 + a);
                probe(x0 + a, z0 + 0.2);
                a += 0.2;
            }
            assert!(
                open,
                "fabric cell ({cell_x},{cell_z}) is sealed on both its \
                 west and north walls"
            );
            cells_checked += 1;
        }
    }
    assert!(cells_checked > 1000, "sample too small: {cells_checked}");
}

/// Red identity is owned by whole rooms: some region must contain a red
/// room whose fixtures voxelize as red lights, and crimson/peeled wall
/// voxels may appear only inside a red-room footprint (plus its wall
/// band) — the approach stain is telegraphy, never leakage into fabric.
#[test]
fn red_rooms_are_lit_red_but_never_built_red() {
    use crate::domain::entities::voxel_grid::{VOXEL_RED_LIGHT, VOXEL_RED_WALL};
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let tuning = LevelTuning::default();

    let mut red_room_seen = false;
    'search: for rx in -4i64..=4 {
        for rz in -4i64..=4 {
            let plans = BackroomsLevel::region_plans_for(
                Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                1.0,
                42,
                &config,
                &noise,
            );
            let plan = &plans.iter().find(|(k, _)| *k == (rx, rz)).unwrap().1;
            for a in &plan.assemblies {
                if !a.corruption.red_room {
                    continue;
                }
                assert!(!a.corruption.abandoned, "a red room must be occupied");
                // Its lit fixtures plan red lights.
                let f = a.fixtures.iter().find(|f| f.lit).expect("lit fixture");
                let c = BackroomsLevel::plan_column(plan, &noise, 42, &tuning, f.at.x, f.at.z);
                if c.light {
                    assert!(c.red_light, "red-room fixture plans a warm light");
                    red_room_seen = true;
                    break 'search;
                }
            }
        }
    }
    assert!(red_room_seen, "no red room found within 81 regions");

    // Red walls stay contained: any crimson voxel must sit inside some
    // red-room footprint (plus wall band), and red lights appear only as
    // ceiling lights.
    for (ox, oz) in [(0.0, 0.0), (30.0, 10.0), (-40.0, 70.0), (150.0, -90.0)] {
        let grid = generate(ox, oz);
        let scale = config.voxel_scale;
        for z in 0..grid.depth() {
            for x in 0..grid.width() {
                for y in 0..grid.height() {
                    if grid.get(x, y, z) == VOXEL_RED_WALL {
                        panic!(
                            "VOXEL_RED_WALL should never be voxelized at {wx}, {y}, {wz}",
                            wx = ox + (x as f32 + 0.5) * scale,
                            wz = oz + (z as f32 + 0.5) * scale
                        );
                    }
                }
                // Red lights sit at ceiling height, never at floor level.
                assert_ne!(grid.get(x, 1, z), VOXEL_RED_LIGHT);
            }
        }
    }
}

fn find_macro_anomaly(kind: AnomalyKind) -> (GeneratorConfig, AnomalyInstance) {
    let mut config = GeneratorConfig::low_spec();
    config.anomalies.frequency = 4.0;
    config.anomalies.pillar_expanses = (kind == AnomalyKind::PillarExpanse) as u8 as f32;
    config.anomalies.blackouts = (kind == AnomalyKind::BlackoutExpanse) as u8 as f32;
    config.anomalies.pit_lattices = (kind == AnomalyKind::PitLattice) as u8 as f32;
    let noise = TestNoise;
    for rz in 3i64..24 {
        for rx in 3i64..24 {
            let plans = BackroomsLevel::region_plans_for(
                Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                1.0,
                42,
                &config,
                &noise,
            );
            if let Some(instance) = plans
                .iter()
                .flat_map(|(_, p)| &p.anomalies)
                .find(|a| a.kind == kind)
            {
                return (config, instance.clone());
            }
        }
    }
    panic!("no {kind:?} fixture found");
}

#[test]
fn pillar_epoch_changes_only_committed_wake_and_preserves_bearing_lane() {
    use crate::domain::entities::anomaly::{AnomalyStateStamp, AxisDirection};
    let (mut config, instance) = find_macro_anomaly(AnomalyKind::PillarExpanse);
    config.anomalies.remap_intensity = 4.0;
    let gate = instance.gates[instance.gates.len() / 2];
    let reality = RealitySnapshot::new(vec![AnomalyStateStamp::new(
        instance.id,
        1,
        gate.plane,
        gate.axis,
        AxisDirection::Positive,
        crate::domain::entities::anomaly::RedRoomPhase::Outside,
        0,
        0,
    )]);
    let empty = RealitySnapshot::empty();
    let noise = TestNoise;
    let mut changed = 0usize;
    let mut unchanged_forward = 0usize;
    let mut lz = -instance.footprint.half_z + 1.0;
    while lz < instance.footprint.half_z - 1.0 {
        let mut lx = -instance.footprint.half_x + 1.0;
        while lx < instance.footprint.half_x - 1.0 {
            let p = instance.world_coords(lx, lz);
            let a = sample_anomaly(&instance, &noise, 42, &config, &empty, p.x, p.z);
            let b = sample_anomaly(&instance, &noise, 42, &config, &reality, p.x, p.z);
            let in_wake =
                reality.stamps()[0].point_is_in_wake(p.x, p.z, config.anomalies.remap_distance);
            if a.solid != b.solid {
                assert!(in_wake, "geometry changed ahead of the crossed gate");
                assert!(lz.abs() > instance.skeleton_half_width);
                changed += 1;
            } else if !in_wake {
                unchanged_forward += 1;
            }
            if lz.abs() <= instance.skeleton_half_width {
                assert!(!a.solid && !b.solid, "bearing lane was blocked");
            }
            lx += 0.4;
        }
        lz += 0.4;
    }
    assert!(changed > 0, "pillar epoch produced no changed wake infill");
    assert!(unchanged_forward > 100);
}

#[test]
fn blackout_has_a_recoverable_glimmer_lane_and_compressed_dark_core() {
    let (config, instance) = find_macro_anomaly(AnomalyKind::BlackoutExpanse);
    let noise = TestNoise;
    let lane = instance.world_coords(0.0, 0.0);
    let lane_plan = sample_anomaly(
        &instance,
        &noise,
        42,
        &config,
        &RealitySnapshot::empty(),
        lane.x,
        lane.z,
    );
    assert!(!lane_plan.solid, "blackout recovery skeleton is blocked");
    let core = instance.world_coords(0.0, instance.skeleton_half_width + 3.0);
    let core_plan = sample_anomaly(
        &instance,
        &noise,
        42,
        &config,
        &RealitySnapshot::empty(),
        core.x,
        core.z,
    );
    assert!(!core_plan.light, "blackout core has an ordinary fixture");
    assert_eq!(core_plan.ceiling_units, 2.6);
}

#[test]
fn pit_lattice_omits_real_floor_and_exports_relocation_hazards() {
    let (config, instance) = find_macro_anomaly(AnomalyKind::PitLattice);
    let hazards = instance.pit_hazards_for_bounds(instance.footprint.bounds());
    assert!(hazards.len() > 20, "pit lattice is not a room-scale hazard");
    let h = hazards[0];
    let plan = sample_anomaly(
        &instance,
        &TestNoise,
        42,
        &config,
        &RealitySnapshot::empty(),
        h.center.x,
        h.center.z,
    );
    assert!(!plan.floor, "pit center still voxelizes a floor slab");
    assert!(!h.contains(h.recovery.x, h.recovery.z));

    let ox = (h.center.x / config.chunk_size).floor() * config.chunk_size;
    let oz = (h.center.z / config.chunk_size).floor() * config.chunk_size;
    let grid = BackroomsLevel.generate_with_reality(
        Position::new(ox, oz),
        42,
        config,
        &TestNoise,
        &RealitySnapshot::empty(),
    );
    assert!(grid.pit_hazards.iter().any(|x| x.id == h.id));
}

#[test]
fn red_threshold_closes_the_remembered_entrance_into_a_loop() {
    use crate::domain::entities::anomaly::AxisDirection;
    let noise = TestNoise;
    let mut config = GeneratorConfig::low_spec();
    config.anomalies.pillar_expanses = 0.0;
    config.anomalies.blackouts = 0.0;
    config.anomalies.pit_lattices = 0.0;
    let mut fixture = None;
    for rz in -6i64..=6 {
        for rx in -6i64..=6 {
            let plans = BackroomsLevel::region_plans_for(
                Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                1.0,
                42,
                &config,
                &noise,
            );
            let plan = &plans.iter().find(|(k, _)| *k == (rx, rz)).unwrap().1;
            if let Some(red) = plan
                .anomalies
                .iter()
                .find(|a| a.kind == AnomalyKind::RedRoom)
            {
                fixture = Some((plan.clone(), red.clone()));
                break;
            }
        }
        if fixture.is_some() {
            break;
        }
    }
    let (plan, red) = fixture.expect("red-room fixture");
    let gate = red.gates[0];
    let reality = RealitySnapshot::new(vec![
        crate::domain::entities::anomaly::AnomalyStateStamp::new(
            red.id,
            1,
            gate.plane,
            gate.axis,
            AxisDirection::Positive,
            crate::domain::entities::anomaly::RedRoomPhase::Sealed,
            0,
            gate.id,
        ),
    ]);
    let entrance = plan
        .assemblies
        .iter()
        .find(|a| a.corruption.red_room)
        .unwrap()
        .entrances[0];
    let open = BackroomsLevel::plan_column_in_reality(
        &plan,
        &noise,
        42,
        &config,
        &RealitySnapshot::empty(),
        entrance.center.x,
        entrance.center.z,
    );
    let closed = BackroomsLevel::plan_column_in_reality(
        &plan,
        &noise,
        42,
        &config,
        &reality,
        entrance.center.x,
        entrance.center.z,
    );
    assert!(!open.solid, "red room is closed before threshold entry");
    assert!(closed.solid, "remembered red-room entrance did not close");

    let ring_point = red.world_coords(red.footprint.half_x - 1.0, 0.0);
    let ring = sample_red_room(
        &red,
        ColumnPlan::open(3.4),
        &config,
        &reality,
        ring_point.x,
        ring_point.z,
    );
    assert!(!ring.solid, "closed red room has no traversable loop");
    assert_eq!(
        ring.floor_material, VOXEL_STICKY_CARPET,
        "committed loop lost its red carpet identity"
    );

    let mut escape_config = config;
    escape_config.anomalies.red_escape_bias = 1.0;
    let escape_reality = RealitySnapshot::new(vec![
        crate::domain::entities::anomaly::AnomalyStateStamp::new(
            red.id,
            4,
            gate.plane,
            gate.axis,
            AxisDirection::Positive,
            crate::domain::entities::anomaly::RedRoomPhase::EscapeOpen,
            3,
            red.gates[1].id,
        ),
    ]);
    let side_extent = if gate.axis == crate::domain::entities::anomaly::Axis2::Z {
        red.footprint.half_x
    } else {
        red.footprint.half_z
    };
    let side_columns = [-side_extent, side_extent].map(|side| {
        let point = if gate.axis == crate::domain::entities::anomaly::Axis2::Z {
            red.world_coords(side, 0.0)
        } else {
            red.world_coords(0.0, side)
        };
        sample_red_room(
            &red,
            ColumnPlan::open(3.4),
            &escape_config,
            &escape_reality,
            point.x,
            point.z,
        )
    });
    assert_eq!(
        side_columns.iter().filter(|column| !column.solid).count(),
        1,
        "escape phase must open exactly one deterministic side wall"
    );
}

/// Arch rooms are the stable contrast: pale walls, deep wet carpet, no
/// gates, and geometry provably identical under any encounter state —
/// even a fabricated stamp for their own instance id changes nothing.
#[test]
fn archway_rooms_are_stable_pale_anchors() {
    use crate::domain::entities::anomaly::{AnomalyStateStamp, Axis2, AxisDirection};
    use crate::domain::entities::voxel_grid::{VOXEL_DEEP_CARPET, VOXEL_PALE_WALL};
    let mut config = GeneratorConfig::low_spec();
    config.anomalies.frequency = 4.0;
    config.anomalies.pillar_expanses = 0.0;
    config.anomalies.blackouts = 0.0;
    config.anomalies.pit_lattices = 0.0;
    let noise = TestNoise;
    let mut found = None;
    'search: for rz in 3i64..24 {
        for rx in 3i64..24 {
            let plans = BackroomsLevel::region_plans_for(
                Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                1.0,
                42,
                &config,
                &noise,
            );
            if let Some(instance) = plans
                .iter()
                .flat_map(|(_, p)| &p.anomalies)
                .find(|a| a.kind == AnomalyKind::ArchwayRoom)
            {
                found = Some(instance.clone());
                break 'search;
            }
        }
    }
    let instance = found.expect("no archway fixture found");
    assert!(instance.gates.is_empty(), "arch rooms must carry no gates");
    assert!(instance.arch.is_some());

    let forged = RealitySnapshot::new(vec![AnomalyStateStamp::new(
        instance.id,
        7,
        0.0,
        Axis2::X,
        AxisDirection::Positive,
        crate::domain::entities::anomaly::RedRoomPhase::Outside,
        0,
        0,
    )]);
    let empty = RealitySnapshot::empty();
    let mut pale_seen = false;
    let mut carpet_seen = false;
    let mut lz = -instance.footprint.half_z + 0.2;
    while lz < instance.footprint.half_z {
        let mut lx = -instance.footprint.half_x + 0.2;
        while lx < instance.footprint.half_x {
            let p = instance.world_coords(lx, lz);
            let a = sample_anomaly(&instance, &noise, 42, &config, &empty, p.x, p.z);
            let b = sample_anomaly(&instance, &noise, 42, &config, &forged, p.x, p.z);
            assert_eq!(a, b, "archway geometry moved under a forged epoch");
            if a.solid && a.wall_material == VOXEL_PALE_WALL {
                pale_seen = true;
            }
            if !a.solid && a.floor_material == VOXEL_DEEP_CARPET {
                carpet_seen = true;
            }
            lx += 0.4;
        }
        lz += 0.4;
    }
    assert!(pale_seen, "no pale arch wall voxelized");
    assert!(carpet_seen, "no deep wet carpet voxelized");
}

/// Arch rooms are stable — and passable and individual. Every non-blind
/// arch of a transition room is a doorway the player can walk through
/// (open at standing height on both long walls), the dead-end variant keeps
/// its one entrance, and profiles vary between instances so no two arcades
/// present the same rhythm.
#[test]
fn archways_are_walkable_and_no_two_arcades_match() {
    use crate::domain::entities::anomaly::ArchLayout;
    let mut config = GeneratorConfig::low_spec();
    config.anomalies.frequency = 4.0;
    config.anomalies.pillar_expanses = 0.0;
    config.anomalies.blackouts = 0.0;
    config.anomalies.pit_lattices = 0.0;
    let noise = TestNoise;
    let empty = RealitySnapshot::empty();

    let mut instances = Vec::new();
    'scan: for rz in 3i64..40 {
        for rx in 3i64..40 {
            let plans = BackroomsLevel::region_plans_for(
                Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                1.0,
                42,
                &config,
                &noise,
            );
            for instance in plans.iter().flat_map(|(_, p)| &p.anomalies) {
                if instance.kind == AnomalyKind::ArchwayRoom
                    && !instances.iter().any(|a: &AnomalyInstance| a.id == instance.id)
                {
                    instances.push(instance.clone());
                    if instances.len() >= 8 {
                        break 'scan;
                    }
                }
            }
        }
    }
    assert!(instances.len() >= 4, "only {} arch rooms found", instances.len());

    let mut profiles = std::collections::BTreeSet::new();
    let mut openings_walked = 0usize;
    for instance in &instances {
        let arch = instance.arch.expect("arch profile");
        profiles.insert((
            (arch.bay * 10.0) as i64,
            (arch.opening * 10.0) as i64,
            arch.blind_every,
            (arch.spring_units * 10.0) as i64,
            arch.layout == ArchLayout::Transition,
        ));
        let half_x = instance.footprint.half_x;
        let half_z = instance.footprint.half_z;
        match arch.layout {
            ArchLayout::Transition => {
                // Every non-blind bay center on both long walls is an open,
                // walkable doorway with standing headroom.
                let mut bay_index = 0i64;
                loop {
                    let center = arch.bay * (bay_index as f32 + 0.5);
                    if center + arch.opening * 0.5 >= 2.0 * half_x {
                        break;
                    }
                    let blind =
                        bay_index.rem_euclid(arch.blind_every as i64) == arch.blind_every as i64 - 1;
                    if !blind && center - arch.opening * 0.5 > 0.0 {
                        for side in [-1.0f32, 1.0] {
                            let p = instance
                                .world_coords(center - half_x, side * (half_z - 0.1));
                            let plan =
                                sample_anomaly(instance, &noise, 42, &config, &empty, p.x, p.z);
                            assert!(
                                !plan.solid,
                                "arch opening walled shut in {:x} bay {bay_index}",
                                instance.id
                            );
                            let head = plan.lintel_from_units.expect("arch head");
                            assert!(
                                head >= 2.0,
                                "arch head {head} too low to walk through in {:x}",
                                instance.id
                            );
                            openings_walked += 1;
                        }
                    }
                    bay_index += 1;
                }
            }
            ArchLayout::DeadEnd => {
                let p = instance.world_coords(-(half_x - 0.1), 0.0);
                let plan = sample_anomaly(instance, &noise, 42, &config, &empty, p.x, p.z);
                assert!(!plan.solid, "dead-end entrance sealed in {:x}", instance.id);
                assert!(plan.lintel_from_units.unwrap_or(0.0) >= 2.0);
            }
        }
    }
    assert!(openings_walked >= 8, "only {openings_walked} openings verified");
    assert!(
        profiles.len() >= instances.len() - 1,
        "arcades repeat themselves: {} profiles for {} rooms",
        profiles.len(),
        instances.len()
    );
}

/// Pillar expanses read calmer than ordinary Level 0 (dry shallow
/// carpet) and the protected bearing lane carries an unbroken light
/// rhythm — the route is architecture, not an invisible collision lane.
#[test]
fn pillar_expanse_is_dry_and_its_bearing_lane_is_lit_in_rhythm() {
    use crate::domain::entities::voxel_grid::VOXEL_DRY_CARPET;
    let (config, instance) = find_macro_anomaly(AnomalyKind::PillarExpanse);
    let noise = TestNoise;
    let empty = RealitySnapshot::empty();
    let interior = instance.world_coords(1.0, 1.0);
    let plan = sample_anomaly(
        &instance, &noise, 42, &config, &empty, interior.x, interior.z,
    );
    assert_eq!(plan.floor_material, VOXEL_DRY_CARPET);

    // Every 4.8u lane module inside the footprint carries a panel.
    let mut modules = 0usize;
    let mut lit = 0usize;
    let mut lx = (-instance.footprint.half_x / 4.8).ceil() * 4.8 + 2.4;
    while lx < instance.footprint.half_x - instance.entry_band {
        if lx.abs() < instance.footprint.half_x - instance.entry_band {
            let p = instance.world_coords(lx, 0.0);
            let c = sample_anomaly(&instance, &noise, 42, &config, &empty, p.x, p.z);
            modules += 1;
            if c.light {
                lit += 1;
            }
        }
        lx += 4.8;
    }
    assert!(modules >= 8, "sample too small: {modules}");
    assert!(
        lit * 10 >= modules * 8,
        "bearing lane rhythm is broken: {lit}/{modules} modules lit"
    );
}

/// Blackout cues are semantic: skeleton fixtures voxelize as cool
/// glimmers, the approach keeps warm office light, and committed-depth
/// floors pool recessed fluid somewhere.
#[test]
fn blackout_cues_are_glimmers_and_floors_pool_fluid() {
    use crate::domain::entities::voxel_grid::{VOXEL_FLUID, VOXEL_GLIMMER};
    let (config, instance) = find_macro_anomaly(AnomalyKind::BlackoutExpanse);
    let noise = TestNoise;
    let empty = RealitySnapshot::empty();

    let mut glimmer_seen = false;
    let mut lx = (-instance.footprint.half_x / 28.0).ceil() * 28.0 + 0.2;
    while lx < instance.footprint.half_x {
        let p = instance.world_coords(lx, 0.0);
        let c = sample_anomaly(&instance, &noise, 42, &config, &empty, p.x, p.z);
        if c.light {
            assert_eq!(
                c.light_material, VOXEL_GLIMMER,
                "skeleton cue is not a glimmer"
            );
            glimmer_seen = true;
        }
        lx += 28.0;
    }
    assert!(glimmer_seen, "no glimmer found on the recovery skeleton");

    let mut fluid_seen = false;
    let mut lz = -instance.footprint.half_z * 0.5;
    while lz < instance.footprint.half_z * 0.5 && !fluid_seen {
        let mut sx = -instance.footprint.half_x * 0.5;
        while sx < instance.footprint.half_x * 0.5 {
            let p = instance.world_coords(sx, lz);
            let c = sample_anomaly(&instance, &noise, 42, &config, &empty, p.x, p.z);
            if !c.solid && c.floor_material == VOXEL_FLUID {
                fluid_seen = true;
                break;
            }
            sx += 0.8;
        }
        lz += 0.8;
    }
    assert!(fluid_seen, "no recessed fluid basin in the blackout core");
}

/// A reality in which every fabric drift cell of a world window has
/// rearranged, with epochs deliberately mixed so cell boundaries between
/// different epochs are exercised.
fn drifted_reality(min_cell: i64, max_cell: i64) -> RealitySnapshot {
    let mut reality = RealitySnapshot::empty();
    for cz in min_cell..=max_cell {
        for cx in min_cell..=max_cell {
            for _ in 0..=((cx + cz).rem_euclid(3) as u32) {
                reality = reality.with_fabric_drift_advanced(cx, cz);
            }
        }
    }
    reality
}

/// The Peripheral Shift: after territory drifts, ordinary fabric has
/// genuinely rearranged — but every planned system (corridors, assemblies
/// and their thresholds, the spawn opening sequence) is byte-identical.
/// "Days of traveled hallways" never replay; the navigation skeleton does.
#[test]
fn peripheral_shift_rearranges_fabric_but_never_the_plan() {
    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let plans =
        BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), REGION_SIZE, 42, &config, &noise);
    let plan_for = |wx: f32, wz: f32| {
        &plans
            .iter()
            .find(|(k, _)| *k == (region_index(wx), region_index(wz)))
            .unwrap()
            .1
    };
    // Drift cells 0..=1 on each axis (world 0..80); leave the rest pristine.
    let reality = drifted_reality(0, 1);
    let sp = crate::use_cases::region_plan::spawn_point(42);

    let mut fabric_changed = 0usize;
    let mut fabric_compared = 0usize;
    for sz in 0..200 {
        for sx in 0..200 {
            let (wx, wz) = (sx as f32 * 0.6 + 0.3, sz as f32 * 0.6 + 0.3);
            let plan = plan_for(wx, wz);
            let before = BackroomsLevel::plan_column_in_reality(
                plan,
                &noise,
                42,
                &config,
                &RealitySnapshot::empty(),
                wx,
                wz,
            );
            let after =
                BackroomsLevel::plan_column_in_reality(plan, &noise, 42, &config, &reality, wx, wz);

            // Corridor interiors never drift. Their edge band is
            // indeterminate here: where the edge opens, the band is fabric
            // and may lawfully drift, so the band is asserted neither way.
            let corridor_interior = plan
                .corridors
                .iter()
                .any(|s| s.distance(wx, wz) <= s.width * 0.5);
            let corridor_band = !corridor_interior
                && plan
                    .corridors
                    .iter()
                    .any(|s| s.distance(wx, wz) <= s.width * 0.5 + PLAN_WALL_T);
            if corridor_band {
                continue;
            }
            let planned = corridor_interior
                || plan.assemblies.iter().any(|a| {
                    let b = a.footprint.bounds();
                    let m = PLAN_WALL_T + 0.05;
                    wx >= b.0 - m && wx <= b.2 + m && wz >= b.1 - m && wz <= b.3 + m
                })
                || plan
                    .anomalies
                    .iter()
                    .any(|a| a.footprint.bounds().expanded(3.2 + 0.1).contains(wx, wz));
            let near_spawn = (wx - sp.x).powi(2) + (wz - sp.z).powi(2) < 26.0 * 26.0;
            // Fabric decisions anchor at their own lattice cell's center, so
            // a column within one fabric cell of the drifted area may share
            // a decision with it. Only columns clear of that band must be
            // untouched; columns inside the band are asserted neither way.
            let drift_edge = 2.0 * 40.0;
            let outside_drift = wx > drift_edge + FABRIC_CELL || wz > drift_edge + FABRIC_CELL;
            let boundary_band = !outside_drift && (wx > drift_edge || wz > drift_edge);
            if boundary_band {
                continue;
            }

            if planned || near_spawn || outside_drift {
                assert_eq!(
                    before, after,
                    "Peripheral Shift touched protected space at ({wx}, {wz})"
                );
            } else {
                fabric_compared += 1;
                if before != after {
                    fabric_changed += 1;
                }
            }
        }
    }
    assert!(fabric_compared > 2000, "sample too small: {fabric_compared}");
    // Walls are 0.4 u bands on a 7.2 u lattice, so even a full re-deal
    // moves only a few percent of *columns* — what matters is that many
    // whole walls and doorways moved, not that the map inverted.
    assert!(
        fabric_changed * 50 >= fabric_compared,
        "drift barely rearranged the fabric: {fabric_changed}/{fabric_compared}"
    );
}

/// The binary-tree connectivity rule holds per fabric cell at *any* epoch,
/// including across boundaries between differently drifted cells: every
/// warren cell still opens through its west or its north wall.
#[test]
fn fabric_stays_connected_through_mixed_drift_epochs() {
    let noise = TestNoise;
    let tuning = LevelTuning::default();
    let reality = drifted_reality(-4, 4);
    let mut cells_checked = 0usize;
    for cell_x in -20i64..20 {
        for cell_z in -20i64..20 {
            let x0 = cell_x as f32 * FABRIC_CELL;
            let z0 = cell_z as f32 * FABRIC_CELL;
            let mut open = false;
            let mut a = PLAN_WALL_T + 0.1;
            while a < FABRIC_CELL - PLAN_WALL_T {
                for (wx, wz) in [(x0 + 0.2, z0 + a), (x0 + a, z0 + 0.2)] {
                    if !BackroomsLevel::column_plan_in_reality(
                        &noise, 42, &tuning, &reality, wx, wz,
                    )
                    .solid
                    {
                        open = true;
                    }
                }
                a += 0.2;
            }
            assert!(
                open,
                "drifted fabric cell ({cell_x},{cell_z}) sealed both its west and north walls"
            );
            cells_checked += 1;
        }
    }
    assert!(cells_checked > 1000, "sample too small: {cells_checked}");
}

/// Strain is the price of mismanaged provisions: at delirium tier 3 the
/// binary-tree doorway guarantee itself erodes, and a meaningful share of
/// warren cells seal both their west and north walls into dead-end
/// pockets. A provisioned wanderer (tier 0) keeps the full guarantee.
#[test]
fn deep_strain_seals_doorways_into_dead_end_pockets() {
    let noise = TestNoise;
    let tuning = LevelTuning::default();
    let sealed_cells = |reality: &RealitySnapshot| {
        let mut sealed = 0usize;
        for cell_x in -20i64..20 {
            for cell_z in -20i64..20 {
                let x0 = cell_x as f32 * FABRIC_CELL;
                let z0 = cell_z as f32 * FABRIC_CELL;
                let mut open = false;
                let mut a = PLAN_WALL_T + 0.1;
                while a < FABRIC_CELL - PLAN_WALL_T {
                    for (wx, wz) in [(x0 + 0.2, z0 + a), (x0 + a, z0 + 0.2)] {
                        if !BackroomsLevel::column_plan_in_reality(
                            &noise, 42, &tuning, reality, wx, wz,
                        )
                        .solid
                        {
                            open = true;
                        }
                    }
                    a += 0.2;
                }
                sealed += usize::from(!open);
            }
        }
        sealed
    };
    assert_eq!(
        sealed_cells(&RealitySnapshot::empty()),
        0,
        "an unstrained fabric must keep the binary-tree guarantee"
    );
    let strained = sealed_cells(&RealitySnapshot::empty().with_delirium(3));
    assert!(
        strained >= 40,
        "deep strain grew too few dead-end pockets: {strained}"
    );
    let mild = sealed_cells(&RealitySnapshot::empty().with_delirium(1));
    assert!(
        mild < strained,
        "strain must escalate: tier 1 sealed {mild}, tier 3 sealed {strained}"
    );
}

/// Deep strain hides slips: sub-door-width slots in walls that offer no
/// doorway at all. Doors are never narrower than 1.4 u, so any open span
/// under ~1.1 u in a surviving wall is a slip — and none may exist while
/// the wanderer is provisioned.
#[test]
fn deep_strain_hides_narrow_slips_in_solid_walls() {
    let noise = TestNoise;
    let tuning = LevelTuning::default();
    let slip_spans = |reality: &RealitySnapshot| {
        let mut slips = 0usize;
        for cell_x in -20i64..20 {
            for cell_z in -20i64..20 {
                // Walk this cell's west wall band and measure contiguous
                // open runs between the junction posts.
                let wx = cell_x as f32 * FABRIC_CELL + 0.2;
                let z0 = cell_z as f32 * FABRIC_CELL;
                let mut run = 0usize;
                let mut step = 0usize;
                while step <= (FABRIC_CELL * 10.0) as usize {
                    let wz = z0 + step as f32 * 0.1;
                    let solid = BackroomsLevel::column_plan_in_reality(
                        &noise, 42, &tuning, reality, wx, wz,
                    )
                    .solid;
                    if !solid {
                        run += 1;
                    } else {
                        if (3..=11).contains(&run) {
                            slips += 1;
                        }
                        run = 0;
                    }
                    step += 1;
                }
            }
        }
        slips
    };
    // A handful of sub-door spans occur naturally where the expanse
    // boundary clips the tail of a wall run; the strained fabric must grow
    // far more of them — the deliberate slips.
    let calm = slip_spans(&RealitySnapshot::empty());
    let strained = slip_spans(&RealitySnapshot::empty().with_delirium(3));
    assert!(
        strained >= calm + 20,
        "deep strain hid too few secret slips: {strained} vs a calm {calm}"
    );
}

/// The corridor's mouths onto the fabric are Peripheral Shift territory:
/// the drift epoch re-deals their width and phase, and deep strain narrows
/// the survivors and seals whole sections — while the spawn opening
/// sequence keeps its walls in every reality.
#[test]
fn corridor_mouths_drift_with_epochs_and_seal_under_strain() {
    use crate::domain::entities::architecture::CirculationSpine;
    use crate::use_cases::region_plan::spawn_point;

    let noise = TestNoise;
    let spine = CirculationSpine {
        id: 7,
        spine_kind: SpaceProgram::MainCorridor,
        path: vec![Position::new(-640.0, 2000.0), Position::new(640.0, 2000.0)],
        width: 5.6,
    };
    let wz = 2002.0;
    let calm = RealitySnapshot::empty();
    let parched = calm.with_delirium(3);
    let mut shifted = RealitySnapshot::empty();
    for cx in -17..=17 {
        shifted = shifted.with_fabric_drift_advanced(cx, (wz / 40.0) as i64);
    }

    let mouth_open = |reality: &RealitySnapshot, wx: f32| {
        BackroomsLevel::corridor_edge_opens(&spine, &noise, 42, reality, wx, true, wx, wz)
    };
    let (mut calm_open, mut parched_open, mut moved) = (0usize, 0usize, 0usize);
    let mut samples = 0usize;
    let mut wx = -640.0f32;
    while wx < 640.0 {
        let open_now = mouth_open(&calm, wx);
        calm_open += usize::from(open_now);
        parched_open += usize::from(mouth_open(&parched, wx));
        moved += usize::from(open_now != mouth_open(&shifted, wx));
        samples += 1;
        wx += 0.4;
    }
    assert!(samples > 3000 && calm_open > 300, "sample too small");
    assert!(
        moved > 0,
        "a drift epoch must re-deal at least one corridor mouth"
    );
    assert!(
        parched_open * 10 < calm_open * 9,
        "deep strain barely closed the mouths: {parched_open}/{calm_open}"
    );

    // The spawn opening sequence stays sacred through drift and strain.
    let sp = spawn_point(42);
    let home = CirculationSpine {
        id: 7,
        spine_kind: SpaceProgram::MainCorridor,
        path: vec![
            Position::new(sp.x - 100.0, sp.z),
            Position::new(sp.x + 100.0, sp.z),
        ],
        width: 5.6,
    };
    for reality in [&calm, &parched, &shifted] {
        assert!(
            !BackroomsLevel::corridor_edge_opens(
                &home, &noise, 42, reality, sp.x, true, sp.x, sp.z
            ),
            "spawn-readable corridor walls must hold in every reality"
        );
    }
}

/// A strained reality is stingier: each tier lowers the supply-cell
/// acceptance threshold under the *same* hash, so deep strain withholds
/// bottles the calm world offered instead of shuffling them elsewhere.
#[test]
fn deep_strain_withholds_supply_cells_the_calm_world_offered() {
    use crate::use_cases::anomalies::determinism::{hash, unit};

    let noise = TestNoise;
    let config = GeneratorConfig::low_spec();
    let calm = RealitySnapshot::empty();
    let parched = calm.with_delirium(3);
    // Default tuning: cell chance is 0.72 calm and 0.72 * 0.55 strained, so
    // a cell whose roll lands between them is offered only to the calm
    // world. Hunt one whose candidate spot actually stamps (open floor).
    let mut verified = false;
    'cells: for cz in 5i64..60 {
        for cx in 5i64..60 {
            let roll = hash(42, 0x0A1A_09D0_57A7_0000, cx, cz);
            if !(0.45..0.65).contains(&unit(roll)) {
                continue;
            }
            let probe = hash(42, 0x0A1A_09D0_0000_0000, cx, cz);
            let px = (cx as f32 + 0.06 + 0.88 * unit(probe)) * 40.0;
            let pz = (cz as f32 + 0.06 + 0.88 * unit(probe.rotate_left(23))) * 40.0;
            let chunk = Position::new(
                (px / config.chunk_size).floor() * config.chunk_size,
                (pz / config.chunk_size).floor() * config.chunk_size,
            );
            let id = roll | 1;
            let offered = BackroomsLevel
                .generate_with_reality(chunk, 42, config, &noise, &calm)
                .supply_items
                .iter()
                .any(|item| item.id == id);
            if !offered {
                // The spot rolled onto solid or anomalous ground; keep
                // hunting — the assertion needs a supply that really exists.
                continue;
            }
            let strained = BackroomsLevel
                .generate_with_reality(chunk, 42, config, &noise, &parched)
                .supply_items
                .iter()
                .any(|item| item.id == id);
            assert!(
                !strained,
                "cell ({cx},{cz}) must be withheld from a strained reality"
            );
            verified = true;
            break 'cells;
        }
    }
    assert!(verified, "no in-band supply cell stamped; widen the hunt");
}

/// Geometry priority is not presentation priority: the main corridor stays
/// carved open through a blackout (you can always walk it), but past the
/// approach band the blackout owns the dark — the corridor's light strips
/// die like every other fixture, so circulation never reads as a lit
/// tunnel through the anomaly.
#[test]
fn a_blackout_owns_the_dark_even_over_the_main_corridor() {
    use crate::domain::entities::architecture::{CirculationSpine, RegionPlan};
    let (config, instance) = find_macro_anomaly(AnomalyKind::BlackoutExpanse);
    let noise = TestNoise;
    let center = instance.footprint.center;
    let start_x = center.x - 300.0;
    let plan = RegionPlan {
        origin_world: Position::new(0.0, 0.0),
        size_world: 80.0,
        architects: Vec::new(),
        assemblies: Vec::new(),
        corridors: vec![CirculationSpine {
            id: 0,
            spine_kind: SpaceProgram::MainCorridor,
            path: vec![
                Position::new(start_x, center.z),
                Position::new(center.x + 300.0, center.z),
            ],
            width: 5.6,
        }],
        anomalies: vec![instance.clone()],
    };
    let reality = RealitySnapshot::empty();

    let mut dark_deep_modules = 0usize;
    let mut lit_outside_modules = 0usize;
    let first_module = (start_x / 4.0).ceil() * 4.0;
    for k in 0..150i64 {
        // Strip modules follow the corridor in *world* space: light where
        // the absolute coordinate along the run satisfies along % 4 < 1.
        let wx = first_module + 4.0 * k as f32 + 0.5;
        let wz = center.z;
        let column =
            BackroomsLevel::plan_column_in_reality(&plan, &noise, 42, &config, &reality, wx, wz);
        assert!(!column.solid, "main corridor blocked at ({wx}, {wz})");
        if instance.contains(wx, wz) && instance.normalized_depth(wx, wz) > 0.25 {
            assert!(
                !column.light,
                "corridor light strip survives deep blackout at ({wx}, {wz})"
            );
            dark_deep_modules += 1;
        } else if !instance.contains(wx, wz) {
            lit_outside_modules += usize::from(column.light);
        }
    }
    assert!(
        dark_deep_modules >= 5,
        "corridor barely crossed the dark core ({dark_deep_modules} modules)"
    );
    assert!(
        lit_outside_modules >= 10,
        "corridor strips outside the blackout must stay lit ({lit_outside_modules})"
    );
}

/// Inside a blackout the same drift stamps rearrange the substrate — the
/// space changes behind the player in real time — while the recovery
/// skeleton stays open in every epoch, so the way out is architecture,
/// never luck.
#[test]
fn blackout_substrate_drifts_but_its_recovery_skeleton_never_does() {
    let (config, instance) = find_macro_anomaly(AnomalyKind::BlackoutExpanse);
    let noise = TestNoise;
    let empty = RealitySnapshot::empty();
    let bounds = instance.footprint.bounds();
    let min_cx = (bounds.min_x / 40.0).floor() as i64 - 1;
    let max_cx = (bounds.max_x / 40.0).floor() as i64 + 1;
    let min_cz = (bounds.min_z / 40.0).floor() as i64 - 1;
    let max_cz = (bounds.max_z / 40.0).floor() as i64 + 1;
    let mut reality = RealitySnapshot::empty();
    for cz in min_cz..=max_cz {
        for cx in min_cx..=max_cx {
            reality = reality.with_fabric_drift_advanced(cx, cz);
        }
    }

    // Step 0.4 u: the fabric's wall bands are 0.4 u wide on a 7.2 u
    // lattice, and a coarser stride can cycle past every band forever.
    let mut changed = 0usize;
    let half_span = instance.footprint.half_x.min(40.0);
    let mut lz = -instance.footprint.half_z + 1.0;
    while lz < instance.footprint.half_z - 1.0 {
        let mut lx = -half_span;
        while lx < half_span {
            let p = instance.world_coords(lx, lz);
            let before = sample_anomaly(&instance, &noise, 42, &config, &empty, p.x, p.z);
            let after = sample_anomaly(&instance, &noise, 42, &config, &reality, p.x, p.z);
            if lz.abs() <= instance.skeleton_half_width {
                assert!(!before.solid && !after.solid, "recovery skeleton blocked");
            } else if before.solid != after.solid {
                changed += 1;
            }
            lx += 0.4;
        }
        lz += 0.4;
    }
    assert!(
        changed > 40,
        "blackout interior barely drifted ({changed} columns changed)"
    );
}

/// Institution age steers decay: old territory keeps meaningfully fewer of
/// its fluorescents alive than young territory. The wiki's endless hum is
/// not uniform — some wings are almost maintained, some run on remnants.
#[test]
fn old_territory_has_more_dead_lights_than_young_territory() {
    use crate::use_cases::world_topology::institution_age_at;
    let noise = TestNoise;
    let tuning = LevelTuning::default();
    let mut young = (0usize, 0usize); // (alive, sites)
    let mut old = (0usize, 0usize);
    for kz in -220i64..220 {
        for kx in -220i64..220 {
            // Dense-grid light sites sit at period/2 + k*period.
            let wx = 1.4 + kx as f32 * 2.8 + 0.2;
            let wz = 1.4 + kz as f32 * 2.8 + 0.2;
            if BackroomsLevel::in_expanse(&noise, 42, wx, wz) {
                continue;
            }
            let column = BackroomsLevel::column_plan(&noise, 42, &tuning, wx, wz);
            if column.solid {
                continue;
            }
            let age = institution_age_at(&noise, 42, wx, wz);
            let bucket = if age < 0.35 {
                &mut young
            } else if age > 0.65 {
                &mut old
            } else {
                continue;
            };
            bucket.0 += usize::from(column.light);
            bucket.1 += 1;
        }
    }
    assert!(young.1 > 400 && old.1 > 400, "sample too small: {young:?} {old:?}");
    let young_rate = young.0 as f32 / young.1 as f32;
    let old_rate = old.0 as f32 / old.1 as f32;
    assert!(
        old_rate < young_rate * 0.82,
        "age barely steers decay: young {young_rate:.3} vs old {old_rate:.3}"
    );
}

#[test]
fn test_print_ascii_map() {
    let noise = TestNoise;
    let tuning = LevelTuning::default();
    let config = GeneratorConfig::low_spec();
    // The whole of region (0,0) at 0.5 u per character.
    let plans =
        BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 80.0, 42, &config, &noise);
    let plan = &plans.iter().find(|(k, _)| *k == (0, 0)).unwrap().1;
    let mut map = String::new();
    for sz in 0..160 {
        for sx in 0..160 {
            let wx = sx as f32 * 0.5 + 0.25;
            let wz = sz as f32 * 0.5 + 0.25;
            let col = BackroomsLevel::plan_column(plan, &noise, 42, &tuning, wx, wz);
            if col.solid {
                map.push('#');
            } else if col.light {
                map.push('*');
            } else if col.lintel_from_units.is_some() {
                map.push('d');
            } else {
                map.push(' ');
            }
        }
        map.push('\n');
    }
    std::fs::write("./ascii_map.txt", map).unwrap();
}

/// The four structural systems must produce genuinely different column
/// layouts, not the same `(bay, bay)` grid relabeled. `CoreAndShell` exists
/// to give a big clear span (`BackroomsLevel::on_column` refuses every
/// column for it — see below); `OffsetGrid` staggers alternate rows by half
/// a bay; both must differ from `RegularGrid`'s plain grid even when every
/// other input (bay size, phase, column side) is identical.
#[test]
fn structural_systems_place_columns_differently() {
    use crate::domain::entities::architecture::{StructuralSystem, StructuralSystemInstance};

    let base = |system: StructuralSystem| StructuralSystemInstance {
        system,
        bay_x: 5.0,
        bay_z: 5.0,
        phase: (0.0, 0.0),
        column_side: 0.6,
    };
    let sample = |system: StructuralSystem| -> Vec<(i32, i32)> {
        let instance = base(system);
        let mut hits = Vec::new();
        for row in 0..6 {
            for col in 0..6 {
                let (wx, wz) = (col as f32 * 1.7, row as f32 * 1.7);
                if BackroomsLevel::on_column(&instance, wx, wz) {
                    hits.push((col, row));
                }
            }
        }
        hits
    };

    let regular = sample(StructuralSystem::RegularGrid);
    let offset = sample(StructuralSystem::OffsetGrid);
    let core_and_shell = sample(StructuralSystem::CoreAndShell);
    let deep_spans = sample(StructuralSystem::DeepSpansWithBeams);

    assert!(
        !regular.is_empty(),
        "a regular grid over this sample area must place at least one column"
    );
    assert!(
        core_and_shell.is_empty(),
        "core-and-shell must have zero interior columns, found {core_and_shell:?}"
    );
    assert_ne!(
        regular, offset,
        "offset grid produced the same column positions as a regular grid"
    );
    assert_eq!(
        regular, deep_spans,
        "deep-spans-with-beams should place columns identically to a regular \
         grid at equal bay_x/bay_z (its bay elongation is decided upstream in \
         structure_for, not by on_column)"
    );
}
