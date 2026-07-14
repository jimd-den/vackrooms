use super::*;
use super::suites::aabb_overlap;
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
    assert!(
        !realized.is_empty(),
        "no red-room event realized in 289 regions"
    );
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
                        let d = ((cx - link.anchor.x).powi(2) + (cz - link.anchor.z).powi(2))
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
