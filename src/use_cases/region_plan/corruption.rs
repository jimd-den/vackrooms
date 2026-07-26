//! The Backrooms corruption pass: the plan was sane; the building is not.
//!
//! Duplicated suites, abandoned expansions, renovation overlays, and the
//! realization of macro-graph red-room events on eligible assemblies.

use crate::domain::entities::anomaly::AnomalyKind;
use crate::domain::entities::architecture::*;
use crate::domain::entities::position::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::world_topology::hash01;

use super::suites::{aabb_overlap, place_suite};
use super::{EDGE_MARGIN, PLAN_WALL_T, REGION_SIZE, pick_index, region_index, snap, spawn_point};

/// Backrooms corruption: the plan was sane; the building is not. Pure
/// sequencing over four independent rules, each its own named function so
/// any one of them is testable without the other three — previously all
/// four were inlined together sharing one hash-key closure, which made it
/// impossible to exercise, say, just the abandonment rule in isolation.
#[allow(clippy::too_many_arguments)]
pub(super) fn corrupt(
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

    if h(1) < 0.6 {
        let src = pick_index(h(2), assemblies.len());
        let shift = snap(12.0 + 12.0 * h(3));
        if let Some(dup) = duplicate_suite_candidate(assemblies, taken, spines, rx, rz, src, shift)
        {
            taken.push(dup.footprint.bounds());
            assemblies.push(dup);
        }
    }

    if h(6) < 0.6 {
        let k = pick_index(h(7), assemblies.len());
        abandon_assembly(&mut assemblies[k]);
    }

    apply_renovation_history(dominant.renovation_history, assemblies, h(8), h(9));

    // Applied last so it respects whatever the earlier passes decided (an
    // abandoned shell stays dark).
    let sp = spawn_point(seed);
    let forced_in_spawn_region = config.anomalies.forced_kind == Some(AnomalyKind::RedRoom)
        && rx == region_index(sp.x)
        && rz == region_index(sp.z);
    let red_room_realized = forced_in_spawn_region
        && force_red_room_in_spawn_region(dominant, rx, rz, assemblies, taken, spines);
    if !red_room_realized {
        let red_room_scale = config.anomalies.frequency * config.anomalies.red_rooms;
        realize_red_room_event(seed, noise, rx, rz, red_room_scale, assemblies);
    }
}

/// Rule 1: a suite repeats itself further down the corridor, slightly wrong.
/// Pure — reads the existing plan, returns the candidate duplicate (already
/// translated, with its misalignment recorded) or `None` if it would fall
/// outside the region, overlap something already placed, or seal itself
/// away from every corridor. The caller decides whether to keep it.
fn duplicate_suite_candidate(
    assemblies: &[AssemblyInstance],
    taken: &[(f32, f32, f32, f32)],
    spines: &[CirculationSpine],
    rx: i64,
    rz: i64,
    src: usize,
    shift: f32,
) -> Option<AssemblyInstance> {
    // Keep the copied threshold flush with its source corridor. The
    // repetition is wrong in its longitudinal position, not sealed away
    // behind an accidental strip of wall.
    let skew = 0.0;
    let mut dup = assemblies[src].clone();
    dup.translate(shift, skew);
    dup.id = assemblies.len() as u32 + 1000;
    dup.corruption.misalignment = (shift, skew);

    let b = dup.footprint.bounds();
    let inside = b.0 > rx as f32 * REGION_SIZE + EDGE_MARGIN
        && b.2 < (rx + 1) as f32 * REGION_SIZE - EDGE_MARGIN
        && b.1 > rz as f32 * REGION_SIZE + EDGE_MARGIN
        && b.3 < (rz + 1) as f32 * REGION_SIZE - EDGE_MARGIN;
    // The skewed entrance must still reach a corridor, or the copy would be
    // a sealed pocket.
    let reachable = dup.entrances().any(|e| {
        spines
            .iter()
            .any(|s| s.distance(e.center.x, e.center.z) <= s.width * 0.5 + PLAN_WALL_T + 0.05)
    });
    (inside && reachable && !taken.iter().any(|t| aabb_overlap(*t, b, 0.4))).then_some(dup)
}

/// Rule 2: one assembly was built and then never occupied — an empty shell,
/// nothing lit.
fn abandon_assembly(assembly: &mut AssemblyInstance) {
    assembly.program = SpaceProgram::AbandonedExpansion;
    assembly.corruption.abandoned = true;
    assembly.spaces.clear();
    assembly
        .hosts
        .retain(|host| host.role != HostRole::Partition);
    assembly.openings.retain(|opening| {
        opening.role != OpeningRole::Interior
            && assembly.hosts.iter().any(|host| host.id == opening.host)
    });
    for f in &mut assembly.fixtures {
        f.lit = false;
    }
}

/// Rule 3: renovation history overlays the original structural grid. A
/// partial refit contradicts one assembly; a fuller history of layered
/// refits leaves a second, distinct assembly contradicted too — otherwise
/// the two `RenovationStyle` variants would be indistinguishable once
/// generated, despite the type's own doc comment promising more.
fn apply_renovation_history(
    history: RenovationStyle,
    assemblies: &mut [AssemblyInstance],
    pick1: f32,
    pick2: f32,
) {
    match history {
        RenovationStyle::Untouched => {}
        RenovationStyle::PartialRefit => {
            let k = pick_index(pick1, assemblies.len());
            overlay_renovation(&mut assemblies[k]);
        }
        RenovationStyle::LayeredRefits => {
            let k1 = pick_index(pick1, assemblies.len());
            overlay_renovation(&mut assemblies[k1]);
            if assemblies.len() > 1 {
                // Offset from k1 rather than re-rolling, so the second pick
                // is guaranteed distinct instead of merely likely to be.
                let k2 = (k1 + 1 + pick_index(pick2, assemblies.len() - 1)) % assemblies.len();
                overlay_renovation(&mut assemblies[k2]);
            }
        }
    }
}

/// A renovation overlays the original structural grid on one assembly.
fn overlay_renovation(assembly: &mut AssemblyInstance) {
    assembly.corruption.renovation_overlay = true;
}

/// Rule 4a: the debug/authored path. `?force_anomaly=redroom` targets the
/// region the player actually spawns in, so this must succeed there even
/// when nothing qualifies yet — mark an existing occupied assembly if one
/// exists, otherwise place a brand new one against the main corridor.
/// Returns whether a red room was realized.
#[allow(clippy::too_many_arguments)]
fn force_red_room_in_spawn_region(
    dominant: &ArchitectGenome,
    rx: i64,
    rz: i64,
    assemblies: &mut Vec<AssemblyInstance>,
    taken: &mut Vec<(f32, f32, f32, f32)>,
    spines: &[CirculationSpine],
) -> bool {
    if let Some(a) = assemblies
        .iter_mut()
        .find(|a| a.primary_entrance().is_some() && !a.corruption.abandoned)
    {
        a.corruption.red_room = true;
        return true;
    }
    let Some(spine) = spines
        .iter()
        .find(|s| s.spine_kind == SpaceProgram::MainCorridor)
    else {
        return false;
    };
    let Some(seg) = spine.path.windows(2).find(|seg| seg[0].z == seg[1].z) else {
        return false;
    };
    let lx0 = seg[0].x.min(seg[1].x);
    let lz = seg[0].z;
    let origin = Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE);
    let Some(mut a) = place_suite(
        9999,
        SpaceProgram::PrivateOffice,
        dominant,
        0.5,
        0.5,
        lx0 + 12.0,
        lz,
        spine.width * 0.5 + PLAN_WALL_T * 0.5,
        1.0,
        origin,
        REGION_SIZE,
        taken,
        spines,
    ) else {
        return false;
    };
    a.corruption.red_room = true;
    taken.push(a.footprint.bounds());
    assemblies.push(a);
    true
}

/// Rule 4b: the ordinary path. Where a red room happens is a *graph event*,
/// not a per-region coin flip — the macro topology plans separated events on
/// its lattice (`world_topology::red_room_event_for_region`), and this rule
/// merely realizes an event that targets this region on one eligible
/// assembly, rotating from a stable start so the choice replays identically
/// for every query of this region. If no assembly qualifies the event stays
/// unrealized — a planned encounter never overwrites navigable circulation.
fn realize_red_room_event(
    seed: u32,
    noise: &dyn NoiseProvider,
    rx: i64,
    rz: i64,
    red_room_scale: f32,
    assemblies: &mut [AssemblyInstance],
) {
    let Some(event) = crate::use_cases::world_topology::red_room_event_for_region(
        seed,
        noise,
        rx,
        rz,
        red_room_scale.clamp(0.0, 4.0),
    ) else {
        return;
    };
    let count = assemblies.len();
    let start = (event.id % count as u64) as usize;
    for offset in 0..count {
        let a = &mut assemblies[(start + offset) % count];
        if !a.corruption.abandoned && a.primary_entrance().is_some() {
            a.corruption.red_room = true;
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::use_cases::ports::NoiseProvider;
    use crate::use_cases::region_plan::derive_genome;

    struct ZeroNoise;
    impl NoiseProvider for ZeroNoise {
        fn evaluate_2d(&self, _seed: u32, _position: Position) -> f32 {
            0.0
        }
    }

    /// A minimal, valid assembly: one rectangular room with one entrance,
    /// one space, one lit fixture, and one ceiling zone — enough for every
    /// corruption rule to have something real to act on, hand-built rather
    /// than routed through `place_suite` so each rule's test needs no
    /// region/corridor context it doesn't actually use.
    fn minimal_assembly(id: u32, x0: f32, z0: f32) -> AssemblyInstance {
        let footprint = Polygon2::rect(x0, z0, 14.0, 10.0);
        AssemblyInstance {
            id,
            program: SpaceProgram::OpenOffice,
            hosts: HostSegment::rectangular_shell(&footprint, PLAN_WALL_T, 3.4),
            openings: vec![Opening {
                id: OpeningId(0),
                host: HostId(0),
                role: OpeningRole::Entrance,
                center: Position::new(x0 + 7.0, z0),
                width: 3.6,
                through_x_wall: true,
                lintel_units: None,
            }],
            door_leaves: Vec::new(),
            spaces: vec![Space {
                program: SpaceProgram::OpenOffice,
                footprint: footprint.clone(),
            }],
            structure: StructuralSystemInstance {
                system: StructuralSystem::RegularGrid,
                bay_x: 5.0,
                bay_z: 5.0,
                phase: (0.0, 0.0),
                column_side: 0.4,
            },
            ceiling_zones: vec![CeilingZone {
                area: footprint.clone(),
                language: CeilingLanguage::FlatTiles,
                height_units: 3.4,
            }],
            fixtures: vec![Fixture {
                at: Position::new(x0 + 7.0, z0 + 5.0),
                half_x: 0.4,
                half_z: 0.4,
                lit: true,
            }],
            service_voids: Vec::new(),
            corruption: CorruptionProfile::default(),
            footprint,
        }
    }

    fn test_genome() -> ArchitectGenome {
        derive_genome(42, 0, 0, 0, &ZeroNoise)
    }

    /// Runs through the region's own center so a suite fronting onto it
    /// (a fixed 2.4u offset either side) never brushes `EDGE_MARGIN` at the
    /// region boundary — a corridor at the very edge is not a shape any
    /// real region plan produces, and would fail `place_suite`'s margin
    /// check for reasons unrelated to whatever a test is actually checking.
    fn corridor_spine() -> CirculationSpine {
        CirculationSpine {
            id: 0,
            spine_kind: SpaceProgram::MainCorridor,
            path: vec![
                Position::new(0.0, REGION_SIZE * 0.5),
                Position::new(REGION_SIZE, REGION_SIZE * 0.5),
            ],
            width: 4.0,
        }
    }

    #[test]
    fn abandonment_clears_spaces_and_darkens_every_fixture() {
        let mut a = minimal_assembly(0, 0.0, 20.0);
        a.hosts.push(HostSegment {
            id: HostId(4),
            start: Position::new(7.0, 20.0),
            end: Position::new(7.0, 30.0),
            thickness: PLAN_WALL_T,
            base_units: 0.0,
            top_units: 3.4,
            role: HostRole::Partition,
        });
        a.openings.push(Opening {
            id: OpeningId(1),
            host: HostId(4),
            role: OpeningRole::Interior,
            center: Position::new(7.0, 25.0),
            width: 1.2,
            through_x_wall: false,
            lintel_units: Some(2.2),
        });
        assert!(!a.spaces.is_empty());
        assert!(a.fixtures.iter().all(|f| f.lit));

        abandon_assembly(&mut a);

        assert_eq!(a.program, SpaceProgram::AbandonedExpansion);
        assert!(a.corruption.abandoned);
        assert!(
            a.spaces.is_empty(),
            "an abandoned shell keeps no program spaces"
        );
        assert!(a.hosts.iter().all(|host| host.role != HostRole::Partition));
        assert!(
            a.openings
                .iter()
                .all(|opening| opening.role != OpeningRole::Interior)
        );
        assert!(
            a.fixtures.iter().all(|f| !f.lit),
            "an abandoned shell has nothing lit"
        );
        // Abandonment doesn't touch the other corruption flags.
        assert!(!a.corruption.renovation_overlay);
        assert!(!a.corruption.red_room);
    }

    #[test]
    fn renovation_overlay_sets_only_its_own_flag() {
        let mut a = minimal_assembly(0, 0.0, 20.0);
        overlay_renovation(&mut a);
        assert!(a.corruption.renovation_overlay);
        assert!(!a.corruption.abandoned);
        assert!(!a.corruption.red_room);
    }

    fn three_assemblies() -> Vec<AssemblyInstance> {
        vec![
            minimal_assembly(0, 0.0, 20.0),
            minimal_assembly(1, 20.0, 20.0),
            minimal_assembly(2, 40.0, 20.0),
        ]
    }

    #[test]
    fn untouched_history_overlays_nothing() {
        let mut assemblies = three_assemblies();
        apply_renovation_history(RenovationStyle::Untouched, &mut assemblies, 0.5, 0.5);
        assert!(assemblies.iter().all(|a| !a.corruption.renovation_overlay));
    }

    #[test]
    fn partial_refit_overlays_exactly_one_assembly() {
        let mut assemblies = three_assemblies();
        apply_renovation_history(RenovationStyle::PartialRefit, &mut assemblies, 0.5, 0.5);
        assert_eq!(
            assemblies
                .iter()
                .filter(|a| a.corruption.renovation_overlay)
                .count(),
            1
        );
    }

    #[test]
    fn layered_refits_overlay_two_distinct_assemblies() {
        // Scan a spread of (pick1, pick2) pairs rather than one fixed pair,
        // so the "distinct" guarantee is checked for more than one lucky
        // roll of pick_index.
        for i in 0..7 {
            for j in 0..7 {
                let mut assemblies = three_assemblies();
                let pick1 = i as f32 / 7.0;
                let pick2 = j as f32 / 7.0;
                apply_renovation_history(
                    RenovationStyle::LayeredRefits,
                    &mut assemblies,
                    pick1,
                    pick2,
                );
                let overlaid: Vec<_> = assemblies
                    .iter()
                    .filter(|a| a.corruption.renovation_overlay)
                    .collect();
                assert_eq!(
                    overlaid.len(),
                    2,
                    "layered refits must contradict two assemblies (pick1={pick1}, pick2={pick2})"
                );
            }
        }
    }

    #[test]
    fn layered_refits_overlay_falls_back_to_one_with_a_single_assembly() {
        let mut assemblies = vec![minimal_assembly(0, 0.0, 20.0)];
        apply_renovation_history(RenovationStyle::LayeredRefits, &mut assemblies, 0.5, 0.5);
        assert!(assemblies[0].corruption.renovation_overlay);
    }

    #[test]
    fn duplicate_candidate_is_translated_and_flagged() {
        let assemblies = vec![minimal_assembly(0, 20.0, REGION_SIZE * 0.5)];
        let taken = [];
        let spines = [corridor_spine()];
        let dup = duplicate_suite_candidate(&assemblies, &taken, &spines, 0, 0, 0, 24.0)
            .expect("an unobstructed duplicate inside the region must be produced");
        assert_eq!(dup.footprint.bounds().0, 44.0, "duplicate was not shifted");
        assert_eq!(dup.corruption.misalignment, (24.0, 0.0));
        assert_ne!(
            dup.id, assemblies[0].id,
            "duplicate must not reuse the source id"
        );
    }

    #[test]
    fn duplicate_candidate_rejects_overlap_with_taken_space() {
        let assemblies = vec![minimal_assembly(0, 20.0, REGION_SIZE * 0.5)];
        // Something already occupies exactly where the duplicate would land.
        let taken = [(44.0, REGION_SIZE * 0.5, 58.0, REGION_SIZE * 0.5 + 10.0)];
        let spines = [corridor_spine()];
        assert!(
            duplicate_suite_candidate(&assemblies, &taken, &spines, 0, 0, 0, 24.0).is_none(),
            "a duplicate overlapping taken space must be rejected"
        );
    }

    #[test]
    fn duplicate_candidate_rejects_leaving_the_region() {
        let assemblies = vec![minimal_assembly(0, REGION_SIZE - 30.0, REGION_SIZE * 0.5)];
        let taken = [];
        let spines = [corridor_spine()];
        // A large shift pushes the copy past the region's far edge.
        assert!(
            duplicate_suite_candidate(&assemblies, &taken, &spines, 0, 0, 0, 40.0).is_none(),
            "a duplicate crossing the region boundary must be rejected"
        );
    }

    #[test]
    fn forced_red_room_marks_an_existing_occupied_assembly_first() {
        let genome = test_genome();
        let mut assemblies = vec![minimal_assembly(0, 20.0, 20.0)];
        let mut taken = vec![];
        let spines = [corridor_spine()];
        let realized =
            force_red_room_in_spawn_region(&genome, 0, 0, &mut assemblies, &mut taken, &spines);
        assert!(realized);
        assert!(assemblies[0].corruption.red_room);
        assert_eq!(
            assemblies.len(),
            1,
            "an existing assembly was reused, not duplicated"
        );
    }

    #[test]
    fn forced_red_room_places_a_new_room_when_nothing_qualifies() {
        let genome = test_genome();
        let mut assemblies: Vec<AssemblyInstance> = vec![];
        let mut taken = vec![];
        let spines = [corridor_spine()];
        let realized =
            force_red_room_in_spawn_region(&genome, 0, 0, &mut assemblies, &mut taken, &spines);
        assert!(
            realized,
            "with a main corridor present, a room must be placeable"
        );
        assert_eq!(assemblies.len(), 1);
        assert!(assemblies[0].corruption.red_room);
    }

    #[test]
    fn red_room_event_marks_one_eligible_assembly_deterministically() {
        let assemblies = vec![
            minimal_assembly(0, 0.0, 20.0),
            minimal_assembly(1, 20.0, 20.0),
            minimal_assembly(2, 40.0, 20.0),
        ];
        // A generous scale over many region coordinates guarantees at least
        // one event fires without depending on a specific lucky (rx, rz).
        let mut fired = false;
        for rx in 0..64 {
            let mut trial = assemblies.clone();
            realize_red_room_event(42, &ZeroNoise, rx, 0, 4.0, &mut trial);
            let marked = trial.iter().filter(|a| a.corruption.red_room).count();
            if marked > 0 {
                assert_eq!(
                    marked, 1,
                    "exactly one assembly should be realized per event"
                );
                fired = true;
            }
        }
        assert!(
            fired,
            "no red room event fired across 64 region samples at scale 4.0"
        );
    }
}
