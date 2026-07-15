//! The Backrooms corruption pass: the plan was sane; the building is not.
//!
//! Duplicated suites, abandoned expansions, renovation overlays, and the
//! realization of macro-graph red-room events on eligible assemblies.

use crate::domain::entities::anomaly::AnomalyKind;
use crate::domain::entities::architecture::*;
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::world_topology::hash01;

use super::suites::{aabb_overlap, place_suite};
use super::{EDGE_MARGIN, PLAN_WALL_T, REGION_SIZE, pick_index, region_index, snap, spawn_point};

/// Backrooms corruption: the plan was sane; the building is not.
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
