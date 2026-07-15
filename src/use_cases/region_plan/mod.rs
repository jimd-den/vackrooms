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
// The planning stages, one intent per module.
// ---------------------------------------------------------------------------

mod circulation;
mod corruption;
mod debug;
mod genome;
mod suites;
#[cfg(test)]
mod tests;

pub use debug::debug_region_ascii;
pub use genome::derive_genome;

use circulation::build_corridors;
use corruption::corrupt;
use suites::{SUITE_PROGRAMS, place_suite};

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
    if let Some(link) =
        crate::use_cases::world_topology::vertical_link_for_region(seed, noise, rx, rz)
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
