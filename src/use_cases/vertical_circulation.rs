//! Stairwells: the first physical layer of the vertical world graph.
//!
//! The macro topology ([`crate::use_cases::world_topology`]) reserves
//! [`VerticalLink`]s — stair and shaft obligations anchored inside regions.
//! This module realizes the *upward* links as concrete architecture:
//!
//! * [`place_stairwell`] gives the region planner a `SpaceProgram::Stair`
//!   assembly beside its primary corridor, near the link's anchor.
//! * [`apply_stair_profile`] shapes the interior per voxel column: a flat
//!   vestibule, a flight of 0.2 u treads, and a landing. An
//!   [`VerticalLinkKind::EndlessAscent`] flight never lands — it keeps
//!   rising under a raised, unlit shaft ceiling until the far wall
//!   swallows it in the dark.
//!
//! What this deliberately does **not** do yet: move the player between
//! elevations. The engine's collision world is a fixed-eye-height slider,
//! so a stair is currently a readable, colliding landmark and a topology
//! contract. Downward links (`EndlessDescent`, `ServiceShaft`) stay
//! reservation-only until chunks can stream below elevation 0; realizing
//! them as fake geometry would spend the mystery without paying it off.
//!
//! Geometry invariants (tested in `region_plan` / here):
//! * treads rise monotonically from the entrance, 0.2 u per 0.4 u of run;
//! * headroom above every tread stays >= 2.2 u;
//! * the entrance column itself is always flat and walkable;
//! * the profile is a pure function of world position and the assembly.

use crate::domain::entities::architecture::{
    AssemblyInstance, CeilingZone, CorruptionProfile, Fixture, Opening, Polygon2, SpaceProgram,
    StructuralSystem, StructuralSystemInstance,
};
use crate::domain::entities::world_topology::{VerticalLink, VerticalLinkKind};
use crate::entities::models::Position;
use crate::use_cases::backrooms_level::ColumnPlan;

/// Stairwell frontage along its corridor, world units.
const STAIR_WIDTH: f32 = 4.8;
/// Stairwell depth away from the corridor.
const STAIR_DEPTH: f32 = 8.8;
/// Flat entry vestibule before the first riser.
const VESTIBULE_DEPTH: f32 = 2.0;
/// One tread: rise per run, both on the 0.4/0.2 plan lattices.
const TREAD_RUN: f32 = 0.4;
const TREAD_RISE: f32 = 0.2;
/// An ordinary flight lands here (half a story on the macro lattice).
const LANDING_UNITS: f32 = 1.6;
/// An endless ascent keeps climbing to this height before the dark takes it.
const ASCENT_MAX_UNITS: f32 = 3.2;
/// Clear space kept above every tread.
const MIN_HEADROOM_UNITS: f32 = 2.2;
/// Stairwell ceiling: high enough that the top of an endless flight still
/// has headroom, low enough to fit the level's 5.8 u grid.
const STAIR_CEILING_UNITS: f32 = 5.4;

fn snap(v: f32) -> f32 {
    (v / 0.4).round() * 0.4
}

/// Should this link become a stair assembly this iteration? Upward links
/// voxelize; downward links remain graph reservations (see module docs).
pub(crate) fn link_wants_geometry(link: &VerticalLink) -> bool {
    matches!(
        link.kind,
        VerticalLinkKind::OrdinaryStair | VerticalLinkKind::EndlessAscent
    )
}

/// Builds the stairwell assembly for one vertical link, beside a horizontal
/// main-corridor leg. Mirrors the suite-placement contract: returns `None`
/// if the footprint would escape the region, collide with a prior assembly,
/// or swallow another corridor — the reservation then stays unrealized
/// rather than corrupting the plan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn place_stairwell(
    id: u32,
    link: &VerticalLink,
    legs: &[(f32, f32, f32, f32)],
    origin: Position,
    size: f32,
    edge_margin: f32,
    wall_t: f32,
    taken: &[(f32, f32, f32, f32)],
    corridor_clearance: impl Fn(f32, f32, f32) -> bool,
) -> Option<AssemblyInstance> {
    // The leg whose axis passes closest to the anchor hosts the stair.
    let (lx0, lx1, lz, lw) = legs.iter().copied().min_by(|a, b| {
        let da = (link.anchor.z - a.2).abs();
        let db = (link.anchor.z - b.2).abs();
        da.total_cmp(&db)
    })?;
    if lx1 - lx0 < STAIR_WIDTH + 8.0 {
        return None;
    }

    let side = if link.anchor.z >= lz { 1.0 } else { -1.0 };
    let corridor_half = lw * 0.5 + wall_t;
    let x0 = snap((link.anchor.x - STAIR_WIDTH * 0.5).clamp(lx0 + 4.0, lx1 - STAIR_WIDTH - 4.0));
    let z_front = snap(if side > 0.0 {
        lz + corridor_half
    } else {
        lz - corridor_half - STAIR_DEPTH
    });
    let footprint = Polygon2::rect(x0, z_front, STAIR_WIDTH, STAIR_DEPTH);
    let b = footprint.bounds();

    if b.0 < origin.x + edge_margin
        || b.1 < origin.z + edge_margin
        || b.2 > origin.x + size - edge_margin
        || b.3 > origin.z + size - edge_margin
    {
        return None;
    }
    if taken
        .iter()
        .any(|t| t.0 < b.2 + 0.8 && b.0 < t.2 + 0.8 && t.1 < b.3 + 0.8 && b.1 < t.3 + 0.8)
    {
        return None;
    }
    let center = ((b.0 + b.2) * 0.5, (b.1 + b.3) * 0.5);
    if !corridor_clearance(center.0, center.1, STAIR_DEPTH.min(STAIR_WIDTH) * 0.25) {
        return None;
    }

    // A broad, framed threshold: the stair announces itself as circulation.
    let front_z = if side > 0.0 { b.1 } else { b.3 };
    let entrances = vec![Opening {
        center: Position::new(snap(x0 + STAIR_WIDTH * 0.5), front_z),
        width: 2.4,
        through_x_wall: true,
        lintel_units: Some(2.6),
    }];

    // One warm fixture over the vestibule; the flight above stays unlit so
    // the climb visibly leaves the light.
    let vestibule_z = if side > 0.0 {
        b.1 + VESTIBULE_DEPTH * 0.5
    } else {
        b.3 - VESTIBULE_DEPTH * 0.5
    };
    let fixtures = vec![Fixture {
        at: Position::new(snap(x0 + STAIR_WIDTH * 0.5), snap(vestibule_z)),
        half_x: 0.4,
        half_z: 0.4,
        lit: true,
    }];

    Some(AssemblyInstance {
        id,
        program: SpaceProgram::Stair,
        spaces: Vec::new(),
        // Core-and-shell: the stair core carries itself; no interior columns
        // may interrupt a flight.
        structure: StructuralSystemInstance {
            system: StructuralSystem::CoreAndShell,
            bay_x: 4.8,
            bay_z: 4.8,
            phase: (0.0, 0.0),
            column_side: 0.4,
        },
        ceiling_zones: vec![CeilingZone {
            area: footprint.clone(),
            language: crate::domain::entities::architecture::CeilingLanguage::ExposedSoffit,
            height_units: STAIR_CEILING_UNITS,
        }],
        fixtures,
        service_voids: Vec::new(),
        corruption: CorruptionProfile::default(),
        entrances,
        footprint,
    })
}

/// Shapes one interior column of a stair assembly: vestibule, flight,
/// landing (or an endless climb). Pure in `(assembly, kind, wx, wz)`.
pub(crate) fn apply_stair_profile(
    assembly: &AssemblyInstance,
    kind: VerticalLinkKind,
    plan: &mut ColumnPlan,
    _wx: f32,
    wz: f32,
) {
    let Some(entrance) = assembly.entrances.first() else {
        return;
    };
    let b = assembly.footprint.bounds();
    // Walk axis runs perpendicular to the entrance wall; depth is measured
    // from the entrance side inward.
    let depth_in = if entrance.through_x_wall {
        if (entrance.center.z - b.1).abs() < (entrance.center.z - b.3).abs() {
            wz - b.1
        } else {
            b.3 - wz
        }
    } else {
        // Stairwells are placed with X-wall entrances today; a Z-wall stair
        // would measure along X the same way.
        return;
    };

    let target = match kind {
        VerticalLinkKind::EndlessAscent => ASCENT_MAX_UNITS,
        _ => LANDING_UNITS,
    };
    let treads = ((depth_in - VESTIBULE_DEPTH) / TREAD_RUN).floor().max(0.0);
    let raw = treads * TREAD_RISE;
    plan.floor_units = raw.min(target);
    // Headroom is a hard invariant: the ceiling zone is planned high enough,
    // but clamp anyway so no future ceiling tweak can pinch a flight.
    plan.floor_units = plan
        .floor_units
        .min(plan.ceiling_units - MIN_HEADROOM_UNITS)
        .max(0.0);

    // The climb leaves the light: no fixture survives above the first riser,
    // and an endless flight rises into an unlit shaft.
    if plan.floor_units > 0.0 {
        plan.light = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stair_assembly() -> AssemblyInstance {
        AssemblyInstance {
            id: 1,
            program: SpaceProgram::Stair,
            footprint: Polygon2::rect(0.0, 0.0, STAIR_WIDTH, STAIR_DEPTH),
            entrances: vec![Opening {
                center: Position::new(STAIR_WIDTH * 0.5, 0.0),
                width: 2.4,
                through_x_wall: true,
                lintel_units: Some(2.6),
            }],
            spaces: Vec::new(),
            structure: StructuralSystemInstance {
                system: StructuralSystem::CoreAndShell,
                bay_x: 4.8,
                bay_z: 4.8,
                phase: (0.0, 0.0),
                column_side: 0.4,
            },
            ceiling_zones: Vec::new(),
            fixtures: Vec::new(),
            service_voids: Vec::new(),
            corruption: CorruptionProfile::default(),
        }
    }

    fn profile_at(kind: VerticalLinkKind, wz: f32) -> ColumnPlan {
        let assembly = stair_assembly();
        let mut plan = ColumnPlan::open(STAIR_CEILING_UNITS);
        apply_stair_profile(&assembly, kind, &mut plan, 2.4, wz);
        plan
    }

    #[test]
    fn flight_rises_monotonically_and_lands() {
        let mut previous = -1.0f32;
        let mut landing_seen = false;
        let mut wz = 0.1;
        while wz < STAIR_DEPTH {
            let plan = profile_at(VerticalLinkKind::OrdinaryStair, wz);
            assert!(
                plan.floor_units >= previous - 1e-6,
                "flight descends at depth {wz}"
            );
            assert!(
                plan.ceiling_units - plan.floor_units >= MIN_HEADROOM_UNITS - 1e-6,
                "headroom pinched at depth {wz}"
            );
            landing_seen |= (plan.floor_units - LANDING_UNITS).abs() < 1e-6;
            previous = plan.floor_units;
            wz += 0.1;
        }
        assert!(landing_seen, "ordinary flight never reaches its landing");
    }

    #[test]
    fn vestibule_is_flat_and_endless_ascent_outclimbs_the_landing() {
        assert_eq!(
            profile_at(VerticalLinkKind::OrdinaryStair, VESTIBULE_DEPTH * 0.5).floor_units,
            0.0,
            "vestibule must stay walkable from the corridor"
        );
        let deep = profile_at(VerticalLinkKind::EndlessAscent, STAIR_DEPTH - 0.2);
        assert!(
            deep.floor_units > LANDING_UNITS,
            "endless ascent stops at an ordinary landing"
        );
        assert!(!deep.light, "the top of an endless flight must be unlit");
    }
}
