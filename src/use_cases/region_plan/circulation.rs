//! Circulation routing: one dominant cross-region spine plus rare branches.
//!
//! The main spine always runs west portal -> east portal, so circulation
//! chains across regions forever; branches are optional and never form a
//! competing street grid.

use crate::domain::entities::architecture::*;
use crate::entities::models::Position;
use crate::use_cases::world_topology::hash01;

use super::{EDGE_MARGIN, snap, snap_width, v_edge_portal_z};

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

pub(super) fn build_corridors(
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
