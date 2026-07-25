//! Route and awareness telemetry: pure selection and bearing math over the
//! authoritative world records the streamer already holds.
//!
//! Nothing here invents targets. A route exists only when a real
//! [`LevelExit`] is resident; an anomaly focus exists only when a real
//! traversal gate or pit hazard is resident. When the data is not streamed
//! in, the truthful answer is `None` — the presenter says "beyond field"
//! instead of pretending.
//!
//! Range is always the straight-line (Euclidean) walking-plane distance and
//! is labelled RANGE by the presenter — there is no navigable route graph
//! in the engine today, so a "path distance" would be a fabrication.

use vackrooms::domain::entities::anomaly::{AnomalyKind, PitHazard, TraversalGate};
use vackrooms::domain::entities::anomaly::LevelExit;

/// A selected navigation objective (a real, resident level door).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteTarget {
    pub exit_id: u64,
    /// Level on the far side of the door.
    pub target_level: u32,
    /// Door trigger center on the walking plane.
    pub position: [f32; 2],
}

/// Straight-line range on the walking plane, meters.
pub fn range_m(from: [f32; 2], to: [f32; 2]) -> f32 {
    let dx = to[0] - from[0];
    let dz = to[1] - from[1];
    (dx * dx + dz * dz).sqrt()
}

/// Bearing of `to` relative to the player's facing, in whole degrees
/// normalized to `0..=359`. 0° is dead ahead; 90° is to the player's right
/// (clockwise positive, viewed from above).
///
/// The facing convention matches [`crate::application::player::Player`]:
/// forward is `[-sin(yaw), -cos(yaw)]` on the XZ plane.
pub fn relative_bearing_deg(yaw: f32, from: [f32; 2], to: [f32; 2]) -> u16 {
    let fwd = [-yaw.sin(), -yaw.cos()];
    let dir = [to[0] - from[0], to[1] - from[1]];
    if !dir[0].is_finite() || !dir[1].is_finite() || (dir[0] == 0.0 && dir[1] == 0.0) {
        return 0;
    }
    // Clockwise-from-forward: right-hand cross on the ground plane, with +Y
    // up, gives the signed turn toward the target.
    let cross = fwd[0] * dir[1] - fwd[1] * dir[0];
    let dot = fwd[0] * dir[0] + fwd[1] * dir[1];
    let deg = cross.atan2(dot).to_degrees();
    let wrapped = ((deg % 360.0) + 360.0) % 360.0;
    (wrapped.round() as i64).rem_euclid(360) as u16
}

/// Selects the navigation objective from the resident level exits: the
/// nearest door, ties broken by id so the choice is deterministic even for
/// coincident duplicates surfaced by chunk-halo overlap.
pub fn select_route_target<'a>(
    exits: impl Iterator<Item = &'a LevelExit>,
    player: [f32; 2],
) -> Option<RouteTarget> {
    exits
        .map(|exit| {
            let pos = [exit.center.x, exit.center.z];
            (range_m(player, pos), exit.id, exit, pos)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
        .map(|(_, _, exit, pos)| RouteTarget {
            exit_id: exit.id,
            target_level: exit.target_level,
            position: pos,
        })
}

/// Debug-only anomaly focus: the single most relevant resident anomaly
/// record near the player. The normal HUD never shows this — the game has
/// no player-facing anomaly-sensing mechanic yet, so surfacing it outside
/// the diagnostic aperture would make the interface psychic. When such a
/// mechanic exists, it can consume this same deterministic selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnomalyFocus {
    pub kind: AnomalyKind,
    pub instance_id: u64,
    /// Nearest point distance is not derived; this is center/plane range, m.
    pub range_m: f32,
    /// Absolute position used for bearing math.
    pub position: [f32; 2],
    /// Larger is more urgent; used only for deterministic ranking.
    pub priority: u8,
}

fn kind_priority(kind: AnomalyKind) -> u8 {
    // Ranked by immediate danger to the wanderer.
    match kind {
        AnomalyKind::RedRoom => 4,
        AnomalyKind::PitLattice => 3,
        AnomalyKind::BlackoutExpanse => 2,
        _ => 1,
    }
}

/// Deterministic single-focus selection over resident gates and pits within
/// `awareness_m`. Ordering: priority class first, then distance (quantized
/// to 0.1 m so float noise cannot flip the choice), then instance id.
pub fn select_anomaly_focus<'a>(
    gates: impl Iterator<Item = &'a TraversalGate>,
    pits: impl Iterator<Item = &'a PitHazard>,
    player: [f32; 2],
    awareness_m: f32,
) -> Option<AnomalyFocus> {
    let gate_candidates = gates.map(|gate| {
        // The gate's reference point: the nearest point of its plane span.
        let (x, z) = match gate.axis {
            vackrooms::domain::entities::anomaly::Axis2::X => {
                (gate.plane, player[1].clamp(gate.span_min, gate.span_max))
            }
            vackrooms::domain::entities::anomaly::Axis2::Z => {
                (player[0].clamp(gate.span_min, gate.span_max), gate.plane)
            }
        };
        AnomalyFocus {
            kind: gate.anomaly_kind,
            instance_id: gate.instance_id,
            range_m: range_m(player, [x, z]),
            position: [x, z],
            priority: kind_priority(gate.anomaly_kind),
        }
    });
    let pit_candidates = pits.map(|pit| AnomalyFocus {
        kind: AnomalyKind::PitLattice,
        instance_id: pit.instance_id,
        range_m: range_m(player, [pit.center.x, pit.center.z]),
        position: [pit.center.x, pit.center.z],
        priority: kind_priority(AnomalyKind::PitLattice),
    });
    gate_candidates
        .chain(pit_candidates)
        .filter(|focus| focus.range_m.is_finite() && focus.range_m <= awareness_m)
        .min_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(((a.range_m * 10.0) as i64).cmp(&((b.range_m * 10.0) as i64)))
                .then(a.instance_id.cmp(&b.instance_id))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vackrooms::domain::entities::anomaly::{
        Axis2, AxisDirection, TraversalGateKind, WorldBounds,
    };
    use vackrooms::domain::entities::position::Position;

    fn exit(id: u64, x: f32, z: f32, target_level: u32) -> LevelExit {
        LevelExit {
            id,
            target_level,
            center: Position::new(x, z),
            half_extent: 0.7,
            arrival: Position::new(3.0, 3.0),
        }
    }

    #[test]
    fn bearing_covers_all_quadrants_and_wraps() {
        // yaw = 0: forward is -Z.
        let p = [0.0, 0.0];
        assert_eq!(relative_bearing_deg(0.0, p, [0.0, -10.0]), 0, "ahead");
        assert_eq!(relative_bearing_deg(0.0, p, [10.0, 0.0]), 90, "right");
        assert_eq!(relative_bearing_deg(0.0, p, [0.0, 10.0]), 180, "behind");
        assert_eq!(relative_bearing_deg(0.0, p, [-10.0, 0.0]), 270, "left");
        // Diagonals.
        assert_eq!(relative_bearing_deg(0.0, p, [10.0, -10.0]), 45);
        assert_eq!(relative_bearing_deg(0.0, p, [-10.0, -10.0]), 315);
        // A quarter-turn of yaw shifts every bearing by the same amount.
        let quarter = std::f32::consts::FRAC_PI_2; // now facing -X
        assert_eq!(relative_bearing_deg(quarter, p, [-10.0, 0.0]), 0);
        assert_eq!(relative_bearing_deg(quarter, p, [0.0, -10.0]), 90);
        // Degenerate/absurd input collapses to 0, never panics.
        assert_eq!(relative_bearing_deg(0.0, p, p), 0);
        assert_eq!(relative_bearing_deg(0.0, p, [f32::NAN, 1.0]), 0);
    }

    #[test]
    fn route_selection_is_nearest_and_deterministic() {
        let doors = [
            exit(9, 30.0, 0.0, 1),
            exit(2, 10.0, 0.0, 1),
            exit(1, 10.0, 0.0, 1), // coincident duplicate: lower id wins
        ];
        let chosen = select_route_target(doors.iter(), [0.0, 0.0]).unwrap();
        assert_eq!(chosen.exit_id, 1);
        assert_eq!(chosen.target_level, 1);
        assert!((range_m([0.0, 0.0], chosen.position) - 10.0).abs() < 1e-4);
        // No resident doors: truthfully nothing.
        assert_eq!(select_route_target([].iter(), [0.0, 0.0]), None);
    }

    #[test]
    fn anomaly_focus_prioritizes_danger_then_distance_then_id() {
        let gate = |id: u64, kind: AnomalyKind, plane: f32| TraversalGate {
            id,
            instance_id: id,
            anomaly_kind: kind,
            kind: TraversalGateKind::Remap,
            axis: Axis2::X,
            plane,
            span_min: -5.0,
            span_max: 5.0,
            forward: AxisDirection::Positive,
            affected_bounds: WorldBounds::new(0.0, 0.0, 10.0, 10.0),
        };
        let pit = PitHazard {
            id: 50,
            instance_id: 50,
            center: Position::new(6.0, 0.0),
            half_side: 2.0,
            depth: 3.0,
            recovery: Position::new(0.0, 0.0),
        };
        // A red room farther away outranks a nearer blackout; the pit sits
        // between them in class.
        let gates = [
            gate(10, AnomalyKind::BlackoutExpanse, 2.0),
            gate(11, AnomalyKind::RedRoom, 9.0),
        ];
        let focus =
            select_anomaly_focus(gates.iter(), [pit].iter(), [0.0, 0.0], 20.0).unwrap();
        assert_eq!(focus.instance_id, 11, "danger class outranks distance");

        // Outside awareness range: honestly nothing.
        assert_eq!(
            select_anomaly_focus(gates.iter(), [].iter(), [0.0, 0.0], 1.0),
            None
        );

        // Equal class and distance: id breaks the tie deterministically.
        let twins = [
            gate(21, AnomalyKind::BlackoutExpanse, 3.0),
            gate(20, AnomalyKind::BlackoutExpanse, -3.0),
        ];
        let focus =
            select_anomaly_focus(twins.iter(), [].iter(), [0.0, 0.0], 20.0).unwrap();
        assert_eq!(focus.instance_id, 20);
    }
}
