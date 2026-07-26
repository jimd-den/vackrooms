//! Assembly-derived Red Room encounter planning.
//!
//! A Red Room is not a macro-lattice shape. It inherits a real assembly's
//! footprint and entrance, then adds stable semantic crossing planes.

use crate::domain::entities::anomaly::{
    AnomalyInstance, AnomalyKind, Axis2, AxisDirection, OrientedFootprint, OrthoBasis, QuarterTurn,
    TraversalGate, TraversalGateKind,
};
use crate::domain::entities::architecture::AssemblyInstance;
use crate::domain::entities::position::Position;
use crate::use_cases::anomalies::determinism::{SNAP, mix64, stable_id};
use crate::use_cases::generate_chunk::GeneratorConfig;

const REGION_SIZE: f32 = 80.0;
const THRESHOLD_DEPTH: f32 = 2.8;
const LOOP_GATE_DEPTH: f32 = 6.0;

fn clamp_inward(desired: f32, far_inner_plane: f32, direction: AxisDirection) -> f32 {
    match direction {
        AxisDirection::Positive => desired.min(far_inner_plane),
        AxisDirection::Negative => desired.max(far_inner_plane),
    }
}

fn red_room_instance(
    seed: u32,
    region_origin: Position,
    assembly: &AssemblyInstance,
) -> Option<AnomalyInstance> {
    let entrance = *assembly.primary_entrance()?;
    let bounds = assembly.footprint.bounds();
    let center = Position::new((bounds.0 + bounds.2) * 0.5, (bounds.1 + bounds.3) * 0.5);
    let region_x = (region_origin.x / REGION_SIZE).floor() as i64;
    let region_z = (region_origin.z / REGION_SIZE).floor() as i64;
    let id = mix64(
        stable_id(seed, AnomalyKind::RedRoom, region_x, region_z)
            ^ (assembly.id as u64).wrapping_mul(0x9E37_79B9),
    );
    let footprint = OrientedFootprint {
        center,
        half_x: (bounds.2 - bounds.0) * 0.5,
        half_z: (bounds.3 - bounds.1) * 0.5,
        basis: OrthoBasis {
            turn: QuarterTurn::Zero,
        },
    };
    let axis = if entrance.through_x_wall {
        Axis2::Z
    } else {
        Axis2::X
    };
    let (edge_coordinate, center_coordinate, half_depth) = match axis {
        Axis2::X => (entrance.center.x, center.x, footprint.half_x),
        Axis2::Z => (entrance.center.z, center.z, footprint.half_z),
    };
    let forward = if center_coordinate >= edge_coordinate {
        AxisDirection::Positive
    } else {
        AxisDirection::Negative
    };

    // Commitment happens inside the room, beyond the externally readable
    // doorway. Entrance placement guarantees enough depth for this vestibule.
    let threshold_plane = edge_coordinate + forward.sign() * THRESHOLD_DEPTH;
    let (span_min, span_max) = if axis == Axis2::Z {
        (
            entrance.center.x - entrance.width * 0.5,
            entrance.center.x + entrance.width * 0.5,
        )
    } else {
        (
            entrance.center.z - entrance.width * 0.5,
            entrance.center.z + entrance.width * 0.5,
        )
    };
    let affected_bounds = footprint.bounds();
    let threshold_gate = TraversalGate {
        id: mix64(id ^ 0x6A7E_0ED0),
        instance_id: id,
        anomaly_kind: AnomalyKind::RedRoom,
        kind: TraversalGateKind::RedThreshold,
        axis,
        plane: threshold_plane,
        span_min,
        span_max,
        forward,
        affected_bounds,
    };

    // The loop checkpoint is six units beyond commitment when the room has
    // that depth. Shallow rooms clamp it one plan voxel inside the far wall,
    // keeping the semantic crossing on geometry owned by this encounter.
    let far_inner_plane = center_coordinate + forward.sign() * (half_depth - SNAP).max(0.0);
    let loop_plane = clamp_inward(
        threshold_plane + forward.sign() * LOOP_GATE_DEPTH,
        far_inner_plane,
        forward,
    );
    let loop_gate = TraversalGate {
        id: mix64(id ^ 0x9B4C_1A2F),
        instance_id: id,
        anomaly_kind: AnomalyKind::RedRoom,
        kind: TraversalGateKind::RedLoop,
        axis,
        plane: loop_plane,
        span_min,
        span_max,
        // `forward` records the authored inward direction. Runtime state owns
        // the separate policy for which traversal directions advance a loop.
        forward,
        affected_bounds,
    };

    Some(AnomalyInstance {
        id,
        kind: AnomalyKind::RedRoom,
        footprint,
        basis: footprint.basis,
        macro_anchor: (region_x, region_z),
        pillar_lattice: None,
        pit_lattice: None,
        arch: None,
        gates: vec![threshold_gate, loop_gate],
        skeleton_half_width: 1.0,
        entry_band: THRESHOLD_DEPTH,
    })
}

pub(crate) fn plan_red_rooms(
    seed: u32,
    region_origin: Position,
    assemblies: &[AssemblyInstance],
    config: &GeneratorConfig,
) -> Vec<AnomalyInstance> {
    let forced = config.anomalies.forced_kind == Some(AnomalyKind::RedRoom);
    let organic_enabled = config.anomalies.frequency > 0.0 && config.anomalies.red_rooms > 0.0;
    if !forced && !organic_enabled {
        return Vec::new();
    }

    assemblies
        .iter()
        .filter(|assembly| assembly.corruption.red_room)
        .filter_map(|assembly| red_room_instance(seed, region_origin, assembly))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::architecture::{
        CorruptionProfile, HostId, HostSegment, Opening, OpeningId, OpeningRole, Polygon2,
        SpaceProgram, StructuralSystem, StructuralSystemInstance,
    };

    fn red_assembly(entrance_z: f32) -> AssemblyInstance {
        let footprint = Polygon2::rect(0.0, 0.0, 12.0, 8.0);
        let host = if entrance_z <= 4.0 {
            HostId(0)
        } else {
            HostId(2)
        };
        AssemblyInstance {
            id: 7,
            program: SpaceProgram::PrivateOffice,
            footprint: footprint.clone(),
            hosts: HostSegment::rectangular_shell(&footprint, 0.4, 3.4),
            openings: vec![Opening {
                id: OpeningId(0),
                host,
                role: OpeningRole::Entrance,
                center: Position::new(6.0, entrance_z),
                width: 1.2,
                through_x_wall: true,
                lintel_units: Some(2.2),
            }],
            door_leaves: Vec::new(),
            spaces: Vec::new(),
            structure: StructuralSystemInstance {
                system: StructuralSystem::RegularGrid,
                bay_x: 4.0,
                bay_z: 4.0,
                phase: (0.0, 0.0),
                column_side: 0.4,
            },
            ceiling_zones: Vec::new(),
            fixtures: Vec::new(),
            service_voids: Vec::new(),
            corruption: CorruptionProfile {
                red_room: true,
                ..CorruptionProfile::default()
            },
        }
    }

    #[test]
    fn forced_red_room_survives_zero_frequency_and_zero_family_weight() {
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.frequency = 0.0;
        config.anomalies.red_rooms = 0.0;
        config.anomalies.forced_kind = Some(AnomalyKind::RedRoom);
        let rooms = plan_red_rooms(42, Position::new(0.0, 0.0), &[red_assembly(0.0)], &config);
        assert_eq!(rooms.len(), 1);
        assert_eq!(rooms[0].kind, AnomalyKind::RedRoom);
        assert_eq!(rooms[0].gates.len(), 2);
    }

    #[test]
    fn loop_gate_is_inside_both_entrance_orientations() {
        let config = GeneratorConfig::low_spec();
        for entrance_z in [0.0, 8.0] {
            let room = plan_red_rooms(
                42,
                Position::new(0.0, 0.0),
                &[red_assembly(entrance_z)],
                &config,
            )
            .pop()
            .unwrap();
            let loop_gate = room
                .gates
                .iter()
                .find(|gate| gate.kind == TraversalGateKind::RedLoop)
                .unwrap();
            let bounds = room.footprint.bounds();
            assert!(loop_gate.plane >= bounds.min_z + SNAP - f32::EPSILON);
            assert!(loop_gate.plane <= bounds.max_z - SNAP + f32::EPSILON);
        }
    }
}
