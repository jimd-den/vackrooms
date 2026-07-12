//! Deterministic macro-scale Level 0 anomaly planning.
//!
//! Candidate identity is anchored to a 160u world lattice, never a region or
//! chunk query. Every region that intersects an instance therefore receives
//! the exact same record, including lattice phase and gate IDs.

use crate::domain::entities::anomaly::*;
use crate::domain::entities::architecture::AssemblyInstance;
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;

const MACRO_CELL: f32 = 160.0;
const SNAP: f32 = 0.4;
const MAX_QUERY_MARGIN: f32 = 400.0;

fn snap(v: f32) -> f32 {
    (v / SNAP).round() * SNAP
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn hash(seed: u32, salt: u64, x: i64, z: i64) -> u64 {
    mix64(
        seed as u64
            ^ salt
            ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (z as u64).rotate_left(31),
    )
}

fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

fn stable_id(seed: u32, kind: AnomalyKind, ax: i64, az: i64) -> u64 {
    hash(seed, 0xA110_4A1E_0000_0000 | kind as u64, ax, az)
}

fn weighted_family(config: &GeneratorConfig, h: f32) -> Option<AnomalyKind> {
    let t = config.anomalies;
    let weights = [
        (AnomalyKind::PillarExpanse, t.pillar_expanses.max(0.0)),
        (AnomalyKind::BlackoutExpanse, t.blackouts.max(0.0)),
        (AnomalyKind::PitLattice, t.pit_lattices.max(0.0)),
    ];
    let total: f32 = weights.iter().map(|p| p.1).sum();
    if total <= 0.0 {
        return None;
    }
    let mut cursor = h * total;
    for (kind, weight) in weights {
        if cursor < weight {
            return Some(kind);
        }
        cursor -= weight;
    }
    Some(AnomalyKind::PillarExpanse)
}

fn gates_for(instance: &AnomalyInstance) -> Vec<TraversalGate> {
    let (step, first, last) = match instance.kind {
        AnomalyKind::PillarExpanse => {
            let bay = instance.pillar_lattice.map_or(4.0, |p| p.bay_x);
            (
                bay * 4.0,
                -instance.footprint.half_x + instance.entry_band,
                instance.footprint.half_x - instance.entry_band,
            )
        }
        AnomalyKind::BlackoutExpanse => (
            22.0,
            -instance.footprint.half_x + instance.entry_band,
            instance.footprint.half_x - instance.entry_band,
        ),
        _ => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut lx = snap((first / step).ceil() * step);
    let world_bounds = instance.footprint.bounds();
    let axis = instance.basis.local_axis_in_world(Axis2::X);
    while lx <= last {
        let at = instance.world_coords(lx, 0.0);
        let (plane, span_min, span_max) = match axis {
            Axis2::X => (at.x, world_bounds.min_z, world_bounds.max_z),
            Axis2::Z => (at.z, world_bounds.min_x, world_bounds.max_x),
        };
        let gate_index = (lx / step).round() as i64;
        out.push(TraversalGate {
            id: mix64(instance.id ^ (gate_index as u64).rotate_left(23) ^ 0x6A7E),
            instance_id: instance.id,
            anomaly_kind: instance.kind,
            kind: TraversalGateKind::Remap,
            axis,
            plane,
            span_min,
            span_max,
            forward: AxisDirection::Positive,
            affected_bounds: world_bounds,
        });
        lx += step;
    }
    out
}

fn macro_instance(
    seed: u32,
    ax: i64,
    az: i64,
    kind: AnomalyKind,
    config: &GeneratorConfig,
) -> AnomalyInstance {
    let id = stable_id(seed, kind, ax, az);
    let h = |salt| hash(seed, salt, ax, az);
    let size_scale = config.anomalies.size.clamp(0.5, 2.0);
    let cx = snap((ax as f32 + 0.5) * MACRO_CELL + (unit(h(0xC1)) - 0.5) * 48.0);
    let cz = snap((az as f32 + 0.5) * MACRO_CELL + (unit(h(0xC2)) - 0.5) * 48.0);
    let basis = OrthoBasis {
        turn: if h(0xC3) & 1 == 0 {
            QuarterTurn::Zero
        } else {
            QuarterTurn::Clockwise
        },
    };
    let (half_x, half_z, entry_band, skeleton_half_width) = match kind {
        AnomalyKind::PillarExpanse => (
            snap((75.0 + 85.0 * unit(h(0xD1))) * size_scale),
            snap((55.0 + 65.0 * unit(h(0xD2))) * size_scale),
            16.0,
            2.0,
        ),
        AnomalyKind::BlackoutExpanse => (
            snap((65.0 + 70.0 * unit(h(0xD3))) * size_scale),
            snap((50.0 + 55.0 * unit(h(0xD4))) * size_scale),
            12.0,
            2.0,
        ),
        AnomalyKind::PitLattice => (
            snap((32.0 + 30.0 * unit(h(0xD5))) * size_scale),
            snap((28.0 + 25.0 * unit(h(0xD6))) * size_scale),
            6.0,
            1.8,
        ),
        // Compact 8-14u stable rooms; entry band/skeleton are meaningless
        // for an anchor that never remaps.
        AnomalyKind::ArchwayRoom => (
            snap(4.0 + 3.0 * unit(h(0xD7))),
            snap(5.0 + 3.0 * unit(h(0xD8))),
            0.0,
            0.0,
        ),
        AnomalyKind::RedRoom => unreachable!("red rooms derive from assemblies"),
    };
    let footprint = OrientedFootprint {
        center: Position::new(cx, cz),
        half_x,
        half_z,
        basis,
    };
    let pillar_lattice = (kind == AnomalyKind::PillarExpanse).then(|| PillarLattice {
        bay_x: snap(3.6 + 1.2 * unit(h(0xE1))),
        bay_z: snap(3.6 + 1.2 * unit(h(0xE2))),
        phase_x: snap((unit(h(0xE3)) - 0.5) * 3.2),
        phase_z: snap((unit(h(0xE4)) - 0.5) * 3.2),
        min_side: 1.2,
        max_side: 1.6,
        variation_seed: id ^ 0xA111_AA55_u64,
    });
    let pit_lattice = (kind == AnomalyKind::PitLattice).then(|| PitLattice {
        spacing_x: snap(2.0 + 0.4 * unit(h(0xF1))),
        spacing_z: snap(2.0 + 0.4 * unit(h(0xF2))),
        phase_x: snap(unit(h(0xF3)) * 1.2),
        phase_z: snap(unit(h(0xF4)) * 1.2),
        side: snap(0.8 + 0.2 * unit(h(0xF5))),
        depth: 2.4,
    });
    let arch = (kind == AnomalyKind::ArchwayRoom).then(|| ArchProfile {
        layout: if h(0xA0) & 1 == 0 {
            ArchLayout::Transition
        } else {
            ArchLayout::DeadEnd
        },
        bay: snap(2.4 + 0.8 * unit(h(0xA1))),
        opening: snap(1.6 + 0.4 * unit(h(0xA2))),
        blind_every: 3 + (h(0xA3) % 2) as u8,
    });
    let mut instance = AnomalyInstance {
        id,
        kind,
        footprint,
        basis,
        macro_anchor: (ax, az),
        pillar_lattice,
        pit_lattice,
        arch,
        gates: Vec::new(),
        skeleton_half_width,
        entry_band,
    };
    instance.gates = gates_for(&instance);
    instance
}

fn red_room_instance(
    seed: u32,
    region_origin: Position,
    a: &AssemblyInstance,
) -> Option<AnomalyInstance> {
    let e = *a.entrances.first()?;
    let b = a.footprint.bounds();
    let center = Position::new((b.0 + b.2) * 0.5, (b.1 + b.3) * 0.5);
    let rx = (region_origin.x / 80.0).floor() as i64;
    let rz = (region_origin.z / 80.0).floor() as i64;
    let id = mix64(
        stable_id(seed, AnomalyKind::RedRoom, rx, rz) ^ (a.id as u64).wrapping_mul(0x9E37_79B9),
    );
    let footprint = OrientedFootprint {
        center,
        half_x: (b.2 - b.0) * 0.5,
        half_z: (b.3 - b.1) * 0.5,
        basis: OrthoBasis {
            turn: QuarterTurn::Zero,
        },
    };
    let axis = if e.through_x_wall { Axis2::Z } else { Axis2::X };
    let (edge_coord, center_coord) = match axis {
        Axis2::X => (e.center.x, center.x),
        Axis2::Z => (e.center.z, center.z),
    };
    let forward = if center_coord >= edge_coord {
        AxisDirection::Positive
    } else {
        AxisDirection::Negative
    };
    // The semantic threshold sits beyond the mandatory vestibule/bend, not
    // in the externally visible doorway.
    let plane = edge_coord + forward.sign() * 2.8;
    let (span_min, span_max) = if axis == Axis2::Z {
        (e.center.x - e.width * 0.5, e.center.x + e.width * 0.5)
    } else {
        (e.center.z - e.width * 0.5, e.center.z + e.width * 0.5)
    };
    let gate = TraversalGate {
        id: mix64(id ^ 0x6A7E_0ED0),
        instance_id: id,
        anomaly_kind: AnomalyKind::RedRoom,
        kind: TraversalGateKind::RedThreshold,
        axis,
        plane,
        span_min,
        span_max,
        forward,
        affected_bounds: footprint.bounds(),
    };
    Some(AnomalyInstance {
        id,
        kind: AnomalyKind::RedRoom,
        footprint,
        basis: footprint.basis,
        macro_anchor: (rx, rz),
        pillar_lattice: None,
        pit_lattice: None,
        arch: None,
        gates: vec![gate],
        skeleton_half_width: 1.0,
        entry_band: 2.8,
    })
}

/// Returns every immutable instance whose footprint overlaps this region.
pub fn plan_anomalies_for_region(
    seed: u32,
    region_origin: Position,
    region_size: f32,
    assemblies: &[AssemblyInstance],
    protected_point: Position,
    config: &GeneratorConfig,
) -> Vec<AnomalyInstance> {
    if config.anomalies.frequency <= 0.0 {
        return Vec::new();
    }
    let query = WorldBounds::new(
        region_origin.x,
        region_origin.z,
        region_origin.x + region_size,
        region_origin.z + region_size,
    );
    let expanded = query.expanded(MAX_QUERY_MARGIN);
    let ax0 = (expanded.min_x / MACRO_CELL).floor() as i64;
    let ax1 = (expanded.max_x / MACRO_CELL).floor() as i64;
    let az0 = (expanded.min_z / MACRO_CELL).floor() as i64;
    let az1 = (expanded.max_z / MACRO_CELL).floor() as i64;
    let chance = (0.20 * config.anomalies.frequency.clamp(0.0, 4.0)).min(0.75);
    let mut out = Vec::new();
    for az in az0..=az1 {
        for ax in ax0..=ax1 {
            let candidate = hash(seed, 0xCAAD_1DA7, ax, az);
            if unit(candidate) >= chance {
                continue;
            }
            let Some(kind) = weighted_family(config, unit(candidate.rotate_left(21))) else {
                continue;
            };
            let instance = macro_instance(seed, ax, az, kind, config);
            if !instance.footprint.bounds().intersects(query) {
                continue;
            }
            // The opening regions teach ordinary Level 0 before corruption.
            if instance
                .footprint
                .bounds()
                .expanded(48.0)
                .contains(protected_point.x, protected_point.z)
            {
                continue;
            }
            out.push(instance);
        }
    }

    // Archway anchors ride an independent candidate stream on the same
    // stable macro lattice. They are deliberately more common than hostile
    // expanses: recurring fixed landmarks.
    if config.anomalies.archways > 0.0 {
        let arch_chance =
            (0.35 * config.anomalies.frequency.clamp(0.0, 4.0) * config.anomalies.archways)
                .min(0.85);
        for az in az0..=az1 {
            for ax in ax0..=ax1 {
                let candidate = hash(seed, 0xAC11_0A7E, ax, az);
                if unit(candidate) >= arch_chance {
                    continue;
                }
                let instance = macro_instance(seed, ax, az, AnomalyKind::ArchwayRoom, config);
                if !instance.footprint.bounds().intersects(query) {
                    continue;
                }
                // Anchors never overlap hostile territory: where they would,
                // the anchor wins by omission of the hostile candidate's
                // interior — but a cheap planner-level exclusion keeps the
                // contrast architecturally readable.
                if out.iter().any(|other| {
                    other.kind != AnomalyKind::ArchwayRoom
                        && other
                            .footprint
                            .bounds()
                            .expanded(3.2)
                            .intersects(instance.footprint.bounds())
                }) {
                    continue;
                }
                out.push(instance);
            }
        }
    }

    if config.anomalies.red_rooms > 0.0 {
        out.extend(
            assemblies
                .iter()
                .filter(|a| a.corruption.red_room)
                .filter_map(|a| red_room_instance(seed, region_origin, a)),
        );
    }
    out.sort_by_key(|a| a.id);
    out.dedup_by_key(|a| a.id);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::use_cases::generate_chunk::GeneratorConfig;

    #[test]
    fn disabled_anomalies_produce_no_instances() {
        let mut c = GeneratorConfig::low_spec();
        c.anomalies.frequency = 0.0;
        assert!(
            plan_anomalies_for_region(
                42,
                Position::new(800.0, 800.0),
                80.0,
                &[],
                Position::new(0.0, 0.0),
                &c,
            )
            .is_empty()
        );
    }

    #[test]
    fn region_partition_keeps_instance_identity() {
        let mut c = GeneratorConfig::low_spec();
        c.anomalies.frequency = 4.0;
        let protected = Position::new(0.0, 0.0);
        let mut shared = false;
        for rz in 4..20 {
            for rx in 4..20 {
                let a = plan_anomalies_for_region(
                    42,
                    Position::new(rx as f32 * 80.0, rz as f32 * 80.0),
                    80.0,
                    &[],
                    protected,
                    &c,
                );
                let b = plan_anomalies_for_region(
                    42,
                    Position::new((rx + 1) as f32 * 80.0, rz as f32 * 80.0),
                    80.0,
                    &[],
                    protected,
                    &c,
                );
                if let Some(one) = a.iter().find(|one| b.iter().any(|two| two.id == one.id)) {
                    let two = b.iter().find(|two| two.id == one.id).unwrap();
                    assert_eq!(one, two);
                    shared = true;
                    break;
                }
            }
            if shared {
                break;
            }
        }
        assert!(shared, "forced-density scan found no cross-region instance");
    }
}
