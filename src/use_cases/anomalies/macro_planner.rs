//! World-lattice planner for region-spanning Level 0 anomalies.
//!
//! Macro candidates live on a 160-unit lattice. Regions merely query that
//! lattice, so overlapping region and chunk requests recover identical
//! instance IDs, footprints, internal phases, and traversal gates.

use crate::domain::entities::anomaly::{
    AnomalyInstance, AnomalyKind, ArchBehavior, ArchLayout, ArchProfile, Axis2, AxisDirection,
    OrientedFootprint, OrthoBasis, PillarLattice, PitLattice, QuarterTurn, TraversalGate,
    TraversalGateKind, WorldBounds,
};
use crate::domain::entities::position::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::world_topology::sample_fields;

use super::determinism::{hash, mix64, snap, stable_id, unit};

/// The anomaly lattice *is* the world-graph lattice, so anomaly anchors and
/// macro-cell fields stay phase-aligned by construction.
const MACRO_CELL: f32 = crate::domain::entities::world_topology::MACRO_CELL_SIZE;
const MAX_QUERY_MARGIN: f32 = 400.0;

fn weighted_family(config: &GeneratorConfig, sample: f32) -> Option<AnomalyKind> {
    let tuning = config.anomalies;
    let weights = [
        (AnomalyKind::PillarExpanse, tuning.pillar_expanses.max(0.0)),
        (AnomalyKind::BlackoutExpanse, tuning.blackouts.max(0.0)),
        (AnomalyKind::PitLattice, tuning.pit_lattices.max(0.0)),
        (AnomalyKind::ArchwayRoom, tuning.archways.max(0.0)),
    ];
    let total: f32 = weights.iter().map(|(_, weight)| weight).sum();
    if total <= 0.0 {
        return None;
    }

    let mut cursor = sample * total;
    for (kind, weight) in weights {
        if cursor < weight {
            return Some(kind);
        }
        cursor -= weight;
    }
    Some(AnomalyKind::PillarExpanse)
}

fn remap_gates(instance: &AnomalyInstance) -> Vec<TraversalGate> {
    let (step, first, last) = match instance.kind {
        AnomalyKind::PillarExpanse => {
            let bay = instance.pillar_lattice.map_or(4.0, |lattice| lattice.bay_x);
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

    let mut gates = Vec::new();
    let mut local_x = snap((first / step).ceil() * step);
    let world_bounds = instance.footprint.bounds();
    let axis = instance.basis.local_axis_in_world(Axis2::X);
    while local_x <= last {
        let at = instance.world_coords(local_x, 0.0);
        let (plane, span_min, span_max) = match axis {
            Axis2::X => (at.x, world_bounds.min_z, world_bounds.max_z),
            Axis2::Z => (at.z, world_bounds.min_x, world_bounds.max_x),
        };
        let gate_index = (local_x / step).round() as i64;
        gates.push(TraversalGate {
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
        local_x += step;
    }
    gates
}

fn macro_instance(
    seed: u32,
    anchor_x: i64,
    anchor_z: i64,
    kind: AnomalyKind,
    config: &GeneratorConfig,
) -> AnomalyInstance {
    let id = stable_id(seed, kind, anchor_x, anchor_z);
    let h = |salt| hash(seed, salt, anchor_x, anchor_z);
    let size_scale = config.anomalies.size.clamp(0.5, 2.0);
    let center_x = snap((anchor_x as f32 + 0.5) * MACRO_CELL + (unit(h(0xC1)) - 0.5) * 48.0);
    let center_z = snap((anchor_z as f32 + 0.5) * MACRO_CELL + (unit(h(0xC2)) - 0.5) * 48.0);
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
        AnomalyKind::ArchwayRoom => (
            snap(4.0 + 3.0 * unit(h(0xD7))),
            snap(5.0 + 3.0 * unit(h(0xD8))),
            0.0,
            0.0,
        ),
        AnomalyKind::RedRoom => unreachable!("red rooms are planned from assemblies"),
    };
    let footprint = OrientedFootprint {
        center: Position::new(center_x, center_z),
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
    let arch = (kind == AnomalyKind::ArchwayRoom).then(|| {
        // Most arch rooms are walk-through connectors; the sheltered dead
        // end is the rarer find. Both stay immutable anchors.
        let layout = if unit(h(0xA0)) < 0.7 {
            ArchLayout::Transition
        } else {
            ArchLayout::DeadEnd
        };
        // Transition rooms are graph connectors: most stay plain anchors,
        // but some are seams into another culture or scale. A dead end
        // connects nothing, so it never claims a seam behavior.
        let behavior = if layout == ArchLayout::DeadEnd {
            ArchBehavior::Anchor
        } else {
            match unit(h(0xA4)) {
                v if v < 0.40 => ArchBehavior::Anchor,
                v if v < 0.75 => ArchBehavior::CultureSeam,
                _ => ArchBehavior::ScaleBreach,
            }
        };
        // Every instance is its own arcade: rhythm, clear width, springing
        // height, and blind cadence all vary, so no two arch rooms present
        // the same silhouette — yet every non-blind opening stays walkable
        // (the opening is capped so piers survive, and the head springs at
        // standing height before rising toward the center).
        let bay = snap(2.4 + 2.0 * unit(h(0xA1)));
        ArchProfile {
            layout,
            behavior,
            bay,
            opening: snap((1.4 + 1.4 * unit(h(0xA2))).min(bay - 0.8)),
            blind_every: 3 + (h(0xA3) % 4) as u8,
            spring_units: ((2.0 + 0.8 * unit(h(0xA5))) / 0.2).round() * 0.2,
        }
    });
    let mut instance = AnomalyInstance {
        id,
        kind,
        footprint,
        basis,
        macro_anchor: (anchor_x, anchor_z),
        pillar_lattice,
        pit_lattice,
        arch,
        gates: Vec::new(),
        skeleton_half_width,
        entry_band,
    };
    instance.gates = remap_gates(&instance);
    instance
}

fn forced_instance(
    seed: u32,
    query: WorldBounds,
    protected_point: Position,
    config: &GeneratorConfig,
) -> Option<AnomalyInstance> {
    let kind = config.anomalies.forced_kind?;
    if kind == AnomalyKind::RedRoom {
        return None;
    }

    if kind == AnomalyKind::ArchwayRoom {
        // A forced anchor is moved next to the protected point so it is both
        // resident and visible; organic anchors keep their macro-cell center.
        let center = Position::new(snap(protected_point.x + 10.0), snap(protected_point.z));
        let anchor_x = (center.x / MACRO_CELL).floor() as i64;
        let anchor_z = (center.z / MACRO_CELL).floor() as i64;
        let mut instance = macro_instance(seed, anchor_x, anchor_z, kind, config);
        instance.footprint.center = center;
        return instance
            .footprint
            .bounds()
            .intersects(query)
            .then_some(instance);
    }

    let anchor_x = (protected_point.x / MACRO_CELL).floor() as i64;
    let anchor_z = (protected_point.z / MACRO_CELL).floor() as i64;
    let instance = macro_instance(seed, anchor_x, anchor_z, kind, config);
    instance
        .footprint
        .bounds()
        .intersects(query)
        .then_some(instance)
}

pub(crate) fn plan_macro_anomalies(
    seed: u32,
    query: WorldBounds,
    protected_point: Position,
    config: &GeneratorConfig,
    noise: &dyn NoiseProvider,
) -> Vec<AnomalyInstance> {
    let mut instances: Vec<_> = forced_instance(seed, query, protected_point, config)
        .into_iter()
        .collect();
    if config.anomalies.frequency <= 0.0 {
        return instances;
    }

    let expanded = query.expanded(MAX_QUERY_MARGIN);
    let anchor_x_min = (expanded.min_x / MACRO_CELL).floor() as i64;
    let anchor_x_max = (expanded.max_x / MACRO_CELL).floor() as i64;
    let anchor_z_min = (expanded.min_z / MACRO_CELL).floor() as i64;
    let anchor_z_max = (expanded.max_z / MACRO_CELL).floor() as i64;
    let base_chance = (0.20 * config.anomalies.frequency.clamp(0.0, 4.0)).min(0.75);

    for anchor_z in anchor_z_min..=anchor_z_max {
        for anchor_x in anchor_x_min..=anchor_x_max {
            // The macro anomaly-pressure field clusters candidates: quiet
            // territory thins toward 0.4× the base chance and pressured
            // territory rises toward 1.6×, so anomalies arrive in loose
            // constellations instead of a uniform sprinkle. The field is a
            // pure world function, so every query that can see this anchor
            // computes the identical decision.
            let pressure = sample_fields(
                noise,
                seed,
                (anchor_x as f32 + 0.5) * MACRO_CELL,
                (anchor_z as f32 + 0.5) * MACRO_CELL,
            )
            .anomaly_pressure;
            let chance = base_chance * (0.40 + 1.20 * pressure);
            let candidate = hash(seed, 0xCAAD_1DA7, anchor_x, anchor_z);
            if unit(candidate) >= chance {
                continue;
            }
            let Some(kind) = weighted_family(config, unit(candidate.rotate_left(21))) else {
                continue;
            };
            let instance = macro_instance(seed, anchor_x, anchor_z, kind, config);
            if !instance.footprint.bounds().intersects(query) {
                continue;
            }
            // Organic candidates honor the opening-region keep-out. Forced
            // candidates intentionally bypass it for developer inspection.
            if instance
                .footprint
                .bounds()
                .expanded(48.0)
                .contains(protected_point.x, protected_point.z)
            {
                continue;
            }
            instances.push(instance);
        }
    }
    instances
}
