//! Chunk generation: voxelize the planned world (plus recursive Red Room
//! addresses) into one output grid, exporting traversal gates, pit hazards,
//! and runtime lights from the same immutable plans as the geometry.

use crate::domain::entities::anomaly::{
    AnomalyInstance, AnomalyKind, Axis2, PitHazard, RealitySnapshot, TraversalGate,
    TraversalGateKind, WorldBounds,
};
use crate::domain::entities::architecture::{LightKind, RegionPlan, RuntimeLight};
use crate::domain::entities::voxel_grid::VoxelGrid;
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::infinite_level::InfiniteRegionWindow;
use crate::use_cases::level_generator::LevelGenerator;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::red_rooms::recursive_level::RecursiveLevelWindow;
use crate::use_cases::region_plan::region_index;

use super::{BackroomsLevel, ColumnField, ColumnPlan, GRID_HEIGHT_UNITS, voxelize_columns};

/// Runtime direct lights sit below their visible ceiling panel. Tall atria
/// need a longer pendant drop so the light reaches occupied space and casts
/// useful column shadows instead of flattening against the vault.
fn runtime_light_height(ceiling_units: f32, is_atrium: bool) -> f32 {
    let pendant_drop = if is_atrium { 1.6 } else { 1.1 };
    (ceiling_units - pendant_drop).max(2.4)
}

fn gate_crosses_bounds(gate: &TraversalGate, bounds: WorldBounds) -> bool {
    match gate.axis {
        Axis2::X => {
            gate.plane >= bounds.min_x - 0.01
                && gate.plane <= bounds.max_x + 0.01
                && gate.span_max >= bounds.min_z
                && gate.span_min <= bounds.max_z
        }
        Axis2::Z => {
            gate.plane >= bounds.min_z - 0.01
                && gate.plane <= bounds.max_z + 0.01
                && gate.span_max >= bounds.min_x
                && gate.span_min <= bounds.max_x
        }
    }
}

/// Deduplicated traversal gates and pit hazards exported from the region
/// plans a chunk overlaps. Pure: reads the already-computed plans and
/// (optionally) the recursive Red Room window, touches no `VoxelGrid` — the
/// caller decides where the results land. Neighboring region plans repeat
/// macro anomaly instances, so gates/hazards are deduplicated here by their
/// stable ids before anything reaches the write stage.
#[allow(clippy::too_many_arguments)]
fn decide_traversal_gates_and_hazards(
    plans: &[((i64, i64), RegionPlan)],
    recursive_level: Option<&RecursiveLevelWindow>,
    chunk_bounds: WorldBounds,
) -> (Vec<TraversalGate>, Vec<PitHazard>) {
    let authored_bounds = recursive_level.map_or(chunk_bounds, |recursive| {
        recursive.to_recursive_bounds(chunk_bounds)
    });
    let mut gates = Vec::new();
    let mut hazards = Vec::new();
    let mut seen_instances = std::collections::HashSet::new();
    let mut seen_gates = std::collections::HashSet::new();
    let mut seen_hazards = std::collections::HashSet::new();
    let mut export_interactions = |plan: &RegionPlan| {
        for anomaly in &plan.anomalies {
            if !seen_instances.insert(anomaly.id)
                || !anomaly
                    .footprint
                    .bounds()
                    .intersects(authored_bounds.expanded(1.0))
            {
                continue;
            }
            for gate in anomaly.traversal_gates() {
                if gate_crosses_bounds(gate, authored_bounds) && seen_gates.insert(gate.id) {
                    gates.push(recursive_level.map_or(*gate, |recursive| recursive.project_gate(*gate)));
                }
            }
            for hazard in anomaly.pit_hazards_for_bounds(authored_bounds) {
                if seen_hazards.insert(hazard.id) {
                    hazards.push(
                        recursive_level.map_or(hazard, |recursive| recursive.project_hazard(hazard)),
                    );
                }
            }
        }
    };
    if let Some(recursive) = recursive_level {
        for (_, plan) in recursive.iter() {
            export_interactions(plan);
        }
    } else {
        for (_, plan) in plans {
            export_interactions(plan);
        }
    }
    drop(export_interactions);

    // The recursive branch supplies its own anomaly semantics, while the
    // encounter that opened it retains one checkpoint in visible space.
    // Alternating across that authored plane advances the closed loop.
    if let Some(recursive) = recursive_level {
        for gate in plans
            .iter()
            .flat_map(|(_, plan)| plan.anomalies.iter())
            .filter(|anomaly| anomaly.id == recursive.instance_id())
            .flat_map(AnomalyInstance::traversal_gates)
            .filter(|gate| gate.kind == TraversalGateKind::RedLoop)
        {
            if gate_crosses_bounds(gate, chunk_bounds) && seen_gates.insert(gate.id) {
                gates.push(*gate);
            }
        }
    }

    (gates, hazards)
}

/// Runtime direct lights exported from the region plans a chunk overlaps.
/// Pure in the same sense as `decide_traversal_gates_and_hazards`: reads
/// plans/config/reality and queries the (already-decided) architectural
/// column plan for burial rejection, but never touches a `VoxelGrid` —
/// runtime lights are decided before voxelization, which is exactly why
/// this queries the column plan directly instead of the still-empty grid.
#[allow(clippy::too_many_arguments)]
fn decide_runtime_lights(
    plans: &[((i64, i64), RegionPlan)],
    recursive_level: Option<&RecursiveLevelWindow>,
    chunk_pos: Position,
    config: &GeneratorConfig,
    seed: u32,
    noise: &dyn NoiseProvider,
    reality: &RealitySnapshot,
) -> Vec<RuntimeLight> {
    let plan_of = |wx: f32, wz: f32| -> &RegionPlan {
        let key = (region_index(wx), region_index(wz));
        plans
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, p)| p)
            .expect("region window covers the requested voxel halo")
    };
    let mut lights = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut collect_runtime_lights = |plan: &RegionPlan| {
        for a in &plan.assemblies {
            if a.corruption.abandoned {
                continue;
            }
            for f in &a.fixtures {
                if !f.lit {
                    continue;
                }
                let rendered_at = recursive_level
                    .map_or(f.at, |recursive| recursive.project_position(f.at));
                let key = (rendered_at.x.to_bits(), rendered_at.z.to_bits());
                if !seen.insert(key) {
                    continue;
                }

                let (sample_at, fixture_plan) = if let Some(recursive) = recursive_level {
                    recursive
                        .plan_at(rendered_at)
                        .expect("recursive region window covers its projected fixtures")
                } else {
                    (f.at, plan_of(f.at.x, f.at.z))
                };

                // Region-scale anomaly interiors own their fixture rhythm.
                // Do not leak an overwritten assembly's runtime light into a
                // blackout/pillar/pit payload.
                if fixture_plan.anomalies.iter().any(|anomaly| {
                    anomaly.kind != AnomalyKind::RedRoom && anomaly.contains(sample_at.x, sample_at.z)
                }) {
                    continue;
                }

                let cx = rendered_at.x - chunk_pos.x;
                let cz = rendered_at.z - chunk_pos.z;
                if cx >= -15.0
                    && cx <= config.chunk_size + 15.0
                    && cz >= -15.0
                    && cz <= config.chunk_size + 15.0
                {
                    let ceiling_units = a
                        .ceiling_zones
                        .iter()
                        .find(|z| z.area.contains(f.at.x, f.at.z))
                        .map(|z| z.height_units)
                        .unwrap_or(4.0);
                    let is_atrium = matches!(
                        a.program,
                        crate::domain::entities::architecture::SpaceProgram::Atrium
                    );
                    let y = runtime_light_height(ceiling_units, is_atrium);

                    let kind = if f.half_x > f.half_z * 2.0 || f.half_z > f.half_x * 2.0 {
                        LightKind::Strip
                    } else {
                        LightKind::CeilingPanel
                    };

                    let is_buried = if let Some(recursive) = recursive_level {
                        BackroomsLevel::plan_column_in_reality(
                            fixture_plan,
                            noise,
                            recursive.seed(),
                            recursive.config(),
                            reality,
                            sample_at.x,
                            sample_at.z,
                        )
                        .solid
                    } else {
                        BackroomsLevel::plan_column_in_reality(
                            fixture_plan,
                            noise,
                            seed,
                            config,
                            reality,
                            sample_at.x,
                            sample_at.z,
                        )
                        .solid
                    };

                    if !is_buried {
                        // Red rooms are exposed purely by their light color;
                        // the fixtures keep the room's spacing.
                        let rgb = if recursive_level.is_some() || a.corruption.red_room {
                            [1.0, 0.22, 0.16]
                        } else {
                            [1.0, 0.95, 0.8]
                        };
                        lights.push(RuntimeLight {
                            world_pos: [rendered_at.x, y, rendered_at.z],
                            half_size: [f.half_x, f.half_z],
                            rgb,
                            range: if is_atrium { 24.0 } else { 16.0 },
                            intensity: if is_atrium { 4.0 } else { 1.0 },
                            enabled: true,
                            kind,
                        });
                    }
                }
            }
        }
    };
    if let Some(recursive) = recursive_level {
        for (_, plan) in recursive.iter() {
            collect_runtime_lights(plan);
        }
    } else {
        for (_, plan) in plans {
            collect_runtime_lights(plan);
        }
    }
    drop(collect_runtime_lights);
    lights
}

impl BackroomsLevel {

    /// Region plans for every region a chunk (plus a margin) overlaps.
    pub(crate) fn region_plans_for(
        chunk_pos: Position,
        chunk_size: f32,
        seed: u32,
        config: &GeneratorConfig,
        noise: &dyn NoiseProvider,
    ) -> Vec<((i64, i64), RegionPlan)> {
        InfiniteRegionWindow::around_chunk(chunk_pos, chunk_size, 1.0, seed, config, noise)
            .iter()
            .map(|(key, plan)| (key, plan.clone()))
            .collect()
    }
}

impl LevelGenerator for BackroomsLevel {
    fn generate(
        &self,
        chunk_pos: Position,
        seed: u32,
        config: GeneratorConfig,
        noise: &dyn NoiseProvider,
    ) -> VoxelGrid {
        self.generate_with_reality(chunk_pos, seed, config, noise, &RealitySnapshot::empty())
    }

    fn generate_with_reality(
        &self,
        chunk_pos: Position,
        seed: u32,
        config: GeneratorConfig,
        noise: &dyn NoiseProvider,
        reality: &RealitySnapshot,
    ) -> VoxelGrid {
        let s = config.voxel_scale;
        let width = (config.chunk_size / s).round() as usize;
        let depth = (config.chunk_size / s).round() as usize;
        let height = (GRID_HEIGHT_UNITS / s) as usize;
        let mut grid = VoxelGrid::new(width, height, depth);

        // Base plans remain available for the authored threshold and for
        // ordinary reality. A committed Red Room adds a second, explicitly
        // addressed Level 0 window instead of smuggling offsets and seeds
        // through the voxel loop.
        let plans =
            BackroomsLevel::region_plans_for(chunk_pos, config.chunk_size, seed, &config, noise);
        let recursive_level = RecursiveLevelWindow::around_chunk(
            chunk_pos,
            config.chunk_size,
            1.0,
            seed,
            config,
            noise,
            reality,
        );

        // Architecture first: plan every region this chunk overlaps.
        let plan_of = |wx: f32, wz: f32| -> &RegionPlan {
            let key = (region_index(wx), region_index(wz));
            plans
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, p)| p)
                .expect("region window covers the requested voxel halo")
        };

        // Export semantic crossings/hazards/lights from the same immutable
        // plans as voxel geometry, decided before anything is voxelized —
        // see decide_traversal_gates_and_hazards and decide_runtime_lights.
        let chunk_bounds = WorldBounds::new(
            chunk_pos.x,
            chunk_pos.z,
            chunk_pos.x + config.chunk_size,
            chunk_pos.z + config.chunk_size,
        );
        let (gates, hazards) =
            decide_traversal_gates_and_hazards(&plans, recursive_level.as_ref(), chunk_bounds);
        grid.traversal_gates.extend(gates);
        grid.pit_hazards.extend(hazards);
        grid.runtime_lights.extend(decide_runtime_lights(
            &plans,
            recursive_level.as_ref(),
            chunk_pos,
            &config,
            seed,
            noise,
            reality,
        ));

        // Plan every column plus a 1-voxel margin: ceiling skirts must seal
        // height steps across chunk borders too.
        let plan_at = |lx: i64, lz: i64| -> ColumnPlan {
            let wx = chunk_pos.x + (lx as f32 + 0.5) * s;
            let wz = chunk_pos.z + (lz as f32 + 0.5) * s;

            if let Some(recursive) = &recursive_level {
                let (recursive_point, plan) = recursive
                    .plan_at(Position::new(wx, wz))
                    .expect("recursive region window covers the voxel halo");
                let mut column = BackroomsLevel::plan_column_in_reality(
                    plan,
                    noise,
                    recursive.seed(),
                    recursive.config(),
                    reality,
                    recursive_point.x,
                    recursive_point.z,
                );
                column.red_light = true;
                column
            } else {
                BackroomsLevel::plan_column_in_reality(
                    plan_of(wx, wz),
                    noise,
                    seed,
                    &config,
                    reality,
                    wx,
                    wz,
                )
            }
        };

        let columns = ColumnField::sample(width, depth, plan_at);
        voxelize_columns(&mut grid, &columns, s);

        // Provisions live only in ordinary Level 0 space: a committed Red
        // Room's recursive address stays barren by design.
        if recursive_level.is_none() {
            super::provisions::stamp_level_zero_provisions(
                &mut grid,
                chunk_pos,
                &super::provisions::ProvisionContext {
                    seed,
                    config: &config,
                    reality,
                    noise,
                    plans: &plans,
                },
            );
        }

        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frameworks_drivers::simple_noise::SimpleNoiseProvider;

    /// The decide-stage functions are exercised directly against real
    /// generated plans — no `VoxelGrid` involved at all — confirming they
    /// are pure and repeatable (the same plans always decide the same
    /// gates/hazards/lights) and that the id-based deduplication they exist
    /// to do actually holds: neighboring region plans repeat macro anomaly
    /// instances, so a naive per-plan export would double-count them.
    #[test]
    fn traversal_gates_and_hazards_are_deterministic_and_deduplicated() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let plans =
            BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 80.0, 42, &config, &noise);
        let chunk_bounds = WorldBounds::new(0.0, 0.0, 80.0, 80.0);

        let (gates_a, hazards_a) = decide_traversal_gates_and_hazards(&plans, None, chunk_bounds);
        let (gates_b, hazards_b) = decide_traversal_gates_and_hazards(&plans, None, chunk_bounds);
        assert_eq!(gates_a, gates_b, "gate export is not deterministic");
        assert_eq!(hazards_a, hazards_b, "hazard export is not deterministic");

        let mut gate_ids: Vec<_> = gates_a.iter().map(|g| g.id).collect();
        gate_ids.sort_unstable();
        gate_ids.dedup();
        assert_eq!(
            gate_ids.len(),
            gates_a.len(),
            "a traversal gate id was exported more than once"
        );

        let mut hazard_ids: Vec<_> = hazards_a.iter().map(|h| h.id).collect();
        hazard_ids.sort_unstable();
        hazard_ids.dedup();
        assert_eq!(
            hazard_ids.len(),
            hazards_a.len(),
            "a pit hazard id was exported more than once"
        );
    }

    #[test]
    fn runtime_lights_are_deterministic_and_deduplicated_by_position() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let reality = RealitySnapshot::empty();
        let mut any_lights = false;

        // Runtime lights come only from authored-assembly fixtures within
        // chunk_size (+15u margin) of chunk_pos, so any single chunk can
        // legitimately have none; sample several real chunks on the actual
        // chunk grid (chunk_pos and config.chunk_size must agree, exactly
        // as generate_with_reality itself always calls this) to confirm
        // the dedup/determinism property on a real, non-trivial result at
        // least once.
        for chunk in 0..40i64 {
            let chunk_pos = Position::new(chunk as f32 * config.chunk_size, 0.0);
            let plans =
                BackroomsLevel::region_plans_for(chunk_pos, config.chunk_size, 42, &config, &noise);

            let lights_a =
                decide_runtime_lights(&plans, None, chunk_pos, &config, 42, &noise, &reality);
            let lights_b =
                decide_runtime_lights(&plans, None, chunk_pos, &config, 42, &noise, &reality);
            assert_eq!(
                lights_a.len(),
                lights_b.len(),
                "light export is not deterministic for chunk {chunk}"
            );

            let mut positions: Vec<_> = lights_a
                .iter()
                .map(|l| (l.world_pos[0].to_bits(), l.world_pos[2].to_bits()))
                .collect();
            positions.sort_unstable();
            let before = positions.len();
            positions.dedup();
            assert_eq!(
                before,
                positions.len(),
                "a fixture position was exported more than once in chunk {chunk}"
            );

            any_lights |= !lights_a.is_empty();
        }
        assert!(any_lights, "no runtime lights found across 40 sampled chunks");
    }
}
