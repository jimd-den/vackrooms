//! Column geometry for immutable, non-Red-Room anomaly families.
//!
//! The region planner decides *where* an anomaly exists. This module decides
//! what one world-space column inside that immutable plan looks like. Red Rooms
//! remain a separate encounter feature because their geometry is coupled to a
//! dedicated phase machine and recursive Level 0 destination.

use crate::domain::entities::anomaly::{
    AnomalyInstance, AnomalyKind, ArchBehavior, ArchLayout, RealitySnapshot,
};
use crate::domain::entities::environment::{EnvironmentProfile, FloorState};
use crate::domain::entities::voxel_grid::{VOXEL_FLUID, VOXEL_GLIMMER, VOXEL_LIGHT};
use crate::use_cases::level_zero::{BackroomsLevel, ColumnPlan};
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::region_plan::PLAN_WALL_T;

const LIGHT_PERIOD: f32 = 2.8;

/// Mutable anomaly infill uses a stable content tile for its seam guard. The
/// tile is a world-generation constant, never the size of an output request.
const GENERATION_TILE_UNITS: f32 = 10.0;

struct SampleContext<'a> {
    instance: &'a AnomalyInstance,
    noise: &'a dyn NoiseProvider,
    seed: u32,
    config: &'a GeneratorConfig,
    reality: &'a RealitySnapshot,
    world_x: f32,
    world_z: f32,
    local_x: f32,
    local_z: f32,
    boundary: f32,
    perimeter: bool,
    skeleton: bool,
    away_from_seam: bool,
}

/// Sample one immutable macro anomaly or stable archway anchor.
///
/// Corridors have already won composition priority before this function is
/// called, so every family may assume its protected bearing route is the only
/// remaining cross-cutting circulation concern. Blackout expanses delegate
/// their substrate to the ordinary Level 0 fabric sampler before applying the
/// anomaly profile.
pub(crate) fn sample_anomaly(
    instance: &AnomalyInstance,
    noise: &dyn NoiseProvider,
    seed: u32,
    config: &GeneratorConfig,
    reality: &RealitySnapshot,
    world_x: f32,
    world_z: f32,
) -> ColumnPlan {
    let (local_x, local_z) = instance.local_coords(world_x, world_z);
    let boundary = instance.boundary_distance(world_x, world_z);
    let seam_x = world_x.rem_euclid(GENERATION_TILE_UNITS);
    let seam_z = world_z.rem_euclid(GENERATION_TILE_UNITS);
    let context = SampleContext {
        instance,
        noise,
        seed,
        config,
        reality,
        world_x,
        world_z,
        local_x,
        local_z,
        boundary,
        perimeter: boundary <= PLAN_WALL_T,
        skeleton: local_z.abs() <= instance.skeleton_half_width,
        away_from_seam: seam_x > PLAN_WALL_T * 2.0
            && seam_x < GENERATION_TILE_UNITS - PLAN_WALL_T * 2.0
            && seam_z > PLAN_WALL_T * 2.0
            && seam_z < GENERATION_TILE_UNITS - PLAN_WALL_T * 2.0,
    };

    match instance.kind {
        AnomalyKind::PillarExpanse => sample_pillar_expanse(&context),
        AnomalyKind::BlackoutExpanse => sample_blackout_expanse(&context),
        AnomalyKind::PitLattice => sample_pit_lattice(&context),
        AnomalyKind::ArchwayRoom => sample_archway_anchor(&context),
        AnomalyKind::RedRoom => unreachable!("red rooms have a dedicated geometry sampler"),
    }
}

fn anomaly_hash(instance: &AnomalyInstance, epoch: u32, salt: u64, x: i64, z: i64) -> f32 {
    let mut hash = instance.id
        ^ salt
        ^ (epoch as u64).rotate_left(19)
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (z as u64).rotate_left(37);
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94D0_49BB_1331_11EB);
    hash ^= hash >> 31;
    (hash >> 40) as f32 / (1u64 << 24) as f32
}

fn remap_epoch(context: &SampleContext<'_>) -> Option<u32> {
    let state = context.instance.state(context.reality)?;
    (state.epoch > 0
        && context.boundary > context.instance.entry_band
        && state.point_is_in_wake(
            context.world_x,
            context.world_z,
            context.config.anomalies.remap_distance,
        ))
    .then_some(state.epoch)
}

fn sample_pillar_expanse(context: &SampleContext<'_>) -> ColumnPlan {
    let instance = context.instance;
    let tuning = &context.config.tuning;
    let (local_x, local_z) = (context.local_x, context.local_z);
    let lattice = instance.pillar_lattice.expect("pillar instance lattice");
    let cell_x = ((local_x - lattice.phase_x) / lattice.bay_x).floor() as i64;
    let cell_z = ((local_z - lattice.phase_z) / lattice.bay_z).floor() as i64;
    let mut within_x = (local_x - lattice.phase_x).rem_euclid(lattice.bay_x);
    let within_z = (local_z - lattice.phase_z).rem_euclid(lattice.bay_z);
    let mut side = instance.pillar_size(cell_x, cell_z).unwrap_or(1.2);

    // Distortion grows only after the entry has taught the player a regular
    // grid. The route and entry band therefore remain a reliable baseline.
    let budget = ((context.boundary - instance.entry_band) / (instance.entry_band.max(8.0) * 2.5))
        .clamp(0.0, 1.0);
    let immutable_zone = context.skeleton || budget <= 0.0;
    let mut dropped = false;
    if !immutable_zone {
        dropped = anomaly_hash(instance, 0, 0x11D0, cell_x, cell_z) < 0.10 * budget;
        if anomaly_hash(instance, 0, 0x11D1, cell_z, 0) < 0.30 * budget {
            within_x = (within_x + lattice.bay_x * 0.5).rem_euclid(lattice.bay_x);
        }
        if anomaly_hash(instance, 0, 0x11D2, cell_x, cell_z) < 0.08 * budget {
            side += 0.4;
        }
    }

    // A committed wake may change relationships between bays, but never at a
    // stable content-tile seam where resident reality cohorts meet.
    let wake_epoch = (!immutable_zone && context.away_from_seam)
        .then(|| remap_epoch(context))
        .flatten();
    if let Some(epoch) = wake_epoch {
        if anomaly_hash(instance, epoch, 0x11F4, cell_x, cell_z) < 0.25 {
            side += 0.4;
        }
        if anomaly_hash(instance, epoch, 0x11F5, cell_z, cell_x) < 0.35 {
            within_x = (within_x + 0.4).rem_euclid(lattice.bay_x);
        }
    }

    let dx = within_x.min(lattice.bay_x - within_x);
    let dz = within_z.min(lattice.bay_z - within_z);
    let mut solid = context.perimeter || (!dropped && dx < side * 0.5 && dz < side * 0.5);
    if !solid && let Some(epoch) = wake_epoch {
        let edge_x = within_x < PLAN_WALL_T || lattice.bay_x - within_x < PLAN_WALL_T;
        let edge_z = within_z < PLAN_WALL_T || lattice.bay_z - within_z < PLAN_WALL_T;
        let threshold = (0.12 * context.config.anomalies.remap_intensity).clamp(0.0, 0.48);
        solid = (edge_x && anomaly_hash(instance, epoch, 0x11F1, cell_x, cell_z) < threshold)
            || (edge_z && anomaly_hash(instance, epoch, 0x11F2, cell_x, cell_z) < threshold);
    }
    if context.skeleton {
        solid = false;
    }

    let lane_light =
        local_z.abs() < 0.45 && (local_x - 2.4 + 0.45).rem_euclid(4.8) < 0.9 && tuning.lights > 0.0;
    let field_light = (local_x - 2.4 + 0.45).rem_euclid(4.8) < 0.9
        && (local_z + 0.45).rem_euclid(4.8) < 0.9
        && anomaly_hash(instance, 0, 0x11A7, cell_x, cell_z) < 0.62 * tuning.lights;
    let ceiling_units = if !immutable_zone
        && anomaly_hash(instance, 0, 0x11D3, cell_x / 2, cell_z / 2) < 0.35 * budget
    {
        4.0
    } else {
        4.2
    };
    ColumnPlan {
        light: !solid && (lane_light || field_light),
        solid,
        ceiling_units,
        floor_material: EnvironmentProfile::pillar_expanse().floor_voxel(),
        ..ColumnPlan::open(ceiling_units)
    }
}

fn sample_blackout_expanse(context: &SampleContext<'_>) -> ColumnPlan {
    let instance = context.instance;
    let tuning = &context.config.tuning;
    let (local_x, local_z) = (context.local_x, context.local_z);
    // The blackout substrate is ordinary fabric under the profile — and it
    // drifts under the same Peripheral Shift stamps as ordinary fabric. In
    // here the engine also advances drift *behind the player in real time*
    // (the darkness hides the swap), so a blackout rearranges more cruelly
    // than the level outside it: turning around is never a way back.
    let mut plan = BackroomsLevel::column_plan_in_reality(
        context.noise,
        context.seed ^ instance.id as u32,
        tuning,
        context.reality,
        context.world_x,
        context.world_z,
    );
    let depth = instance.normalized_depth(context.world_x, context.world_z);
    let profile = EnvironmentProfile::blackout(depth);
    plan.wall_material = profile.wall_voxel();
    plan.light_material = profile.light_voxel();

    // Ordinary fixtures thin through the approach instead of cutting directly
    // to black, so the darkness is spatially earned.
    if depth <= 0.22 {
        let cell_x = (context.world_x / LIGHT_PERIOD).floor() as i64;
        let cell_z = (context.world_z / LIGHT_PERIOD).floor() as i64;
        let keep = 1.0 - depth / 0.22;
        plan.light = plan.light && anomaly_hash(instance, 0, 0xFADE, cell_x, cell_z) < keep;
        plan.light_material = VOXEL_LIGHT;
    } else {
        plan.light = false;
    }
    plan.red_light = false;
    plan.ceiling_units = if depth > 0.68 {
        2.6
    } else if depth > 0.35 {
        3.0
    } else {
        3.4
    };

    if profile.floor == FloorState::RecessedFluid && !plan.solid && !context.skeleton {
        let cell = 7.2;
        let basin_x = (local_x / cell).floor() as i64;
        let basin_z = (local_z / cell).floor() as i64;
        if anomaly_hash(instance, 0, 0xF10D, basin_x, basin_z) < 0.22
            && local_x.rem_euclid(cell) > 1.2
            && local_z.rem_euclid(cell) > 1.2
        {
            plan.floor_material = VOXEL_FLUID;
        }
    }

    plan.solid |= context.perimeter;
    if context.skeleton {
        plan.solid = false;
        plan.light = local_x.rem_euclid(28.0) < 0.45 && tuning.lights > 0.0;
        plan.light_material = VOXEL_GLIMMER;
        return plan;
    }

    let decoys = context.config.anomalies.blackout_decoys.clamp(0.0, 0.5);
    let off_lane = (local_z.abs() - (instance.skeleton_half_width + 7.2)).abs() < 0.45;
    if off_lane
        && depth > 0.3
        && local_x.rem_euclid(28.0) < 0.45
        && tuning.lights > 0.0
        && anomaly_hash(
            instance,
            0,
            0xDEC0,
            (local_x / 28.0).floor() as i64,
            local_z.is_sign_negative() as i64,
        ) < decoys
    {
        plan.light = true;
        plan.light_material = VOXEL_GLIMMER;
    }

    if context.away_from_seam
        && let Some(epoch) = remap_epoch(context)
    {
        let cell = 7.2;
        let cell_x = (local_x / cell).floor() as i64;
        let cell_z = (local_z / cell).floor() as i64;
        let within_x = local_x.rem_euclid(cell);
        let within_z = local_z.rem_euclid(cell);
        let edge = within_x < PLAN_WALL_T || within_z < PLAN_WALL_T;
        let threshold = (0.20 * context.config.anomalies.remap_intensity).clamp(0.0, 0.65);
        if edge && anomaly_hash(instance, epoch, 0xB1AC, cell_x, cell_z) < threshold {
            plan.solid = true;
            plan.light = false;
        }
    }

    // The dark owns a structural layer the Peripheral Shift re-deals: past
    // the approach band, room-scale partitions stand on the instance's own
    // lattice. Their epoch comes from the fabric drift stamps, so the
    // engine's rear shift does not merely nudge 0.4 u fabric bands — it
    // rearranges *rooms* behind the player. Every drawn partition is
    // pierced by its own doorway, so no shift can seal space it creates,
    // and the recovery skeleton stays exempt below.
    if depth > 0.25 && !context.skeleton && tuning.walls > 0.0 {
        const PARTITION_CELL: f32 = 9.6;
        let cell_x = (local_x / PARTITION_CELL).floor() as i64;
        let cell_z = (local_z / PARTITION_CELL).floor() as i64;
        let anchor = instance.world_coords(
            (cell_x as f32 + 0.5) * PARTITION_CELL,
            (cell_z as f32 + 0.5) * PARTITION_CELL,
        );
        let epoch = context.reality.fabric_drift_epoch(anchor.x, anchor.z);
        let within_x = local_x.rem_euclid(PARTITION_CELL);
        let within_z = local_z.rem_euclid(PARTITION_CELL);
        for (in_band, along, wall_salt, door_salt) in [
            (within_x < PLAN_WALL_T, within_z, 0xB1D0u64, 0xB1D1u64),
            (within_z < PLAN_WALL_T, within_x, 0xB1D2u64, 0xB1D3u64),
        ] {
            if !in_band {
                continue;
            }
            if anomaly_hash(instance, epoch, wall_salt, cell_x, cell_z) < 0.5 {
                let gap = anomaly_hash(instance, epoch, door_salt, cell_x, cell_z);
                let gap_pos = 0.8 + (PARTITION_CELL - 3.2) * gap;
                if along < gap_pos || along >= gap_pos + 1.6 {
                    plan.solid = true;
                    plan.light = false;
                }
            }
        }
    }
    plan
}

fn sample_pit_lattice(context: &SampleContext<'_>) -> ColumnPlan {
    let lattice = context.instance.pit_lattice.expect("pit instance lattice");
    let within_x = (context.local_x - lattice.phase_x).rem_euclid(lattice.spacing_x);
    let within_z = (context.local_z - lattice.phase_z).rem_euclid(lattice.spacing_z);
    let dx = within_x.min(lattice.spacing_x - within_x);
    let dz = within_z.min(lattice.spacing_z - within_z);
    let pit = !context.skeleton
        && context.boundary > lattice.side
        && dx < lattice.side * 0.5
        && dz < lattice.side * 0.5;
    ColumnPlan {
        floor: !pit,
        solid: context.perimeter,
        light: !context.perimeter
            && context.local_x.rem_euclid(3.2) < 0.45
            && context.local_z.rem_euclid(3.2) < 0.45
            && context.config.tuning.lights > 0.0,
        ..ColumnPlan::open(3.8)
    }
}

/// Stable pale arch rooms never inspect `RealitySnapshot`, which keeps them a
/// fixed landmark against every mutable anomaly family.
///
/// A `Transition` room is a *graph connector*: its `ArchBehavior` decides
/// what its far half (local `z > 0`) belongs to. The seam is expressed only
/// through legal architectural vocabulary — ceiling regime, floor finish,
/// lintel height, fixture rhythm — never through impossible collision or a
/// repainted wall, so the crossing is felt before it is understood.
fn sample_archway_anchor(context: &SampleContext<'_>) -> ColumnPlan {
    let instance = context.instance;
    let tuning = &context.config.tuning;
    let (local_x, local_z) = (context.local_x, context.local_z);
    let arch = instance.arch.expect("arch instance profile");
    let profile = EnvironmentProfile::archway_room();
    let half_x = instance.footprint.half_x;
    let half_z = instance.footprint.half_z;

    // The far half of a seam room disagrees with the near half.
    let far_side = local_z > 0.0;
    let (ceiling_units, floor_material) = match arch.behavior {
        ArchBehavior::CultureSeam if far_side => {
            (3.0, EnvironmentProfile::pillar_expanse().floor_voxel())
        }
        ArchBehavior::ScaleBreach if far_side => (4.4, profile.floor_voxel()),
        _ => (3.6, profile.floor_voxel()),
    };
    let mut plan = ColumnPlan {
        wall_material: profile.wall_voxel(),
        floor_material,
        ..ColumnPlan::open(ceiling_units)
    };

    let bay_index = ((local_x + half_x) / arch.bay).floor() as i64;
    let along = (local_x + half_x).rem_euclid(arch.bay);
    // The far arcade of a scale breach broadens its openings; a culture
    // seam presses its lintels lower. Both stay on the shared bay rhythm so
    // the two walls visibly disagree about the same structure.
    let effective_opening = match arch.behavior {
        ArchBehavior::ScaleBreach if far_side => arch.opening * 1.5,
        _ => arch.opening,
    };
    let head_drop = match arch.behavior {
        ArchBehavior::CultureSeam if far_side => 0.4,
        _ => 0.0,
    };
    let across_opening = (along - arch.bay * 0.5).abs() < effective_opening * 0.5;
    let blind = bay_index.rem_euclid(arch.blind_every as i64) == arch.blind_every as i64 - 1;
    let arch_head = {
        let t = ((along - arch.bay * 0.5).abs() / (effective_opening * 0.5)).clamp(0.0, 1.0);
        (((arch.spring_units - head_drop) + 0.6 * (1.0 - t)) / 0.2).round() * 0.2
    };

    if context.perimeter {
        plan.solid = tuning.walls > 0.0;
        let on_long_wall = half_z - local_z.abs() <= PLAN_WALL_T;
        match arch.layout {
            ArchLayout::Transition => {
                if on_long_wall && across_opening && !blind {
                    plan.solid = false;
                    plan.lintel_from_units = (tuning.walls > 0.0).then_some(arch_head);
                }
            }
            ArchLayout::DeadEnd => {
                let on_entrance_wall = half_x - local_x.abs() <= PLAN_WALL_T && local_x < 0.0;
                if on_entrance_wall && local_z.abs() < 1.0 {
                    plan.solid = false;
                    plan.lintel_from_units = (tuning.walls > 0.0).then_some(
                        ((arch.spring_units + 0.6 * (1.0 - local_z.abs())) / 0.2).round() * 0.2,
                    );
                }
            }
        }
        return plan;
    }

    if arch.layout == ArchLayout::DeadEnd {
        let arcade_plane = half_z - 1.2 - PLAN_WALL_T;
        let in_arcade = (local_z.abs() - arcade_plane).abs() < PLAN_WALL_T * 0.5;
        let in_alcove = local_z.abs() > arcade_plane;
        if in_arcade && tuning.walls > 0.0 {
            if across_opening && !blind {
                plan.lintel_from_units = Some(arch_head);
            } else {
                plan.solid = true;
            }
        } else if in_alcove {
            plan.ceiling_units = 3.0;
            if ((local_x * 2.5).sin() * (local_z * 1.7).cos()).abs() < 0.3 {
                plan.floor_material = VOXEL_FLUID;
            }
        }
    }

    plan.light = local_z.abs() < 0.45
        && (local_x + half_x - arch.bay * 0.5).rem_euclid(arch.bay * 2.0) < 0.45
        && tuning.lights > 0.0;
    let _ = context.boundary;
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::anomaly::{
        ArchProfile, OrientedFootprint, OrthoBasis, QuarterTurn,
    };
    use crate::entities::models::Position;
    use crate::frameworks_drivers::simple_noise::SimpleNoiseProvider;

    fn arch_room(behavior: ArchBehavior) -> AnomalyInstance {
        let basis = OrthoBasis {
            turn: QuarterTurn::Zero,
        };
        AnomalyInstance {
            id: 0xA5C4_0001,
            kind: AnomalyKind::ArchwayRoom,
            footprint: OrientedFootprint {
                center: Position::new(0.0, 0.0),
                half_x: 6.0,
                half_z: 6.0,
                basis,
            },
            basis,
            macro_anchor: (0, 0),
            pillar_lattice: None,
            pit_lattice: None,
            arch: Some(ArchProfile {
                layout: ArchLayout::Transition,
                behavior,
                bay: 2.8,
                opening: 1.6,
                blind_every: 4,
                spring_units: 2.0,
            }),
            gates: Vec::new(),
            skeleton_half_width: 0.0,
            entry_band: 0.0,
        }
    }

    fn interior(instance: &AnomalyInstance, local_z: f32) -> ColumnPlan {
        sample_anomaly(
            instance,
            &SimpleNoiseProvider::new(),
            42,
            &crate::use_cases::generate_chunk::GeneratorConfig::low_spec(),
            &RealitySnapshot::empty(),
            1.0,
            local_z,
        )
    }

    /// A seam-bearing transition arch is a graph edge you can read: its far
    /// half disagrees with its near half through legal architecture only.
    #[test]
    fn arch_behaviors_split_the_room_across_the_seam() {
        let anchor = arch_room(ArchBehavior::Anchor);
        assert_eq!(
            interior(&anchor, -3.0).ceiling_units,
            interior(&anchor, 3.0).ceiling_units,
            "an anchor room must stay symmetric"
        );

        let culture = arch_room(ArchBehavior::CultureSeam);
        let near = interior(&culture, -3.0);
        let far = interior(&culture, 3.0);
        assert!(
            far.ceiling_units < near.ceiling_units,
            "culture seam far side must change ceiling regime"
        );
        assert_ne!(
            far.floor_material, near.floor_material,
            "culture seam far side must change floor grammar"
        );

        let scale = arch_room(ArchBehavior::ScaleBreach);
        assert!(
            interior(&scale, 3.0).ceiling_units > interior(&scale, -3.0).ceiling_units,
            "scale breach far side must lift its ceiling"
        );
    }
}
