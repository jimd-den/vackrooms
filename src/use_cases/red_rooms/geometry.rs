//! Red Room presentation layered over an authored Level 0 assembly.
//!
//! This sampler knows nothing about chunks or voxel indices.  It receives one
//! world column and returns one architectural decision, leaving recursive
//! world addressing and voxelization to their own stages.

use crate::domain::entities::anomaly::{AnomalyInstance, Axis2, RealitySnapshot, RedRoomPhase};
use crate::domain::entities::environment::EnvironmentProfile;
use crate::domain::entities::voxel_grid::VOXEL_STICKY_CARPET;
use crate::use_cases::backrooms_level::ColumnPlan;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::region_plan::PLAN_WALL_T;

/// Applies avoidable warning cues and the committed encounter topology.
pub(crate) fn sample_red_room(
    instance: &AnomalyInstance,
    mut column: ColumnPlan,
    config: &GeneratorConfig,
    reality: &RealitySnapshot,
    world_x: f32,
    world_z: f32,
) -> ColumnPlan {
    let (local_x, local_z) = instance.local_coords(world_x, world_z);
    let half_x = instance.footprint.half_x;
    let half_z = instance.footprint.half_z;
    let loop_width = 2.0f32.min(half_x * 0.35).min(half_z * 0.35).max(1.2);
    let boundary = instance.boundary_distance(world_x, world_z);

    // Warning cues intensify toward the shell but never replace masonry with
    // a special red wall.  The anomaly is read through carpet, compression,
    // and light; collision remains ordinary Level 0 architecture.
    let contamination = (1.0 - boundary / (loop_width * 2.0)).clamp(0.0, 1.0);
    let profile = EnvironmentProfile::red_room(contamination);
    if boundary > PLAN_WALL_T + 0.05 {
        if contamination > 0.15 {
            column.floor_material = profile.floor_voxel();
        }
        if contamination > 0.33 {
            column.ceiling_units = (column.ceiling_units - 0.2).max(2.6);
        }
        if contamination > 0.66 {
            column.ceiling_units = (column.ceiling_units - 0.4).max(2.4);
        }
    }
    column.red_light = true;

    let Some(state) = instance.state(reality) else {
        return column;
    };
    if state.phase == RedRoomPhase::Outside {
        return column;
    }

    // Commitment closes the remembered entrance and reduces the authored
    // room to one walkable perimeter loop around an opaque core.
    let in_core = local_x.abs() < half_x - loop_width && local_z.abs() < half_z - loop_width;
    column.solid = boundary <= PLAN_WALL_T || in_core;
    column.lintel_from_units = None;
    column.floor_material = VOXEL_STICKY_CARPET;
    column.ceiling_units = column.ceiling_units.min(2.8);
    if column.solid {
        column.light = false;
    }

    if state.phase == RedRoomPhase::EscapeOpen {
        open_single_escape(
            instance,
            state.epoch,
            config.anomalies.red_escape_bias,
            local_x,
            local_z,
            &mut column,
        );
    }
    column
}

/// Opens one deterministic side of the loop.  The bias is an authored width
/// control: zero keeps the encounter sealed; positive values widen a single
/// breach without turning both symmetric walls into exits.
fn open_single_escape(
    instance: &AnomalyInstance,
    epoch: u32,
    bias: f32,
    local_x: f32,
    local_z: f32,
    column: &mut ColumnPlan,
) {
    let width = bias.clamp(0.0, 1.0) * 2.4;
    if width <= 0.0 {
        return;
    }
    let Some(threshold) = instance.gates.first() else {
        return;
    };
    let side_sign = if stable_bit(instance.id, epoch) {
        1.0
    } else {
        -1.0
    };
    let (side_coordinate, side_extent, along) = match threshold.axis {
        Axis2::Z => (local_x, instance.footprint.half_x, local_z),
        Axis2::X => (local_z, instance.footprint.half_z, local_x),
    };
    let on_selected_wall = (side_coordinate - side_sign * side_extent).abs() <= PLAN_WALL_T + 0.1;
    if on_selected_wall && along.abs() < width {
        column.solid = false;
    }
}

fn stable_bit(instance_id: u64, epoch: u32) -> bool {
    let mut value = instance_id ^ (epoch as u64).rotate_left(29) ^ 0xE5CA_9E5C_A9E5_CA9E;
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value & 1 == 1
}
