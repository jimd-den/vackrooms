//! Materialize sampled Level 0 columns at the requested voxel size.

use crate::domain::entities::voxel_grid::{VOXEL_CEILING, VOXEL_LIGHT, VOXEL_RED_LIGHT, VoxelGrid};
use crate::use_cases::level_zero::ColumnField;

const SCONCE_HEIGHT_UNITS: f32 = 2.4;

/// Writes geometry and fixture voxels without knowing seeds, regions,
/// anomalies, chunks, or recursive branches.  Those are planning concerns;
/// this stage only quantizes an already sampled physical area.
pub(crate) fn voxelize_columns(grid: &mut VoxelGrid, field: &ColumnField, voxel_size: f32) {
    let max_y = grid.height().saturating_sub(1);
    let to_voxel = |units: f32| ((units / voxel_size) as usize).clamp(2, max_y);

    for z in 0..field.depth() {
        for x in 0..field.width() {
            let plan = *field.get(x as i64, z as i64);
            let ceiling_y = to_voxel(plan.ceiling_units);

            if plan.floor {
                grid.set(x, 0, z, plan.floor_material);
            }

            if plan.solid {
                for y in 1..ceiling_y {
                    grid.set(x, y, z, plan.wall_material);
                }
                if plan.sconce {
                    let y = to_voxel(SCONCE_HEIGHT_UNITS).min(ceiling_y - 1);
                    grid.set(x, y, z, VOXEL_LIGHT);
                }
            } else if let Some(lintel_height) = plan.lintel_from_units {
                for y in to_voxel(lintel_height)..ceiling_y {
                    grid.set(x, y, z, plan.wall_material);
                }
            }

            // A lower ceiling grows a skirt up to its tallest neighbour.  A
            // request therefore cannot see through a height step that lives
            // just across its output boundary.
            let neighbour_ceiling = [
                field.get(x as i64 - 1, z as i64),
                field.get(x as i64 + 1, z as i64),
                field.get(x as i64, z as i64 - 1),
                field.get(x as i64, z as i64 + 1),
            ]
            .iter()
            .map(|neighbour| to_voxel(neighbour.ceiling_units))
            .max()
            .unwrap_or(ceiling_y);

            let cap_material = if plan.solid {
                plan.wall_material
            } else {
                VOXEL_CEILING
            };
            for y in ceiling_y..=neighbour_ceiling.max(ceiling_y) {
                grid.set(x, y, z, cap_material);
            }

            if plan.light && !plan.solid {
                let material = if plan.red_light {
                    VOXEL_RED_LIGHT
                } else {
                    plan.light_material
                };
                grid.set(x, ceiling_y, z, material);
            }
        }
    }
}
