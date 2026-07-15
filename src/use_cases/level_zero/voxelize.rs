//! Materialize sampled Level 0 columns at the requested voxel size.

use crate::domain::entities::voxel_grid::{VOXEL_CEILING, VOXEL_RED_LIGHT, VoxelGrid};
use crate::use_cases::level_zero::ColumnField;

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
                // Raised floor (stair treads, landings) is solid mass in
                // wall material: collision derives from solid voxels, so a
                // tread must block like architecture, not paint like carpet.
                let raised = (plan.floor_units / voxel_size).round() as usize;
                for y in 1..=raised.min(max_y) {
                    grid.set(x, y, z, plan.wall_material);
                }
            }

            if plan.solid {
                for y in 1..ceiling_y {
                    grid.set(x, y, z, plan.wall_material);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::voxel_grid::{EMISSIVE_MATERIALS, VOXEL_FLOOR, VOXEL_WALL};
    use crate::use_cases::level_zero::ColumnPlan;

    /// Level 0 currently has one authored emitter contract: fixtures are
    /// exposed panels in open ceiling columns. A fixture request on a solid
    /// column must therefore remain ordinary architecture, never a buried
    /// emissive voxel or a second light on the floor.
    #[test]
    fn solid_columns_cannot_voxelize_fake_emitters() {
        let mut column = ColumnPlan::open(4.0);
        column.solid = true;
        column.light = true;
        column.red_light = true;
        column.light_material = VOXEL_RED_LIGHT;

        let field = ColumnField::sample(1, 1, |_, _| column);
        let mut grid = VoxelGrid::new(1, 6, 1);
        voxelize_columns(&mut grid, &field, 1.0);

        assert_eq!(grid.get(0, 0, 0), VOXEL_FLOOR);
        for y in 0..grid.height() {
            let material = grid.get(0, y, 0);
            assert!(
                !EMISSIVE_MATERIALS.contains(&material),
                "solid column contains buried emitter {material} at y={y}"
            );
        }
        for y in 1..=4 {
            let material = grid.get(0, y, 0);
            assert_eq!(material, VOXEL_WALL);
        }
    }
}
