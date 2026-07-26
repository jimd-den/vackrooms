//! Materialize sampled Level 0 columns at the requested voxel size.

use crate::domain::entities::voxel_grid::{
    VOXEL_CEILING, VOXEL_METAL_DOOR, VOXEL_RED_LIGHT, VoxelGrid,
};
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
            if plan.door_leaf && !plan.solid {
                let door_top = to_voxel(plan.lintel_from_units.unwrap_or(2.2));
                for y in 1..door_top {
                    grid.set(x, y, z, VOXEL_METAL_DOOR);
                }
            }

            // A ceiling-height step can only ever be voxelized where a wall
            // or door header already reaches up to carry it: fabric_ceiling_
            // height snaps to the same FABRIC_CELL lattice the wall/lintel
            // decision reads, so a step is only possible exactly at a
            // boundary this column's own `solid`/`lintel_from_units` already
            // has an opinion about. Walls and headers legitimately grow to
            // whichever neighbour is taller — that is just a wall being tall
            // enough. An ordinary open column caps at its own height only:
            // that is a real step in the ceiling plane at a doorway or
            // opening, not a gap, and inventing a floating patch to hide it
            // would be unsupported geometry with nothing holding it up.
            let has_wall_support = plan.solid || plan.lintel_from_units.is_some();
            let cap_top = if has_wall_support {
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
                neighbour_ceiling.max(ceiling_y)
            } else {
                ceiling_y
            };
            let cap_material = if has_wall_support {
                plan.wall_material
            } else {
                VOXEL_CEILING
            };
            for y in ceiling_y..=cap_top {
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
    use crate::domain::entities::voxel_grid::{
        EMISSIVE_MATERIALS, VOXEL_AIR, VOXEL_FLOOR, VOXEL_WALL,
    };
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

    /// The bug this guards: two adjacent open (non-solid, no lintel)
    /// columns at different ceiling heights used to grow a `VOXEL_CEILING`
    /// skirt on the shorter column, bridging up to the taller neighbour's
    /// height with nothing underneath it — a ceiling patch floating over
    /// open, walkable floor. An open column must cap at its own height
    /// only; a real step in the ceiling plane at the boundary, not an
    /// unsupported patch.
    #[test]
    fn open_columns_never_grow_an_unsupported_ceiling_skirt() {
        let short = ColumnPlan::open(4.0);
        let tall = ColumnPlan::open(7.0);
        let field = ColumnField::sample(2, 1, |x, _z| if x == 0 { short } else { tall });
        let mut grid = VoxelGrid::new(2, 8, 1);
        voxelize_columns(&mut grid, &field, 1.0);

        // Column 0's own ceiling cap sits at y=4; nothing above it.
        assert_eq!(grid.get(0, 4, 0), VOXEL_CEILING);
        for y in 5..grid.height() {
            assert_eq!(
                grid.get(0, y, 0),
                VOXEL_AIR,
                "unsupported ceiling patch found at y={y} above an open column"
            );
        }
    }

    /// The support-preserving counterpart: a *solid* boundary column
    /// between two different ceiling heights must still reach the taller
    /// neighbour — that is an ordinary wall being tall enough to actually
    /// separate the two rooms, not the bug above.
    #[test]
    fn solid_boundary_columns_still_reach_the_taller_neighbour() {
        let mut wall = ColumnPlan::open(4.0);
        wall.solid = true;
        let tall = ColumnPlan::open(7.0);
        let field = ColumnField::sample(2, 1, |x, _z| if x == 0 { wall } else { tall });
        let mut grid = VoxelGrid::new(2, 8, 1);
        voxelize_columns(&mut grid, &field, 1.0);

        for y in 1..=7 {
            assert_eq!(
                grid.get(0, y, 0),
                VOXEL_WALL,
                "wall should reach the taller neighbour's ceiling at y={y}"
            );
        }
    }

    #[test]
    fn door_leaves_fill_the_opening_below_the_lintel() {
        let mut door = ColumnPlan::open(4.0);
        door.lintel_from_units = Some(2.2);
        door.door_leaf = true;
        let field = ColumnField::sample(1, 1, |_, _| door);
        let mut grid = VoxelGrid::new(1, 6, 1);
        voxelize_columns(&mut grid, &field, 1.0);

        assert_eq!(grid.get(0, 1, 0), VOXEL_METAL_DOOR);
        assert_eq!(grid.get(0, 2, 0), VOXEL_WALL, "lintel remains structural");
        assert_eq!(grid.get(0, 4, 0), VOXEL_WALL, "header reaches the ceiling");
    }
}
