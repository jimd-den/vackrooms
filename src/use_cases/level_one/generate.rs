//! Level 1 chunk generation: sector-driven columns, supply exports, and the
//! arrival plaza with its door back to Level 0.

use crate::domain::entities::anomaly::{LevelExit, RealitySnapshot};
use crate::domain::entities::supplies::{SupplyItem, SupplyKind};
use crate::domain::entities::voxel_grid::{
    VOXEL_ALMOND_WATER, VOXEL_CONCRETE_FLOOR, VOXEL_CONCRETE_WALL, VOXEL_CRATE, VOXEL_METAL_DOOR,
    VOXEL_PIPE, VOXEL_TILE_FLOOR, VoxelGrid,
};
use crate::entities::models::Position;
use crate::use_cases::anomalies::determinism::{hash, unit};
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::level_generator::{LEVEL_BACKROOMS, LevelGenerator};
use crate::use_cases::level_zero::{ColumnField, ColumnPlan, voxelize_columns};
use crate::use_cases::ports::NoiseProvider;

use super::{GRID_HEIGHT_UNITS, HabitableLevel, Sector};

// -- structural lattices (world-phased so they cross sector borders) --------

/// Aquila / lighting bay: the deep-span parking grid.
const BAY: f32 = 9.6;
/// Massive square parking pillar side.
const PILLAR_SIDE: f32 = 1.6;
/// Gild / Gothic room lattice.
const ROOM: f32 = 7.2;
/// Structural wall thickness.
const WALL_T: f32 = 0.25;
/// Gild doorway width.
const DOOR_W: f32 = 1.4;
/// Gild doorway lintel height.
const DOOR_H: f32 = 2.2;
/// Doorway centers keep this margin from each wall end. The nearest gap
/// edge (margin - DOOR_W/2 = 1.9) must clear the deepest corner-crate
/// reach (1.1 inset + 0.75 max extent = 1.85).
const GAP_BAND_MARGIN: f32 = 2.6;
/// Construction obstacle lattice.
const SCAFFOLD: f32 = 4.8;
/// Gothic circular pillar radius.
const GOTHIC_R: f32 = 0.75;

/// Ceiling pipe runs appear on this lattice of world lines.
const PIPE_SPACING: f32 = 14.4;
const PIPE_HALF_W: f32 = 0.18;
const PIPE_DROP: f32 = 0.3;

// -- the arrival plaza -------------------------------------------------------

/// Radius kept clear around the world origin, where the Level 0 door lands.
const PLAZA_R: f32 = 5.6;
/// The door back to Level 0 stands just off the plaza center, placed to sit
/// wholly inside one chunk at every supported chunk size.
const RETURN_DOOR: (f32, f32) = (2.0, -2.0);
/// Where a wanderer stepping through the Level 0 door arrives (outside the
/// return door's own trigger, facing open floor).
pub(crate) const ARRIVAL_POINT: (f32, f32) = (3.0, 3.0);
/// Where the return door drops the wanderer in Level 0 (the spawn hub).
const LEVEL0_ARRIVAL: (f32, f32) = (5.0, 5.0);
/// Trigger half-extent of a level door.
pub(crate) const DOOR_TRIGGER_HALF: f32 = 0.7;

// -- supplies ----------------------------------------------------------------

/// Supply placement cell (independent of sector cells so caches don't align).
const SUPPLY_CELL: f32 = 24.0;

fn pillar_axis_dist(w: f32, spacing: f32) -> f32 {
    let t = w - (w / spacing).floor() * spacing;
    t.min(spacing - t)
}

/// Distance from a world point to the nearest lattice *node* of `spacing`.
fn node_dist(wx: f32, wz: f32, spacing: f32) -> f32 {
    let dx = pillar_axis_dist(wx, spacing);
    let dz = pillar_axis_dist(wz, spacing);
    (dx * dx + dz * dz).sqrt()
}

/// Stable id for a supply item at a lattice anchor.
fn supply_id(seed: u32, cx: i64, cz: i64, salt: u64) -> u64 {
    hash(seed, 0x5A17_AB1E_0000_0000 ^ salt, cx, cz) | 1
}

impl HabitableLevel {
    /// The architectural plan of one world column. Pure in (seed, position).
    pub(crate) fn plan_column(seed: u32, wx: f32, wz: f32) -> ColumnPlan {
        let sector = Sector::at(seed, wx, wz);
        let mut column = ColumnPlan::open(3.6);
        column.wall_material = VOXEL_CONCRETE_WALL;
        column.floor_material = VOXEL_TILE_FLOOR;

        match sector {
            Sector::Aquila => Self::plan_aquila(seed, wx, wz, &mut column),
            Sector::Gild => Self::plan_gild(seed, wx, wz, &mut column),
            Sector::Gothic => Self::plan_gothic(seed, wx, wz, &mut column),
            Sector::Construction => Self::plan_construction(seed, wx, wz, &mut column),
        }

        Self::apply_pipes(seed, wx, wz, &mut column);
        Self::apply_plaza(wx, wz, &mut column);
        column
    }

    /// Bland parking lot: concrete slab, huge pillars, long sight lines.
    fn plan_aquila(seed: u32, wx: f32, wz: f32, column: &mut ColumnPlan) {
        column.ceiling_units = 3.4;
        column.floor_material = VOXEL_CONCRETE_FLOOR;
        let half = PILLAR_SIDE * 0.5;
        if pillar_axis_dist(wx, BAY) < half && pillar_axis_dist(wz, BAY) < half {
            column.solid = true;
        }
        // Sparse fixtures at bay centers; roughly one in three bays is dark.
        let bx = ((wx - BAY * 0.5) / BAY).round() as i64;
        let bz = ((wz - BAY * 0.5) / BAY).round() as i64;
        let lit = unit(hash(seed, 0xA10A_11A5_0000_0002, bx, bz)) > 0.34;
        let lx = pillar_axis_dist(wx - BAY * 0.5, BAY);
        let lz = pillar_axis_dist(wz - BAY * 0.5, BAY);
        if lit && lx < 0.65 && lz < 0.65 {
            column.light = true;
        }
    }

    /// Dense storage fabric: doorway'd concrete rooms and crate stacks.
    fn plan_gild(seed: u32, wx: f32, wz: f32, column: &mut ColumnPlan) {
        column.ceiling_units = 3.2;

        let cell_x = (wx / ROOM).floor() as i64;
        let cell_z = (wz / ROOM).floor() as i64;
        let in_x_wall = pillar_axis_dist(wx, ROOM) < WALL_T;
        let in_z_wall = pillar_axis_dist(wz, ROOM) < WALL_T;

        if in_x_wall || in_z_wall {
            // A doorway pierces every shared wall segment, so the storage
            // maze is dense but never sealed. The gap position is hashed per
            // wall edge; both cells sharing the edge derive the same gap.
            // Gap centers stay in the wall's central band so a doorway (and
            // the walk lane through it) can never reach a corner crate zone.
            let (edge_x, edge_z, along, salt) = if in_x_wall {
                ((wx / ROOM).round() as i64, cell_z, wz, 0x11)
            } else {
                (cell_x, (wz / ROOM).round() as i64, wx, 0x22)
            };
            let t = unit(hash(seed, 0x91D0_D008_0000_0000 | salt, edge_x, edge_z));
            let gap_center =
                (along / ROOM).floor() * ROOM + GAP_BAND_MARGIN + t * (ROOM - 2.0 * GAP_BAND_MARGIN);
            let in_gap = (along - gap_center).abs() < DOOR_W * 0.5;
            if in_gap {
                column.lintel_from_units = Some(DOOR_H);
            } else {
                column.solid = true;
            }
        } else {
            // Crate stacks hug the room corners. Their maximum reach along
            // either wall (corner inset + extent) stays short of the nearest
            // possible doorway edge (GAP_BAND_MARGIN - DOOR_W/2), so a stack
            // can never stand in a doorway or its walk lane.
            let fx = wx - cell_x as f32 * ROOM;
            let fz = wz - cell_z as f32 * ROOM;
            let corner_x = if fx < ROOM * 0.5 { 1.1 } else { ROOM - 1.1 };
            let corner_z = if fz < ROOM * 0.5 { 1.1 } else { ROOM - 1.1 };
            let quadrant = (u64::from(fx >= ROOM * 0.5) << 1) | u64::from(fz >= ROOM * 0.5);
            let stack = hash(seed, 0xC0A7_E500_0000_0000 | quadrant, cell_x, cell_z);
            // ~55% of room corners hold crates: 1x1 to 1.5x1.5 world units.
            if unit(stack) < 0.55 {
                let extent = 0.5 + unit(stack.rotate_left(17)) * 0.25;
                if (fx - corner_x).abs() < extent && (fz - corner_z).abs() < extent {
                    column.floor_units = if unit(stack.rotate_left(33)) < 0.3 {
                        1.6
                    } else {
                        0.8
                    };
                    column.wall_material = VOXEL_CRATE;
                }
            }
        }

        // Room-center fixtures, dimmer rhythm than Level 0 offices.
        let lx = pillar_axis_dist(wx - ROOM * 0.5, ROOM);
        let lz = pillar_axis_dist(wz - ROOM * 0.5, ROOM);
        if lx < 0.55 && lz < 0.55 && unit(hash(seed, 0x91D0_11A5, cell_x, cell_z)) > 0.42 {
            column.light = true;
        }
    }

    /// Curved masonry: circular pillars joined by parabolic arches under a
    /// vaulted ceiling.
    fn plan_gothic(_seed: u32, wx: f32, wz: f32, column: &mut ColumnPlan) {
        column.ceiling_units = 4.8;

        if node_dist(wx, wz, ROOM) < GOTHIC_R {
            column.solid = true;
            return;
        }

        // Arches span every lattice line between adjacent pillars: the
        // soffit rises from the pillar capitals (2.2u) to 3.6u at midspan.
        let arch = |perp: f32, along: f32| -> Option<f32> {
            if perp >= WALL_T + 0.1 {
                return None;
            }
            let t = {
                let s = along - (along / ROOM).floor() * ROOM;
                (s / ROOM).clamp(0.0, 1.0)
            };
            let rise = (std::f32::consts::PI * t).sin();
            Some(2.2 + 1.4 * rise)
        };
        let x_line = pillar_axis_dist(wx, ROOM);
        let z_line = pillar_axis_dist(wz, ROOM);
        let soffit = match (arch(x_line, wz), arch(z_line, wx)) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        if let Some(h) = soffit {
            column.lintel_from_units = Some(h);
        }

        // Sparse warm pools of light at alternating lattice centers.
        let cx = (wx / ROOM).floor() as i64;
        let cz = (wz / ROOM).floor() as i64;
        let lx = pillar_axis_dist(wx - ROOM * 0.5, ROOM);
        let lz = pillar_axis_dist(wz - ROOM * 0.5, ROOM);
        if lx < 0.5 && lz < 0.5 && (cx + cz).rem_euclid(2) == 0 {
            column.light = true;
        }
    }

    /// Perpetually unfinished: half walls, rebar clusters, dead ends.
    fn plan_construction(seed: u32, wx: f32, wz: f32, column: &mut ColumnPlan) {
        let cell_x = (wx / SCAFFOLD).floor() as i64;
        let cell_z = (wz / SCAFFOLD).floor() as i64;
        let cell = hash(seed, 0xC095_7A0C_7100_0000, cell_x, cell_z);
        column.ceiling_units = 3.2 + unit(cell) * 1.4;
        column.floor_material = VOXEL_CONCRETE_FLOOR;

        // Every third lattice lane each way stays clear, so the chaos can
        // never seal the walkable plane.
        let clear_lane = cell_x.rem_euclid(3) == 0 || cell_z.rem_euclid(3) == 0;
        if clear_lane {
            return;
        }

        let in_x_wall = pillar_axis_dist(wx, SCAFFOLD) < WALL_T;
        let in_z_wall = pillar_axis_dist(wz, SCAFFOLD) < WALL_T;
        if in_x_wall || in_z_wall {
            let salt = if in_x_wall { 0x31 } else { 0x32 };
            let roll = unit(hash(seed, 0xBAD5_CAFF_0000_0000 | salt, cell_x, cell_z));
            if roll < 0.35 {
                column.solid = true; // a finished wall run (dead ends)
            } else if roll < 0.65 {
                column.floor_units = 1.1; // an unfinished half wall
            }
        } else {
            // Rebar clusters: thin full-height pipe bundles.
            let sub_x = (wx / 1.2).floor() as i64;
            let sub_z = (wz / 1.2).floor() as i64;
            if unit(hash(seed, 0x4EBA_4000_0000_0000, sub_x, sub_z)) < 0.045
                && pillar_axis_dist(wx, 1.2) < 0.12
                && pillar_axis_dist(wz, 1.2) < 0.12
            {
                column.solid = true;
                column.wall_material = VOXEL_PIPE;
            }
        }

        // Work lights are rare and mostly failing; long dark stretches are
        // the sector's character.
        let lx = pillar_axis_dist(wx - SCAFFOLD, SCAFFOLD * 2.0);
        let lz = pillar_axis_dist(wz - SCAFFOLD, SCAFFOLD * 2.0);
        if lx < 0.5 && lz < 0.5 && unit(cell.rotate_left(11)) < 0.4 {
            column.light = true;
        }
    }

    /// Exposed service pipes traverse every sector just below the ceiling.
    fn apply_pipes(seed: u32, wx: f32, wz: f32, column: &mut ColumnPlan) {
        if column.solid {
            return;
        }
        let run_z = (wz / PIPE_SPACING).round() as i64;
        let on_x_run = (wz - run_z as f32 * PIPE_SPACING).abs() < PIPE_HALF_W
            && unit(hash(seed, 0x9199_E000_0000_0001, run_z, 0)) < 0.7;
        let run_x = (wx / PIPE_SPACING).round() as i64;
        let on_z_run = (wx - run_x as f32 * PIPE_SPACING).abs() < PIPE_HALF_W
            && unit(hash(seed, 0x9199_E000_0000_0002, run_x, 1)) < 0.7;
        if on_x_run || on_z_run {
            let soffit = column.ceiling_units - PIPE_DROP;
            column.lintel_from_units = Some(
                column
                    .lintel_from_units
                    .map_or(soffit, |existing| existing.min(soffit)),
            );
            column.wall_material = VOXEL_PIPE;
        }
    }

    /// The arrival plaza: an always-open, always-lit clearing at the origin
    /// holding the door back to Level 0.
    fn apply_plaza(wx: f32, wz: f32, column: &mut ColumnPlan) {
        let d2 = wx * wx + wz * wz;
        if d2 > PLAZA_R * PLAZA_R {
            return;
        }
        column.solid = false;
        column.floor_units = 0.0;
        column.lintel_from_units = None;
        column.ceiling_units = column.ceiling_units.max(3.4);
        column.floor_material = VOXEL_TILE_FLOOR;
        column.wall_material = VOXEL_CONCRETE_WALL;
        // One dependable fixture above the arrival point.
        if (wx - ARRIVAL_POINT.0).abs() < 0.6 && (wz - ARRIVAL_POINT.1).abs() < 0.6 {
            column.light = true;
        }
    }

    /// Deterministic supply spot for one supply cell, or None. Supplies are
    /// canon-dense in Gild (crates), present in Aquila, rare elsewhere; the
    /// world tuning scales drink and food frequency independently.
    fn supply_for_cell(
        seed: u32,
        cx: i64,
        cz: i64,
        water: f32,
        food: f32,
    ) -> Option<SupplyItem> {
        let water_weight = 0.72 * water.clamp(0.0, 4.0);
        let food_weight = 0.28 * food.clamp(0.0, 4.0);
        if water_weight + food_weight <= 0.0 {
            return None;
        }
        let roll = hash(seed, 0x5A17_AB1E_00BB_0000, cx, cz);
        let origin_x = cx as f32 * SUPPLY_CELL;
        let origin_z = cz as f32 * SUPPLY_CELL;
        // Probe a few hashed candidate spots; accept the first that lands in
        // open, walkable floor (crate tops count — bottles sit on crates).
        for attempt in 0..6u64 {
            let probe = hash(seed, 0x5A17_AB1E_0000_0100 | attempt, cx, cz);
            let px = origin_x + 1.0 + unit(probe) * (SUPPLY_CELL - 2.0);
            let pz = origin_z + 1.0 + unit(probe.rotate_left(21)) * (SUPPLY_CELL - 2.0);
            let sector = Sector::at(seed, px, pz);
            let chance = match sector {
                Sector::Gild => 0.85,
                Sector::Aquila => 0.35,
                Sector::Gothic => 0.2,
                Sector::Construction => 0.15,
            } * (water_weight + food_weight);
            if unit(roll) > chance {
                return None;
            }
            let column = Self::plan_column(seed, px, pz);
            if column.solid || !column.floor {
                continue;
            }
            let kind = if unit(roll.rotate_left(9)) * (water_weight + food_weight) < water_weight
            {
                SupplyKind::AlmondWater
            } else {
                SupplyKind::Ration
            };
            return Some(SupplyItem {
                id: supply_id(seed, cx, cz, attempt),
                kind,
                position: Position::new(px, pz),
                rest_y: column.floor_units,
            });
        }
        None
    }

    /// All supply items overlapping a chunk, minus what reality consumed.
    #[allow(clippy::too_many_arguments)]
    fn supplies_in_bounds(
        seed: u32,
        min_x: f32,
        min_z: f32,
        max_x: f32,
        max_z: f32,
        water: f32,
        food: f32,
        reality: &RealitySnapshot,
    ) -> Vec<SupplyItem> {
        let mut items = Vec::new();
        let c0x = (min_x / SUPPLY_CELL).floor() as i64;
        let c1x = (max_x / SUPPLY_CELL).floor() as i64;
        let c0z = (min_z / SUPPLY_CELL).floor() as i64;
        let c1z = (max_z / SUPPLY_CELL).floor() as i64;
        for cz in c0z..=c1z {
            for cx in c0x..=c1x {
                let Some(item) = Self::supply_for_cell(seed, cx, cz, water, food) else {
                    continue;
                };
                if reality.supply_consumed(item.id) {
                    continue;
                }
                if item.position.x >= min_x
                    && item.position.x < max_x
                    && item.position.z >= min_z
                    && item.position.z < max_z
                {
                    items.push(item);
                }
            }
        }
        items
    }
}

/// Stamps a supply marker into the grid at a supply position. Shapes are
/// authored in world space so every voxel resolution and every overlapping
/// chunk quantizes them identically:
///
/// * almond water — a milky bottle (~0.16u wide, ~0.3u tall) with a dark
///   screw cap; the body is the faintly emissive bottle material, so it
///   glints in dim fabric.
/// * ration — a squat blue-grey tin (~0.24u wide, ~0.16u tall).
pub(crate) fn stamp_supply_marker(
    grid: &mut VoxelGrid,
    chunk_pos: Position,
    voxel_size: f32,
    item: &SupplyItem,
) {
    let base_y = (item.rest_y / voxel_size).round() as i64 + 1;
    // Fill the voxel columns covering a world-space square around the item
    // center, `rows` layers tall starting at `row`.
    let mut stamp_box = |half_w: f32, row: i64, rows: i64, material: u8| {
        let lo = |c: f32, origin: f32| ((c - half_w - origin) / voxel_size).floor() as i64;
        let hi = |c: f32, origin: f32| ((c + half_w - origin) / voxel_size).ceil() as i64 - 1;
        for z in lo(item.position.z, chunk_pos.z)..=hi(item.position.z, chunk_pos.z) {
            for x in lo(item.position.x, chunk_pos.x)..=hi(item.position.x, chunk_pos.x) {
                for y in row..row + rows {
                    if x >= 0 && y >= 1 && z >= 0 {
                        grid.set(x as usize, y as usize, z as usize, material);
                    }
                }
            }
        }
    };
    match item.kind {
        SupplyKind::AlmondWater => {
            let body_rows = ((0.22 / voxel_size).round() as i64).max(1);
            stamp_box(0.08, base_y, body_rows, VOXEL_ALMOND_WATER);
            stamp_box(0.04, base_y + body_rows, 1, VOXEL_PIPE);
        }
        SupplyKind::Ration => {
            let rows = ((0.16 / voxel_size).round() as i64).max(1);
            stamp_box(0.12, base_y, rows, VOXEL_METAL_DOOR);
        }
    }
}

/// Stamps a free-standing metal door: jambs, lintel, and a walk-through
/// panel. The frame runs along the X axis (panel plane faces Z). The whole
/// footprint stays inside a 1.0u halo of its center so Level 0's region
/// window can always reproduce it across chunk seams.
pub(crate) fn stamp_level_door(
    grid: &mut VoxelGrid,
    chunk_pos: Position,
    voxel_size: f32,
    center_x: f32,
    center_z: f32,
    wall_material: u8,
) {
    let to_local = |w: f32, origin: f32| ((w - origin) / voxel_size).floor() as i64;
    let door_half = 0.6;
    let jamb_half = 0.15;
    let height_v = (2.4 / voxel_size).round() as i64;
    let lintel_v = (2.2 / voxel_size).round() as i64;
    let panel_t = ((0.12 / voxel_size).round() as i64).max(1);

    let x0 = to_local(center_x - door_half - jamb_half * 2.0, chunk_pos.x);
    let x1 = to_local(center_x + door_half + jamb_half * 2.0, chunk_pos.x);
    let z0 = to_local(center_z - panel_t as f32 * voxel_size * 0.5, chunk_pos.z);
    let jamb_from = to_local(center_x - door_half, chunk_pos.x);
    let jamb_to = to_local(center_x + door_half, chunk_pos.x);

    for x in x0..=x1 {
        for dz in 0..panel_t {
            let z = z0 + dz;
            if x < 0 || z < 0 {
                continue;
            }
            for y in 1..=height_v {
                let material = if x < jamb_from || x > jamb_to || y >= lintel_v {
                    wall_material
                } else {
                    VOXEL_METAL_DOOR
                };
                grid.set(x as usize, y as usize, z as usize, material);
            }
        }
    }
}

impl LevelGenerator for HabitableLevel {
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
        _noise: &dyn NoiseProvider,
        reality: &RealitySnapshot,
    ) -> VoxelGrid {
        let s = config.voxel_scale;
        let width = (config.chunk_size / s).round() as usize;
        let depth = (config.chunk_size / s).round() as usize;
        let height = (GRID_HEIGHT_UNITS / s) as usize;
        let mut grid = VoxelGrid::new(width, height, depth);

        let plan_at = |lx: i64, lz: i64| -> ColumnPlan {
            let wx = chunk_pos.x + (lx as f32 + 0.5) * s;
            let wz = chunk_pos.z + (lz as f32 + 0.5) * s;
            HabitableLevel::plan_column(seed, wx, wz)
        };
        let columns = ColumnField::sample(width, depth, plan_at);
        voxelize_columns(&mut grid, &columns, s);

        // Supplies: exported beside the geometry, markers stamped after
        // voxelization so a crate top can hold a bottle. Enumeration runs
        // half a unit past the bounds so a boundary marker stamps its
        // spilled voxels into this grid too; the semantic export stays with
        // the grid whose bounds contain the item.
        let max_x = chunk_pos.x + config.chunk_size;
        let max_z = chunk_pos.z + config.chunk_size;
        for item in HabitableLevel::supplies_in_bounds(
            seed,
            chunk_pos.x - 0.5,
            chunk_pos.z - 0.5,
            max_x + 0.5,
            max_z + 0.5,
            config.tuning.almond_water,
            config.tuning.rations,
            reality,
        ) {
            stamp_supply_marker(&mut grid, chunk_pos, s, &item);
            if item.position.x >= chunk_pos.x
                && item.position.x < max_x
                && item.position.z >= chunk_pos.z
                && item.position.z < max_z
            {
                grid.supply_items.push(item);
            }
        }

        // The return door to Level 0 stands in the arrival plaza.
        let door_bounds_this_chunk = RETURN_DOOR.0 >= chunk_pos.x - 2.0
            && RETURN_DOOR.0 < max_x + 2.0
            && RETURN_DOOR.1 >= chunk_pos.z - 2.0
            && RETURN_DOOR.1 < max_z + 2.0;
        if door_bounds_this_chunk {
            stamp_level_door(
                &mut grid,
                chunk_pos,
                s,
                RETURN_DOOR.0,
                RETURN_DOOR.1,
                VOXEL_CONCRETE_WALL,
            );
        }
        if RETURN_DOOR.0 >= chunk_pos.x
            && RETURN_DOOR.0 < max_x
            && RETURN_DOOR.1 >= chunk_pos.z
            && RETURN_DOOR.1 < max_z
        {
            grid.level_exits.push(LevelExit {
                id: hash(seed, 0xD00E_0000_0000_0001, 1, 0),
                target_level: LEVEL_BACKROOMS,
                center: Position::new(RETURN_DOOR.0, RETURN_DOOR.1),
                half_extent: DOOR_TRIGGER_HALF,
                arrival: Position::new(LEVEL0_ARRIVAL.0, LEVEL0_ARRIVAL.1),
            });
        }

        grid
    }
}
