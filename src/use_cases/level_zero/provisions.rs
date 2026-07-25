//! Level 0 provisions: scattered almond water and the rare physical door.
//!
//! Both are a deterministic post-pass over the already-voxelized chunk,
//! derived purely from (seed, world lattice, reality). Supplies follow the
//! 40u fabric-drift lattice; doors follow the 80u region lattice plus one
//! authored door near spawn, so a fresh wanderer can always find Level 1.
//! Red-room interiors and pit lattices never receive provisions: a sealed
//! loop must stay barren, and nothing may hover over a pit.

use crate::domain::entities::anomaly::{AnomalyKind, LevelExit, RealitySnapshot};
use crate::domain::entities::architecture::RegionPlan;
use crate::domain::entities::supplies::{SupplyItem, SupplyKind};
use crate::domain::entities::voxel_grid::{VOXEL_AIR, VOXEL_WALL, VoxelGrid};
use crate::domain::entities::position::Position;
use crate::use_cases::anomalies::determinism::{hash, unit};
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::infinite_level::InfiniteRegionWindow;
use crate::use_cases::level_generator::LEVEL_HABITABLE;
use crate::use_cases::level_one::generate::{
    ARRIVAL_POINT, DOOR_TRIGGER_HALF, stamp_level_door, stamp_supply_marker,
};
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::region_plan::{REGION_SIZE, region_index};

use super::BackroomsLevel;

/// Supply lattice: one candidate per fabric-drift-sized cell.
const SUPPLY_CELL: f32 = 40.0;
/// Fraction of supply cells that roll a candidate spot. Each cell gets one
/// deterministic spot (never a probe sequence, which would make placement
/// depend on which chunk evaluates it); spots landing on solid columns are
/// simply lost, so this is tuned above the desired field density.
const ALMOND_CHANCE: f32 = 0.72;
/// Of the successful cells, this fraction holds food instead of water.
const RATION_SHARE: f32 = 0.25;

/// Fraction of 80u regions holding an organic door to Level 1 (at the
/// default `level_doors` tuning of 1.0).
const DOOR_CHANCE: f32 = 0.08;
/// How far past its bounds a chunk stamps door geometry. Must stay within
/// the region-plan window margin (1.0) and cover the frame plus clearing.
const DOOR_STAMP_MARGIN: f32 = 1.0;
/// The authored always-present door: guaranteed to exist, but a real hike
/// from spawn — finding it should feel earned, not scripted. Debug runs
/// raise `level_doors` (or walk straight here).
const SPAWN_DOOR: (f32, f32) = (172.0, -116.0);

pub(crate) struct ProvisionContext<'a> {
    pub seed: u32,
    pub config: &'a GeneratorConfig,
    pub reality: &'a RealitySnapshot,
    pub noise: &'a dyn NoiseProvider,
    pub plans: &'a InfiniteRegionWindow,
}

impl ProvisionContext<'_> {
    fn plan_of(&self, wx: f32, wz: f32) -> Option<&RegionPlan> {
        self.plans.plan_at(Position::new(wx, wz))
    }

    /// True when the walking plane at a world point is ordinary open fabric:
    /// not solid, not raised, and not inside a barren anomaly family.
    fn is_open_floor(&self, wx: f32, wz: f32) -> bool {
        let Some(plan) = self.plan_of(wx, wz) else {
            return false;
        };
        let forbidden = plan.anomalies.iter().any(|anomaly| {
            matches!(anomaly.kind, AnomalyKind::RedRoom | AnomalyKind::PitLattice)
                && anomaly.contains(wx, wz)
        });
        if forbidden {
            return false;
        }
        let column = BackroomsLevel::plan_column_in_reality(
            plan,
            self.noise,
            self.seed,
            self.config,
            self.reality,
            wx,
            wz,
        );
        column.floor && !column.solid && column.floor_units < 0.05
    }
}

/// Decides which almond-water/ration supply items exist within stamping
/// range of this chunk, without touching the grid: one deterministic
/// candidate per 40u supply-lattice cell, accepted only on open floor and
/// only where the strain-scaled density and consumption history allow it.
/// Pure — same hash-roll/position logic `stamp_level_zero_provisions` used
/// inline, split out so it (and its rejection reasons) are testable without
/// a `VoxelGrid`. Chunks that cannot even stamp a candidate (outside the
/// 0.5u marker margin) skip without evaluating it further, so no decision
/// ever depends on which chunk happens to be asking.
fn decide_supply_items(chunk_pos: Position, ctx: &ProvisionContext<'_>) -> Vec<SupplyItem> {
    let water = ctx.config.tuning.almond_water.clamp(0.0, 4.0);
    let food = ctx.config.tuning.rations.clamp(0.0, 4.0);
    if water <= 0.0 && food <= 0.0 {
        return Vec::new();
    }
    let max_x = chunk_pos.x + ctx.config.chunk_size;
    let max_z = chunk_pos.z + ctx.config.chunk_size;
    let near_chunk = |x: f32, z: f32, m: f32| {
        x >= chunk_pos.x - m && x < max_x + m && z >= chunk_pos.z - m && z < max_z + m
    };

    let c0x = ((chunk_pos.x - 1.0) / SUPPLY_CELL).floor() as i64;
    let c1x = ((max_x + 1.0) / SUPPLY_CELL).floor() as i64;
    let c0z = ((chunk_pos.z - 1.0) / SUPPLY_CELL).floor() as i64;
    let c1z = ((max_z + 1.0) / SUPPLY_CELL).floor() as i64;
    let mut items = Vec::new();
    for cz in c0z..=c1z {
        for cx in c0x..=c1x {
            let roll = hash(ctx.seed, 0x0A1A_09D0_57A7_0000, cx, cz);
            // Water and food frequencies scale their halves of the density
            // independently; the kind roll then splits proportionally, so
            // e.g. rations=0 yields a pure-water world at water's density.
            let water_weight = (1.0 - RATION_SHARE) * water;
            let food_weight = RATION_SHARE * food;
            // A mismanaged wanderer finds a stingier level: each strain tier
            // withholds a share of the supply cells (same hash, lower
            // threshold, so rising strain removes bottles rather than
            // shuffling them). Tier 0 is the unstrained field.
            let strain_keep = 1.0 - 0.15 * ctx.reality.delirium() as f32;
            let cell_chance = ALMOND_CHANCE * (water_weight + food_weight) * strain_keep;
            if unit(roll) > cell_chance {
                continue;
            }
            let id = roll | 1;
            if ctx.reality.supply_consumed(id) {
                continue;
            }
            // One deterministic spot per cell.
            let probe = hash(ctx.seed, 0x0A1A_09D0_0000_0000, cx, cz);
            let px = (cx as f32 + 0.06 + 0.88 * unit(probe)) * SUPPLY_CELL;
            let pz = (cz as f32 + 0.06 + 0.88 * unit(probe.rotate_left(23))) * SUPPLY_CELL;
            if !near_chunk(px, pz, 0.5) || !ctx.is_open_floor(px, pz) {
                continue;
            }
            let kind = if unit(roll.rotate_left(11)) * (water_weight + food_weight) < food_weight
            {
                SupplyKind::Ration
            } else {
                SupplyKind::AlmondWater
            };
            items.push(SupplyItem {
                id,
                kind,
                position: Position::new(px, pz),
                rest_y: 0.0,
            });
        }
    }
    items
}

/// Stamps supplies and doors into one Level 0 chunk and exports their
/// semantic records. Never called for recursive (red-room-interior) chunks.
pub(crate) fn stamp_level_zero_provisions(
    grid: &mut VoxelGrid,
    chunk_pos: Position,
    ctx: &ProvisionContext<'_>,
) {
    let doors = ctx.config.tuning.level_doors.clamp(0.0, 4.0);
    let has_supplies =
        ctx.config.tuning.almond_water > 0.0 || ctx.config.tuning.rations > 0.0;
    if !has_supplies && doors <= 0.0 {
        return;
    }
    let s = ctx.config.voxel_scale;
    let max_x = chunk_pos.x + ctx.config.chunk_size;
    let max_z = chunk_pos.z + ctx.config.chunk_size;
    let in_chunk = |x: f32, z: f32| x >= chunk_pos.x && x < max_x && z >= chunk_pos.z && z < max_z;
    // Markers span a few voxels, so an item just outside this grid must
    // still stamp its overlapping voxels here or seams (and the halo crop)
    // would disagree with the neighbor that owns it. Stamping quantizes in
    // world space, making the overlap exact; the semantic export stays with
    // every grid whose bounds contain the item (halo overlaps are deduped by
    // id on the consuming side).
    let near_chunk = |x: f32, z: f32, m: f32| {
        x >= chunk_pos.x - m && x < max_x + m && z >= chunk_pos.z - m && z < max_z + m
    };

    for item in decide_supply_items(chunk_pos, ctx) {
        stamp_supply_marker(grid, chunk_pos, s, &item);
        if in_chunk(item.position.x, item.position.z) {
            grid.supply_items.push(item);
        }
    }

    // -- doors to Level 1 on the 80u region lattice ---------------------------
    let r0x = (chunk_pos.x / REGION_SIZE).floor() as i64;
    let r1x = ((max_x - 0.01) / REGION_SIZE).floor() as i64;
    let r0z = (chunk_pos.z / REGION_SIZE).floor() as i64;
    let r1z = ((max_z - 0.01) / REGION_SIZE).floor() as i64;
    // The whole door footprint (frame + clearing) fits inside the region
    // plan window's 1.0u margin, so every chunk that must stamp part of a
    // door can also evaluate its acceptance identically.
    if doors <= 0.0 {
        return;
    }
    for rz in r0z..=r1z {
        for rx in r0x..=r1x {
            let spawn_door_region =
                rx == region_index(SPAWN_DOOR.0) && rz == region_index(SPAWN_DOOR.1);
            let (dx, dz) = if spawn_door_region {
                // The authored guaranteed door carves its own clearing; no
                // floor check, so it exists in every reality.
                SPAWN_DOOR
            } else {
                let roll = hash(ctx.seed, 0xD00E_0000_5EED_0000, rx, rz);
                if unit(roll) > DOOR_CHANCE * doors {
                    continue;
                }
                // One deterministic spot per region; a spot on solid or
                // anomalous ground simply means this region has no door.
                let probe = hash(ctx.seed, 0xD00E_0000_0000_0000, rx, rz);
                let px = (rx as f32 + 0.15 + 0.7 * unit(probe)) * REGION_SIZE;
                let pz = (rz as f32 + 0.15 + 0.7 * unit(probe.rotate_left(19))) * REGION_SIZE;
                if !near_chunk(px, pz, DOOR_STAMP_MARGIN) || !ctx.is_open_floor(px, pz) {
                    continue;
                }
                (px, pz)
            };
            if !near_chunk(dx, dz, DOOR_STAMP_MARGIN) {
                continue;
            }

            carve_door_clearing(grid, chunk_pos, s, dx, dz);
            stamp_level_door(grid, chunk_pos, s, dx, dz, VOXEL_WALL);
            if in_chunk(dx, dz) {
                grid.level_exits.push(LevelExit {
                    id: hash(ctx.seed, 0xD00E_0000_0000_1D00, rx, rz),
                    target_level: LEVEL_HABITABLE,
                    center: Position::new(dx, dz),
                    half_extent: DOOR_TRIGGER_HALF,
                    arrival: Position::new(ARRIVAL_POINT.0, ARRIVAL_POINT.1),
                });
            }
        }
    }
}

/// Guarantees approach space on both sides of a door: air to 2.6u within a
/// small square, with the floor slab restored underneath. The radius stays
/// under `DOOR_STAMP_MARGIN` so the carve never outruns what neighboring
/// chunks can reproduce.
fn carve_door_clearing(
    grid: &mut VoxelGrid,
    chunk_pos: Position,
    voxel_size: f32,
    cx: f32,
    cz: f32,
) {
    let radius = 0.95;
    let height_v = (2.6 / voxel_size).round() as i64;
    let x0 = ((cx - radius - chunk_pos.x) / voxel_size).floor() as i64;
    let x1 = ((cx + radius - chunk_pos.x) / voxel_size).floor() as i64;
    let z0 = ((cz - radius - chunk_pos.z) / voxel_size).floor() as i64;
    let z1 = ((cz + radius - chunk_pos.z) / voxel_size).floor() as i64;
    for z in z0.max(0)..=z1.max(-1) {
        for x in x0.max(0)..=x1.max(-1) {
            let (xu, zu) = (x as usize, z as usize);
            if xu >= grid.width() || zu >= grid.depth() {
                continue;
            }
            for y in 1..=height_v {
                grid.set(xu, y as usize, zu, VOXEL_AIR);
            }
            if grid.get(xu, 0, zu) == VOXEL_AIR {
                grid.set(
                    xu,
                    0,
                    zu,
                    crate::domain::entities::voxel_grid::VOXEL_FLOOR,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frameworks_drivers::simple_noise::SimpleNoiseProvider;

    /// `decide_supply_items` is exercised directly, no `VoxelGrid` involved:
    /// confirms it's deterministic (same chunk always decides the same
    /// items) and that every accepted item actually lands on open floor —
    /// the property `is_open_floor` exists to guarantee.
    #[test]
    fn supply_items_are_deterministic_and_on_open_floor() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let reality = RealitySnapshot::empty();
        let mut any_items = false;

        for chunk in 0..40i64 {
            let chunk_pos = Position::new(chunk as f32 * config.chunk_size, 0.0);
            let plans = BackroomsLevel::region_plans_for(
                chunk_pos,
                config.chunk_size,
                42,
                &config,
                &noise,
            );
            let ctx = ProvisionContext {
                seed: 42,
                config: &config,
                reality: &reality,
                noise: &noise,
                plans: &plans,
            };

            let items_a = decide_supply_items(chunk_pos, &ctx);
            let items_b = decide_supply_items(chunk_pos, &ctx);
            assert_eq!(
                items_a.len(),
                items_b.len(),
                "supply decision is not deterministic for chunk {chunk}"
            );
            for item in &items_a {
                assert!(
                    ctx.is_open_floor(item.position.x, item.position.z),
                    "accepted supply item at {:?} is not on open floor",
                    item.position
                );
            }
            any_items |= !items_a.is_empty();
        }
        assert!(any_items, "no supply items found across 40 sampled chunks");
    }

    /// `almond_water == 0 && rations == 0` must skip the lattice sweep
    /// entirely rather than decide items nobody wanted, matching the early
    /// return `stamp_level_zero_provisions` takes for the same setting.
    #[test]
    fn zero_tuning_decides_no_supply_items() {
        let noise = SimpleNoiseProvider::new();
        let mut config = GeneratorConfig::low_spec();
        config.tuning.almond_water = 0.0;
        config.tuning.rations = 0.0;
        let reality = RealitySnapshot::empty();
        let chunk_pos = Position::new(0.0, 0.0);
        let plans =
            BackroomsLevel::region_plans_for(chunk_pos, config.chunk_size, 42, &config, &noise);
        let ctx = ProvisionContext {
            seed: 42,
            config: &config,
            reality: &reality,
            noise: &noise,
            plans: &plans,
        };
        assert!(decide_supply_items(chunk_pos, &ctx).is_empty());
    }
}
