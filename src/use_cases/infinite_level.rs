//! Deterministic access to the finite window of an infinite planned level.
//!
//! Chunk generation is local, but Level 0 architecture is not: a chunk may
//! sample corridors, assemblies, or anomaly footprints planned by any region
//! that overlaps its voxel halo.  This module makes that query explicit.  The
//! window owns a stable, row-major set of region plans and is otherwise just a
//! world-point-to-plan lookup; callers do not need to reproduce region-range
//! arithmetic or fall back to an unrelated plan when a lookup misses.

use crate::domain::entities::anomaly::WorldBounds;
use crate::domain::entities::architecture::RegionPlan;
use crate::domain::entities::position::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::region_plan::{REGION_SIZE, generate_region_plan, region_index};

/// The region plans needed to sample one bounded area of an infinite level.
///
/// Plans are stored in `(z, x)` row-major order.  That order is deliberately
/// independent of hash maps and request timing, so planning the same area is
/// replayable on the main thread, in a worker, or during a test.
#[derive(Debug)]
pub(crate) struct InfiniteRegionWindow {
    min_region_x: i64,
    min_region_z: i64,
    max_region_x: i64,
    max_region_z: i64,
    plans: Vec<RegionPlan>,
}

impl InfiniteRegionWindow {
    /// Plans every region touched by `area` after expanding it by `halo` world
    /// units.  Boundary-touching regions are included intentionally: a voxel
    /// sampler may ask about the exact far edge while constructing skirts,
    /// meshes, or collision data.
    pub(crate) fn covering(
        area: WorldBounds,
        halo: f32,
        seed: u32,
        config: &GeneratorConfig,
        noise: &dyn NoiseProvider,
    ) -> Self {
        assert!(
            halo.is_finite() && halo >= 0.0,
            "region-window halo must be a finite, non-negative world distance"
        );
        let area = area.expanded(halo);
        assert!(
            [area.min_x, area.min_z, area.max_x, area.max_z]
                .into_iter()
                .all(f32::is_finite),
            "an infinite-region window requires finite world bounds"
        );

        let min_region_x = region_index(area.min_x);
        let min_region_z = region_index(area.min_z);
        let max_region_x = region_index(area.max_x);
        let max_region_z = region_index(area.max_z);
        let width = region_count(min_region_x, max_region_x);
        let height = region_count(min_region_z, max_region_z);
        let capacity = width
            .checked_mul(height)
            .expect("requested region window is too large to address");
        let mut plans = Vec::with_capacity(capacity);

        for region_z in min_region_z..=max_region_z {
            for region_x in min_region_x..=max_region_x {
                plans.push(generate_region_plan(
                    seed,
                    region_origin(region_x, region_z),
                    REGION_SIZE,
                    config,
                    noise,
                ));
            }
        }

        Self {
            min_region_x,
            min_region_z,
            max_region_x,
            max_region_z,
            plans,
        }
    }

    /// Convenience constructor for the common chunk-plus-voxel-halo query.
    pub(crate) fn around_chunk(
        chunk_origin: Position,
        chunk_size: f32,
        halo: f32,
        seed: u32,
        config: &GeneratorConfig,
        noise: &dyn NoiseProvider,
    ) -> Self {
        assert!(
            chunk_size.is_finite() && chunk_size >= 0.0,
            "chunk size must be a finite, non-negative world distance"
        );
        Self::covering(
            WorldBounds::new(
                chunk_origin.x,
                chunk_origin.z,
                chunk_origin.x + chunk_size,
                chunk_origin.z + chunk_size,
            ),
            halo,
            seed,
            config,
            noise,
        )
    }

    /// Returns the authoritative plan for a world point, or `None` when the
    /// caller sampled beyond the area/halo declared at construction time.
    pub(crate) fn plan_at(&self, world: Position) -> Option<&RegionPlan> {
        self.plan_for_region(region_index(world.x), region_index(world.z))
    }

    pub(crate) fn plan_for_region(&self, region_x: i64, region_z: i64) -> Option<&RegionPlan> {
        if region_x < self.min_region_x
            || region_x > self.max_region_x
            || region_z < self.min_region_z
            || region_z > self.max_region_z
        {
            return None;
        }

        let width = region_count(self.min_region_x, self.max_region_x);
        let local_x = usize::try_from(region_x - self.min_region_x).ok()?;
        let local_z = usize::try_from(region_z - self.min_region_z).ok()?;
        self.plans
            .get(local_z.saturating_mul(width).saturating_add(local_x))
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.plans.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    /// Stable row-major iteration for semantic export and deterministic tests.
    pub(crate) fn iter(&self) -> impl Iterator<Item = ((i64, i64), &RegionPlan)> {
        let width = region_count(self.min_region_x, self.max_region_x);
        self.plans.iter().enumerate().map(move |(index, plan)| {
            let local_z = index / width;
            let local_x = index % width;
            (
                (
                    self.min_region_x + local_x as i64,
                    self.min_region_z + local_z as i64,
                ),
                plan,
            )
        })
    }
}

fn region_count(first: i64, last: i64) -> usize {
    let count = last
        .checked_sub(first)
        .and_then(|distance| distance.checked_add(1))
        .expect("requested region window exceeds integer address space");
    usize::try_from(count).expect("requested region window exceeds platform address space")
}

fn region_origin(region_x: i64, region_z: i64) -> Position {
    Position::new(region_x as f32 * REGION_SIZE, region_z as f32 * REGION_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frameworks_drivers::simple_noise::SimpleNoiseProvider;
    use crate::use_cases::region_plan::debug_region_ascii;

    #[test]
    fn area_and_halo_define_an_explicit_negative_coordinate_window() {
        let window = InfiniteRegionWindow::covering(
            WorldBounds::new(-0.1, 0.1, 79.9, 79.9),
            0.4,
            42,
            &GeneratorConfig::low_spec(),
            &SimpleNoiseProvider::new(),
        );

        let keys: Vec<_> = window.iter().map(|(key, _)| key).collect();
        assert_eq!(
            keys,
            vec![
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1)
            ]
        );
        assert_eq!(window.len(), keys.len());
        assert!(!window.is_empty());
    }

    #[test]
    fn lookup_never_falls_back_to_an_unrelated_region() {
        let window = InfiniteRegionWindow::around_chunk(
            Position::new(70.0, -10.0),
            20.0,
            1.0,
            42,
            &GeneratorConfig::low_spec(),
            &SimpleNoiseProvider::new(),
        );

        assert_eq!(
            window
                .plan_at(Position::new(79.9, -0.1))
                .unwrap()
                .origin_world,
            Position::new(0.0, -80.0)
        );
        assert_eq!(
            window
                .plan_at(Position::new(80.1, 0.1))
                .unwrap()
                .origin_world,
            Position::new(80.0, 0.0)
        );
        assert!(window.plan_at(Position::new(400.0, 400.0)).is_none());
    }

    #[test]
    fn planning_the_same_window_replays_the_same_architecture() {
        let build = || {
            InfiniteRegionWindow::around_chunk(
                Position::new(10.0, 10.0),
                10.0,
                0.4,
                42,
                &GeneratorConfig::low_spec(),
                &SimpleNoiseProvider::new(),
            )
        };
        let a = build();
        let b = build();
        let ascii = |window: &InfiniteRegionWindow| {
            window
                .iter()
                .map(|(key, plan)| (key, debug_region_ascii(plan, 2.0)))
                .collect::<Vec<_>>()
        };
        assert_eq!(ascii(&a), ascii(&b));
    }
}
