//! Sampling planned circulation: corridor ceilings drift slowly in long
//! quantized runs, and corridor edge walls persist or dissolve into rooms
//! and fabric on a rhythm that never reads as a doorway grid.

use crate::domain::entities::anomaly::RealitySnapshot;
use crate::domain::entities::architecture::{CirculationSpine, SpaceProgram};
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::region_plan::{PLAN_WALL_T, spawn_point};

use super::BackroomsLevel;

/// Within this radius of the spawn point the main corridor keeps both of its
/// edge walls fully intact: the first thing the player reads is an
/// unambiguous walled corridor, not a dissolved edge into open fabric. The
/// world only starts opening up once that grammar has been established.
pub(super) const SPAWN_READABLE_RADIUS: f32 = 26.0;
const CEILING_ZONE: f32 = 12.0;

impl BackroomsLevel {
    pub(super) fn corridor_ceiling(
        spine: &CirculationSpine,
        noise: &dyn NoiseProvider,
        seed: u32,
        wx: f32,
        wz: f32,
    ) -> f32 {
        // Slow, quantized drift: one sample per 12 u corridor zone snapped
        // to the voxel lattice, so long runs hold one height and then step
        // once — instead of per-column ripple overhead.
        let zone_x = (wx / CEILING_ZONE).floor() * CEILING_ZONE + CEILING_ZONE * 0.5;
        let zone_z = (wz / CEILING_ZONE).floor() * CEILING_ZONE + CEILING_ZONE * 0.5;
        let drift = Self::n(
            noise,
            seed,
            0xCA00_u32.wrapping_add(spine.id),
            zone_x,
            zone_z,
            0.35,
        );
        let (height, lo, hi) = match spine.spine_kind {
            SpaceProgram::MainCorridor => (3.8 + 0.4 * drift, 3.4, 4.2),
            SpaceProgram::SecondaryHall => (3.3 + 0.3 * drift, 3.0, 3.6),
            _ => (3.4, 3.4, 3.4),
        };
        // Snap first, clamp last (f32 lattice snap can overshoot the band).
        ((height / 0.2).round() * 0.2).clamp(lo, hi)
    }

    /// Lower edge of a bulkhead where one corridor ceiling module meets a
    /// different-height module. Each boundary is owned by the module on its
    /// east/south side, so independently sampled chunks author it once.
    pub(super) fn corridor_ceiling_join(
        spine: &CirculationSpine,
        noise: &dyn NoiseProvider,
        seed: u32,
        wx: f32,
        wz: f32,
    ) -> Option<f32> {
        let current = Self::corridor_ceiling(spine, noise, seed, wx, wz);
        let in_west_edge = wx.rem_euclid(CEILING_ZONE) < PLAN_WALL_T;
        let in_north_edge = wz.rem_euclid(CEILING_ZONE) < PLAN_WALL_T;
        let mut join_from: Option<f32> = None;
        for (nx, nz) in [
            in_west_edge.then_some((wx - CEILING_ZONE, wz)),
            in_north_edge.then_some((wx, wz - CEILING_ZONE)),
        ]
        .into_iter()
        .flatten()
        {
            let neighbor = Self::corridor_ceiling(spine, noise, seed, nx, nz);
            if (neighbor - current).abs() > 0.05 {
                let lower = neighbor.min(current);
                join_from = Some(join_from.map_or(lower, |height| height.min(lower)));
            }
        }
        join_from
    }

    /// Whether a corridor-side wall dissolves into the adjacent room/fabric.
    /// Main-route openings take 8--10 u from a 32 u macro span: around a
    /// quarter to a third of an eligible edge is directly open, while the
    /// remaining wall runs stay long enough to avoid a doorway cadence.
    ///
    /// The openings are the fabric's mouths onto circulation, so the
    /// Peripheral Shift re-deals them: the drift epoch (read at the wall
    /// column's own 40 u cell) re-salts each section's width and phase, and
    /// the strain tier narrows them — at deep strain whole sections seal.
    /// The corridor interior itself never closes: a shut mouth costs the
    /// route you knew, never the spine.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn corridor_edge_opens(
        spine: &CirculationSpine,
        noise: &dyn NoiseProvider,
        seed: u32,
        reality: &RealitySnapshot,
        along: f32,
        is_horizontal: bool,
        wx: f32,
        wz: f32,
    ) -> bool {
        // The opening spine sequence is sacred: near spawn the corridor keeps
        // both edge walls so the player's first minute reads as circulation
        // through a building, before any dissolution into fabric.
        let sp = spawn_point(seed);
        let spawn_d2 = (wx - sp.x) * (wx - sp.x) + (wz - sp.z) * (wz - sp.z);
        if spawn_d2 < SPAWN_READABLE_RADIUS * SPAWN_READABLE_RADIUS {
            return false;
        }

        if Self::in_expanse(noise, seed, wx, wz) {
            return true;
        }

        let span = 32.0;
        let section = (along / span).floor() as i64;
        let perpendicular = if is_horizontal { wz } else { wx };
        let side = (perpendicular / spine.width.max(1.0)).floor() as i64;
        let epoch = reality.fabric_drift_epoch(wx, wz);
        let tier = reality.delirium() as u32;
        let salt = (0xCB00_u32.wrapping_add(spine.id)) ^ epoch.wrapping_mul(0x9E37_79B9);
        // Deep strain seals whole sections: the opening the wanderer relied
        // on is simply wall now. The threshold shrinks the same hash's
        // acceptance, so rising tiers close mouths instead of moving them.
        if tier >= 2
            && Self::cell_hash(noise, seed, salt ^ 0x2B, section, side)
                < 0.22 * (tier - 1) as f32
        {
            return false;
        }
        let width_hash = Self::cell_hash(noise, seed, salt, section, side);
        let phase_hash = Self::cell_hash(noise, seed, salt ^ 0x19, section, side);
        let (mut width, start) = match spine.spine_kind {
            SpaceProgram::MainCorridor => (8.0 + 2.0 * width_hash, 6.0 + 10.0 * phase_hash),
            _ => (4.0 + 1.6 * width_hash, 10.0 + 8.0 * phase_hash),
        };
        // Strain narrows every mouth that survives.
        width *= 1.0 - 0.15 * tier as f32;
        let offset = along.rem_euclid(span);
        offset >= start && offset < start + width
    }
}
