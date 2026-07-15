//! Sampling planned circulation: corridor ceilings drift slowly in long
//! quantized runs, and corridor edge walls persist or dissolve into rooms
//! and fabric on a rhythm that never reads as a doorway grid.

use crate::domain::entities::architecture::{CirculationSpine, SpaceProgram};
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::region_plan::spawn_point;

use super::BackroomsLevel;

/// Within this radius of the spawn point the main corridor keeps both of its
/// edge walls fully intact: the first thing the player reads is an
/// unambiguous walled corridor, not a dissolved edge into open fabric. The
/// world only starts opening up once that grammar has been established.
pub(super) const SPAWN_READABLE_RADIUS: f32 = 26.0;

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
        const DRIFT_ZONE: f32 = 12.0;
        let zone_x = (wx / DRIFT_ZONE).floor() * DRIFT_ZONE + DRIFT_ZONE * 0.5;
        let zone_z = (wz / DRIFT_ZONE).floor() * DRIFT_ZONE + DRIFT_ZONE * 0.5;
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

    /// Whether a corridor-side wall dissolves into the adjacent room/fabric.
    /// Main-route openings take 8--10 u from a 32 u macro span: around a
    /// quarter to a third of an eligible edge is directly open, while the
    /// remaining wall runs stay long enough to avoid a doorway cadence.
    pub(super) fn corridor_edge_opens(
        spine: &CirculationSpine,
        noise: &dyn NoiseProvider,
        seed: u32,
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
        let salt = 0xCB00_u32.wrapping_add(spine.id);
        let width_hash = Self::cell_hash(noise, seed, salt, section, side);
        let phase_hash = Self::cell_hash(noise, seed, salt ^ 0x19, section, side);
        let (width, start) = match spine.spine_kind {
            SpaceProgram::MainCorridor => (8.0 + 2.0 * width_hash, 6.0 + 10.0 * phase_hash),
            _ => (4.0 + 1.6 * width_hash, 10.0 + 8.0 * phase_hash),
        };
        let offset = along.rem_euclid(span);
        offset >= start && offset < start + width
    }
}
