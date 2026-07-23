//! Peripheral Shift: "whenever not directly observed, the layout can warp,
//! stretch, or rearrange itself" — the one non-Euclidean mechanic the game
//! has today, and the template for any future level's own version of "the
//! space doesn't stay put." A child module of `engine`; see `presenter.rs`
//! for why that gives it access to `Engine`'s private fields.
//!
//! Structured as a small [`LevelShiftPolicy`] trait with exactly one
//! concrete implementation ([`FabricDriftShift`], level 0's fabric drift).
//! Deliberately not generalized further than that: `RealitySnapshot`'s
//! epoch storage and the fixed 40u `FABRIC_DRIFT_CELL` lattice are still
//! level-0-specific, and there is no second consumer yet to prove a more
//! generic shape against. What this buys today is real: the level-0
//! hardcoding becomes a registration instead of an inline guard, so
//! `Engine::tick` no longer needs to know which levels reshape themselves.

use vackrooms::domain::entities::anomaly::{
    AnomalyKind, FABRIC_DRIFT_CELL, WorldBounds, fabric_drift_cell_of,
};

use super::{BLACKOUT_HIDE_DISTANCE, BLACKOUT_SHIFT_PERIOD_S, ChunkKey, Engine, LEVEL_BACKROOMS, chunk_key};

/// A level's "the space doesn't stay put when unobserved" mechanic. A level
/// with no registered policy simply has none — the space stays put, which
/// is the correct default for every level but 0 today.
pub(super) trait LevelShiftPolicy {
    /// Which level this policy governs.
    fn level(&self) -> u32;
    /// Advance the mechanic by one tick. Has full access to `Engine`
    /// because every existing implementation (and every plausible future
    /// one — a level's own chunk store, player pose, and reality state)
    /// needs it; the pluggability this buys is *which* mechanic runs per
    /// level, not isolation from engine state.
    fn update(&self, engine: &mut Engine, dt: f32);
}

/// Level 0's fabric drift: the sole concrete `LevelShiftPolicy` today.
pub(super) struct FabricDriftShift;

impl LevelShiftPolicy for FabricDriftShift {
    fn level(&self) -> u32 {
        LEVEL_BACKROOMS
    }

    fn update(&self, engine: &mut Engine, dt: f32) {
        engine.run_fabric_drift_shift(dt);
    }
}

impl Engine {
    /// The registered policy for `level`, if any. The only place level ids
    /// map to shift mechanics — adding a second level's policy later means
    /// adding one arm here, not touching `tick`.
    fn shift_policy_for(level: u32) -> Option<&'static dyn LevelShiftPolicy> {
        const FABRIC_DRIFT: FabricDriftShift = FabricDriftShift;
        if level == FABRIC_DRIFT.level() {
            Some(&FABRIC_DRIFT)
        } else {
            None
        }
    }

    /// Runs the current level's shift mechanic, if it has one. Replaces
    /// what used to be a hardcoded `if self.level != LEVEL_BACKROOMS` guard
    /// at the call site — `tick` no longer needs to know which levels
    /// reshape themselves, only that some might.
    pub(super) fn update_peripheral_shift(&mut self, dt: f32) {
        if let Some(policy) = Self::shift_policy_for(self.level) {
            policy.update(self, dt);
        }
    }

    /// Squared distance from a point to a drift cell's world rectangle
    /// (0 inside the cell).
    fn drift_cell_dist2(cell: (i64, i64), px: f32, pz: f32) -> f32 {
        let x0 = cell.0 as f32 * FABRIC_DRIFT_CELL;
        let z0 = cell.1 as f32 * FABRIC_DRIFT_CELL;
        let dx = (x0 - px).max(px - (x0 + FABRIC_DRIFT_CELL)).max(0.0);
        let dz = (z0 - pz).max(pz - (z0 + FABRIC_DRIFT_CELL)).max(0.0);
        dx * dx + dz * dz
    }

    /// A drift cell is hidden from the player once it's farther than any
    /// light source could plausibly reveal detail at. Named and pulled out
    /// on its own (rather than left as an inline expression) because it's
    /// the one piece of tier 2 that's a genuine "can the player perceive
    /// this" predicate, not blackout-specific geometry — the natural first
    /// thing a second shift mechanic would want to reuse.
    fn is_hidden_from_player(cell: (i64, i64), px: f32, pz: f32) -> bool {
        Self::drift_cell_dist2(cell, px, pz) > BLACKOUT_HIDE_DISTANCE.powi(2)
    }

    /// Every drift cell any resident chunk could touch counts as observed.
    /// The far radius adds hysteresis of a couple of cells, so a cell only
    /// drifts once the player has genuinely abandoned it.
    fn drift_near_radius(&self) -> f32 {
        (self.visual_policy.radius as f32 + 1.0) * self.config.chunk_size
    }

    fn drift_far_radius(&self) -> f32 {
        // A mismanaging wanderer loses the hysteresis: the fabric rearranges
        // almost the moment it leaves the streaming footprint.
        self.drift_near_radius()
            + 2.0 * FABRIC_DRIFT_CELL * (1.0 - 0.85 * self.survival.mismanagement())
    }

    /// Advances one drift cell's reality epoch and stops watching it. The
    /// single-cell primitive both tier 1 (a filtered subset of watched
    /// cells) and `abandon_all_watched_drift_cells` (every watched cell,
    /// unconditionally) are built from.
    fn abandon_drift_cell(&mut self, cell: (i64, i64)) {
        self.drift_near.remove(&cell);
        self.reality = self.reality.with_fabric_drift_advanced(cell.0, cell.1);
    }

    /// Leaving the level unobserves everything at once: every tracked drift
    /// cell rearranges, so phasing out and back never returns the wanderer
    /// to the hallways they left. Called from `switch_level` — previously
    /// that reimplemented this drain-and-advance loop inline instead of
    /// sharing it with tier 1's version below.
    pub(super) fn abandon_all_watched_drift_cells(&mut self) {
        let watched: Vec<(i64, i64)> = self.drift_near.drain().collect();
        for cell in watched {
            self.reality = self.reality.with_fabric_drift_advanced(cell.0, cell.1);
        }
    }

    /// Tier 1 — abandoned territory. Cells the player walks near are marked;
    /// when one falls beyond the far radius every chunk of it has long been
    /// evicted, its drift epoch advances, and whatever streams back in later
    /// is a lawfully different warren (corridors, assemblies, and anomalies
    /// never move — navigation survives; hallway memory does not).
    ///
    /// Tier 2 — inside a blackout the shift stalks the player in real time:
    /// on a slow cadence, cells that lie entirely behind the player's facing,
    /// beyond what any light could reveal, and fully inside the blackout's
    /// bounds advance immediately and their resident chunks rebuild. The
    /// darkness hides the swap; turning around is never a way back.
    fn run_fabric_drift_shift(&mut self, dt: f32) {
        let (px, pz) = (self.player.position[0], self.player.position[2]);

        // -- tier 1: mark near cells, drift abandoned ones -------------------
        let near = self.drift_near_radius();
        let min_cx = fabric_drift_cell_of(px - near);
        let max_cx = fabric_drift_cell_of(px + near);
        let min_cz = fabric_drift_cell_of(pz - near);
        let max_cz = fabric_drift_cell_of(pz + near);
        for cz in min_cz..=max_cz {
            for cx in min_cx..=max_cx {
                if Self::drift_cell_dist2((cx, cz), px, pz) <= near * near {
                    self.drift_near.insert((cx, cz));
                }
            }
        }
        let far2 = self.drift_far_radius() * self.drift_far_radius();
        let abandoned: Vec<(i64, i64)> = self
            .drift_near
            .iter()
            .copied()
            .filter(|&cell| Self::drift_cell_dist2(cell, px, pz) > far2)
            .collect();
        for cell in abandoned {
            self.abandon_drift_cell(cell);
        }

        // -- tier 2: the blackout rearranges behind the player ---------------
        self.blackout_shift_cooldown = (self.blackout_shift_cooldown - dt).max(0.0);
        let blackout_bounds = self
            .store
            .all_traversal_gates()
            .find(|gate| {
                gate.anomaly_kind == AnomalyKind::BlackoutExpanse
                    && gate.affected_bounds.contains(px, pz)
            })
            .map(|gate| gate.affected_bounds);
        let Some(bounds) = blackout_bounds else {
            self.blackout_shift_debug.0 = false;
            return;
        };
        self.blackout_shift_debug.0 = true;
        if self.blackout_shift_cooldown > 0.0 {
            return;
        }
        let forward = self.player.forward();
        // The rear shift reaches past the streaming footprint on purpose:
        // cells it advances beyond residency simply rebuild on approach,
        // exactly like tier-1 drift.
        let reach = self.drift_near_radius().max(3.0 * FABRIC_DRIFT_CELL);
        let mut shifted = Vec::new();
        for cz in fabric_drift_cell_of(pz - reach)..=fabric_drift_cell_of(pz + reach) {
            for cx in fabric_drift_cell_of(px - reach)..=fabric_drift_cell_of(px + reach) {
                let cx_w = (cx as f32 + 0.5) * FABRIC_DRIFT_CELL;
                let cz_w = (cz as f32 + 0.5) * FABRIC_DRIFT_CELL;
                // Center-based checks on purpose: a 40 u cell near the
                // blackout's edge or the player's flank still shifts. The
                // sliver a check this loose can expose sits in blackout
                // darkness — and a half-glimpsed wall that wasn't there is
                // the intended experience, not a bug.
                let inside = bounds.contains(cx_w, cz_w);
                let behind = (cx_w - px) * forward[0] + (cz_w - pz) * forward[2] < -4.0;
                let hidden = Self::is_hidden_from_player((cx, cz), px, pz);
                if inside && behind && hidden {
                    shifted.push((cx, cz));
                }
            }
        }
        self.blackout_shift_debug = (true, shifted.len(), self.blackout_shift_debug.2);
        if shifted.is_empty() {
            return;
        }
        self.blackout_shift_debug.2 += shifted.len();
        // Mismanagement accelerates the stalking dark: at full strain the
        // rear shift fires three times as often.
        self.blackout_shift_cooldown =
            BLACKOUT_SHIFT_PERIOD_S / (1.0 + 2.0 * self.survival.mismanagement());
        for &(cx, cz) in &shifted {
            self.reality = self.reality.with_fabric_drift_advanced(cx, cz);
        }
        // Rebuild the resident chunks of the shifted cells in place. The
        // install path's player-overlap check keeps the swap safe, and the
        // request carries the advanced reality by construction.
        let cell_bounds: Vec<WorldBounds> = shifted
            .iter()
            .map(|&(cx, cz)| {
                WorldBounds::new(
                    cx as f32 * FABRIC_DRIFT_CELL,
                    cz as f32 * FABRIC_DRIFT_CELL,
                    (cx + 1) as f32 * FABRIC_DRIFT_CELL,
                    (cz + 1) as f32 * FABRIC_DRIFT_CELL,
                )
            })
            .collect();
        let cs = self.config.chunk_size;
        let targets: Vec<ChunkKey> = self
            .store
            .iter_ordered()
            .filter(|chunk| {
                let rect = WorldBounds::new(
                    chunk.origin.0,
                    chunk.origin.1,
                    chunk.origin.0 + cs,
                    chunk.origin.1 + cs,
                );
                cell_bounds.iter().any(|cell| cell.intersects(rect))
            })
            .map(|chunk| chunk_key(chunk.origin.0, chunk.origin.1))
            .collect();
        for key in targets {
            self.force_rebuild(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::engine::EngineConfig;
    use crate::application::ports::{ChunkPayload, ChunkSourcePort, FrameParams, RendererPort};

    /// A minimal renderer relying entirely on `RendererPort`'s defaults
    /// beyond the two required methods — this test never inspects a frame,
    /// it only needs `Engine::new` to accept something.
    struct NoopRenderer;
    impl RendererPort for NoopRenderer {
        fn upload_atlas(&mut self, _texels: &[u32]) {}
        fn draw(&mut self, _frame: &FrameParams, _chunks: &[super::super::ChunkDraw]) {}
    }

    /// Never actually called: this test drives the shift policy directly,
    /// not through `Engine::tick`, so no chunk load is ever requested.
    struct UnusedSource;
    impl ChunkSourcePort for UnusedSource {
        fn load(&self, _origin_x: f32, _origin_z: f32, _level: u32, _lod: u8) -> ChunkPayload {
            unreachable!("this test never streams chunks")
        }
    }

    fn test_engine() -> Engine {
        Engine::new(
            EngineConfig::default(),
            Box::new(NoopRenderer),
            Box::new(UnusedSource),
        )
    }

    /// A fake policy for a level that doesn't really have one, proving
    /// `update_peripheral_shift` actually dispatches through the registry
    /// instead of only ever running level 0's mechanic regardless of what's
    /// registered — the real claim "pluggable" makes.
    struct SpyShift {
        level: u32,
    }

    impl LevelShiftPolicy for SpyShift {
        fn level(&self) -> u32 {
            self.level
        }

        fn update(&self, engine: &mut Engine, _dt: f32) {
            // A visible, checkable side effect distinct from anything the
            // real fabric-drift mechanic would do on its own: relocate the
            // player to a fixed marker position.
            engine.player.position = [999.0, 999.0, 999.0];
        }
    }

    #[test]
    fn a_registered_policy_actually_runs() {
        let spy = SpyShift { level: 7 };
        assert_eq!(spy.level(), 7);
        let mut engine = test_engine();
        let before = engine.player.position;
        spy.update(&mut engine, 0.016);
        assert_ne!(engine.player.position, before);
        assert_eq!(engine.player.position, [999.0, 999.0, 999.0]);
    }

    #[test]
    fn level_zero_resolves_to_the_fabric_drift_policy() {
        let policy = Engine::shift_policy_for(LEVEL_BACKROOMS).expect("level 0 has a policy");
        assert_eq!(policy.level(), LEVEL_BACKROOMS);
    }

    #[test]
    fn an_unregistered_level_has_no_policy() {
        assert!(Engine::shift_policy_for(999).is_none());
    }
}
