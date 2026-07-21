//! Backrooms lore made mechanical: holding a walk into solid wall long
//! enough occasionally phases the player through reality. A child module of
//! `engine`; see `presenter.rs` for why that gives it access to `Engine`'s
//! private fields.

use super::{
    Engine, InputFrame, LEVEL_BACKROOMS, LEVEL_GRASSLAND, NOCLIP_CHANCE, NOCLIP_PUSH_SECONDS,
};

impl Engine {
    pub(super) fn rng_next01(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Backrooms lore made mechanical: holding a walk into solid wall long
    /// enough occasionally phases the player through reality.
    pub(super) fn update_noclip(&mut self, dt: f32, input: &InputFrame, old_pos: [f32; 3]) {
        self.noclip_cooldown = (self.noclip_cooldown - dt).max(0.0);

        let i = input.intent;
        let pushing = input.locked && (i.forward || i.backward || i.left || i.right);
        let dx = self.player.position[0] - old_pos[0];
        let dz = self.player.position[2] - old_pos[2];
        let pinned = pushing && (dx * dx + dz * dz) < 0.002 * 0.002;

        if !pinned {
            self.push_seconds = 0.0;
            return;
        }
        self.push_seconds += dt;
        if self.push_seconds < NOCLIP_PUSH_SECONDS || self.noclip_cooldown > 0.0 {
            return;
        }
        self.noclip_cooldown = 1.0;
        if self.rng_next01() < NOCLIP_CHANCE {
            self.noclip();
        }
    }

    /// Phase through reality by pushing into walls. From Level 0 the fall is
    /// into the grassland; from anywhere else — including Level 1, exactly
    /// as the wiki warns — a noclip drops the wanderer back into Level 0.
    fn noclip(&mut self) {
        let (target, arrival) = if self.level == LEVEL_BACKROOMS {
            // Into the grassland you phase in place.
            (LEVEL_GRASSLAND, None)
        } else {
            // The way back drops you at the spawn clearing so you can't
            // rematerialize inside a wall.
            (LEVEL_BACKROOMS, Some(self.config.spawn))
        };
        self.switch_level(target, arrival);
    }
}
