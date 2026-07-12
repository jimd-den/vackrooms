//! Input adapter: accumulates raw browser events (key codes, mouse deltas,
//! pointer-lock state) and translates them into one [`InputFrame`] per tick.
//!
//! The driver layer feeds this from DOM event listeners; the frame loop
//! drains it with [`InputCollector::take_frame`]. Keeping the mapping here —
//! not in the driver — means "which key means forward" is a testable rule,
//! not something buried in an event closure.

use crate::application::engine::InputFrame;
use crate::application::player::MoveIntent;

#[derive(Debug, Default)]
pub struct InputCollector {
    intent: MoveIntent,
    pending_dx: f32,
    pending_dy: f32,
    locked: bool,
    flashlight: bool,
    /// One queued flare drop, consumed by the next frame. Edge-triggered:
    /// holding G never queues a second drop until the key is released.
    flare_queued: bool,
    flare_key_held: bool,
}

impl InputCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handles a keydown/keyup pair by `KeyboardEvent.code` value.
    /// Unknown codes are ignored. Both WASD and arrow keys are mapped.
    pub fn key_event(&mut self, code: &str, pressed: bool) {
        #[cfg(target_arch = "wasm32")]
        let doom = crate::DOOM_CONTROLS.load(std::sync::atomic::Ordering::Relaxed);

        #[cfg(not(target_arch = "wasm32"))]
        let doom = false;

        if pressed && code == "KeyF" {
            self.flashlight = !self.flashlight;
        }

        // Flare drop: fires once per press, only while playing (the engine
        // additionally ignores drops when the session is not locked, so a
        // G typed into the settings menu can never reach gameplay).
        if code == "KeyG" {
            if pressed && !self.flare_key_held && self.locked {
                self.flare_queued = true;
            }
            self.flare_key_held = pressed;
        }

        if doom {
            match code {
                "KeyW" | "ArrowUp" => self.intent.forward = pressed,
                "KeyS" | "ArrowDown" => self.intent.backward = pressed,
                "KeyQ" => self.intent.left = pressed,
                "KeyE" => self.intent.right = pressed,
                "KeyA" | "ArrowLeft" => self.intent.turn_left = pressed,
                "KeyD" | "ArrowRight" => self.intent.turn_right = pressed,
                _ => {}
            }
        } else {
            match code {
                "KeyW" | "ArrowUp" => self.intent.forward = pressed,
                "KeyS" | "ArrowDown" => self.intent.backward = pressed,
                "KeyA" | "ArrowLeft" => self.intent.left = pressed,
                "KeyD" | "ArrowRight" => self.intent.right = pressed,
                "KeyQ" => self.intent.turn_left = pressed,
                "KeyE" => self.intent.turn_right = pressed,
                _ => {}
            }
        }
    }

    /// Accumulates a mouse movement (only meaningful while pointer-locked).
    pub fn mouse_delta(&mut self, dx: f32, dy: f32) {
        if self.locked {
            #[cfg(target_arch = "wasm32")]
            let (sens, invert) = {
                use std::sync::atomic::Ordering;
                (
                    f32::from_bits(crate::MOUSE_SENSITIVITY_BITS.load(Ordering::Relaxed)),
                    crate::INVERT_Y.load(Ordering::Relaxed),
                )
            };
            #[cfg(not(target_arch = "wasm32"))]
            let (sens, invert) = (1.0f32, false);

            self.pending_dx += dx * sens;
            self.pending_dy += dy * sens * if invert { -1.0 } else { 1.0 };
        }
    }

    /// Sets movement intent from an analog source (the touch joystick).
    /// `x` is strafe (-1..1, right positive), `y` is walk (-1..1, forward
    /// positive). Values inside the dead zone clear the intent, so lifting
    /// the thumb stops the player.
    pub fn set_move_axes(&mut self, x: f32, y: f32) {
        const DEAD_ZONE: f32 = 0.25;
        self.intent.forward = y > DEAD_ZONE;
        self.intent.backward = y < -DEAD_ZONE;
        self.intent.right = x > DEAD_ZONE;
        self.intent.left = x < -DEAD_ZONE;
    }

    /// Toggles the flashlight (touch button; keyboard uses KeyF).
    pub fn toggle_flashlight(&mut self) {
        self.flashlight = !self.flashlight;
    }

    /// Queues one flare drop (touch button; keyboard uses KeyG). One queued
    /// drop per frame regardless of how many events bubble in.
    pub fn queue_flare(&mut self) {
        if self.locked {
            self.flare_queued = true;
        }
    }

    /// Pointer lock engaged/released. Releasing clears held keys so the
    /// player doesn't keep walking while the overlay is up.
    pub fn set_locked(&mut self, locked: bool) {
        self.locked = locked;
        if !locked {
            self.intent = MoveIntent::default();
            self.pending_dx = 0.0;
            self.pending_dy = 0.0;
            self.flare_queued = false;
        }
    }

    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Drains accumulated input into a frame snapshot. Mouse deltas reset;
    /// key intent persists until the matching keyup.
    pub fn take_frame(&mut self) -> InputFrame {
        let frame = InputFrame {
            intent: self.intent,
            look_dx: self.pending_dx,
            look_dy: self.pending_dy,
            locked: self.locked,
            flashlight: self.flashlight,
            drop_flare: self.flare_queued,
        };
        self.pending_dx = 0.0;
        self.pending_dy = 0.0;
        self.flare_queued = false;
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasd_maps_to_intent_and_persists_across_frames() {
        let mut input = InputCollector::new();
        input.set_locked(true);
        input.key_event("KeyW", true);
        input.key_event("KeyD", true);
        let f1 = input.take_frame();
        assert!(f1.intent.forward && f1.intent.right);
        let f2 = input.take_frame();
        assert!(f2.intent.forward, "held keys persist");
        input.key_event("KeyW", false);
        assert!(!input.take_frame().intent.forward);
    }

    #[test]
    fn mouse_deltas_accumulate_then_drain() {
        let mut input = InputCollector::new();
        input.set_locked(true);
        input.mouse_delta(3.0, -1.0);
        input.mouse_delta(2.0, 1.0);
        let f = input.take_frame();
        assert_eq!((f.look_dx, f.look_dy), (5.0, 0.0));
        let f2 = input.take_frame();
        assert_eq!((f2.look_dx, f2.look_dy), (0.0, 0.0));
    }

    #[test]
    fn unlock_clears_held_keys_and_ignores_mouse() {
        let mut input = InputCollector::new();
        input.set_locked(true);
        input.key_event("KeyW", true);
        input.set_locked(false);
        input.mouse_delta(10.0, 10.0);
        let f = input.take_frame();
        assert!(!f.intent.forward);
        assert_eq!(f.look_dx, 0.0);
        assert!(!f.locked);
    }
}
