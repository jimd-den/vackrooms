//! Adaptive internal-resolution policy, independent of renderer and browser.

/// Adaptive internal-resolution governor. Low-spec machines get fewer rays
/// per frame by rendering at a reduced backing-store resolution; the driver
/// applies the scale to the canvas. Hysteresis + cooldown prevent flapping.
#[derive(Debug, Clone, Copy)]
pub struct PerfGovernor {
    ema_frame_ms: f32,
    scale: f32,
    cooldown_frames: u32,
    /// Frames since the last scale-up attempt (saturating).
    frames_since_raise: u32,
    /// While positive, scale-ups are locked out because a recent raise
    /// immediately overloaded the GPU and had to be reverted (anti-flap).
    raise_lockout_frames: u32,
}

impl PerfGovernor {
    const SCALES: [f32; 4] = [0.5, 0.65, 0.8, 1.0];
    /// A drop this soon after a raise counts as a failed raise.
    const RAISE_PROBE_FRAMES: u32 = 600;
    /// How long a failed raise blocks further raise attempts (~1 min).
    const RAISE_LOCKOUT_FRAMES: u32 = 3600;

    pub fn new() -> Self {
        Self {
            ema_frame_ms: 16.0,
            scale: 0.8,
            cooldown_frames: 0,
            frames_since_raise: u32::MAX,
            raise_lockout_frames: 0,
        }
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn update(&mut self, frame_ms: f32) {
        self.ema_frame_ms = self.ema_frame_ms * 0.95 + frame_ms.clamp(0.0, 100.0) * 0.05;
        self.frames_since_raise = self.frames_since_raise.saturating_add(1);
        self.raise_lockout_frames = self.raise_lockout_frames.saturating_sub(1);
        if self.cooldown_frames > 0 {
            self.cooldown_frames -= 1;
            return;
        }
        let idx = Self::SCALES
            .iter()
            .position(|&s| s == self.scale)
            .unwrap_or(2);
        if self.ema_frame_ms > 33.0 && idx > 0 {
            // Sustained under ~30 fps: drop one resolution step. If this
            // happens right after a raise, the raise failed — lock raises
            // out for a while so the scale doesn't oscillate.
            if self.frames_since_raise < Self::RAISE_PROBE_FRAMES {
                self.raise_lockout_frames = Self::RAISE_LOCKOUT_FRAMES;
            }
            self.scale = Self::SCALES[idx - 1];
            self.cooldown_frames = 120;
        } else if self.ema_frame_ms < 17.5
            && idx + 1 < Self::SCALES.len()
            && self.raise_lockout_frames == 0
        {
            // Holding 60Hz vsync (rAF EMA floors at ~16.7 ms, so a lower
            // threshold would never fire): probe one resolution step up.
            self.scale = Self::SCALES[idx + 1];
            self.cooldown_frames = 240;
            self.frames_since_raise = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn governor_drops_resolution_under_sustained_load() {
        let mut governor = PerfGovernor::new();
        for _ in 0..200 {
            governor.update(50.0); // 20 fps
        }
        assert!(governor.scale() < 0.8);
    }

    #[test]
    fn governor_raises_resolution_when_holding_60hz_vsync() {
        let mut governor = PerfGovernor::new();
        // rAF on a 60Hz display floors at ~16.7ms even with GPU headroom.
        for _ in 0..600 {
            governor.update(16.7);
        }
        assert_eq!(governor.scale(), 1.0, "must reach native res under vsync");
    }

    #[test]
    fn governor_locks_out_raises_after_a_failed_probe() {
        let mut governor = PerfGovernor::new();
        for _ in 0..300 {
            governor.update(16.7); // raises to 1.0 almost immediately
        }
        assert_eq!(governor.scale(), 1.0);
        for _ in 0..600 {
            governor.update(40.0); // raise fails: overloaded at 1.0
        }
        assert!(governor.scale() < 1.0);
        let settled = governor.scale();
        for _ in 0..600 {
            governor.update(16.7); // healthy again, but inside the lockout
        }
        assert_eq!(governor.scale(), settled, "raise must stay locked out");
    }
}
