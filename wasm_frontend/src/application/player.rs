//! First-person player controller: mouse look, WASD movement with friction,
//! and axis-separated sliding collision against the [`CollisionWorld`].
//!
//! The tuning constants are carried over 1:1 from the original JS client so
//! the wasm port feels identical.

use crate::application::collision::CollisionWorld;

/// Acceleration applied while a movement key is held (units/s^2 toward the
/// wish direction). Terminal speed = ACCELERATION / FRICTION = 2.5 u/s (increased from 0.5 u/s).
const ACCELERATION: f32 = 20.0;
/// Exponential velocity damping factor (1/s).
const FRICTION: f32 = 8.0;
/// Radians of yaw/pitch per pixel of mouse movement.
const MOUSE_SENSITIVITY: f32 = 0.002;
/// Pitch is clamped just short of straight up/down to avoid gimbal flip.
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.05;

/// Which movement keys are currently held.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoveIntent {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub turn_left: bool,
    pub turn_right: bool,
}

/// Player state. Position is the *eye* position in world space.
#[derive(Debug, Clone)]
pub struct Player {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    /// Velocity in camera-local space: x = strafe, z = forward.
    vel_strafe: f32,
    vel_forward: f32,
}

impl Player {
    pub fn new(position: [f32; 3]) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            vel_strafe: 0.0,
            vel_forward: 0.0,
        }
    }

    /// Applies a mouse movement delta (in pixels) to yaw/pitch.
    pub fn apply_look(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * MOUSE_SENSITIVITY;
        self.pitch = (self.pitch - dy * MOUSE_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// World-space unit forward vector projected on the ground plane.
    pub fn forward(&self) -> [f32; 3] {
        [-self.yaw.sin(), 0.0, -self.yaw.cos()]
    }

    /// World-space unit right vector on the ground plane.
    pub fn right(&self) -> [f32; 3] {
        [self.yaw.cos(), 0.0, -self.yaw.sin()]
    }

    /// Integrates one physics step and resolves collision by sliding:
    /// the X and Z axes are moved and tested independently, so hitting a
    /// wall diagonally slides the player along it instead of stopping dead.
    pub fn step(&mut self, dt: f32, intent: &MoveIntent, world: &CollisionWorld) {
        // Friction: exponential-style decay, matching `v -= v * 8 * dt`.
        self.vel_strafe -= self.vel_strafe * FRICTION * dt;
        self.vel_forward -= self.vel_forward * FRICTION * dt;

        // Turn via keyboard (Doom controls)
        let turn_speed = 3.0; // radians per second
        if intent.turn_left {
            self.yaw += turn_speed * dt;
        }
        if intent.turn_right {
            self.yaw -= turn_speed * dt;
        }

        let mut wish_forward = (intent.forward as i32 - intent.backward as i32) as f32;
        let mut wish_strafe = (intent.right as i32 - intent.left as i32) as f32;
        let len = (wish_forward * wish_forward + wish_strafe * wish_strafe).sqrt();
        if len > 0.0 {
            wish_forward /= len;
            wish_strafe /= len;
        }

        self.vel_forward += wish_forward * ACCELERATION * dt;
        self.vel_strafe += wish_strafe * ACCELERATION * dt;

        let fwd = self.forward();
        let rgt = self.right();
        let step_x = (fwd[0] * self.vel_forward + rgt[0] * self.vel_strafe) * dt;
        let step_z = (fwd[2] * self.vel_forward + rgt[2] * self.vel_strafe) * dt;

        let old = self.position;

        self.position[0] = old[0] + step_x;
        if world.collides(self.position) {
            self.position[0] = old[0];
        }

        self.position[2] = old[2] + step_z;
        if world.collides(self.position) {
            self.position[2] = old[2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::collision::Aabb;

    fn walk(player: &mut Player, world: &CollisionWorld, intent: MoveIntent, seconds: f32) {
        let dt = 1.0 / 60.0;
        let steps = (seconds / dt) as usize;
        for _ in 0..steps {
            player.step(dt, &intent, world);
        }
    }

    #[test]
    fn forward_intent_moves_along_negative_z_at_zero_yaw() {
        let world = CollisionWorld::new();
        let mut p = Player::new([0.0, 1.7, 0.0]);
        walk(
            &mut p,
            &world,
            MoveIntent {
                forward: true,
                ..Default::default()
            },
            2.0,
        );
        assert!(
            p.position[2] < -0.5,
            "expected forward motion, z = {}",
            p.position[2]
        );
        assert!(p.position[0].abs() < 1e-3);
    }

    #[test]
    fn speed_converges_to_terminal_velocity() {
        let world = CollisionWorld::new();
        let mut p = Player::new([0.0, 1.7, 0.0]);
        walk(
            &mut p,
            &world,
            MoveIntent {
                forward: true,
                ..Default::default()
            },
            5.0,
        );
        let before = p.position[2];
        walk(
            &mut p,
            &world,
            MoveIntent {
                forward: true,
                ..Default::default()
            },
            1.0,
        );
        let speed = (before - p.position[2]).abs();
        // Terminal speed = ACCELERATION / FRICTION = 2.5 u/s (was 0.5 u/s).
        assert!((speed - 2.5).abs() < 0.05, "speed was {speed}");
    }

    #[test]
    fn diagonal_wall_hit_slides_along_wall() {
        let mut world = CollisionWorld::new();
        // Wall blocking motion in -z ahead of the player.
        world.rebuild(&[Aabb::new([-10.0, 0.0, -2.0], [10.0, 3.0, -1.8])]);
        let mut p = Player::new([0.0, 1.7, 0.0]);
        // Move forward-right: z blocked by the wall, x should still advance.
        walk(
            &mut p,
            &world,
            MoveIntent {
                forward: true,
                right: true,
                ..Default::default()
            },
            4.0,
        );
        assert!(p.position[2] > -1.5, "should be stopped by wall");
        assert!(
            p.position[0] > 0.5,
            "should have slid along the wall, x = {}",
            p.position[0]
        );
    }
}
