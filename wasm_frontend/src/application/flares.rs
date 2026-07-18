//! Dropped-flare lifecycle and per-frame light selection.
//!
//! Flares are transient world objects rather than chunk data. This module
//! owns their age, insertion order, collision-aware placement, retirement,
//! and clockless renderer payload. The engine only supplies player/world
//! context and the shared simulation time.

use crate::application::collision::{Aabb, CollisionWorld};
use crate::application::ports::{DynamicLight, MAX_DYNAMIC_LIGHTS};

/// Flare lifetime in seconds; intensity fades during the final tenth.
const FLARE_LIFETIME_S: f32 = 90.0;
/// Global active-flare cap. A new drop retires the oldest flare at the cap.
const FLARE_CAP: usize = 12;
/// Flares farther than this horizontal distance do not light the frame.
const FLARE_CULL_DISTANCE: f32 = 40.0;
/// Local warm glow radius of one flare.
const FLARE_LIGHT_RADIUS: f32 = 9.0;
/// Horizontal distance from the player at which a drop is attempted.
const FLARE_DROP_DISTANCE: f32 = 0.9;
/// Flare flame height above the floor slab.
const FLARE_HEIGHT: f32 = 0.3;
/// Height above the flame used to detect geometry remapped over a flare.
const BURIED_PROBE_HEIGHT: f32 = 0.2;

/// One dropped flare: pure runtime world-space state. Never serialized into
/// chunks, never part of the reality snapshot, and preserved untouched by
/// ordinary chunk streaming.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flare {
    pub position: [f32; 3],
    pub age_seconds: f32,
    /// Monotonic drop order; doubles as the deterministic flicker phase.
    pub sequence: u64,
}

impl Flare {
    pub fn remaining_seconds(&self) -> f32 {
        (FLARE_LIFETIME_S - self.age_seconds).max(0.0)
    }
}

/// Fixed renderer payload and the meaningful prefix length.
pub(crate) struct FlareLightSet {
    pub(crate) lights: [DynamicLight; MAX_DYNAMIC_LIGHTS],
    pub(crate) count: u8,
}

/// Active flare collection, kept oldest first.
#[derive(Debug, Default)]
pub(crate) struct FlareField {
    active: Vec<Flare>,
    sequence: u64,
}

impl FlareField {
    /// Advances every flare and retires those whose lifetime has elapsed.
    pub(crate) fn advance(&mut self, dt_seconds: f32) {
        for flare in &mut self.active {
            flare.age_seconds += dt_seconds;
        }
        self.active
            .retain(|flare| flare.age_seconds < FLARE_LIFETIME_S);
    }

    /// Drops a flare slightly ahead of the player unless that placement
    /// would collide, in which case the known-valid player position wins.
    pub(crate) fn drop_from(
        &mut self,
        eye: [f32; 3],
        forward: [f32; 3],
        collision: &CollisionWorld,
    ) {
        let ahead = [
            eye[0] + forward[0] * FLARE_DROP_DISTANCE,
            eye[1],
            eye[2] + forward[2] * FLARE_DROP_DISTANCE,
        ];
        let spot = if collision.collides(ahead) {
            eye
        } else {
            ahead
        };

        if self.active.len() >= FLARE_CAP {
            self.active.remove(0);
        }
        self.sequence += 1;
        self.active.push(Flare {
            position: [spot[0], FLARE_HEIGHT, spot[2]],
            age_seconds: 0.0,
            sequence: self.sequence,
        });
    }

    /// Extinguishes flares covered by newly installed collision geometry.
    /// The inclusive point test intentionally matches the former engine
    /// policy at box faces.
    pub(crate) fn extinguish_buried(&mut self, boxes: &[Aabb]) {
        self.active.retain(|flare| {
            let probe = [
                flare.position[0],
                flare.position[1] + BURIED_PROBE_HEIGHT,
                flare.position[2],
            ];
            !boxes.iter().any(|bounds| point_in_aabb(probe, bounds))
        });
    }

    /// Builds the nearest-light payload once. Selection, count, fade, and
    /// deterministic flicker therefore cannot disagree within a frame.
    pub(crate) fn lights(&self, camera: [f32; 3], time_seconds: f32) -> FlareLightSet {
        let nearest = self.nearest(camera);
        let count = nearest.len() as u8;
        let mut lights = [DynamicLight::default(); MAX_DYNAMIC_LIGHTS];

        for (slot, flare) in nearest.into_iter().enumerate() {
            let fade = (flare.remaining_seconds() / (FLARE_LIFETIME_S * 0.1)).clamp(0.0, 1.0);
            let phase = time_seconds * 13.0 + flare.sequence as f32 * 1.7;
            let flicker = 0.85 + 0.15 * phase.sin() * (phase * 0.37).cos();
            lights[slot] = DynamicLight {
                position: flare.position,
                color: [1.0, 0.45, 0.18],
                radius: FLARE_LIGHT_RADIUS,
                intensity: 1.6 * fade * flicker,
            };
        }

        FlareLightSet { lights, count }
    }

    /// Removes active flares without rewinding their monotonic identity.
    pub(crate) fn clear(&mut self) {
        self.active.clear();
    }

    pub(crate) fn active(&self) -> &[Flare] {
        &self.active
    }

    pub(crate) fn len(&self) -> usize {
        self.active.len()
    }

    fn nearest(&self, camera: [f32; 3]) -> Vec<&Flare> {
        let cull_distance_squared = FLARE_CULL_DISTANCE * FLARE_CULL_DISTANCE;
        let distance_squared = |flare: &Flare| {
            let dx = flare.position[0] - camera[0];
            let dz = flare.position[2] - camera[2];
            dx * dx + dz * dz
        };
        let mut nearest: Vec<_> = self
            .active
            .iter()
            .filter(|flare| distance_squared(flare) < cull_distance_squared)
            .collect();
        nearest.sort_by(|a, b| {
            distance_squared(a)
                .total_cmp(&distance_squared(b))
                .then_with(|| a.sequence.cmp(&b.sequence))
        });
        nearest.truncate(MAX_DYNAMIC_LIGHTS);
        nearest
    }
}

fn point_in_aabb(point: [f32; 3], bounds: &Aabb) -> bool {
    point[0] >= bounds.min[0]
        && point[0] <= bounds.max[0]
        && point[1] >= bounds.min[1]
        && point[1] <= bounds.max[1]
        && point[2] >= bounds.min[2]
        && point[2] <= bounds.max[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_world() -> CollisionWorld {
        CollisionWorld::new()
    }

    #[test]
    fn aging_retires_a_flare_at_its_exact_lifetime() {
        let mut field = FlareField::default();
        field.drop_from([0.0, 1.7, 0.0], [0.0, 0.0, -1.0], &empty_world());

        field.advance(FLARE_LIFETIME_S - 0.25);
        assert_eq!(field.len(), 1);
        assert_eq!(field.active()[0].remaining_seconds(), 0.25);

        field.advance(0.25);
        assert!(field.active().is_empty());
    }

    #[test]
    fn cap_retires_the_oldest_and_preserves_drop_order() {
        let mut field = FlareField::default();
        let world = empty_world();
        for x in 0..=FLARE_CAP {
            field.drop_from([x as f32, 1.7, 0.0], [0.0; 3], &world);
        }

        assert_eq!(field.len(), FLARE_CAP);
        assert_eq!(field.active().first().unwrap().sequence, 2);
        assert_eq!(
            field.active().last().unwrap().sequence,
            (FLARE_CAP + 1) as u64
        );
    }

    #[test]
    fn blocked_drop_falls_back_and_new_geometry_extinguishes_it() {
        let mut world = CollisionWorld::new();
        world.rebuild(&[Aabb::new([0.7, 0.0, -0.2], [1.1, 3.0, 0.2])]);
        let mut field = FlareField::default();
        field.drop_from([0.0, 1.7, 0.0], [1.0, 0.0, 0.0], &world);
        assert_eq!(field.active()[0].position, [0.0, FLARE_HEIGHT, 0.0]);

        field.extinguish_buried(&[Aabb::new(
            [-0.1, FLARE_HEIGHT + BURIED_PROBE_HEIGHT, -0.1],
            [0.1, 1.0, 0.1],
        )]);
        assert!(field.active().is_empty());
    }

    #[test]
    fn nearest_lights_are_selected_once_with_stable_distance_ties() {
        let mut field = FlareField::default();
        let world = empty_world();
        for x in [4.0, 1.0, -1.0, 3.0, 2.0, 39.0, 40.0] {
            field.drop_from([x, 1.7, 0.0], [0.0; 3], &world);
        }

        let selected = field.lights([0.0, 1.7, 0.0], 2.5);
        assert_eq!(selected.count as usize, MAX_DYNAMIC_LIGHTS);
        let positions: Vec<f32> = selected.lights[..selected.count as usize]
            .iter()
            .map(|light| light.position[0])
            .collect();
        assert_eq!(positions, vec![1.0, -1.0, 2.0, 3.0]);
        assert!(
            selected.lights[..selected.count as usize]
                .iter()
                .all(|light| light.radius == FLARE_LIGHT_RADIUS && light.intensity > 0.0)
        );
    }

    #[test]
    fn clearing_a_level_keeps_sequence_monotonic() {
        let mut field = FlareField::default();
        let world = empty_world();
        field.drop_from([0.0, 1.7, 0.0], [0.0; 3], &world);
        field.clear();
        field.drop_from([0.0, 1.7, 0.0], [0.0; 3], &world);
        assert_eq!(field.active()[0].sequence, 2);
    }
}
