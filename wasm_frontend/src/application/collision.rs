//! Axis-aligned collision primitives and the sliding-collision world.
//!
//! Collision geometry is *derived from the SVO itself* (solid leaf nodes of
//! type WALL / RED_WALL become world-space boxes), so the physical world and
//! the rendered world can never drift apart.

/// Axis-aligned bounding box in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        self.max[0] > other.min[0]
            && self.min[0] < other.max[0]
            && self.max[1] > other.min[1]
            && self.min[1] < other.max[1]
            && self.max[2] > other.min[2]
            && self.min[2] < other.max[2]
    }
}

/// Player capsule approximated as an AABB, matching the original client:
/// radius 0.35 around the eye, extending 1.65 below and 0.1 above eye level.
pub const PLAYER_RADIUS: f32 = 0.35;
pub const PLAYER_EYE_TO_FEET: f32 = 1.65;
pub const PLAYER_EYE_TO_HEAD: f32 = 0.1;

/// Builds the player's AABB from an eye-level position.
pub fn player_aabb(eye_pos: [f32; 3]) -> Aabb {
    Aabb::new(
        [
            eye_pos[0] - PLAYER_RADIUS,
            eye_pos[1] - PLAYER_EYE_TO_FEET,
            eye_pos[2] - PLAYER_RADIUS,
        ],
        [
            eye_pos[0] + PLAYER_RADIUS,
            eye_pos[1] + PLAYER_EYE_TO_HEAD,
            eye_pos[2] + PLAYER_RADIUS,
        ],
    )
}

/// The set of solid boxes the player can collide with.
/// Rebuilt whenever the chunk set changes.
#[derive(Debug, Default)]
pub struct CollisionWorld {
    boxes: Vec<Aabb>,
}

impl CollisionWorld {
    pub fn new() -> Self {
        Self { boxes: Vec::new() }
    }

    pub fn rebuild<'a, I: IntoIterator<Item = &'a Aabb>>(&mut self, boxes: I) {
        self.boxes.clear();
        self.boxes.extend(boxes.into_iter().copied());
    }

    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }

    pub fn boxes(&self) -> &[Aabb] {
        &self.boxes
    }

    /// True if a player standing at `eye_pos` overlaps any solid box.
    pub fn collides(&self, eye_pos: [f32; 3]) -> bool {
        let p = player_aabb(eye_pos);
        self.boxes.iter().any(|b| p.intersects(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_intersection_is_exclusive_on_touching_faces() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let touching = Aabb::new([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        let overlapping = Aabb::new([0.9, 0.0, 0.0], [2.0, 1.0, 1.0]);
        assert!(!a.intersects(&touching));
        assert!(a.intersects(&overlapping));
    }

    #[test]
    fn player_collides_with_wall_at_waist_height() {
        let mut world = CollisionWorld::new();
        world.rebuild(&[Aabb::new([1.0, 0.0, 0.0], [1.2, 3.0, 5.0])]);
        // Eye at 1.7 — wall spans full height, player is 0.35 wide.
        assert!(world.collides([1.0, 1.7, 2.0]));
        assert!(!world.collides([0.0, 1.7, 2.0]));
    }

    #[test]
    fn player_does_not_collide_with_floor_slab_below_feet() {
        let mut world = CollisionWorld::new();
        // Floor slab 0.0..0.05 — feet bottom is at 1.7 - 1.65 = 0.05 exactly:
        // touching, not overlapping.
        world.rebuild(&[Aabb::new([-10.0, 0.0, -10.0], [10.0, 0.05, 10.0])]);
        assert!(!world.collides([0.0, 1.7, 0.0]));
    }
}
