//! Axis-aligned collision primitives and the sliding-collision world.
//!
//! Collision geometry is *derived from the SVO itself* (solid leaf nodes of
//! type WALL / RED_WALL become world-space boxes), so the physical world and
//! the rendered world can never drift apart.

use std::collections::HashMap;

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

type SpatialCell = (i64, i64);

const SPATIAL_CELL_SIZE: f32 = 4.0;
const MAX_CELLS_PER_BOX: usize = 256;

/// The set of solid boxes the player can collide with.
///
/// Boxes remain canonical in insertion order for diagnostics. Two derived
/// X/Z indexes serve the hot queries: occupied cells for exact collision and
/// center cells for thermal enclosure. Rebuilding is event-driven when the
/// resident chunk set changes; steady-state frames only query the hashes.
#[derive(Debug)]
pub struct CollisionWorld {
    boxes: Vec<Aabb>,
    occupied_cells: HashMap<SpatialCell, Vec<usize>>,
    center_cells: HashMap<SpatialCell, Vec<usize>>,
    fallback: Vec<usize>,
}

impl CollisionWorld {
    pub fn new() -> Self {
        Self {
            boxes: Vec::new(),
            occupied_cells: HashMap::new(),
            center_cells: HashMap::new(),
            fallback: Vec::new(),
        }
    }

    pub fn rebuild<'a, I: IntoIterator<Item = &'a Aabb>>(&mut self, boxes: I) {
        self.boxes.clear();
        self.boxes.extend(boxes.into_iter().copied());
        self.occupied_cells.clear();
        self.center_cells.clear();
        self.fallback.clear();

        for (index, aabb) in self.boxes.iter().enumerate() {
            let Some((min_cell, max_cell)) = cells_for_bounds(aabb.min, aabb.max) else {
                self.fallback.push(index);
                continue;
            };
            let width = cell_count(min_cell.0, max_cell.0).unwrap_or(usize::MAX);
            let depth = cell_count(min_cell.1, max_cell.1).unwrap_or(usize::MAX);
            if width.saturating_mul(depth) > MAX_CELLS_PER_BOX {
                self.fallback.push(index);
                continue;
            }

            for cell_z in min_cell.1..=max_cell.1 {
                for cell_x in min_cell.0..=max_cell.0 {
                    self.occupied_cells
                        .entry((cell_x, cell_z))
                        .or_default()
                        .push(index);
                }
            }

            let center_x = (aabb.min[0] + aabb.max[0]) * 0.5;
            let center_z = (aabb.min[2] + aabb.max[2]) * 0.5;
            self.center_cells
                .entry(spatial_cell(center_x, center_z))
                .or_default()
                .push(index);
        }
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
        self.intersects(&player_aabb(eye_pos))
    }

    /// Counts box centers within an X/Z radius without scanning distant
    /// chunks. This preserves the thermal model's exact center-distance rule.
    pub fn count_centers_within_xz(&self, point: [f32; 3], radius: f32) -> usize {
        if !point[0].is_finite() || !point[2].is_finite() || !radius.is_finite() || radius < 0.0 {
            return 0;
        }
        let min = spatial_cell(point[0] - radius, point[2] - radius);
        let max = spatial_cell(point[0] + radius, point[2] + radius);
        let radius_squared = radius * radius;
        let mut count = 0;
        for cell_z in min.1..=max.1 {
            for cell_x in min.0..=max.0 {
                let Some(indices) = self.center_cells.get(&(cell_x, cell_z)) else {
                    continue;
                };
                count += indices
                    .iter()
                    .filter(|&&index| center_within_xz(self.boxes[index], point, radius_squared))
                    .count();
            }
        }
        count
            + self
                .fallback
                .iter()
                .filter(|&&index| center_within_xz(self.boxes[index], point, radius_squared))
                .count()
    }

    fn intersects(&self, query: &Aabb) -> bool {
        let indexed_hit = cells_for_bounds(query.min, query.max).is_some_and(|(min, max)| {
            (min.1..=max.1).any(|cell_z| {
                (min.0..=max.0).any(|cell_x| {
                    self.occupied_cells
                        .get(&(cell_x, cell_z))
                        .is_some_and(|indices| {
                            indices
                                .iter()
                                .any(|&index| query.intersects(&self.boxes[index]))
                        })
                })
            })
        });
        indexed_hit
            || self
                .fallback
                .iter()
                .any(|&index| query.intersects(&self.boxes[index]))
    }
}

impl Default for CollisionWorld {
    fn default() -> Self {
        Self::new()
    }
}

fn spatial_cell(x: f32, z: f32) -> SpatialCell {
    (
        (x / SPATIAL_CELL_SIZE).floor() as i64,
        (z / SPATIAL_CELL_SIZE).floor() as i64,
    )
}

fn cells_for_bounds(min: [f32; 3], max: [f32; 3]) -> Option<(SpatialCell, SpatialCell)> {
    if ![min[0], min[2], max[0], max[2]]
        .into_iter()
        .all(f32::is_finite)
        || max[0] < min[0]
        || max[2] < min[2]
    {
        return None;
    }
    Some((spatial_cell(min[0], min[2]), spatial_cell(max[0], max[2])))
}

fn cell_count(first: i64, last: i64) -> Option<usize> {
    last.checked_sub(first)
        .and_then(|distance| distance.checked_add(1))
        .and_then(|count| usize::try_from(count).ok())
}

fn center_within_xz(aabb: Aabb, point: [f32; 3], radius_squared: f32) -> bool {
    let center_x = (aabb.min[0] + aabb.max[0]) * 0.5;
    let center_z = (aabb.min[2] + aabb.max[2]) * 0.5;
    let dx = center_x - point[0];
    let dz = center_z - point[2];
    dx * dx + dz * dz <= radius_squared
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

    #[test]
    fn spatial_index_handles_negative_and_cross_cell_boxes() {
        let boxes = [
            Aabb::new([-4.2, 0.0, -0.2], [-3.8, 3.0, 0.2]),
            Aabb::new([7.8, 0.0, 7.8], [8.2, 3.0, 8.2]),
        ];
        let mut world = CollisionWorld::new();
        world.rebuild(&boxes);

        assert!(world.collides([-4.0, 1.7, 0.0]));
        assert!(world.collides([8.0, 1.7, 8.0]));
        assert!(!world.collides([2.0, 1.7, 2.0]));
    }

    #[test]
    fn oversized_boxes_use_the_exact_fallback() {
        let huge = Aabb::new([-100.0, 0.0, -100.0], [100.0, 3.0, 100.0]);
        let mut world = CollisionWorld::new();
        world.rebuild(&[huge]);

        assert!(world.collides([0.0, 1.7, 0.0]));
        assert_eq!(world.count_centers_within_xz([0.0, 0.0, 0.0], 0.1), 1);
    }

    #[test]
    fn center_query_matches_brute_force() {
        let boxes: Vec<_> = (-20..=20)
            .map(|x| Aabb::new([x as f32, 0.0, -0.2], [x as f32 + 0.4, 3.0, 0.2]))
            .collect();
        let mut world = CollisionWorld::new();
        world.rebuild(&boxes);
        let point = [-3.25, 1.7, 0.0];
        let radius = 6.0;
        let expected = boxes
            .iter()
            .filter(|&&aabb| center_within_xz(aabb, point, radius * radius))
            .count();

        assert_eq!(world.count_centers_within_xz(point, radius), expected);
    }
}
