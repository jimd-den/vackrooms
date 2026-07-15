//! Conservative finite-range fixture clustering for a world-space AABB.
//!
//! Both rewritten GPU paths use this exact predicate. It removes work, not
//! light: a fixture is omitted only when no point on its rectangle can be
//! within its compact-support range of any receiver in the bounds.

use crate::application::ports::LightSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightRange {
    pub first: i32,
    pub count: i32,
}

pub fn append_lights_reaching_bounds(
    lights: &[LightSource],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    clustered: &mut Vec<LightSource>,
) -> LightRange {
    let first = clustered.len();
    clustered.extend(
        lights
            .iter()
            .copied()
            .filter(|light| reaches_bounds(light, bounds_min, bounds_max)),
    );
    LightRange {
        first: first.min(i32::MAX as usize) as i32,
        count: clustered.len().saturating_sub(first).min(i32::MAX as usize) as i32,
    }
}

fn reaches_bounds(light: &LightSource, bounds_min: [f32; 3], bounds_max: [f32; 3]) -> bool {
    // Expanding X/Z by the emitter half-size converts rectangle-to-box
    // distance into point-to-box distance (the Minkowski-sum construction).
    let expanded_min = [
        bounds_min[0] - light.half_size[0],
        bounds_min[1],
        bounds_min[2] - light.half_size[1],
    ];
    let expanded_max = [
        bounds_max[0] + light.half_size[0],
        bounds_max[1],
        bounds_max[2] + light.half_size[1],
    ];
    let mut distance_squared = 0.0;
    for axis in 0..3 {
        let delta = if light.position[axis] < expanded_min[axis] {
            expanded_min[axis] - light.position[axis]
        } else if light.position[axis] > expanded_max[axis] {
            light.position[axis] - expanded_max[axis]
        } else {
            0.0
        };
        distance_squared += delta * delta;
    }
    distance_squared <= light.radius * light.radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ports::{LightKind, LightSource};

    fn light(id: u64, position: [f32; 3]) -> LightSource {
        LightSource {
            id,
            position,
            half_size: [0.5, 0.25],
            color: [1.0; 3],
            radius: 2.0,
            intensity: 1.0,
            kind: LightKind::CeilingPanel,
            flicker_mode: 0,
            enabled: true,
        }
    }

    #[test]
    fn clustering_keeps_every_reaching_light_without_a_slot_cap() {
        let lights: Vec<_> = (0..40)
            .map(|id| light(id, [0.5, 1.0, 0.5]))
            .chain(std::iter::once(light(99, [20.0, 1.0, 20.0])))
            .collect();
        let mut clustered = Vec::new();
        let range = append_lights_reaching_bounds(
            &lights,
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
            &mut clustered,
        );
        assert_eq!(
            range,
            LightRange {
                first: 0,
                count: 40
            }
        );
        assert_eq!(clustered.len(), 40);
    }

    #[test]
    fn rectangle_extent_is_included_in_the_conservative_distance() {
        let wide = LightSource {
            half_size: [2.0, 0.25],
            radius: 1.0,
            ..light(1, [2.5, 1.0, 0.5])
        };
        let mut clustered = Vec::new();
        let range = append_lights_reaching_bounds(
            &[wide],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
            &mut clustered,
        );
        assert_eq!(range.count, 1);
    }
}
