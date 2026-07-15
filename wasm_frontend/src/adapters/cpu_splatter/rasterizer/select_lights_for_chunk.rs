//! Build an exact finite-support fixture cluster for one voxel chunk.
//!
//! Traversal shades many splats from the same chunk. Scanning every world
//! fixture for every splat would dominate the CPU path, so the frame scans
//! the fixture list once per chunk. A light is rejected only when the minimum
//! distance between its emitter geometry and the chunk AABB exceeds its
//! authored finite range; this changes no radiometric result.

use crate::application::ports::{ChunkDraw, LightKind, LightSource};

pub(super) fn select_lights_for_chunk(
    scene_lights: &[LightSource],
    chunk: &ChunkDraw,
    selected: &mut Vec<LightSource>,
) {
    selected.clear();
    selected.extend(
        scene_lights
            .iter()
            .copied()
            .filter(|light| light_can_reach_chunk(light, chunk)),
    );
}

fn light_can_reach_chunk(light: &LightSource, chunk: &ChunkDraw) -> bool {
    if !light.enabled
        || !light.radius.is_finite()
        || light.radius <= 0.0
        || light
            .position
            .iter()
            .any(|coordinate| !coordinate.is_finite())
    {
        return false;
    }

    let emitter_half_size = match light.kind {
        LightKind::Point => [0.0, 0.0],
        LightKind::CeilingPanel | LightKind::Strip | LightKind::Emergency => {
            if light
                .half_size
                .iter()
                .any(|half| !half.is_finite() || *half <= 0.0)
            {
                return false;
            }
            light.half_size
        }
    };

    let chunk_max = chunk.origin.map(|minimum| minimum + chunk.world_size);
    let dx = interval_distance(
        light.position[0] - emitter_half_size[0],
        light.position[0] + emitter_half_size[0],
        chunk.origin[0],
        chunk_max[0],
    );
    let dy = point_interval_distance(light.position[1], chunk.origin[1], chunk_max[1]);
    let dz = interval_distance(
        light.position[2] - emitter_half_size[1],
        light.position[2] + emitter_half_size[1],
        chunk.origin[2],
        chunk_max[2],
    );
    dx * dx + dy * dy + dz * dz <= light.radius * light.radius
}

fn interval_distance(a_min: f32, a_max: f32, b_min: f32, b_max: f32) -> f32 {
    if a_max < b_min {
        b_min - a_max
    } else if b_max < a_min {
        a_min - b_max
    } else {
        0.0
    }
}

fn point_interval_distance(point: f32, minimum: f32, maximum: f32) -> f32 {
    if point < minimum {
        minimum - point
    } else if point > maximum {
        point - maximum
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use crate::application::ports::{ChunkDraw, LightKind, LightSource};

    use super::select_lights_for_chunk;

    fn chunk() -> ChunkDraw {
        ChunkDraw {
            origin: [0.0; 3],
            root_index: 0,
            world_size: 4.0,
            voxel_size: 0.5,
            svo_depth: 3,
        }
    }

    fn light(position: [f32; 3], radius: f32) -> LightSource {
        LightSource {
            id: 7,
            position,
            radius,
            intensity: 1.0,
            color: [1.0; 3],
            enabled: true,
            ..LightSource::default()
        }
    }

    #[test]
    fn point_support_rejects_only_lights_outside_the_chunk_range() {
        let lights = [light([5.0, 2.0, 2.0], 1.0), light([5.1, 2.0, 2.0], 1.0)];
        let mut selected = Vec::new();

        select_lights_for_chunk(&lights, &chunk(), &mut selected);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].position, [5.0, 2.0, 2.0]);
    }

    #[test]
    fn rectangular_extent_is_part_of_the_conservative_intersection() {
        let panel = LightSource {
            kind: LightKind::CeilingPanel,
            half_size: [2.0, 0.5],
            ..light([6.5, 2.0, 2.0], 0.5)
        };
        let mut selected = Vec::new();

        select_lights_for_chunk(&[panel], &chunk(), &mut selected);

        assert_eq!(selected, vec![panel]);
    }

    #[test]
    fn disabled_lights_are_not_carried_into_splat_shading() {
        let disabled = LightSource {
            enabled: false,
            ..light([2.0; 3], 10.0)
        };
        let mut selected = Vec::new();

        select_lights_for_chunk(&[disabled], &chunk(), &mut selected);

        assert!(selected.is_empty());
    }
}
