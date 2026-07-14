//! Prepare the immutable fixture list consumed by a render frame.
//!
//! Chunk payloads overlap on purpose, so the same ceiling panel can arrive
//! from several neighboring halos. Selection belongs here, before any GPU
//! driver: all renderers must see the same deduplicated physical scene.

use std::collections::HashMap;

use crate::application::ports::LightSource;

/// Collects every finite, enabled fixture and orders it by conservative
/// contribution at `camera_position`.
///
/// The ordering is not a budget: no fixture is discarded. It keeps the hero
/// shadow choice stable and makes output independent of hash-map/chunk arrival
/// order. Projected emitter area and authored intensity raise importance while
/// squared distance lowers it.
pub fn select_scene_lights<'a>(
    lights: impl Iterator<Item = &'a LightSource>,
    camera_position: [f32; 3],
) -> Vec<LightSource> {
    let mut unique = HashMap::<u64, LightSource>::new();
    for light in lights.copied() {
        if light.enabled && valid(&light) {
            unique.entry(light.id).or_insert(light);
        }
    }

    let mut ranked: Vec<(LightSource, f32)> = unique
        .into_values()
        .map(|light| {
            let delta = [
                light.position[0] - camera_position[0],
                light.position[1] - camera_position[1],
                light.position[2] - camera_position[2],
            ];
            let distance_squared = delta.into_iter().map(|v| v * v).sum::<f32>();
            let area = (4.0 * light.half_size[0] * light.half_size[1]).max(0.01);
            let importance = light.intensity * area / (distance_squared + area);
            (light, importance)
        })
        .collect();
    ranked.sort_by(|(a, a_score), (b, b_score)| {
        b_score.total_cmp(a_score).then_with(|| a.id.cmp(&b.id))
    });

    ranked.into_iter().map(|(light, _)| light).collect()
}

fn valid(light: &LightSource) -> bool {
    light.position.into_iter().all(f32::is_finite)
        && light.color.into_iter().all(f32::is_finite)
        && light
            .half_size
            .into_iter()
            .all(|v| v.is_finite() && v >= 0.0)
        && light.radius.is_finite()
        && light.radius > 0.0
        && light.intensity.is_finite()
        && light.intensity > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ports::LightKind;

    fn light(id: u64, x: f32) -> LightSource {
        LightSource {
            id,
            position: [x, 3.0, 0.0],
            half_size: [1.0, 0.3],
            color: [1.0, 0.9, 0.7],
            radius: 12.0,
            intensity: 10.0,
            kind: LightKind::CeilingPanel,
            flicker_mode: 0,
            enabled: true,
        }
    }

    #[test]
    fn duplicate_halo_records_become_one_fixture() {
        let a = light(7, 1.0);
        let lights = [a, a];
        let selected = select_scene_lights(lights.iter(), [0.0; 3]);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, 7);
    }

    #[test]
    fn selection_is_nearest_first_and_order_independent() {
        let near = light(1, 1.0);
        let far = light(2, 30.0);
        let a = select_scene_lights([&far, &near].into_iter(), [0.0; 3]);
        let b = select_scene_lights([&near, &far].into_iter(), [0.0; 3]);
        assert_eq!(a, b);
        assert_eq!(a[0].id, near.id);
    }

    #[test]
    fn invalid_and_disabled_lights_never_reach_a_driver() {
        let mut disabled = light(1, 1.0);
        disabled.enabled = false;
        let mut invalid = light(2, 2.0);
        invalid.position[0] = f32::NAN;
        let selected = select_scene_lights([&disabled, &invalid].into_iter(), [0.0; 3]);
        assert!(selected.is_empty());
    }

    #[test]
    fn collection_never_truncates_valid_fixtures() {
        let lights: Vec<_> = (0..40).map(|id| light(id, id as f32)).collect();
        let selected = select_scene_lights(lights.iter(), [0.0; 3]);
        assert_eq!(selected.len(), lights.len());
    }
}
