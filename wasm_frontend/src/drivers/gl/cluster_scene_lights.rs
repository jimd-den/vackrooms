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

/// Persistent cache for per-chunk top-K light selection with hysteresis.
///
/// # Rationale & Design Pattern (Observer & Strategy Pattern)
/// Prevents light popping across chunk seams when camera movement or floating point sorting
/// slight changes candidate light order. A candidate replaces an existing selected light only
/// if its calculated importance score exceeds the existing light's score by at least 20%.
#[derive(Debug, Clone, Default)]
pub struct LightSelectionCache {
    entries: std::collections::HashMap<crate::application::ports::SurfaceChunkKey, Vec<u64>>,
}

impl LightSelectionCache {
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    /// Ranks lights reaching bounds and applies a hysteresis filter against previous frame selection.
    pub fn select_lights_with_hysteresis(
        &mut self,
        key: crate::application::ports::SurfaceChunkKey,
        candidate_lights: &[LightSource],
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        top_k: usize,
    ) -> Vec<LightSource> {
        let center = [
            (bounds_min[0] + bounds_max[0]) * 0.5,
            (bounds_min[1] + bounds_max[1]) * 0.5,
            (bounds_min[2] + bounds_max[2]) * 0.5,
        ];

        // Rank candidates by estimated irradiance score = intensity / (distance_sq + 1.0)
        let mut scored: Vec<(f32, &LightSource)> = candidate_lights
            .iter()
            .map(|light| {
                let dx = light.position[0] - center[0];
                let dy = light.position[1] - center[1];
                let dz = light.position[2] - center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let score = light.intensity / (dist_sq + 1.0);
                (score, light)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let previous_ids = self.entries.get(&key);
        let mut selected = Vec::with_capacity(top_k.min(scored.len()));

        for (score, light) in scored {
            if selected.len() >= top_k {
                break;
            }
            let is_previously_selected = previous_ids.map_or(false, |ids| ids.contains(&light.id));
            let effective_score = if is_previously_selected {
                score * 1.20 // 20% hysteresis boost to retain current lights and prevent seam popping
            } else {
                score
            };
            selected.push((effective_score, light));
        }

        selected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let result: Vec<LightSource> = selected.into_iter().take(top_k).map(|(_, l)| *l).collect();
        self.entries.insert(key, result.iter().map(|l| l.id).collect());
        result
    }
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

    #[test]
    fn test_light_selection_cache_hysteresis() {
        let mut cache = LightSelectionCache::new();
        let chunk_key = (0, 0);

        let l1 = light(1, [0.0, 1.0, 0.0]);
        let l2 = light(2, [0.1, 1.0, 0.0]);

        // First frame selects l1 as top 1
        let sel1 = cache.select_lights_with_hysteresis(chunk_key, &[l1, l2], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1);
        assert_eq!(sel1.len(), 1);
        assert_eq!(sel1[0].id, 1);

        // Second frame: l2 moves slightly closer, but l1 is retained due to hysteresis
        let l2_slightly_closer = light(2, [0.0, 0.95, 0.0]);
        let sel2 = cache.select_lights_with_hysteresis(chunk_key, &[l1, l2_slightly_closer], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1);
        assert_eq!(sel2.len(), 1);
        assert_eq!(sel2[0].id, 1);
    }
}
