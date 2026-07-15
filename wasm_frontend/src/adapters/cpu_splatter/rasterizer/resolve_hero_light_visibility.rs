//! Resolve the CPU renderer's optional hero-light visibility term.
//!
//! Hero rays are intentionally a policy layer around the exact SVO raycast:
//! they are limited to nearby, non-emissive receivers, stopped when the frame
//! is already expensive, and amortized through a position-hashed cache. The
//! traversal only asks for a visibility factor; it does not know how that
//! factor is budgeted or cached.

use crate::application::ports::{ChunkDraw, LightSource};

use super::super::camera::dot;
use super::super::raycast::{ray_box_interval, svo_segment_is_occluded};
use super::super::settings::CpuShadowMode;
use super::super::shading::HeroLightVisibility;
use super::SoftwareRasterizer;

const CACHE_ENTRIES: usize = 65_536;
const RETRACE_PERIOD_FRAMES: usize = 4;
const MAX_RECEIVER_DISTANCE: f32 = 24.0;
const RECEIVER_EXIT_BIAS: f32 = 0.02;
const EMITTER_ENDPOINT_BIAS: f32 = 0.02;
/// Above this write level, hero shadow rays are no longer affordable.
const SHADOW_PIXEL_BUDGET: usize = 1_500_000;

/// Persistent visibility samples and the staggered refresh clock.
pub(super) struct HeroLightVisibilityCache {
    samples: Vec<CachedVisibility>,
    frame_index: usize,
}

#[derive(Clone, Copy, Default)]
struct CachedVisibility {
    query_key: u64,
    visibility: f32,
    valid: bool,
}

impl HeroLightVisibilityCache {
    pub(super) fn new() -> Self {
        Self {
            samples: vec![CachedVisibility::default(); CACHE_ENTRIES],
            frame_index: 0,
        }
    }

    pub(super) fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl SoftwareRasterizer {
    /// Returns no override for the reference/no-shadow path, otherwise an
    /// id-matched binary SVO visibility sample for one real scene fixture.
    pub(super) fn resolve_hero_light_visibility(
        &mut self,
        chunks: &[ChunkDraw],
        center: [f32; 3],
        receiver_world_size: f32,
        is_emissive: bool,
        receiver_distance: f32,
        hero: Option<&LightSource>,
    ) -> Option<HeroLightVisibility> {
        let hero = hero?;
        let affordable =
            receiver_distance <= MAX_RECEIVER_DISTANCE && self.pixel_writes < SHADOW_PIXEL_BUDGET;
        if self.settings.shadows != CpuShadowMode::Hero || is_emissive || !affordable {
            return None;
        }

        let query_key = hash_receiver_and_light(center, receiver_world_size, hero.id);
        let cache_index = query_key as usize % self.hero_light_visibility.samples.len();
        let should_trace = (query_key as usize + self.hero_light_visibility.frame_index)
            .is_multiple_of(RETRACE_PERIOD_FRAMES);
        let cached = self.hero_light_visibility.samples[cache_index];
        if !should_trace && cached.valid && cached.query_key == query_key {
            return Some(HeroLightVisibility {
                light_id: hero.id,
                visibility: cached.visibility,
            });
        }

        let visibility = trace_hero_fixture(
            &self.atlas,
            chunks,
            center,
            receiver_world_size,
            hero.position,
        );
        self.hero_light_visibility.samples[cache_index] = CachedVisibility {
            query_key,
            visibility,
            valid: true,
        };
        Some(HeroLightVisibility {
            light_id: hero.id,
            visibility,
        })
    }
}

fn hash_receiver_and_light(center: [f32; 3], world_size: f32, light_id: u64) -> u64 {
    let mut key = light_id ^ 0xcbf2_9ce4_8422_2325;
    for value in [center[0], center[1], center[2], world_size] {
        let bits = if value == 0.0 { 0 } else { value.to_bits() };
        key ^= u64::from(bits);
        key = key.wrapping_mul(0x0000_0100_0000_01b3);
    }
    key ^= key >> 33;
    key = key.wrapping_mul(0xff51_afd7_ed55_8ccd);
    key ^ (key >> 33)
}

fn trace_hero_fixture(
    atlas: &[u32],
    chunks: &[ChunkDraw],
    center: [f32; 3],
    receiver_world_size: f32,
    light_position: [f32; 3],
) -> f32 {
    let to_light = [
        light_position[0] - center[0],
        light_position[1] - center[1],
        light_position[2] - center[2],
    ];
    let distance_to_light = dot(to_light, to_light).sqrt();
    if distance_to_light <= 0.001 {
        return 1.0;
    }

    let inverse_distance = 1.0 / distance_to_light;
    let direction = to_light.map(|component| component * inverse_distance);
    let receiver_half = receiver_world_size * 0.5;
    let receiver_min = center.map(|coordinate| coordinate - receiver_half);
    let receiver_max = center.map(|coordinate| coordinate + receiver_half);
    let Some(receiver_interval) = ray_box_interval(center, direction, receiver_min, receiver_max)
    else {
        return 1.0;
    };
    let origin_offset = receiver_interval.exit.max(0.0) + RECEIVER_EXIT_BIAS;
    let trace_length = distance_to_light - origin_offset - EMITTER_ENDPOINT_BIAS;
    if trace_length <= 0.0 {
        return 1.0;
    }
    let origin = [
        center[0] + direction[0] * origin_offset,
        center[1] + direction[1] * origin_offset,
        center[2] + direction[2] * origin_offset,
    ];
    if svo_segment_is_occluded(atlas, chunks, origin, direction, trace_length) {
        0.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use crate::application::ports::ChunkDraw;

    use super::trace_hero_fixture;

    #[test]
    fn coarse_receiver_does_not_occlude_its_own_hero_ray() {
        // One solid root leaf fills the whole 4u chunk. A center-biased ray
        // would immediately hit it; exiting the represented receiver first
        // leaves no geometry between its top face and the fixture.
        let atlas = [1, 1, 0xFF_FF_FF, 15];
        let chunks = [ChunkDraw {
            origin: [0.0; 3],
            root_index: 0,
            world_size: 4.0,
            voxel_size: 4.0,
            svo_depth: 0,
        }];

        let visibility = trace_hero_fixture(&atlas, &chunks, [2.0, 2.0, 2.0], 4.0, [2.0, 8.0, 2.0]);

        assert_eq!(visibility, 1.0);
    }
}
