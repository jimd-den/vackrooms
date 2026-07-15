//! Exact finite SVO visibility segments for analytic fixture samples.
//!
//! The expensive CPU `Full` shadow policy calls this once for a point light
//! and once for each of a rectangle's four Gauss samples. Receiver and
//! emitter biases mirror the strict GPU raymarch reference so neither the
//! receiver voxel nor the luminous endpoint shadows itself.

use crate::application::ports::ChunkDraw;

use super::super::camera::dot;
use super::super::raycast::svo_segment_is_occluded;

#[derive(Clone, Copy)]
pub(super) struct DirectLightVisibility<'a> {
    atlas: &'a [u32],
    chunks: &'a [ChunkDraw],
    receiver_world_size: f32,
    trace_scene: bool,
}

impl<'a> DirectLightVisibility<'a> {
    pub(super) fn unoccluded() -> Self {
        Self {
            atlas: &[],
            chunks: &[],
            receiver_world_size: 0.0,
            trace_scene: false,
        }
    }

    pub(super) fn trace_scene(
        atlas: &'a [u32],
        chunks: &'a [ChunkDraw],
        receiver_world_size: f32,
    ) -> Self {
        Self {
            atlas,
            chunks,
            receiver_world_size,
            trace_scene: true,
        }
    }

    pub(super) fn sample_is_visible(
        self,
        surface_position: [f32; 3],
        surface_normal: [f32; 3],
        sample_position: [f32; 3],
    ) -> bool {
        if !self.trace_scene {
            return true;
        }

        let bias = (self.receiver_world_size.abs() * 1.0e-3).max(1.0e-5);
        let origin = [
            surface_position[0] + surface_normal[0] * bias,
            surface_position[1] + surface_normal[1] * bias,
            surface_position[2] + surface_normal[2] * bias,
        ];
        let segment = [
            sample_position[0] - origin[0],
            sample_position[1] - origin[1],
            sample_position[2] - origin[2],
        ];
        let segment_length = dot(segment, segment).sqrt();
        if !segment_length.is_finite() || segment_length <= bias * 2.0 {
            return true;
        }
        let inverse_length = 1.0 / segment_length;
        let direction = segment.map(|component| component * inverse_length);
        let blocker_limit = segment_length - bias * 2.0;
        !svo_segment_is_occluded(self.atlas, self.chunks, origin, direction, blocker_limit)
    }
}

#[cfg(test)]
mod tests {
    use super::DirectLightVisibility;
    use crate::application::ports::ChunkDraw;

    #[test]
    fn finite_sample_segment_detects_only_blockers_before_the_endpoint() {
        let atlas = [1, 1, 0x00ff_ffff, 0];
        let chunks = [ChunkDraw {
            origin: [0.0; 3],
            root_index: 0,
            world_size: 1.0,
            voxel_size: 1.0,
            svo_depth: 0,
        }];
        let visibility = DirectLightVisibility::trace_scene(&atlas, &chunks, 0.5);

        assert!(!visibility.sample_is_visible([0.5, 0.5, -1.0], [0.0, 0.0, 1.0], [0.5, 0.5, 2.0],));
        assert!(
            visibility.sample_is_visible([0.5, 0.5, -1.0], [0.0, 0.0, 1.0], [0.5, 0.5, -0.25],)
        );
    }
}
