//! A finite, one-sided spotlight used by the CPU splatter.
//!
//! The module keeps the four independent responsibilities visible:
//!
//! 1. [`cone_falloff`] describes the lamp's angular intensity profile;
//! 2. inverse-square attenuation describes transport through distance;
//! 3. a Lambert cosine describes how the receiver faces the lamp;
//! 4. [`segment_is_occluded`] optionally tests geometric visibility.
//!
//! Occlusion is deliberately not folded into the light equation. Disabling
//! `rt_beam_occlusion` removes only the visibility query; it cannot change
//! the cone, range, color, or receiver response.

use crate::application::ports::ChunkDraw;
#[cfg(test)]
use crate::application::rendering::compact_support_window;
use crate::application::rendering::{
    finite_range_inverse_square_attenuation, lambertian_receiver_cosine,
};

use super::camera::{Camera, dot};
use super::raycast::{ray_box_interval, svo_segment_is_occluded};
use super::settings::CpuFlashlightVisibility;

/// Full-strength core and soft outer edge of the angular profile.
pub const INNER_DEG: f32 = 11.0;
pub const OUTER_DEG: f32 = 24.0;
/// Finite support of the lamp in world units.
pub const RANGE_END: f32 = 14.0;
/// Radiometric scale before inverse-square transport. It is intentionally a
/// named calibration constant rather than a hidden post-lighting boost.
pub const INTENSITY: f32 = 32.0;
/// Finite source-radius clamp, preventing a singularity against the lamp.
pub const MINIMUM_DISTANCE: f32 = 0.25;
/// Lamp offset from the eye: a small hand-held parallax, not an eye glow.
pub const LAMP_FORWARD: f32 = 0.18;
pub const LAMP_DOWN: f32 = 0.10;
/// Warm-white spectral tint in linear RGB.
pub const TINT: [f32; 3] = [1.0, 0.91, 0.72];

/// Fixed world-space endpoint tolerances for the visibility segment.
const OCCLUSION_ORIGIN_BIAS: f32 = 0.02;
const OCCLUSION_RECEIVER_BIAS: f32 = 0.02;

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Angular intensity: one in the core, zero outside the outer cone, and a
/// smooth transition between them. `alignment` is `cos(theta)`.
pub fn cone_falloff(alignment: f32) -> f32 {
    let inner_cos = INNER_DEG.to_radians().cos();
    let outer_cos = OUTER_DEG.to_radians().cos();
    if !alignment.is_finite() || alignment <= outer_cos {
        return 0.0;
    }
    smoothstep01((alignment - outer_cos) / (inner_cos - outer_cos))
}

/// Compact-support window used by the inverse-square term. Exposing it
/// separately makes range behavior easy to inspect in diagnostics/tests.
#[cfg(test)]
pub fn range_falloff(distance: f32) -> f32 {
    compact_support_window(distance, RANGE_END)
}

fn lamp_position(cam: &Camera) -> [f32; 3] {
    [
        cam.pos[0] + cam.forward[0] * LAMP_FORWARD,
        cam.pos[1] + cam.forward[1] * LAMP_FORWARD - LAMP_DOWN,
        cam.pos[2] + cam.forward[2] * LAMP_FORWARD,
    ]
}

fn normalized_direction(from: [f32; 3], to: [f32; 3]) -> Option<([f32; 3], f32)> {
    let delta = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    let distance = dot(delta, delta).sqrt();
    if !distance.is_finite() || distance <= 1.0e-4 {
        return None;
    }
    Some((delta.map(|component| component / distance), distance))
}

/// The splatter shades an axis-aligned cube at one representative surface
/// point and normal. Full bounds stop the visibility ray at the physical
/// near face; transport is evaluated at `sample_position`.
#[derive(Debug, Clone, Copy)]
pub struct BeamReceiver {
    center: [f32; 3],
    sample_position: [f32; 3],
    half_extent: f32,
    normal: [f32; 3],
}

impl BeamReceiver {
    #[cfg(test)]
    pub fn cube(center: [f32; 3], world_size: f32, normal: [f32; 3]) -> Self {
        Self {
            center,
            sample_position: center,
            half_extent: (world_size * 0.5).max(0.0),
            normal,
        }
    }

    /// Uses an explicit representative point on the cube's visible surface
    /// while retaining the full bounds for the occlusion endpoint.
    pub fn surface_sample(
        center: [f32; 3],
        world_size: f32,
        sample_position: [f32; 3],
        normal: [f32; 3],
    ) -> Self {
        Self {
            center,
            sample_position,
            half_extent: (world_size * 0.5).max(0.0),
            normal,
        }
    }

    fn bounds(self) -> ([f32; 3], [f32; 3]) {
        (
            self.center.map(|value| value - self.half_extent),
            self.center.map(|value| value + self.half_extent),
        )
    }
}

/// Pure unoccluded spotlight irradiance at one representative receiver.
///
/// ```text
/// E = tint * I * cone(theta) * window(d, R) / max(d, d_min)^2
///          * max(dot(n, omega_to_lamp), 0)
/// ```
pub fn unoccluded_irradiance(cam: &Camera, receiver: BeamReceiver) -> [f32; 3] {
    let lamp = lamp_position(cam);
    let Some((lamp_to_receiver, distance)) = normalized_direction(lamp, receiver.sample_position)
    else {
        return [0.0; 3];
    };

    let cone = cone_falloff(dot(lamp_to_receiver, cam.forward));
    let facing = lambertian_receiver_cosine(receiver.normal, lamp_to_receiver.map(|v| -v));
    let attenuation =
        finite_range_inverse_square_attenuation(distance, RANGE_END, MINIMUM_DISTANCE);
    let scalar = INTENSITY * cone * facing * attenuation;
    TINT.map(|channel| channel * scalar)
}

fn occlusion_trace_length(lamp_to_receiver_face: f32) -> Option<f32> {
    let length = lamp_to_receiver_face - OCCLUSION_ORIGIN_BIAS - OCCLUSION_RECEIVER_BIAS;
    (length > 0.0).then_some(length)
}

/// Tests only visibility of the finite lamp-to-receiver segment.
fn segment_is_occluded(
    cam: &Camera,
    atlas: &[u32],
    chunks: &[ChunkDraw],
    receiver: BeamReceiver,
) -> bool {
    let lamp = lamp_position(cam);
    let Some((direction, _)) = normalized_direction(lamp, receiver.sample_position) else {
        return false;
    };
    let (bounds_min, bounds_max) = receiver.bounds();
    let Some(interval) = ray_box_interval(lamp, direction, bounds_min, bounds_max) else {
        return false;
    };
    let receiver_face = interval.entry.max(0.0);
    let Some(max_trace) = occlusion_trace_length(receiver_face) else {
        return false;
    };
    let origin = [
        lamp[0] + direction[0] * OCCLUSION_ORIGIN_BIAS,
        lamp[1] + direction[1] * OCCLUSION_ORIGIN_BIAS,
        lamp[2] + direction[2] * OCCLUSION_ORIGIN_BIAS,
    ];
    svo_segment_is_occluded(atlas, chunks, origin, direction, max_trace)
}

/// Spotlight irradiance with optional SVO visibility. The toggle is an
/// independent cost/quality switch: off is exactly the pure unoccluded term.
pub fn beam_contribution(
    cam: &Camera,
    atlas: &[u32],
    chunks: &[ChunkDraw],
    receiver: BeamReceiver,
    visibility: CpuFlashlightVisibility,
) -> [f32; 3] {
    let irradiance = unoccluded_irradiance(cam, receiver);
    if irradiance == [0.0; 3] {
        return irradiance;
    }
    if visibility == CpuFlashlightVisibility::TraceScene
        && segment_is_occluded(cam, atlas, chunks, receiver)
    {
        [0.0; 3]
    } else {
        irradiance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ports::FrameParams;

    use super::super::settings::CpuRenderSettings;

    fn camera_facing_positive_z() -> Camera {
        let frame = FrameParams {
            camera_pos: [0.0, 1.1, 0.0],
            yaw: std::f32::consts::PI,
            pitch: 0.0,
            flashlight: true,
            ..FrameParams::default()
        };
        Camera::new(&frame, 64, 64, &CpuRenderSettings::default())
    }

    fn receiver(center: [f32; 3], normal: [f32; 3]) -> BeamReceiver {
        BeamReceiver::cube(center, 0.1, normal)
    }

    fn max_component(color: [f32; 3]) -> f32 {
        color.into_iter().fold(0.0, f32::max)
    }

    #[test]
    fn cone_has_a_full_core_soft_edge_and_zero_exterior() {
        assert_eq!(cone_falloff(1.0), 1.0);
        assert_eq!(cone_falloff((INNER_DEG - 1.0).to_radians().cos()), 1.0);
        assert_eq!(cone_falloff((OUTER_DEG + 1.0).to_radians().cos()), 0.0);
        let mid = cone_falloff(((INNER_DEG + OUTER_DEG) * 0.5).to_radians().cos());
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn range_window_is_monotone_and_zero_at_support() {
        let mut previous = 1.0;
        for step in 0..=140 {
            let current = range_falloff(step as f32 * 0.1);
            assert!(current <= previous + 1.0e-6);
            previous = current;
        }
        assert_eq!(range_falloff(RANGE_END), 0.0);
        assert_eq!(range_falloff(RANGE_END + 1.0), 0.0);
    }

    #[test]
    fn transport_is_inverse_square_away_from_the_cutoff() {
        let cam = camera_facing_positive_z();
        let near = unoccluded_irradiance(&cam, receiver([0.0, 1.0, 2.0], [0.0, 0.0, -1.0]));
        let far = unoccluded_irradiance(&cam, receiver([0.0, 1.0, 4.0], [0.0, 0.0, -1.0]));
        assert!(near[0] > far[0] * 3.0, "near={near:?}, far={far:?}");
    }

    #[test]
    fn back_faces_receive_no_flashlight_energy() {
        let cam = camera_facing_positive_z();
        let back = unoccluded_irradiance(&cam, receiver([0.0, 1.0, 5.0], [0.0, 0.0, 1.0]));
        assert_eq!(back, [0.0; 3]);
    }

    #[test]
    fn coarse_center_outside_the_cone_gets_no_phantom_energy() {
        let cam = camera_facing_positive_z();
        let outside = BeamReceiver::cube([4.0, 1.0, 8.0], 4.0, [-0.45, 0.0, -0.89]);
        assert_eq!(unoccluded_irradiance(&cam, outside), [0.0; 3]);
    }

    #[test]
    fn occlusion_toggle_changes_visibility_only() {
        let cam = camera_facing_positive_z();
        let atlas = [1, 1, 0x00FF_FFFF, 0];
        let chunks = [ChunkDraw {
            origin: [-0.25, 0.7, 3.0],
            root_index: 0,
            world_size: 0.5,
            voxel_size: 0.5,
            svo_depth: 0,
        }];
        let receiver = receiver([0.0, 1.0, 5.05], [0.0, 0.0, -1.0]);
        let expected = unoccluded_irradiance(&cam, receiver);

        assert_eq!(
            beam_contribution(
                &cam,
                &atlas,
                &chunks,
                receiver,
                CpuFlashlightVisibility::Unoccluded,
            ),
            expected
        );

        assert_eq!(
            beam_contribution(
                &cam,
                &atlas,
                &chunks,
                receiver,
                CpuFlashlightVisibility::TraceScene,
            ),
            [0.0; 3]
        );
        assert!(max_component(expected) > 0.0);
    }

    #[test]
    fn receiver_geometry_does_not_self_occlude() {
        let cam = camera_facing_positive_z();
        let atlas = [1, 1, 0x00FF_FFFF, 0];
        let chunks = [ChunkDraw {
            origin: [-1.0, 0.0, 4.0],
            root_index: 0,
            world_size: 2.0,
            voxel_size: 2.0,
            svo_depth: 0,
        }];
        let receiver = BeamReceiver::cube([0.0, 1.0, 5.0], 2.0, [0.0, 0.0, -1.0]);
        assert!(
            max_component(beam_contribution(
                &cam,
                &atlas,
                &chunks,
                receiver,
                CpuFlashlightVisibility::TraceScene,
            )) > 0.0
        );
    }
}
