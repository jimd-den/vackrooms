//! The player's flashlight — a true spotlight cone.
//!
//! This module is the **single specification** of the cone light's shape.
//! The GPU shaders implement the same formula in GLSL (see
//! `drivers::shaders::chunks::SPOT_CONE_GLSL`); the constants below are
//! quoted there, so a tuning change here must be mirrored there and
//! vice versa. The beam is defined by four independent factors, multiplied:
//!
//! 1. **Cone** — smoothstep between an inner angle (full strength) and an
//!    outer angle (zero), measured between the beam direction and the ray
//!    from the lamp to the surface point. This is what makes it a *cone*
//!    rather than a screen-centered glow.
//! 2. **Range** — smoothstep fade from `RANGE_FULL` to `RANGE_END` world
//!    units, so the beam dies in fog instead of clipping.
//! 3. **Facing** — Lambert-ish response of the surface normal against the
//!    beam, floored at `FACING_FLOOR` so grazing surfaces inside the cone
//!    still read (the beam has width in reality).
//! 4. **Occlusion** — one SVO shadow ray from the lamp toward the surface
//!    (CPU path only; toggleable via `RenderToggles::flashlight_occlusion`).
//!
//! The lamp sits slightly forward of and below the camera (a hand-held
//! light, not an eye-mounted one), which gives near surfaces a believable
//! parallax between view and beam.

use crate::application::ports::ChunkDraw;

use super::camera::{Camera, dot};
use super::raycast::{ray_box_interval, trace_svo};

/// Inner cone half-angle: full-strength core of the beam.
pub const INNER_DEG: f32 = 11.0;
/// Outer cone half-angle: intensity reaches zero here.
pub const OUTER_DEG: f32 = 24.0;
/// Distance (world units) up to which the beam is at full strength.
pub const RANGE_FULL: f32 = 3.0;
/// Distance at which the beam has fully faded.
pub const RANGE_END: f32 = 14.0;
/// Minimum facing response inside the cone (beam width fake).
pub const FACING_FLOOR: f32 = 0.08;
/// Lamp offset from the eye: forward and slightly down.
pub const LAMP_FORWARD: f32 = 0.18;
pub const LAMP_DOWN: f32 = 0.10;
/// Move secondary rays out of geometry immediately surrounding the lamp.
const OCCLUSION_ORIGIN_BIAS: f32 = 0.15;
/// Stop immediately before the physical receiver face. This is deliberately
/// independent of splat/LOD size: representation changes must not change the
/// lamp-to-receiver segment being tested.
const OCCLUSION_RECEIVER_BIAS: f32 = 0.02;
/// Warm-white beam tint, matching the GPU paths.
pub const TINT: [f32; 3] = [1.0, 0.96, 0.85];

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Angular falloff: 1 inside the inner cone, 0 outside the outer cone,
/// smooth in between. `alignment` is cos(angle between beam and forward).
pub fn cone_falloff(alignment: f32) -> f32 {
    let inner_cos = INNER_DEG.to_radians().cos();
    let outer_cos = OUTER_DEG.to_radians().cos();
    if alignment <= outer_cos {
        0.0
    } else {
        smoothstep01((alignment - outer_cos) / (inner_cos - outer_cos))
    }
}

/// Distance falloff: 1 out to [`RANGE_FULL`], 0 at [`RANGE_END`].
pub fn range_falloff(dist: f32) -> f32 {
    1.0 - smoothstep01((dist - RANGE_FULL) / (RANGE_END - RANGE_FULL))
}

/// Length passed to `trace_svo`, whose origin has already moved
/// [`OCCLUSION_ORIGIN_BIAS`] away from the lamp.
fn occlusion_trace_length(lamp_to_receiver: f32) -> Option<f32> {
    let length = lamp_to_receiver - OCCLUSION_ORIGIN_BIAS - OCCLUSION_RECEIVER_BIAS;
    (length > 0.0).then_some(length)
}

/// The CPU splatter shades an axis-aligned cube at one representative
/// normal. Keeping the receiver bounds explicit lets the shadow ray stop at
/// the actual near face even when that normal is diagonally blended.
#[derive(Debug, Clone, Copy)]
pub struct BeamReceiver {
    center: [f32; 3],
    half_extent: f32,
    normal: [f32; 3],
}

impl BeamReceiver {
    pub fn cube(center: [f32; 3], world_size: f32, normal: [f32; 3]) -> Self {
        Self {
            center,
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

    /// Chooses the point on this box nearest the spotlight's center axis.
    ///
    /// A CPU splat is flat-shaded as one unit. Sampling only its center makes
    /// a coarse splat abruptly go dark when the center leaves the outer cone,
    /// even while part of its visible footprint is still inside. Projecting
    /// the box center onto the axis and clamping that point to the AABB is a
    /// continuous, allocation-free approximation of the closest point.
    fn sample_nearest_axis(self, lamp: [f32; 3], axis: [f32; 3]) -> [f32; 3] {
        let to_center = [
            self.center[0] - lamp[0],
            self.center[1] - lamp[1],
            self.center[2] - lamp[2],
        ];
        let along_axis = dot(to_center, axis).max(0.0);
        let point_on_axis = [
            lamp[0] + axis[0] * along_axis,
            lamp[1] + axis[1] * along_axis,
            lamp[2] + axis[2] * along_axis,
        ];
        let (bounds_min, bounds_max) = self.bounds();
        [
            point_on_axis[0].clamp(bounds_min[0], bounds_max[0]),
            point_on_axis[1].clamp(bounds_min[1], bounds_max[1]),
            point_on_axis[2].clamp(bounds_min[2], bounds_max[2]),
        ]
    }
}

/// RGB contribution of the flashlight on one cubic splat receiver.
///
/// The ray aims at an in-bounds sample nearest the cone axis, then intersects
/// the receiver AABB to recover the physical near face. Occlusion depends on
/// that geometric segment, never on subtracting an arbitrary fraction of the
/// LOD size. `toggles` controls both the optional SVO ray and its near-to-far
/// traversal order.
pub fn beam_contribution(
    cam: &Camera,
    atlas: &[u32],
    chunks: &[ChunkDraw],
    receiver: BeamReceiver,
    toggles: &crate::application::render_settings::RenderToggles,
) -> [f32; 3] {
    // The hand-held lamp position (forward of and below the eye).
    let lamp = [
        cam.pos[0] + cam.forward[0] * LAMP_FORWARD,
        cam.pos[1] + cam.forward[1] * LAMP_FORWARD - LAMP_DOWN,
        cam.pos[2] + cam.forward[2] * LAMP_FORWARD,
    ];

    let sample = receiver.sample_nearest_axis(lamp, cam.forward);
    let to_sample = [
        sample[0] - lamp[0],
        sample[1] - lamp[1],
        sample[2] - lamp[2],
    ];
    let sample_distance = dot(to_sample, to_sample).sqrt();
    if sample_distance <= 0.0001 {
        return [0.0; 3];
    }
    let beam = [
        to_sample[0] / sample_distance,
        to_sample[1] / sample_distance,
        to_sample[2] / sample_distance,
    ];
    let (bounds_min, bounds_max) = receiver.bounds();
    let Some(receiver_interval) = ray_box_interval(lamp, beam, bounds_min, bounds_max) else {
        return [0.0; 3];
    };
    if receiver_interval.exit < 0.0 {
        return [0.0; 3];
    }
    let receiver_distance = receiver_interval.entry.max(0.0);

    let cone = cone_falloff(dot(beam, cam.forward));
    if cone <= 0.0 {
        return [0.0; 3];
    }
    let range = range_falloff(receiver_distance);
    if range <= 0.0 {
        return [0.0; 3];
    }
    let facing = (-dot(beam, receiver.normal)).max(FACING_FLOOR);

    // Occlusion: trace the geometric lamp-to-face segment. Both endpoint
    // biases are fixed world-space tolerances; tying the receiver bias to a
    // coarse node's half-extent made shadows jump when LOD changed.
    if toggles.flashlight_occlusion {
        let ray_origin = [
            lamp[0] + beam[0] * OCCLUSION_ORIGIN_BIAS,
            lamp[1] + beam[1] * OCCLUSION_ORIGIN_BIAS,
            lamp[2] + beam[2] * OCCLUSION_ORIGIN_BIAS,
        ];
        let blocked = occlusion_trace_length(receiver_distance).is_some_and(|max_trace| {
            trace_svo(
                atlas,
                chunks,
                ray_origin,
                beam,
                max_trace,
                toggles.front_to_back,
            )
            .is_some()
        });
        if blocked {
            return [0.0; 3];
        }
    }

    let intensity = cone * range * facing;
    [
        intensity * TINT[0],
        intensity * TINT[1],
        intensity * TINT[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ports::FrameParams;
    use crate::application::render_settings::RenderToggles;

    use super::super::settings::CpuRenderSettings;

    fn camera_facing_positive_z() -> Camera {
        let frame = FrameParams {
            // Lamp y = 1.0 after the hand-held down offset, exactly level
            // with the test receivers.
            camera_pos: [0.0, 1.1, 0.0],
            yaw: std::f32::consts::PI,
            pitch: 0.0,
            flashlight: true,
            ..FrameParams::default()
        };
        Camera::new(&frame, 64, 64, &CpuRenderSettings::default())
    }

    fn max_component(color: [f32; 3]) -> f32 {
        color.into_iter().fold(0.0, f32::max)
    }

    #[test]
    fn cone_is_full_inside_inner_angle_and_zero_outside_outer() {
        assert_eq!(cone_falloff(1.0), 1.0);
        assert_eq!(cone_falloff((INNER_DEG - 1.0).to_radians().cos()), 1.0);
        assert_eq!(cone_falloff((OUTER_DEG + 1.0).to_radians().cos()), 0.0);
        let mid = cone_falloff(((INNER_DEG + OUTER_DEG) * 0.5).to_radians().cos());
        assert!(mid > 0.0 && mid < 1.0, "smooth transition, got {mid}");
    }

    #[test]
    fn range_fades_between_full_and_end() {
        assert_eq!(range_falloff(0.0), 1.0);
        assert_eq!(range_falloff(RANGE_FULL), 1.0);
        assert_eq!(range_falloff(RANGE_END), 0.0);
        let mid = range_falloff((RANGE_FULL + RANGE_END) * 0.5);
        assert!((mid - 0.5).abs() < 1e-5, "smoothstep midpoint, got {mid}");
    }

    #[test]
    fn trace_length_accounts_for_both_endpoint_biases() {
        let distance = 5.0;
        let trace = occlusion_trace_length(distance).expect("ordinary receiver distance");
        let endpoint_from_lamp = OCCLUSION_ORIGIN_BIAS + trace;
        assert!((endpoint_from_lamp - (distance - OCCLUSION_RECEIVER_BIAS)).abs() < 1.0e-6);
    }

    /// Regression: a 0.1-unit leaf used to trace through its own near face,
    /// while a coarse 8-unit representation stopped several units early.
    /// Both representations describe the same receiver face at z=5 and must
    /// therefore receive the same unoccluded beam.
    #[test]
    fn occlusion_is_independent_of_receiver_lod_size() {
        let cam = camera_facing_positive_z();
        let atlas = [1, 1, 0x00FF_FFFF, 0]; // one solid leaf
        let normal = [0.0, 0.0, -1.0];
        let toggles = RenderToggles::default();
        let mut contributions = Vec::new();

        for receiver_size in [0.1, 8.0] {
            let chunks = [ChunkDraw {
                origin: [-receiver_size * 0.5, 1.0 - receiver_size * 0.5, 5.0],
                root_index: 0,
                world_size: receiver_size,
            }];
            contributions.push(beam_contribution(
                &cam,
                &atlas,
                &chunks,
                BeamReceiver::cube([0.0, 1.0, 5.0 + receiver_size * 0.5], receiver_size, normal),
                &toggles,
            ));
        }

        assert!(
            max_component(contributions[0]) > 0.0,
            "fine receiver self-shadowed"
        );
        assert!(
            max_component(contributions[1]) > 0.0,
            "coarse receiver lost the beam"
        );
        for channel in 0..3 {
            assert!((contributions[0][channel] - contributions[1][channel]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn fixed_receiver_bias_still_detects_a_real_blocker() {
        let cam = camera_facing_positive_z();
        let atlas = [1, 1, 0x00FF_FFFF, 0];
        let chunks = [ChunkDraw {
            origin: [-0.25, 0.7, 3.0],
            root_index: 0,
            world_size: 0.5,
        }];
        let contribution = beam_contribution(
            &cam,
            &atlas,
            &chunks,
            BeamReceiver::cube([0.0, 1.0, 5.05], 0.1, [0.0, 0.0, -1.0]),
            &RenderToggles::default(),
        );
        assert_eq!(contribution, [0.0; 3]);
    }

    /// A blended normal points inside a cube except on a cardinal axis. The
    /// occlusion endpoint must still use the AABB's real near face, otherwise
    /// a coarse diagonal receiver hits itself and the cone goes black.
    #[test]
    fn diagonal_receiver_uses_its_aabb_entry() {
        let cam = camera_facing_positive_z();
        let atlas = [1, 1, 0x00FF_FFFF, 0];
        let chunks = [ChunkDraw {
            origin: [0.5, 0.0, 4.0],
            root_index: 0,
            world_size: 2.0,
        }];
        let contribution = beam_contribution(
            &cam,
            &atlas,
            &chunks,
            BeamReceiver::cube([1.5, 1.0, 5.0], 2.0, [-0.3, 0.0, -0.954]),
            &RenderToggles::default(),
        );
        assert!(
            max_component(contribution) > 0.0,
            "diagonal receiver self-shadowed: {contribution:?}"
        );
    }

    /// Regression: the receiver center is outside the 24-degree cone, but
    /// the coarse box visibly overlaps it. Center-only flat shading made the
    /// whole splat disappear at this camera angle.
    #[test]
    fn coarse_receiver_overlapping_cone_uses_an_in_bounds_sample() {
        let cam = camera_facing_positive_z();
        let atlas = [1, 1, 0x00FF_FFFF, 0];
        let chunks = [ChunkDraw {
            origin: [2.0, -1.0, 6.0],
            root_index: 0,
            world_size: 4.0,
        }];
        let receiver = BeamReceiver::cube([4.0, 1.0, 8.0], 4.0, [-0.45, 0.0, -0.89]);

        let lamp = [0.0, 1.0, LAMP_FORWARD];
        let center_direction = [4.0, 0.0, 8.0 - LAMP_FORWARD];
        let center_length = dot(center_direction, center_direction).sqrt();
        let center_alignment = center_direction[2] / center_length;
        assert_eq!(
            cone_falloff(center_alignment),
            0.0,
            "fixture must reproduce the old center-only angular dropout"
        );
        let sample = receiver.sample_nearest_axis(lamp, cam.forward);
        assert!(sample[0] >= 2.0 && sample[0] <= 6.0);

        let contribution =
            beam_contribution(&cam, &atlas, &chunks, receiver, &RenderToggles::default());
        assert!(
            max_component(contribution) > 0.0,
            "box overlaps the cone but its flat sample went dark"
        );
    }
}
