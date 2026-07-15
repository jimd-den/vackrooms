//! Analytic static-fixture irradiance. Point emitters and finite rectangular
//! ceiling emitters share one sanitized radiometric boundary.

use crate::application::ports::{LightKind, LightSource};
use crate::application::rendering::{LinearRgb, finite_range_inverse_square_attenuation};

use super::super::camera::dot;
use super::FrameLighting;
use super::trace_direct_light_visibility::DirectLightVisibility;

const GAUSS_ABSCISSA: f32 = 0.577_350_26;
const MINIMUM_LIGHT_DISTANCE: f32 = 0.2;

/// Exact zero test for the four quadrature samples. This is not a lighting
/// approximation: it rejects work only when every sample is outside the
/// finite support, behind the one-sided emitter, or behind the receiver.
fn rectangle_can_contribute(
    surface_position: [f32; 3],
    normal: [f32; 3],
    light: &LightSource,
) -> bool {
    if !light.radius.is_finite() || light.radius <= 0.0 {
        return false;
    }
    let center_delta = [
        light.position[0] - surface_position[0],
        light.position[1] - surface_position[1],
        light.position[2] - surface_position[2],
    ];
    if center_delta
        .into_iter()
        .chain(normal)
        .any(|component| !component.is_finite())
    {
        return false;
    }

    // Every sample shares the emitter height. A horizontal panel emits only
    // downward, so receivers level with or above it get exactly zero.
    if center_delta[1] <= 0.0 {
        return false;
    }

    let sample_extent_x = GAUSS_ABSCISSA * light.half_size[0];
    let sample_extent_z = GAUSS_ABSCISSA * light.half_size[1];
    let nearest_x = (center_delta[0].abs() - sample_extent_x).max(0.0);
    let nearest_z = (center_delta[2].abs() - sample_extent_z).max(0.0);
    let nearest_squared =
        nearest_x * nearest_x + center_delta[1] * center_delta[1] + nearest_z * nearest_z;
    if nearest_squared >= light.radius * light.radius {
        return false;
    }

    // Dot products are affine over the sample rectangle, so this bound is
    // the exact maximum receiver cosine numerator among its four corners.
    let maximum_receiver_dot = dot(normal, center_delta)
        + normal[0].abs() * sample_extent_x
        + normal[2].abs() * sample_extent_z;
    maximum_receiver_dot > 0.0
}

fn sanitized_light_scale(color: LinearRgb, intensity: f32) -> LinearRgb {
    if !intensity.is_finite() || intensity <= 0.0 {
        return [0.0; 3];
    }
    color.map(|channel| {
        if channel.is_finite() {
            channel.max(0.0) * intensity
        } else {
            0.0
        }
    })
}

fn normalized(vector: [f32; 3]) -> Option<[f32; 3]> {
    if vector.into_iter().any(|component| !component.is_finite()) {
        return None;
    }
    let scale = vector
        .into_iter()
        .fold(0.0_f32, |largest, component| largest.max(component.abs()));
    if scale == 0.0 {
        return None;
    }
    let scaled = vector.map(|component| component / scale);
    let length = dot(scaled, scaled).sqrt();
    Some(scaled.map(|component| component / length))
}

/// Reuses the one surface-to-light normalization for every geometric term.
/// The generic rendering helpers normalize each argument independently;
/// doing that twice per Gauss sample is mathematically redundant and was the
/// dominant arithmetic cost in fixture-heavy CPU frames.
pub(super) fn sample_geometry(
    receiver_normal: [f32; 3],
    surface_to_light: [f32; 3],
) -> Option<(f32, f32, f32)> {
    let distance = dot(surface_to_light, surface_to_light).sqrt();
    if !distance.is_finite() || distance <= 0.0 {
        return None;
    }
    let inverse_distance = 1.0 / distance;
    let direction = surface_to_light.map(|component| component * inverse_distance);
    let receiver_cosine = dot(receiver_normal, direction).clamp(0.0, 1.0);
    // Horizontal rectangular fixtures point down. For direction L from the
    // receiver to the light, dot(-Y, direction_light_to_receiver) == L.y.
    let downward_emitter_cosine = direction[1].clamp(0.0, 1.0);
    Some((distance, receiver_cosine, downward_emitter_cosine))
}

pub(super) fn point_light_irradiance(
    surface_position: [f32; 3],
    normal: [f32; 3],
    light_position: [f32; 3],
    color: LinearRgb,
    range: f32,
    intensity: f32,
) -> LinearRgb {
    let delta = [
        light_position[0] - surface_position[0],
        light_position[1] - surface_position[1],
        light_position[2] - surface_position[2],
    ];
    let distance_squared = dot(delta, delta);
    if !range.is_finite()
        || range <= 0.0
        || !distance_squared.is_finite()
        || distance_squared >= range * range
    {
        return [0.0; 3];
    }
    let Some(normal) = normalized(normal) else {
        return [0.0; 3];
    };
    let Some((distance, geometry, _)) = sample_geometry(normal, delta) else {
        return [0.0; 3];
    };
    let attenuation =
        finite_range_inverse_square_attenuation(distance, range, MINIMUM_LIGHT_DISTANCE);
    let scale = sanitized_light_scale(color, intensity);
    scale.map(|channel| channel * geometry * attenuation)
}

/// Deterministic 2x2 Gauss quadrature over a physical XZ rectangle.
#[cfg(test)]
pub(super) fn rectangle_light_irradiance(
    surface_position: [f32; 3],
    normal: [f32; 3],
    light: &LightSource,
) -> LinearRgb {
    rectangle_light_irradiance_with_visibility(
        surface_position,
        normal,
        light,
        DirectLightVisibility::unoccluded(),
    )
}

fn rectangle_light_irradiance_with_visibility(
    surface_position: [f32; 3],
    normal: [f32; 3],
    light: &LightSource,
    visibility: DirectLightVisibility<'_>,
) -> LinearRgb {
    let [half_x, half_z] = light.half_size;
    if !half_x.is_finite() || !half_z.is_finite() || half_x <= 0.0 || half_z <= 0.0 {
        return [0.0; 3];
    }
    if !rectangle_can_contribute(surface_position, normal, light) {
        return [0.0; 3];
    }
    let mut integral = 0.0;
    for x in [-GAUSS_ABSCISSA, GAUSS_ABSCISSA] {
        for z in [-GAUSS_ABSCISSA, GAUSS_ABSCISSA] {
            let sample = [
                light.position[0] + x * half_x,
                light.position[1],
                light.position[2] + z * half_z,
            ];
            let surface_to_light = [
                sample[0] - surface_position[0],
                sample[1] - surface_position[1],
                sample[2] - surface_position[2],
            ];
            let Some((distance, receiver_cosine, emitter_cosine)) =
                sample_geometry(normal, surface_to_light)
            else {
                continue;
            };
            if receiver_cosine <= 0.0
                || emitter_cosine <= 0.0
                || !visibility.sample_is_visible(surface_position, normal, sample)
            {
                continue;
            }
            integral += receiver_cosine
                * emitter_cosine
                * finite_range_inverse_square_attenuation(
                    distance,
                    light.radius,
                    MINIMUM_LIGHT_DISTANCE,
                );
        }
    }
    let area = 4.0 * half_x * half_z;
    let scale = sanitized_light_scale(light.color, light.intensity);
    scale.map(|channel| channel * area * integral * 0.25)
}

fn scene_light_irradiance(
    light: &LightSource,
    surface_position: [f32; 3],
    normal: [f32; 3],
    visibility: DirectLightVisibility<'_>,
) -> LinearRgb {
    if !light.enabled {
        return [0.0; 3];
    }
    match light.kind {
        LightKind::Point => {
            let contribution = point_light_irradiance(
                surface_position,
                normal,
                light.position,
                light.color,
                light.radius,
                light.intensity,
            );
            if contribution != [0.0; 3]
                && !visibility.sample_is_visible(surface_position, normal, light.position)
            {
                [0.0; 3]
            } else {
                contribution
            }
        }
        LightKind::CeilingPanel | LightKind::Strip | LightKind::Emergency => {
            rectangle_light_irradiance_with_visibility(surface_position, normal, light, visibility)
        }
    }
}

#[cfg(test)]
pub(super) fn fixture_irradiance(
    lighting: FrameLighting<'_>,
    surface_position: [f32; 3],
    normal: [f32; 3],
) -> LinearRgb {
    fixture_irradiance_with_visibility(
        lighting,
        surface_position,
        normal,
        DirectLightVisibility::unoccluded(),
    )
}

pub(super) fn fixture_irradiance_with_visibility(
    lighting: FrameLighting<'_>,
    surface_position: [f32; 3],
    normal: [f32; 3],
    direct_visibility: DirectLightVisibility<'_>,
) -> LinearRgb {
    let mut sum = [0.0; 3];
    for light in lighting.scene_lights {
        let visibility = lighting
            .hero_visibility
            .filter(|hero| hero.light_id == light.id)
            .map_or(1.0, |hero| hero.visibility.clamp(0.0, 1.0));
        let contribution =
            scene_light_irradiance(light, surface_position, normal, direct_visibility);
        for channel in 0..3 {
            sum[channel] += contribution[channel] * visibility;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use crate::application::ports::{ChunkDraw, LightKind, LightSource};
    use crate::application::rendering::{
        downward_rectangular_emitter_cosine, lambertian_receiver_cosine,
    };

    use super::super::trace_direct_light_visibility::DirectLightVisibility;
    use super::{
        normalized, rectangle_can_contribute, rectangle_light_irradiance,
        rectangle_light_irradiance_with_visibility, sample_geometry,
    };

    #[test]
    fn shared_direction_geometry_matches_the_reference_cosines() {
        for normal in [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.3, 0.4, -0.5]] {
            let normal = normalized(normal).unwrap();
            for delta in [[1.0, 3.0, -2.0], [-4.0, 0.5, 1.0], [0.0, -2.0, 0.5]] {
                let (_, receiver, emitter) = sample_geometry(normal, delta).unwrap();
                let reference_receiver = lambertian_receiver_cosine(normal, delta);
                let reference_emitter =
                    downward_rectangular_emitter_cosine(delta.map(|component| -component));
                assert!((receiver - reference_receiver).abs() < 1.0e-6);
                assert!((emitter - reference_emitter).abs() < 1.0e-6);
            }
        }
    }

    fn panel(position: [f32; 3], radius: f32) -> LightSource {
        LightSource {
            position,
            half_size: [1.0, 0.5],
            color: [1.0; 3],
            radius,
            intensity: 1.0,
            kind: LightKind::CeilingPanel,
            enabled: true,
            ..LightSource::default()
        }
    }

    #[test]
    fn exact_rectangle_cull_rejects_only_zero_contribution_regions() {
        let light = panel([0.0, 3.0, 0.0], 5.0);
        assert!(rectangle_can_contribute(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            &light,
        ));
        assert!(!rectangle_can_contribute(
            [0.0, 4.0, 0.0],
            [0.0, -1.0, 0.0],
            &light,
        ));
        assert!(!rectangle_can_contribute(
            [20.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            &light,
        ));
        assert!(!rectangle_can_contribute(
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            &light,
        ));
    }

    #[test]
    fn full_visibility_traces_each_rectangle_quadrature_endpoint() {
        let mut light = panel([0.0, 4.0, 0.0], 10.0);
        light.half_size = [2.0, 2.0];
        let receiver = [0.0, 0.0, 0.0];
        let normal = [0.0, 1.0, 0.0];
        let unoccluded = rectangle_light_irradiance(receiver, normal, &light);

        // This half-unit blocker lies on only the (-X,-Z) Gauss segment.
        // A center-only visibility query would miss it and return the full
        // integral; endpoint visibility removes only that sample.
        let atlas = [1, 1, 0x00ff_ffff, 0];
        let chunks = [ChunkDraw {
            origin: [-0.83, 1.75, -0.83],
            root_index: 0,
            world_size: 0.5,
            voxel_size: 0.5,
            svo_depth: 0,
        }];
        let partial = rectangle_light_irradiance_with_visibility(
            receiver,
            normal,
            &light,
            DirectLightVisibility::trace_scene(&atlas, &chunks, 0.5),
        );

        assert!(partial[0] > 0.0);
        assert!(partial[0] < unoccluded[0]);
    }
}
