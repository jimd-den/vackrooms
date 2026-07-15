//! Runtime point-light irradiance (currently dropped flares).

use crate::application::rendering::LinearRgb;

use super::super::camera::Camera;
use super::evaluate_fixture_irradiance::point_light_irradiance;

pub(super) fn dynamic_irradiance(
    cam: &Camera,
    surface_position: [f32; 3],
    normal: [f32; 3],
) -> LinearRgb {
    let mut sum = [0.0; 3];
    for light in &cam.dynamic_lights[..cam.dynamic_light_count] {
        let contribution = point_light_irradiance(
            surface_position,
            normal,
            light.position,
            light.color,
            light.radius,
            light.intensity,
        );
        for channel in 0..3 {
            sum[channel] += contribution[channel];
        }
    }
    sum
}
