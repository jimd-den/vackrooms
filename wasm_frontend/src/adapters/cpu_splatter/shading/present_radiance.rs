//! Compose the level atmosphere in linear RGB, tone-map, and encode once.

use crate::application::ports::Environment;
use crate::application::rendering::{
    LinearRgb, compose_fog, encode_display_color, fog_transmittance,
};

use super::super::camera::{Camera, dot};
use super::SplatSurface;

pub(super) fn view_distance(
    cam: &Camera,
    surface: &SplatSurface,
    surface_position: [f32; 3],
) -> f32 {
    let delta = [
        surface_position[0] - cam.pos[0],
        surface_position[1] - cam.pos[1],
        surface_position[2] - cam.pos[2],
    ];
    let distance = dot(delta, delta).sqrt();
    if distance.is_finite() {
        distance
    } else {
        surface.dist.max(0.0)
    }
}

pub(super) fn present_radiance(
    radiance: LinearRgb,
    distance: f32,
    environment: &Environment,
) -> [u8; 3] {
    let transmittance = fog_transmittance(distance, environment.fog_start, environment.fog_density);
    let through_fog = compose_fog(radiance, environment.fog_color, transmittance);
    encode_display_color(through_fog)
        .map(|channel| (channel * 255.0).round().clamp(0.0, 255.0) as u8)
}
