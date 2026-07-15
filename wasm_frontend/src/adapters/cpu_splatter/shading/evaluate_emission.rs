//! Emitted radiance for exposed fixture undersides.

use crate::application::rendering::LinearRgb;

use super::SplatSurface;

pub(crate) fn emission_strength(base_color: [f32; 3]) -> f32 {
    let near = |reference: [f32; 3]| {
        base_color
            .into_iter()
            .zip(reference)
            .all(|(actual, expected)| (actual - expected).abs() <= 0.5)
    };
    if near([255.0, 248.0, 214.0]) {
        10.0
    } else if near([255.0, 68.0, 51.0]) {
        8.0
    } else if near([159.0, 196.0, 232.0]) {
        0.9
    } else {
        1.0
    }
}

#[cfg(test)]
pub(super) fn emitted_radiance(
    surface: &SplatSurface,
    albedo: LinearRgb,
    normal: [f32; 3],
) -> Option<LinearRgb> {
    emitted_radiance_with_strength(
        surface,
        albedo,
        normal,
        surface
            .is_emissive
            .then(|| emission_strength(surface.base_color)),
    )
}

/// Prepared-material variant used by production traversal. Its scalar comes
/// from the material id, so no encoded color has to travel through every
/// virtual-node visit merely to identify one of three fixture strengths.
pub(super) fn emitted_radiance_with_strength(
    surface: &SplatSurface,
    albedo: LinearRgb,
    normal: [f32; 3],
    strength: Option<f32>,
) -> Option<LinearRgb> {
    if !surface.is_emissive || normal[1] >= -0.5 {
        return None;
    }
    Some(albedo.map(|channel| channel * strength.unwrap_or(1.0)))
}
