//! Reconstruct one representative point and normal from an axis-aligned
//! splat, then decode its authored albedo into linear RGB.

pub(super) use super::super::decode_srgb_albedo::decode_albedo;
#[cfg(test)]
pub(super) use super::super::decode_srgb_albedo::srgb_channel_to_linear;
pub(super) use super::super::surface_geometry::representative_surface_position;
use super::super::surface_geometry::select_surface_normal;
use super::SplatSurface;

/// Exact normal of the AABB face entered by the center-directed camera ray.
/// For a centered cube this is the dominant ray axis. Ties use X, then Y,
/// then Z, making edge/corner views deterministic without inventing a
/// diagonal receiver cosine.
#[cfg(test)]
pub(super) fn camera_facing_aabb_normal(camera_to_surface: [f32; 3]) -> Option<[f32; 3]> {
    if camera_to_surface
        .into_iter()
        .any(|component| !component.is_finite())
    {
        return None;
    }
    let absolute = camera_to_surface.map(f32::abs);
    let axis = if absolute[0] >= absolute[1] && absolute[0] >= absolute[2] {
        0
    } else if absolute[1] >= absolute[2] {
        1
    } else {
        2
    };
    if absolute[axis] <= 1.0e-6 {
        return None;
    }
    let mut normal = [0.0; 3];
    normal[axis] = if camera_to_surface[axis] > 0.0 {
        -1.0
    } else {
        1.0
    };
    Some(normal)
}

/// Selects one actual exposed face, with authored orientation for floor,
/// ceiling, and emissive panel materials. Camera dominance is used only for
/// cube-like materials and only among faces present in the geometry contract.
pub(super) fn representative_normal(
    surface: &SplatSurface,
    camera_to_surface: [f32; 3],
) -> Option<[f32; 3]> {
    select_surface_normal(
        surface.voxel_type,
        surface.is_emissive,
        surface.exposure,
        camera_to_surface,
    )
}
