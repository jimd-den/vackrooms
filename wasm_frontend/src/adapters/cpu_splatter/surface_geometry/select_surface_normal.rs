//! Choose a lighting normal from prepared surface geometry.

use vackrooms::domain::entities::voxel_grid::{
    VOXEL_CEILING, VOXEL_DEEP_CARPET, VOXEL_DRY_CARPET, VOXEL_FLOOR, VOXEL_FLUID, VOXEL_GRASS,
    VOXEL_STICKY_CARPET, VOXEL_WATER,
};

use super::{SurfaceExposure, SurfaceFace};

/// Material semantics orient thin architectural layers.
///
/// Floors and carpets are authored walking surfaces, while ceiling voxels
/// are authored undersides. Cube-like walls deliberately return no hint and
/// use the most camera-visible exposed face instead.
fn material_face(voxel_type: u32, is_emissive: bool) -> Option<SurfaceFace> {
    if is_emissive {
        return Some(SurfaceFace::NegativeY);
    }
    let voxel_type = u8::try_from(voxel_type).ok()?;
    match voxel_type {
        VOXEL_FLOOR | VOXEL_DRY_CARPET | VOXEL_DEEP_CARPET | VOXEL_STICKY_CARPET | VOXEL_GRASS
        | VOXEL_WATER | VOXEL_FLUID => Some(SurfaceFace::PositiveY),
        VOXEL_CEILING => Some(SurfaceFace::NegativeY),
        _ => None,
    }
}

/// Select one exact axis normal that belongs to an exposed physical face.
///
/// Thin-layer material orientation wins when that authored face is exposed;
/// this is why a floor in front of a mostly-horizontal camera remains a floor
/// and receives overhead Lambertian light. Generic cube materials choose the
/// exposed face with the largest positive surface-to-camera cosine. Area is
/// the deterministic tie-breaker and improves coarse MIP choices.
pub(crate) fn select_surface_normal(
    voxel_type: u32,
    is_emissive: bool,
    exposure: SurfaceExposure,
    camera_to_surface: [f32; 3],
) -> Option<[f32; 3]> {
    if exposure.is_empty()
        || camera_to_surface
            .into_iter()
            .any(|component| !component.is_finite())
    {
        return None;
    }

    if let Some(face) = material_face(voxel_type, is_emissive) {
        let normal = face.normal();
        let facing = -(normal[0] * camera_to_surface[0]
            + normal[1] * camera_to_surface[1]
            + normal[2] * camera_to_surface[2]);
        if exposure.contains(face) && facing > 1.0e-6 {
            return Some(face.normal());
        }
    }

    let mut best: Option<(f32, u8, SurfaceFace)> = None;
    for face in SurfaceFace::ALL {
        let area = exposure.area(face);
        if area == 0 {
            continue;
        }
        let normal = face.normal();
        let facing = -(normal[0] * camera_to_surface[0]
            + normal[1] * camera_to_surface[1]
            + normal[2] * camera_to_surface[2]);
        if facing <= 1.0e-6 {
            continue;
        }
        let candidate = (facing, area, face);
        let replace = best.is_none_or(|current| {
            candidate.0 > current.0 || (candidate.0 == current.0 && candidate.1 > current.1)
        });
        if replace {
            best = Some(candidate);
        }
    }
    best.map(|(_, _, face)| face.normal())
}
