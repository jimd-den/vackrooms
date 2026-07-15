//! Propagate exposed parent and sibling boundaries into one octant.

use super::{SurfaceExposure, SurfaceFace};

/// Surface area visible on one occupied child octant.
///
/// A face on the parent's outside inherits that parent's prepared exposure.
/// A face inside the parent is exposed only when the adjacent sibling is
/// absent from the occupancy mask. Passing `0xff` for a virtually subdivided
/// solid leaf is important: artificial children then inherit the real box's
/// exterior, but their internal cut planes stay closed.
pub(crate) fn exposure_for_octant(
    parent: SurfaceExposure,
    occupied_siblings: u32,
    octant: u32,
) -> SurfaceExposure {
    debug_assert!(octant < 8);
    let mut result = SurfaceExposure::NONE;

    for face in SurfaceFace::ALL {
        let bit = face.axis_bit();
        let child_on_positive_side = octant & bit != 0;
        let child_on_parent_face = child_on_positive_side == face.is_positive();
        let area = if child_on_parent_face {
            parent.area(face)
        } else {
            let adjacent_octant = octant ^ bit;
            if occupied_siblings & (1 << adjacent_octant) == 0 {
                255
            } else {
                0
            }
        };
        result.set_area(face, area);
    }

    result
}
