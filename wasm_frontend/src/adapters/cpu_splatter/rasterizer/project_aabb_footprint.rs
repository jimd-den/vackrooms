//! Exact conservative screen bounds for one axis-aligned voxel box.
//!
//! Projecting only the box center and multiplying its world half-size by a
//! fudge factor under-covers oblique boxes.  Perspective projection is a
//! linear-fractional map, so each screen-coordinate extremum over the AABB
//! clipped to the near half-space occurs at a clipped-polyhedron vertex.
//! Those vertices are the original corners in front of the near plane plus
//! intersections of the twelve box edges with that plane. Projecting this
//! small fixed set gives the exact axis-aligned screen rectangle used by
//! culling, LOD selection, splitting, and splat writing.

use super::super::camera::{Camera, dot};

pub(super) const NEAR_PLANE_DEPTH: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ScreenFootprint {
    pub(super) center: [f32; 2],
    pub(super) half_extent: [f32; 2],
    /// Nearest camera-space depth of any AABB corner.
    pub(super) nearest_depth: f32,
}

impl ScreenFootprint {
    /// Scalar radius used by the square-splat target and square HZ tiles.
    /// It encloses the exact projected rectangle without an empirical pad.
    pub(super) fn enclosing_radius(self) -> f32 {
        self.half_extent[0].max(self.half_extent[1])
    }

    pub(super) fn outside_framebuffer(self, width: usize, height: usize) -> bool {
        self.center[0] + self.half_extent[0] < 0.0
            || self.center[0] - self.half_extent[0] >= width as f32
            || self.center[1] + self.half_extent[1] < 0.0
            || self.center[1] - self.half_extent[1] >= height as f32
    }
}

/// Exact camera-depth interval of an AABB.  This support-function form is
/// equivalent to testing all corners but avoids eight dot products for the
/// behind/near-plane decision.
pub(super) fn aabb_camera_depth_interval(
    camera: &Camera,
    center_relative_to_camera: [f32; 3],
    half_size: f32,
) -> [f32; 2] {
    let center_depth = dot(center_relative_to_camera, camera.forward);
    let depth_radius =
        half_size * (camera.forward[0].abs() + camera.forward[1].abs() + camera.forward[2].abs());
    [center_depth - depth_radius, center_depth + depth_radius]
}

/// Projects the exact vertices of an AABB clipped against the near plane.
/// Returns `None` for non-finite/malformed or wholly-behind geometry.
pub(super) fn project_screen_footprint(
    camera: &Camera,
    min: [f32; 3],
    size: f32,
    nearest_depth: f32,
) -> Option<ScreenFootprint> {
    if !size.is_finite()
        || size <= 0.0
        || !nearest_depth.is_finite()
        || min.into_iter().any(|coordinate| !coordinate.is_finite())
    {
        return None;
    }

    let mut relative_corners = [[0.0; 3]; 8];
    let mut depths = [0.0; 8];
    for octant in 0..8usize {
        let corner = [
            min[0] + if octant & 1 != 0 { size } else { 0.0 },
            min[1] + if octant & 2 != 0 { size } else { 0.0 },
            min[2] + if octant & 4 != 0 { size } else { 0.0 },
        ];
        let relative = [
            corner[0] - camera.pos[0],
            corner[1] - camera.pos[1],
            corner[2] - camera.pos[2],
        ];
        let depth = dot(relative, camera.forward);
        if !depth.is_finite() || relative.into_iter().any(|component| !component.is_finite()) {
            return None;
        }
        relative_corners[octant] = relative;
        depths[octant] = depth;
    }

    let mut screen_min = [f32::INFINITY; 2];
    let mut screen_max = [f32::NEG_INFINITY; 2];
    let mut include_projection = |relative: [f32; 3], depth: f32| -> Option<()> {
        let inverse_depth = 1.0 / depth;
        let screen = [
            camera.half_w + dot(relative, camera.right) * inverse_depth * camera.focal_px,
            camera.half_h - dot(relative, camera.up) * inverse_depth * camera.focal_px,
        ];
        if screen.into_iter().any(|coordinate| !coordinate.is_finite()) {
            return None;
        }
        for axis in 0..2 {
            screen_min[axis] = screen_min[axis].min(screen[axis]);
            screen_max[axis] = screen_max[axis].max(screen[axis]);
        }
        Some(())
    };

    for octant in 0..8usize {
        if depths[octant] >= NEAR_PLANE_DEPTH {
            include_projection(relative_corners[octant], depths[octant])?;
        }
    }

    // Each undirected AABB edge appears once: its low endpoint has the
    // corresponding axis bit clear. Crossing edges add the vertices created
    // by clipping the box against camera-space z = near.
    for axis_bit in [1usize, 2, 4] {
        for low in 0..8usize {
            if low & axis_bit != 0 {
                continue;
            }
            let high = low | axis_bit;
            let low_depth = depths[low];
            let high_depth = depths[high];
            let crosses = (low_depth < NEAR_PLANE_DEPTH && high_depth > NEAR_PLANE_DEPTH)
                || (high_depth < NEAR_PLANE_DEPTH && low_depth > NEAR_PLANE_DEPTH);
            if !crosses {
                continue;
            }
            let t = (NEAR_PLANE_DEPTH - low_depth) / (high_depth - low_depth);
            let intersection = [
                relative_corners[low][0]
                    + (relative_corners[high][0] - relative_corners[low][0]) * t,
                relative_corners[low][1]
                    + (relative_corners[high][1] - relative_corners[low][1]) * t,
                relative_corners[low][2]
                    + (relative_corners[high][2] - relative_corners[low][2]) * t,
            ];
            include_projection(intersection, NEAR_PLANE_DEPTH)?;
        }
    }

    if screen_min[0] == f32::INFINITY {
        return None;
    }

    let center = [
        (screen_min[0] + screen_max[0]) * 0.5,
        (screen_min[1] + screen_max[1]) * 0.5,
    ];
    let half_extent = [
        (screen_max[0] - screen_min[0]) * 0.5,
        (screen_max[1] - screen_min[1]) * 0.5,
    ];
    Some(ScreenFootprint {
        center,
        half_extent,
        nearest_depth: nearest_depth.max(NEAR_PLANE_DEPTH),
    })
}

#[cfg(test)]
mod tests {
    use crate::application::ports::{DynamicLight, MAX_DYNAMIC_LIGHTS};

    use super::*;

    fn camera(right: [f32; 3], forward: [f32; 3]) -> Camera {
        Camera {
            pos: [0.0; 3],
            right,
            up: [0.0, 1.0, 0.0],
            forward,
            focal_px: 100.0,
            half_w: 100.0,
            half_h: 100.0,
            flashlight: false,
            dynamic_lights: [DynamicLight::default(); MAX_DYNAMIC_LIGHTS],
            dynamic_light_count: 0,
        }
    }

    fn projected_corner(camera: &Camera, corner: [f32; 3]) -> [f32; 2] {
        let depth = dot(corner, camera.forward);
        [
            camera.half_w + dot(corner, camera.right) / depth * camera.focal_px,
            camera.half_h - dot(corner, camera.up) / depth * camera.focal_px,
        ]
    }

    #[test]
    fn every_oblique_aabb_corner_is_inside_the_returned_footprint() {
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        let camera = camera([diagonal, 0.0, -diagonal], [diagonal, 0.0, diagonal]);
        let min = [-1.0, -1.0, 9.0];
        let size = 2.0;
        let center = [0.0, 0.0, 10.0];
        let interval = aabb_camera_depth_interval(&camera, center, 1.0);
        let footprint = project_screen_footprint(&camera, min, size, interval[0]).unwrap();

        for octant in 0..8usize {
            let corner = [
                min[0] + if octant & 1 != 0 { size } else { 0.0 },
                min[1] + if octant & 2 != 0 { size } else { 0.0 },
                min[2] + if octant & 4 != 0 { size } else { 0.0 },
            ];
            let pixel = projected_corner(&camera, corner);
            assert!((pixel[0] - footprint.center[0]).abs() <= footprint.half_extent[0] + 1.0e-4);
            assert!((pixel[1] - footprint.center[1]).abs() <= footprint.half_extent[1] + 1.0e-4);
        }
    }

    #[test]
    fn oblique_support_is_larger_than_the_removed_one_point_fudge() {
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        let camera = camera([diagonal, 0.0, -diagonal], [diagonal, 0.0, diagonal]);
        let center = [0.0, 0.0, 10.0];
        let interval = aabb_camera_depth_interval(&camera, center, 1.0);
        let footprint =
            project_screen_footprint(&camera, [-1.0, -1.0, 9.0], 2.0, interval[0]).unwrap();
        let old_padded_half = 1.15 * camera.focal_px / dot(center, camera.forward);

        assert!(footprint.enclosing_radius() > old_padded_half);
    }

    #[test]
    fn depth_interval_is_exact_for_an_oblique_camera() {
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        let camera = camera([diagonal, 0.0, -diagonal], [diagonal, 0.0, diagonal]);
        let interval = aabb_camera_depth_interval(&camera, [0.0, 0.0, 10.0], 1.0);
        assert!((interval[0] - 8.0 * diagonal).abs() < 1.0e-5);
        assert!((interval[1] - 12.0 * diagonal).abs() < 1.0e-5);
    }

    #[test]
    fn box_crossing_near_plane_uses_clipped_edge_vertices() {
        let camera = camera([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let min = [0.02, -0.01, -0.01];
        let footprint = project_screen_footprint(&camera, min, 0.02, -0.01).unwrap();

        // At z=near the clipped rectangle spans x=.02..04 and y=-.01..01.
        assert!((footprint.center[0] - 400.0).abs() < 1.0e-3);
        assert!((footprint.half_extent[0] - 100.0).abs() < 1.0e-3);
        assert!((footprint.center[1] - 100.0).abs() < 1.0e-3);
        assert!((footprint.half_extent[1] - 100.0).abs() < 1.0e-3);
        assert_eq!(footprint.nearest_depth, NEAR_PLANE_DEPTH);
    }
}
