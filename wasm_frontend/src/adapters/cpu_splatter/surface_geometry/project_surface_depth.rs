//! Project one selected box face into an exact per-pixel depth plane.
//!
//! A square splat covers more screen pixels than the projected voxel face.
//! Giving that whole square the voxel-center depth makes adjacent coplanar
//! voxels disagree about the depth of the same physical plane.  The error is
//! especially visible when a ceiling voxel overlaps a luminous panel.
//!
//! For a pinhole camera, reciprocal camera depth on a world-space plane is
//! affine in screen coordinates.  The three coefficients below therefore
//! recover the exact ray/plane intersection depth at every pixel without a
//! per-pixel world-space ray construction.

use super::super::camera::{Camera, dot};
use super::{SurfaceExposure, select_surface_normal};

/// Camera-depth information for one selected physical box face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProjectedSurfaceDepth {
    representative_depth: f32,
    /// `1 / depth = ax + by + c`, evaluated at pixel centers.
    reciprocal_depth: [f32; 3],
}

impl ProjectedSurfaceDepth {
    /// Constant-depth constructor retained for non-surface flare splats and
    /// narrow renderer tests. Production voxel surfaces use [`project_surface_depth`].
    pub(crate) fn constant(depth: f32) -> Self {
        Self {
            representative_depth: depth,
            reciprocal_depth: [0.0, 0.0, depth.recip()],
        }
    }

    pub(crate) fn representative_depth(self) -> f32 {
        self.representative_depth
    }

    /// Exact positive reciprocal camera depth at a target-space sample.
    /// Fine depth comparison consumes this directly, avoiding a division for
    /// every candidate pixel.
    pub(crate) fn reciprocal_at_pixel(self, pixel: [f32; 2]) -> Option<f32> {
        let inverse_depth = self.reciprocal_depth[0].mul_add(
            pixel[0],
            self.reciprocal_depth[1].mul_add(pixel[1], self.reciprocal_depth[2]),
        );
        if !inverse_depth.is_finite() || inverse_depth <= 0.0 {
            return None;
        }
        Some(inverse_depth)
    }
}

/// Selects one exposed face and projects its plane into target space.
pub(crate) fn project_surface_depth(
    camera: &Camera,
    center: [f32; 3],
    world_size: f32,
    voxel_type: u32,
    is_emissive: bool,
    exposure: SurfaceExposure,
) -> Option<ProjectedSurfaceDepth> {
    let camera_to_surface = [
        center[0] - camera.pos[0],
        center[1] - camera.pos[1],
        center[2] - camera.pos[2],
    ];
    let normal = select_surface_normal(voxel_type, is_emissive, exposure, camera_to_surface)?;
    let position = representative_surface_position(center, world_size, normal);
    let position_from_camera = [
        position[0] - camera.pos[0],
        position[1] - camera.pos[1],
        position[2] - camera.pos[2],
    ];
    let representative_depth = dot(position_from_camera, camera.forward);
    let plane_offset = dot(normal, position_from_camera);
    if !representative_depth.is_finite()
        || representative_depth <= 0.0
        || !plane_offset.is_finite()
        || plane_offset.abs() <= 1.0e-8
        || !camera.focal_px.is_finite()
        || camera.focal_px <= 0.0
    {
        return None;
    }

    let normal_right = dot(normal, camera.right);
    let normal_up = dot(normal, camera.up);
    let reciprocal_scale = (camera.focal_px * plane_offset).recip();
    let reciprocal_depth = [
        normal_right * reciprocal_scale,
        -normal_up * reciprocal_scale,
        dot(normal, camera.forward) / plane_offset
            + (-normal_right * camera.half_w + normal_up * camera.half_h) * reciprocal_scale,
    ];
    if reciprocal_depth
        .into_iter()
        .any(|coefficient| !coefficient.is_finite())
    {
        return None;
    }

    Some(ProjectedSurfaceDepth {
        representative_depth,
        reciprocal_depth,
    })
}

/// Intersects the normal ray from the box center with the AABB boundary.
/// Lighting and fine depth both use this same physical surface sample.
pub(crate) fn representative_surface_position(
    center: [f32; 3],
    world_size: f32,
    normal: [f32; 3],
) -> [f32; 3] {
    let dominant = normal
        .into_iter()
        .fold(0.0_f32, |largest, component| largest.max(component.abs()));
    if !world_size.is_finite() || world_size <= 0.0 || dominant <= 1.0e-6 {
        return center;
    }
    let distance_to_boundary = world_size * 0.5 / dominant;
    [
        center[0] + normal[0] * distance_to_boundary,
        center[1] + normal[1] * distance_to_boundary,
        center[2] + normal[2] * distance_to_boundary,
    ]
}

#[cfg(test)]
mod tests {
    use crate::application::ports::{DynamicLight, MAX_DYNAMIC_LIGHTS};
    use vackrooms::domain::entities::voxel_grid::VOXEL_CEILING;

    use super::*;

    fn camera() -> Camera {
        Camera {
            pos: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, 1.0],
            focal_px: 100.0,
            half_w: 100.0,
            half_h: 50.0,
            flashlight: false,
            dynamic_lights: [DynamicLight::default(); MAX_DYNAMIC_LIGHTS],
            dynamic_light_count: 0,
        }
    }

    #[test]
    fn coplanar_ceiling_voxels_produce_identical_pixel_depths() {
        let camera = camera();
        let exposure = SurfaceExposure::from_area([0, 0, 255, 0, 0, 0]);
        let near = project_surface_depth(
            &camera,
            [0.25, 2.25, 2.25],
            0.5,
            VOXEL_CEILING as u32,
            false,
            exposure,
        )
        .unwrap();
        let far = project_surface_depth(
            &camera,
            [0.25, 2.25, 4.25],
            0.5,
            VOXEL_CEILING as u32,
            false,
            exposure,
        )
        .unwrap();

        for pixel in [[10.5, 10.5], [100.5, 50.5], [190.5, 90.5]] {
            assert_eq!(
                near.reciprocal_at_pixel(pixel),
                far.reciprocal_at_pixel(pixel)
            );
        }
    }

    #[test]
    fn reciprocal_plane_reconstructs_the_representative_face_depth() {
        let camera = camera();
        let exposure = SurfaceExposure::from_area([0, 0, 255, 0, 0, 0]);
        let projected = project_surface_depth(
            &camera,
            [0.25, 2.25, 4.25],
            0.5,
            VOXEL_CEILING as u32,
            false,
            exposure,
        )
        .unwrap();
        let face_position = [0.25, 2.0, 4.25];
        let relative = [
            face_position[0] - camera.pos[0],
            face_position[1] - camera.pos[1],
            face_position[2] - camera.pos[2],
        ];
        let depth = dot(relative, camera.forward);
        let pixel = [
            camera.half_w + dot(relative, camera.right) / depth * camera.focal_px,
            camera.half_h - dot(relative, camera.up) / depth * camera.focal_px,
        ];

        assert!((projected.reciprocal_at_pixel(pixel).unwrap() - depth.recip()).abs() < 1.0e-5);
        assert_eq!(projected.representative_depth(), depth);
    }
}
