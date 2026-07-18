//! Camera transforms shared by every presentation strategy.
//!
//! This module deliberately knows nothing about a graphics API.  Raster
//! renderers consume the WebGPU projection matrix while ray renderers consume
//! the orthonormal basis.  Keeping both products behind one function prevents
//! switching renderer from subtly changing the camera.

use crate::application::ports::FrameParams;

/// `tan(vertical field of view / 2)` for the 75 degree default.
pub const DEFAULT_FOV_TAN: f32 = 0.767_326_95;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraBasis {
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
}

/// Builds the canonical right/up/forward basis from engine yaw and pitch.
pub fn camera_basis(yaw: f32, pitch: f32) -> CameraBasis {
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    CameraBasis {
        right: [cos_yaw, 0.0, -sin_yaw],
        up: [sin_pitch * sin_yaw, cos_pitch, sin_pitch * cos_yaw],
        forward: [-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw],
    }
}

/// Returns a column-major WebGPU view-projection matrix.
///
/// WebGPU's clip-space depth is `0..1`, unlike the old OpenGL `-1..1`
/// convention.  Naming the convention here makes the migration testable and
/// keeps backend-specific correction matrices out of individual pipelines.
pub fn webgpu_view_projection(
    frame: &FrameParams,
    width: u32,
    height: u32,
    far: f32,
    fov_tan: f32,
) -> [f32; 16] {
    let basis = camera_basis(frame.yaw, frame.pitch);
    let p = frame.camera_pos;
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let view = [
        basis.right[0],
        basis.up[0],
        -basis.forward[0],
        0.0,
        basis.right[1],
        basis.up[1],
        -basis.forward[1],
        0.0,
        basis.right[2],
        basis.up[2],
        -basis.forward[2],
        0.0,
        -dot(basis.right, p),
        -dot(basis.up, p),
        dot(basis.forward, p),
        1.0,
    ];

    let near = 0.03;
    let far = far.max(near + 0.01);
    let f = 1.0 / fov_tan.clamp(0.05, 4.0);
    let aspect = width.max(1) as f32 / height.max(1) as f32;
    let projection = [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        far / (near - far),
        -1.0,
        0.0,
        0.0,
        (far * near) / (near - far),
        0.0,
    ];
    multiply_column_major(&projection, &view)
}

pub fn multiply_column_major(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut result = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            result[column * 4 + row] = a[row] * b[column * 4]
                + a[4 + row] * b[column * 4 + 1]
                + a[8 + row] * b[column * 4 + 2]
                + a[12 + row] * b[column * 4 + 3];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn basis_is_orthonormal_for_oblique_camera() {
        let basis = camera_basis(-1.1, 0.37);
        for axis in [basis.right, basis.up, basis.forward] {
            assert!((dot(axis, axis) - 1.0).abs() < 1e-5);
        }
        assert!(dot(basis.right, basis.up).abs() < 1e-5);
        assert!(dot(basis.right, basis.forward).abs() < 1e-5);
        assert!(dot(basis.up, basis.forward).abs() < 1e-5);
    }

    #[test]
    fn webgpu_projection_maps_near_and_far_to_zero_and_one() {
        let frame = FrameParams::default();
        let matrix = webgpu_view_projection(&frame, 160, 90, 100.0, DEFAULT_FOV_TAN);
        // At yaw/pitch zero, forward is -Z and the view is identity.  Apply
        // just the depth rows to two points in front of the camera.
        let project_depth = |z: f32| {
            let clip_z = matrix[10] * z + matrix[14];
            let clip_w = matrix[11] * z + matrix[15];
            clip_z / clip_w
        };
        assert!(project_depth(-0.03).abs() < 1e-4);
        assert!((project_depth(-100.0) - 1.0).abs() < 1e-4);
    }
}
