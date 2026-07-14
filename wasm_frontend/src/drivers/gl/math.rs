//! Camera and light matrices — column-major, ready for `uniformMatrix4fv`.

use crate::application::ports::FrameParams;

/// View + projection for the raster paths, from the same yaw/pitch basis
/// the raymarcher and the CPU splatter use, so every renderer frames the
/// world identically.
pub fn camera_matrices(
    frame: &FrameParams,
    width: i32,
    height: i32,
    max_draw_distance: f32,
) -> ([f32; 16], [f32; 16]) {
    let (sp, cp) = frame.pitch.sin_cos();
    let (sy, cy) = frame.yaw.sin_cos();
    let right = [cy, 0.0, -sy];
    let up = [sp * sy, cp, sp * cy];
    let forward = [-cp * sy, sp, -cp * cy];
    let p = frame.camera_pos;
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let view = [
        right[0],
        up[0],
        -forward[0],
        0.0,
        right[1],
        up[1],
        -forward[1],
        0.0,
        right[2],
        up[2],
        -forward[2],
        0.0,
        -dot(right, p),
        -dot(up, p),
        dot(forward, p),
        1.0,
    ];
    let near = 0.03;
    let far = max_draw_distance + 10.0;
    let f = 1.0 / crate::drivers::webgl::fov_tan();
    let aspect = width as f32 / height.max(1) as f32;
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
        (far + near) / (near - far),
        -1.0,
        0.0,
        0.0,
        (2.0 * far * near) / (near - far),
        0.0,
    ];
    (projection, view)
}

pub fn ortho_matrix(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [f32; 16] {
    let mut m = [0.0; 16];
    m[0] = 2.0 / (right - left);
    m[5] = 2.0 / (top - bottom);
    m[10] = -2.0 / (far - near);
    m[12] = -(right + left) / (right - left);
    m[13] = -(top + bottom) / (top - bottom);
    m[14] = -(far + near) / (far - near);
    m[15] = 1.0;
    m
}

/// View matrix for a light looking straight down (-Y), up = -Z. Used by
/// the hero-light shadow pass (ceiling fixtures shine downward).
pub fn look_at_matrix_down(eye: [f32; 3]) -> [f32; 16] {
    let forward = [0.0, -1.0, 0.0];
    let up = [0.0, 0.0, -1.0f32];
    let right = [1.0, 0.0, 0.0f32];

    let mut m = [0.0; 16];
    m[0] = right[0];
    m[1] = up[0];
    m[2] = -forward[0];

    m[4] = right[1];
    m[5] = up[1];
    m[6] = -forward[1];

    m[8] = right[2];
    m[9] = up[2];
    m[10] = -forward[2];

    m[12] = -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]);
    m[13] = -(up[0] * eye[0] + up[1] * eye[1] + up[2] * eye[2]);
    m[14] = -(-forward[0] * eye[0] + -forward[1] * eye[1] + -forward[2] * eye[2]);
    m[15] = 1.0;
    m
}

pub fn multiply_matrices(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut m = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            m[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    m
}
