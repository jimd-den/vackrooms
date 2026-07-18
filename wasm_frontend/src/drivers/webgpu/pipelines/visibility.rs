use crate::application::ports::FrameParams;
use crate::application::rendering::camera_basis;

/// Conservative chunk/cell sphere test shared by raster strategies.
pub(super) fn sphere_is_visible(
    center: [f32; 3],
    radius: f32,
    frame: &FrameParams,
    max_distance: f32,
) -> bool {
    let delta = [
        center[0] - frame.camera_pos[0],
        center[1] - frame.camera_pos[1],
        center[2] - frame.camera_pos[2],
    ];
    let distance_squared = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    let reach = max_distance + radius.max(0.0);
    if distance_squared > reach * reach {
        return false;
    }
    let forward = camera_basis(frame.yaw, frame.pitch).forward;
    let forward_distance = delta[0] * forward[0] + delta[1] * forward[1] + delta[2] * forward[2];
    forward_distance + radius >= -0.03
}

pub(super) fn bounds_sphere(origin: [f32; 3], bounds_max: [f32; 3]) -> ([f32; 3], f32) {
    let half = [
        bounds_max[0] * 0.5,
        bounds_max[1] * 0.5,
        bounds_max[2] * 0.5,
    ];
    (
        [
            origin[0] + half[0],
            origin[1] + half[1],
            origin[2] + half[2],
        ],
        (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservative_test_keeps_intersecting_and_rejects_far_behind() {
        let frame = FrameParams::default();
        assert!(sphere_is_visible([0.0, 0.0, -5.0], 1.0, &frame, 10.0));
        assert!(!sphere_is_visible([0.0, 0.0, 5.0], 1.0, &frame, 10.0));
        assert!(!sphere_is_visible([0.0, 0.0, -30.0], 1.0, &frame, 10.0));
    }
}
