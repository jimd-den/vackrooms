//! Conservative visibility tests shared by the raster drivers.

use crate::application::ports::FrameParams;

/// Draw-distance cap for the raster paths (world units). Their fog reaches
/// full opacity before this, so the cull is invisible.
pub const MAX_DRAW_DISTANCE: f32 = 100.0;

/// Sphere visibility: inside the draw distance and not entirely behind the
/// camera plane. Returns the squared distance for near-to-far sorting, or
/// `None` when culled. Deliberately conservative (no side-plane frustum
/// test): a false "visible" costs a few draws, a false cull costs a hole
/// in the world.
pub fn sphere_visible(
    center: [f32; 3],
    radius: f32,
    frame: &FrameParams,
    max_draw_distance: f32,
) -> Option<f32> {
    let to = [
        center[0] - frame.camera_pos[0],
        center[1] - frame.camera_pos[1],
        center[2] - frame.camera_pos[2],
    ];
    let distance2 = to[0] * to[0] + to[1] * to[1] + to[2] * to[2];
    // Cull only when the *whole sphere* lies beyond the distance boundary.
    // Testing the center against `max_draw_distance` cuts chunks/cells that
    // still intersect the visible volume and creates a hard pop at the fog
    // edge.
    let farthest_visible_center = max_draw_distance + radius;
    if distance2 > farthest_visible_center * farthest_visible_center {
        return None;
    }
    let (sp, cp) = frame.pitch.sin_cos();
    let (sy, cy) = frame.yaw.sin_cos();
    let forward = [-cp * sy, sp, -cp * cy];
    let dot = to[0] * forward[0] + to[1] * forward[1] + to[2] * forward[2];
    if dot >= -radius { Some(distance2) } else { None }
}

/// Center + bounding-sphere radius of a chunk from its origin and local
/// bounds-max — the shape both mesh and splat chunk records share.
pub fn chunk_bounding_sphere(origin: [f32; 3], bounds_max: [f32; 3]) -> ([f32; 3], f32) {
    let center = [
        origin[0] + bounds_max[0] * 0.5,
        origin[1] + bounds_max[1] * 0.5,
        origin[2] + bounds_max[2] * 0.5,
    ];
    let radius = (bounds_max[0] * bounds_max[0]
        + bounds_max[1] * bounds_max[1]
        + bounds_max[2] * bounds_max[2])
        .sqrt()
        * 0.5;
    (center, radius)
}
