//! Project one voxel node and conservatively reject invisible bounds.

use super::super::super::camera::{Camera, dot};
use super::super::SoftwareRasterizer;
use super::super::project_aabb_footprint::{
    NEAR_PLANE_DEPTH, ScreenFootprint, aabb_camera_depth_interval, project_screen_footprint,
};
use super::NodeVisit;

/// Screen-space facts shared by traversal, LOD selection, and splat drawing.
pub(super) struct ProjectedVoxelNode {
    pub(super) center: [f32; 3],
    pub(super) relative_to_camera: [f32; 3],
    pub(super) camera_depth: f32,
    /// One eight-corner projection shared by culling, LOD, splitting, HZ,
    /// deferred-depth rejection, and the final splat write.
    pub(super) footprint: ScreenFootprint,
}

/// Returns `None` when the node cannot contribute a visible pixel.
pub(super) fn project_voxel_node(
    renderer: &SoftwareRasterizer,
    camera: &Camera,
    visit: NodeVisit,
) -> Option<ProjectedVoxelNode> {
    let half_size = visit.size * 0.5;
    let center = [
        visit.min[0] + half_size,
        visit.min[1] + half_size,
        visit.min[2] + half_size,
    ];
    let relative_to_camera = [
        center[0] - camera.pos[0],
        center[1] - camera.pos[1],
        center[2] - camera.pos[2],
    ];
    let camera_depth = dot(relative_to_camera, camera.forward);
    let depth_interval = aabb_camera_depth_interval(camera, relative_to_camera, half_size);
    if depth_interval[1] <= NEAR_PLANE_DEPTH {
        return None;
    }

    let footprint = project_screen_footprint(camera, visit.min, visit.size, depth_interval[0])?;

    if footprint.outside_framebuffer(renderer.target.width(), renderer.target.height()) {
        return None;
    }

    // OPTIMIZATION (hierarchical z): every overlapping tile must be fully
    // covered and its farthest pixel nearer than the node's nearest bound.
    if renderer.settings.toggles.hierarchical_z
        && renderer.target.coarse_rect_occludes(
            footprint.center,
            footprint.half_extent,
            footprint.nearest_depth.max(0.0),
        )
    {
        return None;
    }

    Some(ProjectedVoxelNode {
        center,
        relative_to_camera,
        camera_depth,
        footprint,
    })
}

/// True when a projected node may use the bounded flashlight-focus reserve
/// between the soft degradation threshold and the absolute hard cap.
pub(super) fn node_is_in_flashlight_focus(camera: &Camera, projected: &ProjectedVoxelNode) -> bool {
    if !camera.flashlight {
        return false;
    }
    let distance = projected.camera_depth.max(0.001);
    if distance >= 15.0 {
        return false;
    }
    let direction = projected
        .relative_to_camera
        .map(|component| component / distance);
    dot(direction, camera.forward) > 0.88
}
