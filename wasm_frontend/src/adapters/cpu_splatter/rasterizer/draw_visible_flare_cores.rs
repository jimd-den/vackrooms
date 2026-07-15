//! Draw the visible cores of dynamic flare lights.
//!
//! A core is an ordinary depth-tested splat. Drawing it after scene geometry
//! lets the fine depth buffer provide wall occlusion without another raycast.

use crate::application::ports::FrameParams;

use super::super::camera::{Camera, dot};
use super::SoftwareRasterizer;

impl SoftwareRasterizer {
    pub(super) fn draw_visible_flare_cores(&mut self, frame: &FrameParams, camera: &Camera) {
        for light in frame.active_dynamic_lights() {
            let Some(core) = project_flare_core(light.position, light.intensity, camera) else {
                continue;
            };
            if core.depth > self.settings.max_draw_distance {
                continue;
            }

            self.splat(
                core.pixel[0],
                core.pixel[1],
                core.half_extent,
                core.depth - 0.05,
                core.color,
            );
        }
    }
}

struct ProjectedFlareCore {
    pixel: [f32; 2],
    half_extent: f32,
    depth: f32,
    color: [u8; 3],
}

fn project_flare_core(
    position: [f32; 3],
    intensity: f32,
    camera: &Camera,
) -> Option<ProjectedFlareCore> {
    let relative = [
        position[0] - camera.pos[0],
        position[1] - camera.pos[1],
        position[2] - camera.pos[2],
    ];
    let depth = dot(relative, camera.forward);
    if depth <= 0.1 {
        return None;
    }

    let normalized_intensity = intensity.clamp(0.0, 1.6) / 1.6;
    Some(ProjectedFlareCore {
        pixel: [
            camera.half_w + dot(relative, camera.right) / depth * camera.focal_px,
            camera.half_h - dot(relative, camera.up) / depth * camera.focal_px,
        ],
        half_extent: (0.06 / depth * camera.focal_px).clamp(1.0, 6.0),
        depth,
        color: [
            (255.0 * normalized_intensity) as u8,
            (150.0 * normalized_intensity) as u8,
            (60.0 * normalized_intensity) as u8,
        ],
    })
}
