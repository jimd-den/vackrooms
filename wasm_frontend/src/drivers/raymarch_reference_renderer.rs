use crate::adapters::cpu_splatter::atlas::{decode_node, is_emissive};
use crate::adapters::cpu_splatter::camera::Camera;
use crate::adapters::cpu_splatter::settings::CpuRenderSettings;
use crate::adapters::cpu_splatter::shading::{SplatSurface, shade};
use crate::application::ports::{ChunkDraw, Environment, FrameParams};
use crate::core::ports::reference_renderer::{
    ReferenceRenderSettings, ReferenceRendererPort, RenderSceneSnapshot, RenderedImage,
};

pub struct RaymarchReferenceRenderer;

impl RaymarchReferenceRenderer {
    pub fn new() -> Self {
        Self
    }
}

fn raymarch_pixel(
    cam: &Camera,
    env: &Environment,
    settings: &CpuRenderSettings,
    atlas: &[u32],
    chunks: &[ChunkDraw],
    ro: [f32; 3],
    rd: [f32; 3],
) -> Option<[u8; 3]> {
    let safe = |v: f32| {
        if v.abs() < 1e-4 {
            1e-4f32.copysign(v)
        } else {
            v
        }
    };
    let rd = [safe(rd[0]), safe(rd[1]), safe(rd[2])];
    let inv = [1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2]];

    let mut hits: Vec<(usize, f32)> = Vec::new();
    for (i, d) in chunks.iter().enumerate() {
        let lro = [
            ro[0] - d.origin[0],
            ro[1] - d.origin[1],
            ro[2] - d.origin[2],
        ];
        let mut t_entry = f32::MIN;
        let mut t_exit = f32::MAX;
        for a in 0..3 {
            let t1 = (0.0 - lro[a]) * inv[a];
            let t2 = (d.world_size - lro[a]) * inv[a];
            t_entry = t_entry.max(t1.min(t2));
            t_exit = t_exit.min(t1.max(t2));
        }
        if t_entry < t_exit && t_exit > 0.0 {
            hits.push((i, t_entry.max(0.0)));
        }
    }
    hits.sort_by(|a, b| a.1.total_cmp(&b.1));

    for (i, t_min) in hits {
        let d = &chunks[i];
        let lro = [
            ro[0] - d.origin[0],
            ro[1] - d.origin[1],
            ro[2] - d.origin[2],
        ];
        if let Some((t, bmin, bmax, _vt, hit_leaf_node_idx)) =
            raymarch_svo(atlas, lro, rd, d.root_index as usize, t_min, d.world_size)
        {
            let decoded = decode_node(atlas, hit_leaf_node_idx)?;

            let center = [
                (bmin[0] + bmax[0]) * 0.5 + d.origin[0],
                (bmin[1] + bmax[1]) * 0.5 + d.origin[1],
                (bmin[2] + bmax[2]) * 0.5 + d.origin[2],
            ];
            let world_size = bmax[0] - bmin[0];

            let surface = SplatSurface {
                center,
                world_size,
                dist: t,
                base_color: decoded.color,
                light_level: decoded.light,
                is_emissive: is_emissive(decoded.voxel_type),
                crowded_siblings: 1, // Default/fallback sibling density for raymarcher
                shadow_factor: 1.0,  // Unshadowed baseline
            };

            let rgb = shade(cam, env, settings, atlas, chunks, &surface);
            return Some(rgb);
        }
    }
    None
}

fn raymarch_svo(
    atlas: &[u32],
    ro: [f32; 3],
    rd: [f32; 3],
    chunk_root_idx: usize,
    t_entry: f32,
    world_size: f32,
) -> Option<(f32, [f32; 3], [f32; 3], u32, usize)> {
    let mut t = t_entry;
    let mut p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];

    let mut stack: Vec<(usize, [f32; 3], [f32; 3])> = Vec::new();
    let mut current_node = chunk_root_idx;
    let mut cmin = [0.0f32; 3];
    let mut cmax = [world_size; 3];

    for _ in 0..160 {
        if current_node * 4 + 2 >= atlas.len() {
            return None;
        }
        let tx = current_node * 4;
        let (is_leaf, a, b) = (atlas[tx] == 1, atlas[tx + 1], atlas[tx + 2]);

        if is_leaf {
            if a != 0 {
                return Some((t, cmin, cmax, a, current_node));
            }
            let exit = |lo: f32, hi: f32, o: f32, d: f32| (if d > 0.0 { hi } else { lo } - o) / d;
            let tx = exit(cmin[0], cmax[0], ro[0], rd[0]);
            let ty = exit(cmin[1], cmax[1], ro[1], rd[1]);
            let tz = exit(cmin[2], cmax[2], ro[2], rd[2]);
            let t_exit = tx.min(ty).min(tz);
            t = t_exit;
            p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];
            if (t_exit - tx).abs() < 1e-4 {
                p[0] = (if rd[0] > 0.0 { cmax[0] } else { cmin[0] })
                    + (if rd[0] > 0.0 { 0.001 } else { -0.001 });
            }
            if (t_exit - ty).abs() < 1e-4 {
                p[1] = (if rd[1] > 0.0 { cmax[1] } else { cmin[1] })
                    + (if rd[1] > 0.0 { 0.001 } else { -0.001 });
            }
            if (t_exit - tz).abs() < 1e-4 {
                p[2] = (if rd[2] > 0.0 { cmax[2] } else { cmin[2] })
                    + (if rd[2] > 0.0 { 0.001 } else { -0.001 });
            }
            while outside(p, cmin, cmax) {
                let Some(f) = stack.pop() else { return None };
                current_node = f.0;
                cmin = f.1;
                cmax = f.2;
            }
        } else {
            let center = [
                (cmin[0] + cmax[0]) * 0.5,
                (cmin[1] + cmax[1]) * 0.5,
                (cmin[2] + cmax[2]) * 0.5,
            ];
            let ox = (p[0] >= center[0]) as usize;
            let oy = (p[1] >= center[1]) as usize;
            let oz = (p[2] >= center[2]) as usize;
            let child_idx = (oz << 2) | (oy << 1) | ox;

            if b & (1 << child_idx) != 0 {
                if stack.len() < 8 {
                    stack.push((current_node, cmin, cmax));
                }
                if ox == 1 {
                    cmin[0] = center[0]
                } else {
                    cmax[0] = center[0]
                }
                if oy == 1 {
                    cmin[1] = center[1]
                } else {
                    cmax[1] = center[1]
                }
                if oz == 1 {
                    cmin[2] = center[2]
                } else {
                    cmax[2] = center[2]
                }
                current_node = a as usize + child_idx;
            } else {
                let omin = [
                    if ox == 1 { center[0] } else { cmin[0] },
                    if oy == 1 { center[1] } else { cmin[1] },
                    if oz == 1 { center[2] } else { cmin[2] },
                ];
                let omax = [
                    if ox == 1 { cmax[0] } else { center[0] },
                    if oy == 1 { cmax[1] } else { center[1] },
                    if oz == 1 { cmax[2] } else { center[2] },
                ];
                let exit =
                    |lo: f32, hi: f32, o: f32, d: f32| (if d > 0.0 { hi } else { lo } - o) / d;
                let tx = exit(omin[0], omax[0], ro[0], rd[0]);
                let ty = exit(omin[1], omax[1], ro[1], rd[1]);
                let tz = exit(omin[2], omax[2], ro[2], rd[2]);
                let t_exit = tx.min(ty).min(tz);
                t = t_exit;
                p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];
                if (t_exit - tx).abs() < 1e-4 {
                    p[0] = (if rd[0] > 0.0 { omax[0] } else { omin[0] })
                        + (if rd[0] > 0.0 { 0.001 } else { -0.001 });
                }
                if (t_exit - ty).abs() < 1e-4 {
                    p[1] = (if rd[1] > 0.0 { omax[1] } else { omin[1] })
                        + (if rd[1] > 0.0 { 0.001 } else { -0.001 });
                }
                if (t_exit - tz).abs() < 1e-4 {
                    p[2] = (if rd[2] > 0.0 { omax[2] } else { omin[2] })
                        + (if rd[2] > 0.0 { 0.001 } else { -0.001 });
                }
                while outside(p, cmin, cmax) {
                    let Some(f) = stack.pop() else { return None };
                    current_node = f.0;
                    cmin = f.1;
                    cmax = f.2;
                }
            }
        }
    }
    None
}

fn outside(p: [f32; 3], min: [f32; 3], max: [f32; 3]) -> bool {
    p[0] < min[0]
        || p[0] > max[0]
        || p[1] < min[1]
        || p[1] > max[1]
        || p[2] < min[2]
        || p[2] > max[2]
}

impl ReferenceRendererPort for RaymarchReferenceRenderer {
    fn render(
        &mut self,
        scene: &RenderSceneSnapshot,
        settings: &ReferenceRenderSettings,
    ) -> RenderedImage {
        let mut rgba = vec![0u8; (settings.width * settings.height * 4) as usize];

        let mut frame_params = FrameParams::default();
        frame_params.camera_pos = settings.camera.position;
        frame_params.yaw = settings.camera.yaw;
        frame_params.pitch = settings.camera.pitch;
        frame_params.environment = settings.environment;

        let camera = Camera::new(
            &frame_params,
            settings.width as usize,
            settings.height as usize,
            &settings.cpu,
        );

        for y in 0..settings.height {
            for x in 0..settings.width {
                let relative_x = x as f32 - camera.half_w;
                let relative_y = camera.half_h - y as f32;

                let dir_camera_space = [
                    relative_x / camera.focal_px,
                    relative_y / camera.focal_px,
                    1.0f32,
                ];
                let length = (dir_camera_space[0].powi(2)
                    + dir_camera_space[1].powi(2)
                    + dir_camera_space[2].powi(2))
                .sqrt();
                let dir_camera_space_normalized = [
                    dir_camera_space[0] / length,
                    dir_camera_space[1] / length,
                    dir_camera_space[2] / length,
                ];

                let rd = [
                    dir_camera_space_normalized[0] * camera.right[0]
                        + dir_camera_space_normalized[1] * camera.up[0]
                        + dir_camera_space_normalized[2] * camera.forward[0],
                    dir_camera_space_normalized[0] * camera.right[1]
                        + dir_camera_space_normalized[1] * camera.up[1]
                        + dir_camera_space_normalized[2] * camera.forward[1],
                    dir_camera_space_normalized[0] * camera.right[2]
                        + dir_camera_space_normalized[1] * camera.up[2]
                        + dir_camera_space_normalized[2] * camera.forward[2],
                ];

                let color = raymarch_pixel(
                    &camera,
                    &settings.environment,
                    &settings.cpu,
                    &scene.atlas,
                    &scene.chunks,
                    settings.camera.position,
                    rd,
                )
                .unwrap_or([0, 0, 0]);

                let idx = ((y * settings.width + x) * 4) as usize;
                rgba[idx] = color[0];
                rgba[idx + 1] = color[1];
                rgba[idx + 2] = color[2];
                rgba[idx + 3] = 255;
            }
        }

        RenderedImage {
            width: settings.width,
            height: settings.height,
            rgba,
        }
    }
}
