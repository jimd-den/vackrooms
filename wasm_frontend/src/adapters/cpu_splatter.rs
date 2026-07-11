//! CPU microvoxel splatting renderer — a software implementation of
//! [`RendererPort`] for machines with no usable GPU (or `?renderer=cpu`).
//!
//! Instead of marching a ray per pixel (Family 2/3), the CPU path inverts
//! the loop: it walks the SVO front-to-back and *splats* nodes onto a
//! z-buffered framebuffer, sized by their projected footprint:
//!
//! * **Coarse-to-fine LOD** — an interior node whose bounding sphere projects
//!   to under ~a pixel is drawn as a single splat using MIP-filtered
//!   attributes (average child color, max light, occupancy), computed once
//!   per atlas upload by [`build_mips`]. Distant geometry costs O(pixels),
//!   not O(voxels).
//! * **Virtual subdivision** — a *large* solid leaf (the SVO collapses
//!   uniform regions) is recursively split into eight virtual sub-boxes
//!   until each splat is a few pixels wide, so nearby walls stay crisp.
//! * **Front-to-back traversal** — children are visited nearest-octant-first
//!   so the z-buffer rejects most occluded splats before shading.
//!
//! Lighting is *richer* than the GPU path precisely because it runs per
//! splat, not per pixel:
//! * BFS-propagated light level (same data as the GPU path),
//! * per-face directional shading (tops bright, bottoms dark),
//! * contact AO from how crowded the parent octant is,
//! * a player-held torch with quadratic falloff,
//! * exponential distance fog.
//!
//! This module is platform-free (no web-sys): the browser driver only blits
//! the RGBA buffer. All geometry/shading logic is natively unit-tested.

use crate::application::ports::{ChunkDraw, FrameParams, RendererPort};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuShadowMode {
    Off,
    Hero,
}

#[derive(Debug, Clone, Copy)]
pub struct CpuRenderSettings {
    pub internal_scale: f32,      // e.g. 0.25 .. 1.0
    pub lod_cutoff_px: f32,      // e.g. 0.25 .. 2.0
    pub max_splat_half_px: f32,  // e.g. 2.0 .. 16.0
    pub max_virtual_depth: u8,   // e.g. 3 .. 8
    pub min_split_size: f32,     // world-space lower bound
    pub fog_density: f32,        // 0.003 .. 0.03
    pub fog_start: f32,
    pub max_draw_distance: f32,  // chunk/node rejection
    pub min_mip_occupancy: f32,  // preserve distant sparse detail
    pub shadows: CpuShadowMode,  // Off, Hero
    pub fov_tan: f32,            // vertical fov tangent
}

impl Default for CpuRenderSettings {
    fn default() -> Self {
        Self {
            internal_scale: 1.0,
            lod_cutoff_px: 1.0,
            max_splat_half_px: 12.0,
            max_virtual_depth: 5,
            min_split_size: 0.02,
            fog_density: 0.015,
            fog_start: 0.0,
            max_draw_distance: 96.0,
            min_mip_occupancy: 0.25,
            shadows: CpuShadowMode::Off,
            fov_tan: 0.767, // default 75 deg FOV
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    pub t: f32,
    pub voxel_type: u32,
}

struct DecodedNode {
    is_leaf: bool,
    child_base: usize,
    child_mask: u32,
    voxel_type: u32,
}

fn decode_node(atlas: &[u32], node_idx: usize) -> Option<DecodedNode> {
    let t = node_idx * 4;
    if t + 3 >= atlas.len() {
        return None;
    }
    let is_leaf = atlas[t] == 1;
    if is_leaf {
        Some(DecodedNode {
            is_leaf: true,
            child_base: 0,
            child_mask: 0,
            voxel_type: atlas[t + 1],
        })
    } else {
        Some(DecodedNode {
            is_leaf: false,
            child_base: atlas[t + 1] as usize,
            child_mask: atlas[t + 2],
            voxel_type: 0,
        })
    }
}

fn raymarch_svo_single(
    atlas: &[u32],
    ro: [f32; 3],
    rd: [f32; 3],
    chunk_root_idx: usize,
    t_entry: f32,
    t_exit: f32,
    world_size: f32,
) -> Option<RayHit> {
    let mut t = t_entry;
    let mut p = [
        ro[0] + t * rd[0],
        ro[1] + t * rd[1],
        ro[2] + t * rd[2],
    ];

    #[derive(Clone, Copy)]
    struct StackFrame {
        node_idx: usize,
        b_min: [f32; 3],
        b_max: [f32; 3],
    }

    let mut stack = [StackFrame { node_idx: 0, b_min: [0.0; 3], b_max: [0.0; 3] }; 9];
    let mut stack_ptr = 0;

    let mut current_node = chunk_root_idx;
    let mut current_min = [0.0, 0.0, 0.0];
    let mut current_max = [world_size, world_size, world_size];

    let mut steps = 0;
    const MAX_STEPS: i32 = 160;

    while steps < MAX_STEPS {
        steps += 1;

        let node = decode_node(atlas, current_node)?;

        if node.is_leaf {
            if node.voxel_type != 0 {
                return Some(RayHit {
                    t,
                    voxel_type: node.voxel_type,
                });
            } else {
                // Empty-space skip: jump straight to this leaf's exit plane.
                let t_max_planes = [
                    ((if rd[0] > 0.0 { current_max[0] } else { current_min[0] }) - ro[0]) / rd[0],
                    ((if rd[1] > 0.0 { current_max[1] } else { current_min[1] }) - ro[1]) / rd[1],
                    ((if rd[2] > 0.0 { current_max[2] } else { current_min[2] }) - ro[2]) / rd[2],
                ];

                let t_exit_box = t_max_planes[0].min(t_max_planes[1]).min(t_max_planes[2]);
                t = t_exit_box;
                p = [
                    ro[0] + t * rd[0],
                    ro[1] + t * rd[1],
                    ro[2] + t * rd[2],
                ];

                if (t_exit_box - t_max_planes[0]).abs() < 0.0001 {
                    p[0] = (if rd[0] > 0.0 { current_max[0] } else { current_min[0] }) + (if rd[0] > 0.0 { 0.001 } else { -0.001 });
                }
                if (t_exit_box - t_max_planes[1]).abs() < 0.0001 {
                    p[1] = (if rd[1] > 0.0 { current_max[1] } else { current_min[1] }) + (if rd[1] > 0.0 { 0.001 } else { -0.001 });
                }
                if (t_exit_box - t_max_planes[2]).abs() < 0.0001 {
                    p[2] = (if rd[2] > 0.0 { current_max[2] } else { current_min[2] }) + (if rd[2] > 0.0 { 0.001 } else { -0.001 });
                }

                while p[0] < current_min[0] || p[0] > current_max[0] ||
                       p[1] < current_min[1] || p[1] > current_max[1] ||
                       p[2] < current_min[2] || p[2] > current_max[2] {

                    if stack_ptr == 0 {
                        return None;
                    }
                    stack_ptr -= 1;
                    current_node = stack[stack_ptr].node_idx;
                    current_min = stack[stack_ptr].b_min;
                    current_max = stack[stack_ptr].b_max;
                }
            }
        } else {
            let center = [
                (current_min[0] + current_max[0]) * 0.5,
                (current_min[1] + current_max[1]) * 0.5,
                (current_min[2] + current_max[2]) * 0.5,
            ];
            let ox = if p[0] >= center[0] { 1 } else { 0 };
            let oy = if p[1] >= center[1] { 1 } else { 0 };
            let oz = if p[2] >= center[2] { 1 } else { 0 };
            let child_idx = (oz << 2) | (oy << 1) | ox;

            if (node.child_mask & (1 << child_idx)) != 0 {
                if stack_ptr < 8 {
                    stack[stack_ptr] = StackFrame {
                        node_idx: current_node,
                        b_min: current_min,
                        b_max: current_max,
                    };
                    stack_ptr += 1;
                }
                current_min[0] = if ox == 1 { center[0] } else { current_min[0] };
                current_max[0] = if ox == 1 { current_max[0] } else { center[0] };
                current_min[1] = if oy == 1 { center[1] } else { current_min[1] };
                current_max[1] = if oy == 1 { current_max[1] } else { center[1] };
                current_min[2] = if oz == 1 { center[2] } else { current_min[2] };
                current_max[2] = if oz == 1 { current_max[2] } else { center[2] };
                current_node = node.child_base + child_idx;
            } else {
                let oct_max = [
                    if ox == 1 { current_max[0] } else { center[0] },
                    if oy == 1 { current_max[1] } else { center[1] },
                    if oz == 1 { current_max[2] } else { center[2] },
                ];
                let oct_min = [
                    if ox == 1 { center[0] } else { current_min[0] },
                    if oy == 1 { center[1] } else { center[1] },
                    if oz == 1 { center[2] } else { current_min[2] },
                ];
                let t_max_planes = [
                    ((if rd[0] > 0.0 { oct_max[0] } else { oct_min[0] }) - ro[0]) / rd[0],
                    ((if rd[1] > 0.0 { oct_max[1] } else { oct_min[1] }) - ro[1]) / rd[1],
                    ((if rd[2] > 0.0 { oct_max[2] } else { oct_min[2] }) - ro[2]) / rd[2],
                ];

                let t_exit_oct = t_max_planes[0].min(t_max_planes[1]).min(t_max_planes[2]);
                t = t_exit_oct;
                p = [
                    ro[0] + t * rd[0],
                    ro[1] + t * rd[1],
                    ro[2] + t * rd[2],
                ];
                if (t_exit_oct - t_max_planes[0]).abs() < 0.0001 {
                    p[0] = (if rd[0] > 0.0 { oct_max[0] } else { oct_min[0] }) + (if rd[0] > 0.0 { 0.001 } else { -0.001 });
                }
                if (t_exit_oct - t_max_planes[1]).abs() < 0.0001 {
                    p[1] = (if rd[1] > 0.0 { oct_max[1] } else { oct_min[1] }) + (if rd[1] > 0.0 { 0.001 } else { -0.001 });
                }
                if (t_exit_oct - t_max_planes[2]).abs() < 0.0001 {
                    p[2] = (if rd[2] > 0.0 { oct_max[2] } else { oct_min[2] }) + (if rd[2] > 0.0 { 0.001 } else { -0.001 });
                }

                while p[0] < current_min[0] || p[0] > current_max[0] ||
                       p[1] < current_min[1] || p[1] > current_max[1] ||
                       p[2] < current_min[2] || p[2] > current_max[2] {

                    if stack_ptr == 0 {
                        return None;
                    }
                    stack_ptr -= 1;
                    current_node = stack[stack_ptr].node_idx;
                    current_min = stack[stack_ptr].b_min;
                    current_max = stack[stack_ptr].b_max;
                }
            }
        }
    }
    None
}

pub fn trace_svo(
    atlas: &[u32],
    chunks: &[ChunkDraw],
    origin: [f32; 3],
    direction: [f32; 3],
    max_t: f32,
) -> Option<RayHit> {
    struct ChunkHit {
        idx: usize,
        t_min: f32,
        t_max: f32,
    }

    let mut hits = Vec::with_capacity(chunks.len());

    let safe_rd = [
        if direction[0].abs() < 1e-4 { direction[0].signum() * 1e-4 } else { direction[0] },
        if direction[1].abs() < 1e-4 { direction[1].signum() * 1e-4 } else { direction[1] },
        if direction[2].abs() < 1e-4 { direction[2].signum() * 1e-4 } else { direction[2] },
    ];
    let inv_rd = [1.0 / safe_rd[0], 1.0 / safe_rd[1], 1.0 / safe_rd[2]];

    for (i, chunk) in chunks.iter().enumerate() {
        let local_ro = [
            origin[0] - chunk.origin[0],
            origin[1] - chunk.origin[1],
            origin[2] - chunk.origin[2],
        ];

        let box_min = [0.0, 0.0, 0.0];
        let box_max = [chunk.world_size, chunk.world_size, chunk.world_size];

        let t1 = [
            (box_min[0] - local_ro[0]) * inv_rd[0],
            (box_min[1] - local_ro[1]) * inv_rd[1],
            (box_min[2] - local_ro[2]) * inv_rd[2],
        ];
        let t2 = [
            (box_max[0] - local_ro[0]) * inv_rd[0],
            (box_max[1] - local_ro[1]) * inv_rd[1],
            (box_max[2] - local_ro[2]) * inv_rd[2],
        ];

        let t_min_p = [
            t1[0].min(t2[0]),
            t1[1].min(t2[1]),
            t1[2].min(t2[2]),
        ];
        let t_max_p = [
            t1[0].max(t2[0]),
            t1[1].max(t2[1]),
            t1[2].max(t2[2]),
        ];

        let t_entry = t_min_p[0].max(t_min_p[1]).max(t_min_p[2]);
        let t_exit = t_max_p[0].min(t_max_p[1]).min(t_max_p[2]);

        if t_entry < t_exit && t_exit > 0.0 && t_entry < max_t {
            let real_entry = t_entry.max(0.0);
            hits.push(ChunkHit {
                idx: i,
                t_min: real_entry,
                t_max: t_exit.min(max_t),
            });
        }
    }

    hits.sort_by(|a, b| a.t_min.total_cmp(&b.t_min));

    for hit in hits {
        let chunk = &chunks[hit.idx];
        let local_ro = [
            origin[0] - chunk.origin[0],
            origin[1] - chunk.origin[1],
            origin[2] - chunk.origin[2],
        ];

        if let Some(ray_hit) = raymarch_svo_single(
            atlas,
            local_ro,
            safe_rd,
            chunk.root_index as usize,
            hit.t_min,
            hit.t_max,
            chunk.world_size,
        ) {
            return Some(ray_hit);
        }
    }

    None
}

const VOXEL_AIR: u32 = 0;
const VOXEL_LIGHT: u32 = 4;

/// Per-node MIP-filtered attributes, rebuilt on every atlas upload.
#[derive(Debug, Clone, Copy, Default)]
struct MipNode {
    /// Occupancy-weighted average RGB of the subtree, 0..255 per channel.
    color: [f32; 3],
    /// Max BFS light level in the subtree (0..15).
    light: f32,
    /// Fraction of the subtree volume that is solid, 0..1.
    occupancy: f32,
}

/// Camera basis matching the GPU shader's yaw/pitch rotation exactly.
struct Camera {
    pos: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
    /// Focal length in pixels: h / (2 * tan(fov/2)).
    focal_px: f32,
    half_w: f32,
    half_h: f32,
    flashlight: bool,
}

impl Camera {
    fn new(frame: &FrameParams, width: usize, height: usize, settings: &CpuRenderSettings) -> Self {
        let (sy, cy) = frame.yaw.sin_cos();
        let (sp, cp) = frame.pitch.sin_cos();
        Self {
            pos: frame.camera_pos,
            right: [cy, 0.0, -sy],
            up: [sp * sy, cp, sp * cy],
            forward: [-cp * sy, sp, -cp * cy],
            focal_px: height as f32 / (2.0 * settings.fov_tan),
            half_w: width as f32 / 2.0,
            half_h: height as f32 / 2.0,
            flashlight: frame.flashlight,
        }
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SoftwareRasterizerTelemetry {
    pub visited_nodes: usize,
    pub budget_exhausted: bool,
    pub max_virtual_depth: usize,
    pub splat_count: usize,
    pub pixel_writes: usize,
}

pub struct SoftwareRasterizer {
    width: usize,
    height: usize,
    /// RGBA8 framebuffer, ready for ImageData blit.
    rgba: Vec<u8>,
    depth: Vec<f32>,
    /// SVO atlas texels: 4 u32 per node (see OctreeGpuSerializer).
    atlas: Vec<u32>,
    mips: Vec<MipNode>,
    
    // Telemetry and budgets
    pub visited_nodes: usize,
    pub budget_exhausted: bool,
    pub max_virtual_depth_reached: usize,
    pub splat_count: usize,
    pub pixel_writes: usize,

    // Configurable settings, shadow cache, and frame state
    pub settings: CpuRenderSettings,
    shadow_cache: Vec<f32>,
    frame_index: usize,
}

impl SoftwareRasterizer {
    pub fn new(width: usize, height: usize) -> Self {
        let mut r = Self {
            width: 0,
            height: 0,
            rgba: Vec::new(),
            depth: Vec::new(),
            atlas: Vec::new(),
            mips: Vec::new(),
            visited_nodes: 0,
            budget_exhausted: false,
            max_virtual_depth_reached: 0,
            splat_count: 0,
            pixel_writes: 0,
            settings: CpuRenderSettings::default(),
            shadow_cache: vec![1.0; 65536],
            frame_index: 0,
        };
        r.resize(width, height);
        r
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.rgba = vec![0; self.width * self.height * 4];
        self.depth = vec![f32::INFINITY; self.width * self.height];
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// The finished frame as RGBA8 rows, top-down.
    pub fn framebuffer(&self) -> &[u8] {
        &self.rgba
    }

    fn clear(&mut self) {
        for px in self.rgba.chunks_exact_mut(4) {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
            px[3] = 255;
        }
        self.depth.fill(f32::INFINITY);
    }

    /// Fills a depth-tested square splat. `half` is the half-extent in px.
    #[allow(clippy::too_many_arguments)]
    fn splat(&mut self, cx: f32, cy: f32, half: f32, z: f32, r: u8, g: u8, b: u8) {
        if self.pixel_writes >= 2_000_000 {
            self.budget_exhausted = true;
            return;
        }
        self.splat_count += 1;

        // Cap the maximum splat size to 32.0 pixels
        let half = half.min(32.0);

        let x0 = (cx - half).floor().max(0.0) as usize;
        let x1 = ((cx + half).ceil() as usize).min(self.width);
        let y0 = (cy - half).floor().max(0.0) as usize;
        let y1 = ((cy + half).ceil() as usize).min(self.height);

        for y in y0..y1 {
            let row = y * self.width;
            for x in x0..x1 {
                let idx = row + x;
                if z < self.depth[idx] {
                    self.depth[idx] = z;
                    let p = idx * 4;
                    self.rgba[p] = r;
                    self.rgba[p + 1] = g;
                    self.rgba[p + 2] = b;
                    self.pixel_writes += 1;
                }
            }
        }
    }

    /// Shades and splats one solid box (leaf, virtual sub-leaf, or MIP node).
    #[allow(clippy::too_many_arguments)]
    fn shade_and_splat(
        &mut self,
        cam: &Camera,
        center: [f32; 3],
        px: f32,
        py: f32,
        half_px: f32,
        dist: f32,
        base_color: [f32; 3],
        light_level: f32,
        is_emissive: bool,
        crowded_siblings: u32,
        shadow_factor: f32,
    ) {
        let to_cam = [
            center[0] - cam.pos[0],
            center[1] - cam.pos[1],
            center[2] - cam.pos[2],
        ];

        let dx_sq = to_cam[0] * to_cam[0];
        let dy_sq = to_cam[1] * to_cam[1];
        let dz_sq = to_cam[2] * to_cam[2];
        let sum = dx_sq + dy_sq + dz_sq;

        // Reconstruct face response and normal approximation
        let face_response = if sum > 1e-6 {
            let nx_sign = if to_cam[0] > 0.0 { -1.0 } else { 1.0 };
            let ny_sign = if to_cam[1] > 0.0 { -1.0 } else { 1.0 };
            let nz_sign = if to_cam[2] > 0.0 { -1.0 } else { 1.0 };

            let wx = dx_sq / sum;
            let wy = dy_sq / sum;
            let wz = dz_sq / sum;

            // Upward-facing surfaces (floor normal points up): 1.0
            // Downward-facing surfaces (ceiling normal points down): 0.35
            // Walls: 0.65 + 0.1 * nx + 0.05 * nz
            let y_resp = if ny_sign > 0.0 { 1.0 } else { 0.35 };
            let wall_resp = 0.65 + 0.1 * nx_sign + 0.05 * nz_sign;
            wy * y_resp + (wx + wz) * wall_resp
        } else {
            0.8
        };

        // Strengthen voxel AO, clamped to prevent pure black
        let raw_ao = 1.0 - 0.04 * crowded_siblings.saturating_sub(1) as f32;
        let ao = (raw_ao * 1.3).clamp(0.22, 1.0);

        let light_norm = (light_level / 15.0) * shadow_factor;

        // Fluorescent yellow-green ambient top vs dark olive ambient bottom mix
        let n_y = if sum > 1e-6 {
            (dy_sq / sum) * (if to_cam[1] > 0.0 { -1.0 } else { 1.0 })
        } else {
            0.0
        };
        let t_mix = n_y * 0.5 + 0.5;
        
        let ambient_up = [1.05 * light_norm, 1.0 * light_norm, 0.7 * light_norm];
        let ambient_down = [0.6 * light_norm, 0.55 * light_norm, 0.35 * light_norm];
        
        let ambient = [
            (ambient_down[0] * (1.0 - t_mix) + ambient_up[0] * t_mix) * ao * face_response,
            (ambient_down[1] * (1.0 - t_mix) + ambient_up[1] * t_mix) * ao * face_response,
            (ambient_down[2] * (1.0 - t_mix) + ambient_up[2] * t_mix) * ao * face_response,
        ];

        // Flashlight beam (spotlight cone of ~30 degrees)
        let mut flashlight_color = [0.0; 3];
        if cam.flashlight && sum > 1e-6 {
            let inv_dist = 1.0 / dist;
            let dir_to_cam = [to_cam[0] * inv_dist, to_cam[1] * inv_dist, to_cam[2] * inv_dist];
            
            // Camera forward vector points away from cam.pos towards looking dir.
            // Alignment checks if the splat is within the spotlight cone.
            let alignment = dot(dir_to_cam, cam.forward);
            if alignment > 0.85 {
                let edge_fade = ((alignment - 0.85) / 0.15).min(1.0);
                // Inverse quadratic flashlight attenuation up to 24 units
                let dist_fade = (1.0 - (dist / 24.0)).max(0.0).powi(2);
                let intensity = edge_fade * dist_fade * 1.5;
                
                // Warm cream highlights
                flashlight_color = [intensity * 1.0, intensity * 0.96, intensity * 0.85];
            }
        }

        // Apply lighting to base color
        let final_r = if is_emissive {
            base_color[0] * 1.5
        } else {
            base_color[0] * (ambient[0] + flashlight_color[0])
        };
        let final_g = if is_emissive {
            base_color[1] * 1.5
        } else {
            base_color[1] * (ambient[1] + flashlight_color[1])
        };
        let final_b = if is_emissive {
            base_color[2] * 1.5
        } else {
            base_color[2] * (ambient[2] + flashlight_color[2])
        };

        // Exponential fog blending
        let fog = (-self.settings.fog_density * (dist - self.settings.fog_start).max(0.0)).exp();

        // Blend with the background sickly-brown clear color: (0.15, 0.125, 0.055)
        let clear_color = [0.15 * 255.0, 0.125 * 255.0, 0.055 * 255.0];
        let r = (final_r * fog + clear_color[0] * (1.0 - fog)).clamp(0.0, 255.0) as u8;
        let g = (final_g * fog + clear_color[1] * (1.0 - fog)).clamp(0.0, 255.0) as u8;
        let b = (final_b * fog + clear_color[2] * (1.0 - fog)).clamp(0.0, 255.0) as u8;

        self.splat(px, py, half_px, dist, r, g, b);
    }

    pub fn telemetry(&self) -> SoftwareRasterizerTelemetry {
        SoftwareRasterizerTelemetry {
            visited_nodes: self.visited_nodes,
            budget_exhausted: self.budget_exhausted,
            max_virtual_depth: self.max_virtual_depth_reached,
            splat_count: self.splat_count,
            pixel_writes: self.pixel_writes,
        }
    }

    fn get_shadow_factor(&mut self, chunks: &[ChunkDraw], center: [f32; 3], is_emissive: bool, dist: f32) -> f32 {
        if self.settings.shadows == CpuShadowMode::Hero && !is_emissive && dist <= 24.0 && self.pixel_writes < 1_500_000 {
            let ix = (center[0] * 100.0) as i32;
            let iy = (center[1] * 100.0) as i32;
            let iz = (center[2] * 100.0) as i32;
            let hash_key = (ix.wrapping_mul(73856093) ^ iy.wrapping_mul(19349663) ^ iz.wrapping_mul(83492791)) as usize;
            let cache_idx = hash_key % self.shadow_cache.len();

            let should_trace = (hash_key + self.frame_index) % 4 == 0;
            if should_trace || self.shadow_cache[cache_idx] == 0.0 {
                let light_pos = [
                    (center[0] / 3.0).floor() * 3.0 + 1.5,
                    3.6,
                    (center[2] / 3.0).floor() * 3.0 + 1.5,
                ];
                let mut shadow_dir = [
                    light_pos[0] - center[0],
                    light_pos[1] - center[1],
                    light_pos[2] - center[2],
                ];
                let dist_to_light = (shadow_dir[0]*shadow_dir[0] + shadow_dir[1]*shadow_dir[1] + shadow_dir[2]*shadow_dir[2]).sqrt();
                let shadow_factor = if dist_to_light > 0.001 {
                    let inv_dist = 1.0 / dist_to_light;
                    let shadow_dir_norm = [
                        shadow_dir[0] * inv_dist,
                        shadow_dir[1] * inv_dist,
                        shadow_dir[2] * inv_dist,
                    ];
                    let ray_origin = [
                        center[0] + shadow_dir_norm[0] * 0.05,
                        center[1] + shadow_dir_norm[1] * 0.05,
                        center[2] + shadow_dir_norm[2] * 0.05,
                    ];
                    if let Some(_hit) = trace_svo(&self.atlas, chunks, ray_origin, shadow_dir_norm, dist_to_light - 0.05) {
                        0.25
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                self.shadow_cache[cache_idx] = shadow_factor;
                shadow_factor
            } else {
                self.shadow_cache[cache_idx]
            }
        } else {
            1.0
        }
    }

    /// Recursive front-to-back node renderer.
    #[allow(clippy::too_many_arguments)]
    fn render_node(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        node_idx: usize,
        min: [f32; 3],
        size: f32,
        crowded_siblings: u32,
        leaf_attrs: Option<(u32, [f32; 3], f32)>, // (voxel_type, color, light)
        virtual_depth: usize,
    ) {
        self.visited_nodes += 1;

        let half_size = size * 0.5;
        let center = [min[0] + half_size, min[1] + half_size, min[2] + half_size];
        let radius = size * 0.866; // bounding sphere

        let rel = [
            center[0] - cam.pos[0],
            center[1] - cam.pos[1],
            center[2] - cam.pos[2],
        ];
        let z = dot(rel, cam.forward);
        // Entirely behind the camera.
        if z + radius <= 0.01 {
            return;
        }

        let inside = z - radius <= 0.0; // camera inside the bounding sphere
        let (px, py, proj_half, proj_radius) = if inside {
            // Camera is near or inside the node.
            // Compute projection with a clamped, safe depth to avoid division by zero or infinity.
            let z_safe = z.max(half_size * 0.5).max(0.05);
            let inv_z = 1.0 / z_safe;
            let px = cam.half_w + dot(rel, cam.right) * inv_z * cam.focal_px;
            let py = cam.half_h - dot(rel, cam.up) * inv_z * cam.focal_px;
            let proj_half = half_size * cam.focal_px * inv_z;
            let proj_radius = radius * cam.focal_px * inv_z;
            (px, py, proj_half, proj_radius)
        } else {
            let inv_z = 1.0 / z;
            let px = cam.half_w + dot(rel, cam.right) * inv_z * cam.focal_px;
            let py = cam.half_h - dot(rel, cam.up) * inv_z * cam.focal_px;
            let proj_radius = radius * cam.focal_px / (z - radius).max(0.001);
            // Conservative screen-bounds cull.
            if px + proj_radius < 0.0
                || px - proj_radius >= self.width as f32
                || py + proj_radius < 0.0
                || py - proj_radius >= self.height as f32
            {
                return;
            }
            (px, py, half_size * cam.focal_px * inv_z, proj_radius)
        };

        // Resolve node attributes (real atlas node or virtual sub-leaf).
        let (is_leaf, payload) = if let Some(attrs) = leaf_attrs {
            (true, attrs)
        } else {
            let t = node_idx * 4;
            if t + 3 >= self.atlas.len() {
                return;
            }
            if self.atlas[t] == 1 {
                let vt = self.atlas[t + 1];
                let c = self.atlas[t + 2];
                let color = [
                    ((c >> 16) & 0xFF) as f32,
                    ((c >> 8) & 0xFF) as f32,
                    (c & 0xFF) as f32,
                ];
                (true, (vt, color, (self.atlas[t + 3] & 0xFF) as f32))
            } else {
                (false, (0, [0.0; 3], 0.0))
            }
        };

        // Strict per-frame visited node budget
        if self.visited_nodes >= 150_000 {
            self.budget_exhausted = true;
            if is_leaf {
                let (voxel_type, color, light) = payload;
                if voxel_type != VOXEL_AIR {
                    let shadow_factor = self.get_shadow_factor(chunks, center, voxel_type == VOXEL_LIGHT, z);
                    self.shade_and_splat(
                        cam,
                        center,
                        px,
                        py,
                        (proj_half * 1.15).max(0.85),
                        z,
                        color,
                        light,
                        voxel_type == VOXEL_LIGHT,
                        crowded_siblings,
                        shadow_factor,
                    );
                }
            } else {
                let mip = self.mips.get(node_idx).copied().unwrap_or_default();
                if mip.occupancy >= self.settings.min_mip_occupancy {
                    let shadow_factor = self.get_shadow_factor(chunks, center, false, z);
                    self.shade_and_splat(
                        cam,
                        center,
                        px,
                        py,
                        1.0,
                        z,
                        mip.color,
                        mip.light,
                        false,
                        crowded_siblings,
                        shadow_factor,
                    );
                }
            }
            return;
        }

        if is_leaf {
            let (voxel_type, color, light) = payload;
            if voxel_type == VOXEL_AIR {
                return;
            }

            if virtual_depth > self.max_virtual_depth_reached {
                self.max_virtual_depth_reached = virtual_depth;
            }

            // Large collapsed leaf: virtually subdivide until splats are small.
            if proj_half > self.settings.max_splat_half_px && size > self.settings.min_split_size && virtual_depth < self.settings.max_virtual_depth as usize && !inside {
                self.recurse_children_front_to_back(
                    cam,
                    chunks,
                    usize::MAX,
                    min,
                    half_size,
                    0xFF,
                    crowded_siblings,
                    Some((voxel_type, color, light)),
                    virtual_depth,
                );
            } else {
                let shadow_factor = self.get_shadow_factor(chunks, center, voxel_type == VOXEL_LIGHT, z);
                self.shade_and_splat(
                    cam,
                    center,
                    px,
                    py,
                    (proj_half * 1.15).max(0.85),
                    z,
                    color,
                    light,
                    voxel_type == VOXEL_LIGHT,
                    crowded_siblings,
                    shadow_factor,
                );
            }
            return;
        }

        // Interior node.
        let t = node_idx * 4;
        let child_base = self.atlas[t + 1] as usize;
        let child_mask = self.atlas[t + 2];

        // LOD cutoff: subtree fits in ~a pixel -> one MIP splat.
        if proj_radius < self.settings.lod_cutoff_px {
            let mip = self.mips.get(node_idx).copied().unwrap_or_default();
            if mip.occupancy >= self.settings.min_mip_occupancy {
                let shadow_factor = self.get_shadow_factor(chunks, center, false, z);
                self.shade_and_splat(
                    cam,
                    center,
                    px,
                    py,
                    1.0,
                    z,
                    mip.color,
                    mip.light,
                    false,
                    crowded_siblings,
                    shadow_factor,
                );
            }
            return;
        }

        self.recurse_children_front_to_back(
            cam,
            chunks,
            child_base,
            min,
            half_size,
            child_mask,
            child_mask.count_ones(),
            None,
            virtual_depth,
        );
    }

    /// Visits the eight octants nearest-first (camera octant XOR popcount
    /// order) so the z-buffer culls occluded splats early.
    #[allow(clippy::too_many_arguments)]
    fn recurse_children_front_to_back(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        child_base: usize, // usize::MAX -> virtual subdivision
        min: [f32; 3],
        half_size: f32,
        child_mask: u32,
        crowding: u32,
        leaf_attrs: Option<(u32, [f32; 3], f32)>,
        virtual_depth: usize,
    ) {
        let cx = min[0] + half_size;
        let cy = min[1] + half_size;
        let cz = min[2] + half_size;
        let near_octant = (cam.pos[0] >= cx) as u32
            | (((cam.pos[1] >= cy) as u32) << 1)
            | (((cam.pos[2] >= cz) as u32) << 2);

        // XOR distances grouped by popcount: near octant first, far last.
        const ORDER: [u32; 8] = [0, 1, 2, 4, 3, 5, 6, 7];
        for &flip in &ORDER {
            let child = near_octant ^ flip;
            if child_mask & (1 << child) == 0 {
                continue;
            }
            let child_min = [
                min[0] + (child & 1) as f32 * half_size,
                min[1] + ((child >> 1) & 1) as f32 * half_size,
                min[2] + ((child >> 2) & 1) as f32 * half_size,
            ];
            let child_idx = if child_base == usize::MAX {
                usize::MAX
            } else {
                child_base + child as usize
            };
            self.render_node(
                cam,
                chunks,
                child_idx,
                child_min,
                half_size,
                crowding,
                leaf_attrs,
                if child_base == usize::MAX { virtual_depth + 1 } else { 0 },
            );
        }
    }
}

/// Bottom-up MIP filtering over the whole atlas. Node order in the arena is
/// not guaranteed (`SparseVoxelOctree::set` appends children after parents,
/// `BuildOctreeUseCase` pushes the root last), so each subtree is resolved by
/// memoized recursion instead of a single directional sweep. Still O(n): every
/// node is computed exactly once.
fn build_mips(atlas: &[u32]) -> Vec<MipNode> {
    let node_count = atlas.len() / 4;
    let mut mips = vec![MipNode::default(); node_count];
    let mut done = vec![false; node_count];
    for i in 0..node_count {
        compute_mip(atlas, i, &mut mips, &mut done);
    }
    mips
}

fn compute_mip(atlas: &[u32], i: usize, mips: &mut [MipNode], done: &mut [bool]) {
    if done[i] {
        return;
    }
    // Marked before descending: a malformed cycle degrades to a zero mip
    // instead of infinite recursion.
    done[i] = true;

    let t = i * 4;
    if atlas[t] == 1 {
        // Leaf.
        let voxel_type = atlas[t + 1];
        if voxel_type != VOXEL_AIR {
            let c = atlas[t + 2];
            mips[i] = MipNode {
                color: [
                    ((c >> 16) & 0xFF) as f32,
                    ((c >> 8) & 0xFF) as f32,
                    (c & 0xFF) as f32,
                ],
                // Bits 0-7 are the scalar level; occlusion and the RGB
                // channels live in the higher bits (see OctreeGpuSerializer).
                light: (atlas[t + 3] & 0xFF) as f32,
                occupancy: 1.0,
            };
        }
    } else {
        let child_base = atlas[t + 1] as usize;
        let child_mask = atlas[t + 2];
        let mut acc = MipNode::default();
        for child in 0..8usize {
            if child_mask & (1 << child) == 0 {
                continue;
            }
            let idx = child_base + child;
            if idx >= mips.len() {
                continue;
            }
            compute_mip(atlas, idx, mips, done);
            let m = mips[idx];
            let w = m.occupancy / 8.0;
            acc.color[0] += m.color[0] * w;
            acc.color[1] += m.color[1] * w;
            acc.color[2] += m.color[2] * w;
            acc.light = acc.light.max(m.light);
            acc.occupancy += w;
        }
        if acc.occupancy > 0.0 {
            let inv = 1.0 / acc.occupancy;
            acc.color[0] *= inv;
            acc.color[1] *= inv;
            acc.color[2] *= inv;
        }
        mips[i] = acc;
    }
}

fn chunk_visible(chunk: &ChunkDraw, cam: &Camera, width: usize, height: usize, max_draw_dist: f32) -> bool {
    let half_size = chunk.world_size * 0.5;
    let center = [
        chunk.origin[0] + half_size,
        chunk.origin[1] + half_size,
        chunk.origin[2] + half_size,
    ];
    let radius = chunk.world_size * 0.866;

    let rel = [
        center[0] - cam.pos[0],
        center[1] - cam.pos[1],
        center[2] - cam.pos[2],
    ];
    
    // Distance check (3D distance to bounding sphere)
    let dist = (rel[0]*rel[0] + rel[1]*rel[1] + rel[2]*rel[2]).sqrt();
    if dist - radius > max_draw_dist {
        return false;
    }

    let z = dot(rel, cam.forward);
    // Entirely behind the camera.
    if z + radius <= 0.01 {
        return false;
    }

    let inside = z - radius <= 0.0;
    if !inside {
        let inv_z = 1.0 / z;
        let px = cam.half_w + dot(rel, cam.right) * inv_z * cam.focal_px;
        let py = cam.half_h - dot(rel, cam.up) * inv_z * cam.focal_px;
        let proj_radius = radius * cam.focal_px / (z - radius).max(0.001);
        // Conservative screen-bounds cull.
        if px + proj_radius < 0.0
            || px - proj_radius >= width as f32
            || py + proj_radius < 0.0
            || py - proj_radius >= height as f32
        {
            return false;
        }
    }
    true
}

impl RendererPort for SoftwareRasterizer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        self.atlas = texels.to_vec();
        self.mips = build_mips(&self.atlas);
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        self.clear();
        self.visited_nodes = 0;
        self.budget_exhausted = false;
        self.max_virtual_depth_reached = 0;
        self.splat_count = 0;
        self.pixel_writes = 0;

        if self.atlas.is_empty() {
            return;
        }
        let cam = Camera::new(frame, self.width, self.height, &self.settings);

        // Nearest chunk first: maximizes early z-rejection across chunks.
        let mut order: Vec<&ChunkDraw> = chunks.iter().collect();
        order.sort_by(|a, b| {
            let da = chunk_distance_sq(a, frame.camera_pos);
            let db = chunk_distance_sq(b, frame.camera_pos);
            da.total_cmp(&db)
        });

        for chunk in order {
            if !chunk_visible(chunk, &cam, self.width, self.height, self.settings.max_draw_distance) {
                continue;
            }

            self.render_node(
                &cam,
                chunks,
                chunk.root_index as usize,
                chunk.origin,
                chunk.world_size,
                1,
                None,
                0,
            );
        }

        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

fn chunk_distance_sq(chunk: &ChunkDraw, pos: [f32; 3]) -> f32 {
    let half = chunk.world_size * 0.5;
    let dx = chunk.origin[0] + half - pos[0];
    let dz = chunk.origin[2] + half - pos[2];
    dx * dx + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ports::{FrameParams, RendererPort};
    use vackrooms::adapters::octree_gpu_serializer::OctreeGpuSerializer;
    use vackrooms::domain::entities::sparse_voxel_octree::SparseVoxelOctree;

    /// depth-2 SVO (4^3) with one solid voxel, serialized like production.
    fn one_voxel_atlas(color: u32, light: u8) -> (Vec<u32>, u32) {
        let mut svo = SparseVoxelOctree::new(2, 4.0);
        svo.set(1, 1, 1, 1, color, [light; 3], 0);
        let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
        (gpu.texel_data, svo.root as u32)
    }

    fn frame_at(pos: [f32; 3], yaw: f32) -> FrameParams {
        FrameParams {
            camera_pos: pos,
            yaw,
            pitch: 0.0,
            flashlight: false,
        }
    }

    fn center_pixel(r: &SoftwareRasterizer) -> [u8; 4] {
        let idx = (r.height() / 2 * r.width() + r.width() / 2) * 4;
        let fb = r.framebuffer();
        [fb[idx], fb[idx + 1], fb[idx + 2], fb[idx + 3]]
    }

    #[test]
    fn voxel_in_front_of_camera_covers_center_pixel() {
        let (atlas, root) = one_voxel_atlas(0xFF8040, 15);
        let mut r = SoftwareRasterizer::new(64, 64);
        r.upload_atlas(&atlas);

        // Voxel spans (1..2)^3; look at its center from -z (yaw = PI faces +z).
        let chunks = [ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: root as i32,
            world_size: 4.0,
        }];
        r.draw(&frame_at([1.5, 1.5, -2.0], std::f32::consts::PI), &chunks);

        let px = center_pixel(&r);
        assert!(px[0] > 60, "red channel lit, got {:?}", px);
        assert!(px[0] > px[2], "red voxel must stay reddish, got {:?}", px);

        // A corner pixel must remain background black.
        let fb = r.framebuffer();
        assert_eq!(&fb[0..3], &[0, 0, 0]);
    }

    #[test]
    fn camera_facing_away_sees_nothing() {
        let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
        let mut r = SoftwareRasterizer::new(32, 32);
        r.upload_atlas(&atlas);
        let chunks = [ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: root as i32,
            world_size: 4.0,
        }];
        // yaw = 0 looks toward -z; the voxel is at +z relative to the camera.
        r.draw(&frame_at([1.5, 1.5, -2.0], 0.0), &chunks);
        assert!(
            r.framebuffer()
                .chunks_exact(4)
                .all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0)
        );
    }

    #[test]
    fn nearer_voxel_wins_depth_test() {
        // Two stacked chunks: a red voxel near, a white voxel behind it.
        let (red, red_root) = one_voxel_atlas(0xFF0000, 15);
        let (white, white_root) = one_voxel_atlas(0xFFFFFF, 15);
        let red_nodes = red.len() as u32 / 4;

        let mut atlas = red.clone();
        atlas.extend_from_slice(&white);

        let mut r = SoftwareRasterizer::new(64, 64);
        r.upload_atlas(&atlas);
        let chunks = [
            // Far chunk listed first to prove sorting/z-buffer handles order.
            ChunkDraw {
                origin: [0.0, 0.0, 6.0],
                root_index: (red_nodes + white_root) as i32,
                world_size: 4.0,
            },
            ChunkDraw {
                origin: [0.0, 0.0, 0.0],
                root_index: red_root as i32,
                world_size: 4.0,
            },
        ];
        r.draw(&frame_at([1.5, 1.5, -2.0], std::f32::consts::PI), &chunks);

        let px = center_pixel(&r);
        assert!(
            px[0] > 60 && px[2] < px[0] / 2,
            "near red voxel must win: {:?}",
            px
        );
    }

    #[test]
    fn mip_aggregation_works_for_root_last_node_order() {
        // BuildOctreeUseCase (the production chunk pipeline) pushes children
        // BEFORE their parent, so the root is the LAST node — the opposite
        // order of SparseVoxelOctree::set. build_mips must handle both.
        use vackrooms::domain::entities::voxel_grid::{VOXEL_WALL, VoxelGrid};
        use vackrooms::domain::use_cases::build_octree::BuildOctreeUseCase;

        let mut grid = VoxelGrid::new(4, 4, 4);
        grid.set(0, 0, 0, VOXEL_WALL);
        grid.set(3, 3, 3, VOXEL_WALL);
        let svo = BuildOctreeUseCase::new().execute(&grid, 2, 4.0);
        let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
        let mips = build_mips(&gpu.texel_data);

        let root = &mips[svo.root];
        assert!(
            root.occupancy > 0.0,
            "root mip must see its solid descendants, got occupancy 0"
        );
    }

    #[test]
    fn mip_aggregation_averages_child_colors() {
        let mut svo = SparseVoxelOctree::new(1, 2.0);
        // Two solid children: pure red + pure blue -> average purple-ish.
        svo.set(0, 0, 0, 1, 0xFF0000, [10; 3], 0);
        svo.set(1, 1, 1, 1, 0x0000FF, [4; 3], 0);
        let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
        let mips = build_mips(&gpu.texel_data);

        let root = &mips[svo.root];
        assert!((root.occupancy - 2.0 / 8.0).abs() < 1e-6);
        assert!((root.color[0] - 127.5).abs() < 1.0);
        assert!((root.color[2] - 127.5).abs() < 1.0);
        assert_eq!(root.light, 10.0);
    }

    #[test]
    fn distant_geometry_lods_to_single_splats() {
        let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
        let mut r = SoftwareRasterizer::new(64, 64);
        r.upload_atlas(&atlas);
        let chunks = [ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: root as i32,
            world_size: 4.0,
        }];
        // Very far away: the voxel projects to well under a pixel, and the
        // root's occupancy (1/64) is below MIN_SPLAT_OCCUPANCY -> no draw,
        // proving the LOD path (not the leaf path) handled it.
        r.draw(&frame_at([1.5, 1.5, -400.0], std::f32::consts::PI), &chunks);
        let lit = r.framebuffer().chunks_exact(4).filter(|p| p[0] > 0).count();
        assert_eq!(lit, 0);
    }

    #[test]
    fn test_face_shading_discontinuity() {
        let mut r = SoftwareRasterizer::new(16, 16);
        let center = [0.0, 0.0, 0.0];

        // Case A: Camera Y is 1.01 (above the center, Y-dominant)
        let frame_a = FrameParams {
            camera_pos: [-1.0, 1.01, -0.1],
            yaw: 0.0,
            pitch: 0.0,
            flashlight: false,
        };
        let cam_a = Camera::new(&frame_a, 16, 16, &CpuRenderSettings::default());

        // Case B: Camera Y is 0.99 (below Case A, X-dominant)
        let frame_b = FrameParams {
            camera_pos: [-1.0, 0.99, -0.1],
            yaw: 0.0,
            pitch: 0.0,
            flashlight: false,
        };
        let cam_b = Camera::new(&frame_b, 16, 16, &CpuRenderSettings::default());

        // Clear rasterizer
        r.clear();
        // Call shade_and_splat for Case A
        r.shade_and_splat(
            &cam_a,
            center,
            8.0,
            8.0,
            1.0,
            1.0,                   // dist
            [100.0, 100.0, 100.0], // base color
            15.0,                  // light level
            false,                 // is_emissive
            1,                     // crowded siblings
            1.0,                   // shadow factor
        );
        let color_a = center_pixel(&r);

        // Clear rasterizer
        r.clear();
        // Call shade_and_splat for Case B
        r.shade_and_splat(
            &cam_b,
            center,
            8.0,
            8.0,
            1.0,
            1.0,                   // dist
            [100.0, 100.0, 100.0], // base color
            15.0,                  // light level
            false,                 // is_emissive
            1,                     // crowded siblings
            1.0,                   // shadow factor
        );
        let color_b = center_pixel(&r);

        // Check difference. Under the old code, Case A gives face = 1.0, Case B gives face = 0.8.
        // This is a 20% difference, leading to a difference in color (e.g. 20 out of 255).
        // Assert that the difference is very small (e.g., <= 2) to prove continuity/smoothness.
        let diff = (color_a[0] as i32 - color_b[0] as i32).abs();
        assert!(
            diff <= 2,
            "Discontinuity found: diff was {} (color_a = {:?}, color_b = {:?})",
            diff,
            color_a,
            color_b
        );
    }

    #[test]
    fn test_adjacent_voxels_shading_variation() {
        let mut r = SoftwareRasterizer::new(16, 16);

        // Camera positioned above the floor plane (Y = 5.0)
        let frame = FrameParams {
            camera_pos: [1.0, 5.0, 1.0],
            yaw: 0.0,
            pitch: 0.0,
            flashlight: false,
        };
        let cam = Camera::new(&frame, 16, 16, &CpuRenderSettings::default());

        // Voxel 1 center: [0.5, 0.0, 0.5]
        // dist = sqrt(0.5^2 + 5.0^2 + 0.5^2) = sqrt(25.5) = 5.0497
        r.clear();
        r.shade_and_splat(
            &cam,
            [0.5, 0.0, 0.5],
            8.0,
            8.0,
            1.0,
            5.0497,
            [100.0, 100.0, 100.0],
            15.0,
            false,
            1,
            1.0,
        );
        let color_1 = center_pixel(&r);

        // Voxel 2 center: [2.5, 0.0, 0.5]
        // dist = sqrt(1.5^2 + 5.0^2 + 0.5^2) = sqrt(27.5) = 5.2440
        r.clear();
        r.shade_and_splat(
            &cam,
            [2.5, 0.0, 0.5],
            8.0,
            8.0,
            1.0,
            5.2440,
            [100.0, 100.0, 100.0],
            15.0,
            false,
            1,
            1.0,
        );
        let color_2 = center_pixel(&r);

        // Verify that the two adjacent voxels differ in color/shading.
        // This diagnostic test proves that rendering flat surfaces voxel-by-voxel
        // without quad-merging causes individual tiles to have slightly different
        // shading, creating a visible grid pattern.
        assert_ne!(
            color_1[0], color_2[0],
            "Expected adjacent voxels to have slightly different shading due to center-based lighting calculations"
        );
    }

    #[test]
    fn test_splat_flat_shading_limitation() {
        let mut r = SoftwareRasterizer::new(16, 16);
        let frame = FrameParams {
            camera_pos: [1.0, 5.0, 1.0],
            yaw: 0.0,
            pitch: 0.0,
            flashlight: false,
        };
        let cam = Camera::new(&frame, 16, 16, &CpuRenderSettings::default());

        // Draw a single large splat (half_px = 4.0) centered at [8.0, 8.0]
        r.clear();
        r.shade_and_splat(
            &cam,
            [0.5, 0.0, 0.5],
            8.0,
            8.0,
            4.0, // large size
            5.0,
            [100.0, 100.0, 100.0],
            15.0,
            false,
            1,
            1.0,
        );

        // Read pixels at different parts of the splat (left inside vs right inside)
        let idx_left = (8 * r.width() + 5) * 4;
        let idx_right = (8 * r.width() + 10) * 4;
        let fb = r.framebuffer();
        let color_left = [fb[idx_left], fb[idx_left + 1], fb[idx_left + 2]];
        let color_right = [fb[idx_right], fb[idx_right + 1], fb[idx_right + 2]];

        // Confirm they are identical, which proves that the entire projected area
        // of a single splat is displayed with one flat color, causing lighting to look
        // like big blocks/squares on screen.
        assert_eq!(
            color_left, color_right,
            "Expected a single splat to be flat-shaded (all its pixels have identical color)"
        );
    }

    #[test]
    fn camera_inside_large_solid_leaf_terminates_under_budget() {
        let mut atlas = vec![0u32; 4];
        atlas[0] = 1; // Leaf
        atlas[1] = 1; // Solid voxel type
        atlas[2] = 0xFFFFFF; // White color
        atlas[3] = 15; // BFS light

        let mut r = SoftwareRasterizer::new(64, 64);
        r.upload_atlas(&atlas);

        let chunks = [ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: 0,
            world_size: 16.0, // Very large leaf
        }];

        // Camera positioned inside the bounding sphere of the root chunk (e.g. at center [8.0, 8.0, 8.0])
        r.draw(&frame_at([8.0, 8.0, 8.0], 0.0), &chunks);

        let stats = r.telemetry();
        // Telemetry must show we did NOT trigger infinite recursive virtual subdivision
        // because camera was inside the node. It should have splatted directly or terminated
        // with small node count (e.g., visited_nodes <= 10, not 150,000 budget exhausted).
        assert!(!stats.budget_exhausted);
        assert!(stats.visited_nodes < 50, "Visited nodes was {}, expected very low", stats.visited_nodes);
    }

    #[test]
    fn test_cpu_render_settings_default() {
        let settings = CpuRenderSettings::default();
        assert_eq!(settings.max_virtual_depth, 5);
        assert_eq!(settings.shadows, CpuShadowMode::Off);
    }

    #[test]
    fn test_trace_svo_basic_intersect() {
        let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
        let chunks = [ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: root as i32,
            world_size: 4.0,
        }];

        // Ray passing through the center of the voxel (1.5, 1.5, 1.5):
        // Origin [1.5, 1.5, -1.0], direction [0.0, 0.0, 1.0], max_t = 10.0
        let hit = trace_svo(&atlas, &chunks, [1.5, 1.5, -1.0], [0.0, 0.0, 1.0], 10.0);
        assert!(hit.is_some());
        let hit_val = hit.unwrap();
        assert_eq!(hit_val.voxel_type, 1); // VOXEL_WALL
        assert!((hit_val.t - 2.0).abs() < 1e-4); // voxel starts at z=1.0

        // Ray missing the voxel:
        // Origin [0.5, 0.5, -1.0], direction [0.0, 0.0, 1.0]
        let miss = trace_svo(&atlas, &chunks, [0.5, 0.5, -1.0], [0.0, 0.0, 1.0], 10.0);
        assert!(miss.is_none());
    }

    #[test]
    fn test_draw_distance_culling() {
        let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
        let mut r = SoftwareRasterizer::new(16, 16);
        r.upload_atlas(&atlas);
        let chunks = [ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: root as i32,
            world_size: 4.0,
        }];
        // Draw with large draw distance
        r.settings.max_draw_distance = 100.0;
        let frame = frame_at([1.5, 1.5, -2.0], std::f32::consts::PI);
        r.draw(&frame, &chunks);
        assert!(r.telemetry().visited_nodes > 0);

        // Draw with tiny draw distance (should cull the chunk)
        r.settings.max_draw_distance = 0.1;
        r.draw(&frame, &chunks);
        assert_eq!(r.telemetry().visited_nodes, 0);
    }
}
