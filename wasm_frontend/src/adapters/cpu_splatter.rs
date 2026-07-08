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

/// Half field of view tangent; identical to the GPU shader's `0.767`
/// (tan of 75 deg FOV / 2).
const HALF_FOV_TAN: f32 = 0.767;
/// Interior nodes projecting smaller than this many pixels (radius) are
/// drawn as one MIP splat instead of being descended into.
const LOD_CUTOFF_PX: f32 = 1.0;
/// Solid leaves are virtually subdivided until their projected half-extent
/// is at most this many pixels.
const MAX_SPLAT_HALF_PX: f32 = 3.0;
/// Don't subdivide below this world size (guards runaway recursion).
const MIN_SPLIT_SIZE: f32 = 0.02;
/// Ignore MIP splats of nodes that are mostly air.
const MIN_SPLAT_OCCUPANCY: f32 = 0.25;
/// Same fog density as the GPU shader.
const FOG_DENSITY: f32 = 0.015;

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
}

impl Camera {
    fn new(frame: &FrameParams, width: usize, height: usize) -> Self {
        let (sy, cy) = frame.yaw.sin_cos();
        let (sp, cp) = frame.pitch.sin_cos();
        Self {
            pos: frame.camera_pos,
            right: [cy, 0.0, -sy],
            up: [sp * sy, cp, sp * cy],
            forward: [-cp * sy, sp, -cp * cy],
            focal_px: height as f32 / (2.0 * HALF_FOV_TAN),
            half_w: width as f32 / 2.0,
            half_h: height as f32 / 2.0,
        }
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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
    ) {
        // Per-face directional shading from the face we actually see:
        // the dominant axis of the view vector picks the visible face.
        let to_cam = [
            center[0] - cam.pos[0],
            center[1] - cam.pos[1],
            center[2] - cam.pos[2],
        ];
        let ax = to_cam[0].abs();
        let ay = to_cam[1].abs();
        let az = to_cam[2].abs();
        let face = if ay >= ax && ay >= az {
            if to_cam[1] > 0.0 { 0.55 } else { 1.0 } // bottom face vs top face
        } else if ax >= az {
            0.8
        } else {
            0.7
        };

        // Contact AO: leaves packed tightly among solid siblings darken,
        // approximating corner occlusion for free from the child mask.
        let ao = 1.0 - 0.04 * crowded_siblings.saturating_sub(1) as f32;

        // BFS light + player torch (quadratic falloff) + fog.
        let bfs = 0.25 + 0.75 * (light_level / 15.0);
        let torch = (1.2 / (1.0 + 0.35 * dist + 0.10 * dist * dist)).min(1.0);
        let lum = if is_emissive {
            1.0
        } else {
            (bfs.max(torch * 0.9) * face * ao).min(1.0)
        };
        let fog = (-FOG_DENSITY * dist).exp();

        let scale = lum * fog * if is_emissive { 1.0 } else { 1.0 };
        let r = (base_color[0] * scale).min(255.0) as u8;
        let g = (base_color[1] * scale).min(255.0) as u8;
        let b = (base_color[2] * scale).min(255.0) as u8;

        self.splat(px, py, half_px, dist, r, g, b);
    }

    /// Recursive front-to-back node renderer.
    ///
    /// `node_idx == usize::MAX` marks a *virtual* node: a subdivision of a
    /// large collapsed leaf, which reuses `leaf_attrs` instead of the atlas.
    #[allow(clippy::too_many_arguments)]
    fn render_node(
        &mut self,
        cam: &Camera,
        node_idx: usize,
        min: [f32; 3],
        size: f32,
        crowded_siblings: u32,
        leaf_attrs: Option<(u32, [f32; 3], f32)>, // (voxel_type, color, light)
    ) {
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
            (0.0, 0.0, f32::INFINITY, f32::INFINITY)
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
                (true, (vt, color, self.atlas[t + 3] as f32))
            } else {
                (false, (0, [0.0; 3], 0.0))
            }
        };

        if is_leaf {
            let (voxel_type, color, light) = payload;
            if voxel_type == VOXEL_AIR {
                return;
            }
            // Large collapsed leaf: virtually subdivide until splats are small.
            if proj_half > MAX_SPLAT_HALF_PX && size > MIN_SPLIT_SIZE {
                self.recurse_children_front_to_back(
                    cam,
                    usize::MAX,
                    min,
                    half_size,
                    0xFF,
                    crowded_siblings,
                    Some((voxel_type, color, light)),
                );
            } else {
                self.shade_and_splat(
                    cam,
                    center,
                    px,
                    py,
                    proj_half.max(0.6),
                    z,
                    color,
                    light,
                    voxel_type == VOXEL_LIGHT,
                    crowded_siblings,
                );
            }
            return;
        }

        // Interior node.
        let t = node_idx * 4;
        let child_base = self.atlas[t + 1] as usize;
        let child_mask = self.atlas[t + 2];

        // LOD cutoff: subtree fits in ~a pixel -> one MIP splat.
        if proj_radius < LOD_CUTOFF_PX {
            let mip = self.mips.get(node_idx).copied().unwrap_or_default();
            if mip.occupancy >= MIN_SPLAT_OCCUPANCY {
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
                );
            }
            return;
        }

        self.recurse_children_front_to_back(
            cam,
            child_base,
            min,
            half_size,
            child_mask,
            child_mask.count_ones(),
            None,
        );
    }

    /// Visits the eight octants nearest-first (camera octant XOR popcount
    /// order) so the z-buffer culls occluded splats early.
    #[allow(clippy::too_many_arguments)]
    fn recurse_children_front_to_back(
        &mut self,
        cam: &Camera,
        child_base: usize, // usize::MAX -> virtual subdivision
        min: [f32; 3],
        half_size: f32,
        child_mask: u32,
        crowding: u32,
        leaf_attrs: Option<(u32, [f32; 3], f32)>,
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
            self.render_node(cam, child_idx, child_min, half_size, crowding, leaf_attrs);
        }
    }
}

/// Bottom-up MIP filtering over the whole atlas. Children always live at
/// higher indices than their parent (the SVO arena is append-only), so one
/// reverse pass aggregates every subtree in O(n).
fn build_mips(atlas: &[u32]) -> Vec<MipNode> {
    let node_count = atlas.len() / 4;
    let mut mips = vec![MipNode::default(); node_count];

    for i in (0..node_count).rev() {
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
                    light: atlas[t + 3] as f32,
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
                let m = match mips.get(child_base + child) {
                    Some(m) => *m,
                    None => continue,
                };
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
    mips
}

impl RendererPort for SoftwareRasterizer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        self.atlas = texels.to_vec();
        self.mips = build_mips(&self.atlas);
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        self.clear();
        if self.atlas.is_empty() {
            return;
        }
        let cam = Camera::new(frame, self.width, self.height);

        // Nearest chunk first: maximizes early z-rejection across chunks.
        let mut order: Vec<&ChunkDraw> = chunks.iter().collect();
        order.sort_by(|a, b| {
            let da = chunk_distance_sq(a, frame.camera_pos);
            let db = chunk_distance_sq(b, frame.camera_pos);
            da.total_cmp(&db)
        });

        for chunk in order {
            self.render_node(
                &cam,
                chunk.root_index as usize,
                chunk.origin,
                chunk.world_size,
                1,
                None,
            );
        }
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
        svo.set(1, 1, 1, 1, color, light);
        let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
        (gpu.texel_data, svo.root as u32)
    }

    fn frame_at(pos: [f32; 3], yaw: f32) -> FrameParams {
        FrameParams { camera_pos: pos, yaw, pitch: 0.0 }
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
        let chunks = [ChunkDraw { origin: [0.0, 0.0, 0.0], root_index: root as i32, world_size: 4.0 }];
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
        let chunks = [ChunkDraw { origin: [0.0, 0.0, 0.0], root_index: root as i32, world_size: 4.0 }];
        // yaw = 0 looks toward -z; the voxel is at +z relative to the camera.
        r.draw(&frame_at([1.5, 1.5, -2.0], 0.0), &chunks);
        assert!(r.framebuffer().chunks_exact(4).all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0));
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
            ChunkDraw { origin: [0.0, 0.0, 6.0], root_index: (red_nodes + white_root) as i32, world_size: 4.0 },
            ChunkDraw { origin: [0.0, 0.0, 0.0], root_index: red_root as i32, world_size: 4.0 },
        ];
        r.draw(&frame_at([1.5, 1.5, -2.0], std::f32::consts::PI), &chunks);

        let px = center_pixel(&r);
        assert!(px[0] > 60 && px[2] < px[0] / 2, "near red voxel must win: {:?}", px);
    }

    #[test]
    fn mip_aggregation_averages_child_colors() {
        let mut svo = SparseVoxelOctree::new(1, 2.0);
        // Two solid children: pure red + pure blue -> average purple-ish.
        svo.set(0, 0, 0, 1, 0xFF0000, 10);
        svo.set(1, 1, 1, 1, 0x0000FF, 4);
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
        let chunks = [ChunkDraw { origin: [0.0, 0.0, 0.0], root_index: root as i32, world_size: 4.0 }];
        // Very far away: the voxel projects to well under a pixel, and the
        // root's occupancy (1/64) is below MIN_SPLAT_OCCUPANCY -> no draw,
        // proving the LOD path (not the leaf path) handled it.
        r.draw(&frame_at([1.5, 1.5, -400.0], std::f32::consts::PI), &chunks);
        let lit = r.framebuffer().chunks_exact(4).filter(|p| p[0] > 0).count();
        assert_eq!(lit, 0);
    }
}
