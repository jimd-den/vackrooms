//! Primary SVO traversal, projection, LOD selection, and virtual subdivision.
//!
//! `render_node` is deliberately a short dispatcher. Projection, atlas
//! decoding, budget fallback, leaf handling, interior handling, MIP drawing,
//! and child ordering each live in a small helper with an explicit data type.

use crate::application::ports::ChunkDraw;

use super::super::atlas::{VOXEL_AIR, decode_node, is_emissive};
use super::super::camera::{Camera, dot};
use super::{MAX_VISITED_NODES, SoftwareRasterizer};

/// Attributes carried by a real atlas leaf or a virtually subdivided leaf.
#[derive(Clone, Copy)]
struct LeafPayload {
    voxel_type: u32,
    color: [f32; 3],
    light: f32,
}

/// Everything needed to visit one real or virtual node.
#[derive(Clone, Copy)]
struct NodeVisit {
    node_idx: usize,
    min: [f32; 3],
    size: f32,
    crowded_siblings: u32,
    virtual_leaf: Option<LeafPayload>,
    virtual_depth: usize,
}

impl NodeVisit {
    fn root(chunk: &ChunkDraw) -> Self {
        Self {
            node_idx: chunk.root_index as usize,
            min: chunk.origin,
            size: chunk.world_size,
            crowded_siblings: 1,
            virtual_leaf: None,
            virtual_depth: 0,
        }
    }
}

#[derive(Clone, Copy)]
enum NodeKind {
    Leaf(LeafPayload),
    Interior { child_base: usize, child_mask: u32 },
}

/// Conservative screen projection of a node's bounding sphere and splat.
struct ProjectedNode {
    center: [f32; 3],
    relative_to_camera: [f32; 3],
    camera_depth: f32,
    pixel_x: f32,
    pixel_y: f32,
    splat_half: f32,
    projected_radius: f32,
    camera_inside: bool,
}

#[derive(Clone, Copy)]
enum Children {
    Atlas { base: usize, mask: u32 },
    Virtual(LeafPayload),
}

impl Children {
    fn mask(self) -> u32 {
        match self {
            Self::Atlas { mask, .. } => mask,
            Self::Virtual(_) => 0xFF,
        }
    }

    fn child(
        self,
        octant: u32,
        parent_virtual_depth: usize,
    ) -> (usize, Option<LeafPayload>, usize) {
        match self {
            Self::Atlas { base, .. } => (base + octant as usize, None, 0),
            Self::Virtual(payload) => (usize::MAX, Some(payload), parent_virtual_depth + 1),
        }
    }
}

impl SoftwareRasterizer {
    /// Traversal entry point kept narrow for the frame orchestrator.
    pub(super) fn render_chunk(&mut self, cam: &Camera, chunks: &[ChunkDraw], chunk: &ChunkDraw) {
        self.render_node(cam, chunks, NodeVisit::root(chunk));
    }

    /// Projects, resolves, and dispatches one node. All branch-specific work
    /// is delegated so the recursive control flow remains readable.
    fn render_node(&mut self, cam: &Camera, chunks: &[ChunkDraw], visit: NodeVisit) {
        let Some(projected) = self.project_node(cam, visit) else {
            return;
        };

        // Preserve telemetry semantics: a projected malformed atlas node is
        // counted before its failed decode, just as the original traversal.
        self.visited_nodes += 1;
        let Some(kind) = self.resolve_node(visit) else {
            return;
        };

        if self.visited_nodes >= MAX_VISITED_NODES && !budget_exempt(cam, &projected) {
            self.budget_exhausted = true;
            self.render_budget_fallback(cam, chunks, visit, &projected, kind);
            return;
        }

        match kind {
            NodeKind::Leaf(payload) => self.render_leaf(cam, chunks, visit, &projected, payload),
            NodeKind::Interior {
                child_base,
                child_mask,
            } => {
                self.render_interior(cam, chunks, visit, &projected, child_base, child_mask);
            }
        }
    }

    /// Projects a node and applies correctness clipping plus optional HZ
    /// rejection. `None` means the node cannot contribute a visible pixel.
    fn project_node(&self, cam: &Camera, visit: NodeVisit) -> Option<ProjectedNode> {
        let half_size = visit.size * 0.5;
        let center = [
            visit.min[0] + half_size,
            visit.min[1] + half_size,
            visit.min[2] + half_size,
        ];
        let radius = visit.size * 0.866;
        let relative_to_camera = [
            center[0] - cam.pos[0],
            center[1] - cam.pos[1],
            center[2] - cam.pos[2],
        ];
        let camera_depth = dot(relative_to_camera, cam.forward);
        if camera_depth + radius <= 0.01 {
            return None;
        }

        let camera_inside = camera_depth - radius <= 0.0;
        if camera_inside {
            return Some(project_inside_node(
                cam,
                center,
                relative_to_camera,
                camera_depth,
                half_size,
                radius,
            ));
        }

        let inverse_depth = 1.0 / camera_depth;
        let pixel_x =
            cam.half_w + dot(relative_to_camera, cam.right) * inverse_depth * cam.focal_px;
        let pixel_y = cam.half_h - dot(relative_to_camera, cam.up) * inverse_depth * cam.focal_px;
        let projected_radius = radius * cam.focal_px / (camera_depth - radius).max(0.001);

        if outside_target(
            pixel_x,
            pixel_y,
            projected_radius,
            self.target.width(),
            self.target.height(),
        ) {
            return None;
        }

        // OPTIMIZATION (hierarchical z): the target summary is conservative:
        // every overlapping tile must be fully covered and its farthest pixel
        // must still be nearer than this node's nearest possible depth.
        if self.settings.toggles.hierarchical_z
            && self.target.coarse_occludes(
                pixel_x,
                pixel_y,
                projected_radius,
                (camera_depth - radius).max(0.0),
            )
        {
            return None;
        }

        Some(ProjectedNode {
            center,
            relative_to_camera,
            camera_depth,
            pixel_x,
            pixel_y,
            splat_half: half_size * cam.focal_px * inverse_depth,
            projected_radius,
            camera_inside: false,
        })
    }

    fn resolve_node(&self, visit: NodeVisit) -> Option<NodeKind> {
        if let Some(payload) = visit.virtual_leaf {
            return Some(NodeKind::Leaf(payload));
        }

        let node = decode_node(&self.atlas, visit.node_idx)?;
        if node.is_leaf {
            Some(NodeKind::Leaf(LeafPayload {
                voxel_type: node.voxel_type,
                color: node.color,
                light: node.light,
            }))
        } else {
            Some(NodeKind::Interior {
                child_base: node.child_base,
                child_mask: node.child_mask,
            })
        }
    }

    /// Safety-cap behavior: retain the current fidelity and stop descending.
    fn render_budget_fallback(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        visit: NodeVisit,
        projected: &ProjectedNode,
        kind: NodeKind,
    ) {
        match kind {
            NodeKind::Leaf(payload) if payload.voxel_type != VOXEL_AIR => {
                self.draw_leaf_splat(cam, chunks, visit, projected, payload);
            }
            NodeKind::Leaf(_) => {}
            NodeKind::Interior { .. } => self.draw_mip_splat(cam, chunks, visit, projected),
        }
    }

    fn render_leaf(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        visit: NodeVisit,
        projected: &ProjectedNode,
        payload: LeafPayload,
    ) {
        if payload.voxel_type == VOXEL_AIR {
            return;
        }
        self.max_virtual_depth_reached = self.max_virtual_depth_reached.max(visit.virtual_depth);

        let should_subdivide = projected.splat_half > self.settings.max_splat_half_px
            && visit.size > self.settings.min_split_size
            && visit.virtual_depth < self.settings.max_virtual_depth as usize
            && !projected.camera_inside;
        if should_subdivide {
            self.visit_children(
                cam,
                chunks,
                visit,
                Children::Virtual(payload),
                visit.crowded_siblings,
            );
        } else {
            self.draw_leaf_splat(cam, chunks, visit, projected, payload);
        }
    }

    fn draw_leaf_splat(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        visit: NodeVisit,
        projected: &ProjectedNode,
        payload: LeafPayload,
    ) {
        let emissive = is_emissive(payload.voxel_type);
        let shadow = self.shadow_factor(chunks, projected.center, emissive, projected.camera_depth);
        self.shade_and_splat(
            cam,
            chunks,
            projected.center,
            visit.size,
            projected.pixel_x,
            projected.pixel_y,
            (projected.splat_half * 1.15).max(0.85),
            projected.camera_depth,
            payload.color,
            payload.light,
            emissive,
            visit.crowded_siblings,
            shadow,
        );
    }

    fn render_interior(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        visit: NodeVisit,
        projected: &ProjectedNode,
        child_base: usize,
        child_mask: u32,
    ) {
        // OPTIMIZATION (LOD): a subtree fitting in roughly a pixel becomes
        // one prefiltered MIP splat instead of a full descendant walk.
        if self.settings.toggles.mip_lod && projected.projected_radius < self.settings.lod_cutoff_px
        {
            self.draw_mip_splat(cam, chunks, visit, projected);
            return;
        }

        self.visit_children(
            cam,
            chunks,
            visit,
            Children::Atlas {
                base: child_base,
                mask: child_mask,
            },
            child_mask.count_ones(),
        );
    }

    fn draw_mip_splat(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        visit: NodeVisit,
        projected: &ProjectedNode,
    ) {
        let mip = self.mips.get(visit.node_idx).copied().unwrap_or_default();
        if mip.occupancy < self.settings.min_mip_occupancy {
            return;
        }
        let shadow = self.shadow_factor(chunks, projected.center, false, projected.camera_depth);
        self.shade_and_splat(
            cam,
            chunks,
            projected.center,
            visit.size,
            projected.pixel_x,
            projected.pixel_y,
            1.0,
            projected.camera_depth,
            mip.color,
            mip.light,
            false,
            visit.crowded_siblings,
            shadow,
        );
    }

    /// Visits occupied octants either near-to-far or in reference index
    /// order. Virtual subdivision reuses one leaf payload at the next depth.
    fn visit_children(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        parent: NodeVisit,
        children: Children,
        crowding: u32,
    ) {
        let half_size = parent.size * 0.5;
        let parent_center = [
            parent.min[0] + half_size,
            parent.min[1] + half_size,
            parent.min[2] + half_size,
        ];
        let near_octant = (cam.pos[0] >= parent_center[0]) as u32
            | (((cam.pos[1] >= parent_center[1]) as u32) << 1)
            | (((cam.pos[2] >= parent_center[2]) as u32) << 2);

        // XOR distances grouped by popcount: camera octant first, opposite
        // octant last. Plain octant order is the toggle-off reference path.
        const FRONT_TO_BACK_ORDER: [u32; 8] = [0, 1, 2, 4, 3, 5, 6, 7];
        let mask = children.mask();
        for (index, relative_octant) in FRONT_TO_BACK_ORDER.into_iter().enumerate() {
            let octant = if self.settings.toggles.front_to_back {
                near_octant ^ relative_octant
            } else {
                index as u32
            };
            if mask & (1 << octant) == 0 {
                continue;
            }

            let min = [
                parent.min[0] + (octant & 1) as f32 * half_size,
                parent.min[1] + ((octant >> 1) & 1) as f32 * half_size,
                parent.min[2] + ((octant >> 2) & 1) as f32 * half_size,
            ];
            let (node_idx, virtual_leaf, virtual_depth) =
                children.child(octant, parent.virtual_depth);
            self.render_node(
                cam,
                chunks,
                NodeVisit {
                    node_idx,
                    min,
                    size: half_size,
                    crowded_siblings: crowding,
                    virtual_leaf,
                    virtual_depth,
                },
            );
        }
    }
}

fn project_inside_node(
    cam: &Camera,
    center: [f32; 3],
    relative_to_camera: [f32; 3],
    camera_depth: f32,
    half_size: f32,
    radius: f32,
) -> ProjectedNode {
    let safe_depth = camera_depth.max(half_size * 0.5).max(0.05);
    let inverse_depth = 1.0 / safe_depth;
    ProjectedNode {
        center,
        relative_to_camera,
        camera_depth,
        pixel_x: cam.half_w + dot(relative_to_camera, cam.right) * inverse_depth * cam.focal_px,
        pixel_y: cam.half_h - dot(relative_to_camera, cam.up) * inverse_depth * cam.focal_px,
        splat_half: half_size * cam.focal_px * inverse_depth,
        projected_radius: radius * cam.focal_px * inverse_depth,
        camera_inside: true,
    }
}

fn outside_target(px: f32, py: f32, radius: f32, width: usize, height: usize) -> bool {
    px + radius < 0.0
        || px - radius >= width as f32
        || py + radius < 0.0
        || py - radius >= height as f32
}

/// The spotlight area is exempt from the node cap so the player's focus does
/// not visibly degrade when unrelated geometry exhausts the frame budget.
fn budget_exempt(cam: &Camera, projected: &ProjectedNode) -> bool {
    if !cam.flashlight {
        return false;
    }
    let distance = projected.camera_depth.max(0.001);
    if distance >= 15.0 {
        return false;
    }
    let direction = projected
        .relative_to_camera
        .map(|component| component / distance);
    dot(direction, cam.forward) > 0.88
}

/// Chunk-level visibility: distance cap plus the same conservative screen
/// bounds used for individual nodes.
pub(super) fn chunk_visible(
    chunk: &ChunkDraw,
    cam: &Camera,
    width: usize,
    height: usize,
    max_draw_distance: f32,
) -> bool {
    let half_size = chunk.world_size * 0.5;
    let center = [
        chunk.origin[0] + half_size,
        chunk.origin[1] + half_size,
        chunk.origin[2] + half_size,
    ];
    let radius = chunk.world_size * 0.866;
    let relative_to_camera = [
        center[0] - cam.pos[0],
        center[1] - cam.pos[1],
        center[2] - cam.pos[2],
    ];

    if dot(relative_to_camera, relative_to_camera).sqrt() - radius > max_draw_distance {
        return false;
    }
    let camera_depth = dot(relative_to_camera, cam.forward);
    if camera_depth + radius <= 0.01 {
        return false;
    }
    if camera_depth - radius <= 0.0 {
        return true;
    }

    let inverse_depth = 1.0 / camera_depth;
    let pixel_x = cam.half_w + dot(relative_to_camera, cam.right) * inverse_depth * cam.focal_px;
    let pixel_y = cam.half_h - dot(relative_to_camera, cam.up) * inverse_depth * cam.focal_px;
    let projected_radius = radius * cam.focal_px / (camera_depth - radius).max(0.001);
    !outside_target(pixel_x, pixel_y, projected_radius, width, height)
}

pub(super) fn chunk_distance_sq(chunk: &ChunkDraw, position: [f32; 3]) -> f32 {
    let half = chunk.world_size * 0.5;
    let dx = chunk.origin[0] + half - position[0];
    let dz = chunk.origin[2] + half - position[2];
    dx * dx + dz * dz
}
