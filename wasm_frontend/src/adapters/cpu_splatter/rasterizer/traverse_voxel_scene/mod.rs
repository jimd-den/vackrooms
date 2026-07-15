//! Traverse the voxel scene and turn selected SVO nodes into shaded splats.
//!
//! `render_node` is deliberately a short dispatcher. Projection, atlas
//! decoding, budget fallback, leaf handling, interior handling, MIP drawing,
//! and child ordering each live in a small helper with an explicit data type.

use crate::application::ports::{ChunkDraw, LightSource};

mod project_voxel_node;
mod select_voxel_detail;
mod visit_voxel_octants;

use super::super::atlas::{VOXEL_AIR, decode_node, emission_strength};
use super::super::camera::Camera;
use super::super::shading::{FrameLighting, HeroLightVisibility};
use super::super::surface_geometry::{SurfaceExposure, project_surface_depth};
use super::SoftwareRasterizer;
use project_voxel_node::{ProjectedVoxelNode, node_is_in_flashlight_focus, project_voxel_node};
use select_voxel_detail::{should_draw_mip_splat, should_subdivide_leaf};
use visit_voxel_octants::{VoxelChildren, voxel_child_visits};

/// Attributes carried by a real atlas leaf or a virtually subdivided leaf.
#[derive(Clone, Copy)]
pub(super) struct LeafPayload {
    voxel_type: u32,
    linear_albedo: [f32; 3],
    baked_rgb: [u8; 3],
}

/// Everything needed to visit one real or virtual node.
#[derive(Clone, Copy)]
pub(super) struct NodeVisit {
    node_idx: usize,
    min: [f32; 3],
    size: f32,
    crowded_siblings: u32,
    virtual_leaf: Option<LeafPayload>,
    virtual_depth: usize,
    exposure: SurfaceExposure,
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
            exposure: SurfaceExposure::ALL_FACES,
        }
    }
}

#[derive(Clone, Copy)]
enum NodeKind {
    Leaf {
        payload: LeafPayload,
        exposure: SurfaceExposure,
    },
    Interior {
        child_base: usize,
        child_mask: u32,
    },
}

/// The exact finite-support cluster for one chunk plus the frame-global hero
/// when (and only when) that same fixture belongs to the cluster.
#[derive(Clone, Copy)]
struct ChunkLighting<'a> {
    fixtures: &'a [LightSource],
    hero: Option<&'a LightSource>,
}

impl<'a> ChunkLighting<'a> {
    fn with_global_hero(fixtures: &'a [LightSource], global_hero_id: Option<u64>) -> Self {
        let hero = global_hero_id.and_then(|id| fixtures.iter().find(|light| light.id == id));
        Self { fixtures, hero }
    }
}

impl SoftwareRasterizer {
    /// Traversal entry point kept narrow for the frame orchestrator.
    pub(super) fn traverse_voxel_chunk(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        chunk: &ChunkDraw,
        scene_lights: &[LightSource],
        global_hero_id: Option<u64>,
    ) {
        let lighting = ChunkLighting::with_global_hero(scene_lights, global_hero_id);
        self.traverse_voxel_node(cam, chunks, lighting, NodeVisit::root(chunk));
    }

    /// Projects, resolves, and dispatches one node. All branch-specific work
    /// is delegated so the recursive control flow remains readable.
    fn traverse_voxel_node(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        lighting: ChunkLighting<'_>,
        visit: NodeVisit,
    ) {
        // These are absolute frame limits, checked before projection, atlas
        // decode, lighting, or a secondary visibility ray. In particular,
        // reaching the pixel-write cap must not leave an invisible tail of
        // expensive shading work in the recursion.
        if self.frame_work_budget.writes_exhausted(self.pixel_writes)
            || self.frame_work_budget.nodes_exhausted(self.visited_nodes)
        {
            self.budget_exhausted = true;
            return;
        }

        let Some(projected) = project_voxel_node(self, cam, visit) else {
            return;
        };

        // Preserve telemetry semantics: a projected malformed atlas node is
        // counted before its failed decode, just as the original traversal.
        self.visited_nodes += 1;
        let Some(kind) = self.resolve_node(visit) else {
            return;
        };

        let at_hard_cap = self.frame_work_budget.nodes_exhausted(self.visited_nodes);
        let outside_focus_reserve = self
            .frame_work_budget
            .outside_focus_reserve(self.visited_nodes)
            && !node_is_in_flashlight_focus(cam, &projected);
        if at_hard_cap || outside_focus_reserve {
            self.budget_exhausted = true;
            self.render_budget_fallback(cam, chunks, lighting, visit, &projected, kind);
            return;
        }

        match kind {
            NodeKind::Leaf { payload, exposure } => {
                let visit = NodeVisit { exposure, ..visit };
                self.render_leaf(cam, chunks, lighting, visit, &projected, payload)
            }
            NodeKind::Interior {
                child_base,
                child_mask,
            } => {
                self.render_interior(
                    cam, chunks, lighting, visit, &projected, child_base, child_mask,
                );
            }
        }
    }

    fn resolve_node(&self, visit: NodeVisit) -> Option<NodeKind> {
        if let Some(payload) = visit.virtual_leaf {
            return Some(NodeKind::Leaf {
                payload,
                exposure: visit.exposure,
            });
        }

        let node = decode_node(&self.atlas, visit.node_idx)?;
        if node.is_leaf {
            let linear_albedo = self
                .mips
                .get(visit.node_idx)
                .map_or([0.0; 3], |mip| mip.linear_albedo);
            Some(NodeKind::Leaf {
                payload: LeafPayload {
                    voxel_type: node.voxel_type,
                    linear_albedo,
                    baked_rgb: node.baked_rgb,
                },
                exposure: node.exposure,
            })
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
        lighting: ChunkLighting<'_>,
        visit: NodeVisit,
        projected: &ProjectedVoxelNode,
        kind: NodeKind,
    ) {
        match kind {
            NodeKind::Leaf { payload, exposure } if payload.voxel_type != VOXEL_AIR => {
                let visit = NodeVisit { exposure, ..visit };
                self.draw_leaf_splat(cam, chunks, lighting, visit, projected, payload);
            }
            NodeKind::Leaf { .. } => {}
            NodeKind::Interior { .. } => {
                self.draw_mip_splat(cam, chunks, lighting, visit, projected)
            }
        }
    }

    fn render_leaf(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        lighting: ChunkLighting<'_>,
        visit: NodeVisit,
        projected: &ProjectedVoxelNode,
        payload: LeafPayload,
    ) {
        if payload.voxel_type == VOXEL_AIR {
            return;
        }
        self.max_virtual_depth_reached = self.max_virtual_depth_reached.max(visit.virtual_depth);

        let should_subdivide = should_subdivide_leaf(&self.settings, visit, projected);
        if should_subdivide {
            self.visit_children(
                cam,
                chunks,
                lighting,
                visit,
                VoxelChildren::Virtual(payload),
                visit.crowded_siblings,
            );
        } else {
            self.draw_leaf_splat(cam, chunks, lighting, visit, projected, payload);
        }
    }

    fn draw_leaf_splat(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        lighting: ChunkLighting<'_>,
        visit: NodeVisit,
        projected: &ProjectedVoxelNode,
        payload: LeafPayload,
    ) {
        let half_px = projected.footprint.enclosing_radius().max(0.85);
        let emission_strength = emission_strength(payload.voxel_type);
        let emissive = emission_strength.is_some();
        let Some(surface_depth) = project_surface_depth(
            cam,
            projected.center,
            visit.size,
            payload.voxel_type,
            emissive,
            visit.exposure,
        ) else {
            return;
        };
        if self.settings.toggles.deferred_shading
            && !self.surface_splat_may_contribute(
                projected.footprint.center[0],
                projected.footprint.center[1],
                half_px,
                surface_depth,
                emissive,
            )
        {
            return;
        }
        let hero_visibility = self.resolve_hero_light_visibility(
            chunks,
            projected.center,
            visit.size,
            emissive,
            surface_depth.representative_depth(),
            lighting.hero,
        );
        self.shade_predecoded_and_splat_with_frame_lighting(
            cam,
            chunks,
            frame_lighting(lighting.fixtures, hero_visibility),
            projected.center,
            visit.size,
            projected.footprint.center[0],
            projected.footprint.center[1],
            half_px,
            surface_depth,
            payload.linear_albedo,
            payload.baked_rgb.map(f32::from),
            emission_strength,
            payload.voxel_type,
            visit.exposure,
            visit.crowded_siblings,
        );
    }

    fn render_interior(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        lighting: ChunkLighting<'_>,
        visit: NodeVisit,
        projected: &ProjectedVoxelNode,
        child_base: usize,
        child_mask: u32,
    ) {
        // OPTIMIZATION (LOD): a subtree fitting in roughly a pixel becomes
        // one prefiltered MIP splat instead of a full descendant walk.
        let contains_emissive = self
            .mips
            .get(visit.node_idx)
            .is_some_and(|mip| mip.emissive_coverage > 0.0);
        if should_draw_mip_splat(&self.settings, projected) && !contains_emissive {
            self.draw_mip_splat(cam, chunks, lighting, visit, projected);
            return;
        }

        self.visit_children(
            cam,
            chunks,
            lighting,
            visit,
            VoxelChildren::Atlas {
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
        lighting: ChunkLighting<'_>,
        visit: NodeVisit,
        projected: &ProjectedVoxelNode,
    ) {
        let mip = self.mips.get(visit.node_idx).copied().unwrap_or_default();
        if mip.occupancy < self.settings.min_mip_occupancy {
            return;
        }
        let half_px = projected.footprint.enclosing_radius().max(0.85);
        let emissive = mip.emissive_coverage > 0.0;
        let mip_voxel_type = if emissive {
            vackrooms::domain::entities::voxel_grid::VOXEL_LIGHT as u32
        } else {
            mip.dominant_voxel_type
        };
        let Some(surface_depth) = project_surface_depth(
            cam,
            projected.center,
            visit.size,
            mip_voxel_type,
            emissive,
            mip.exposed_area,
        ) else {
            return;
        };
        if self.settings.toggles.deferred_shading
            && !self.surface_splat_may_contribute(
                projected.footprint.center[0],
                projected.footprint.center[1],
                half_px,
                surface_depth,
                emissive,
            )
        {
            return;
        }
        let hero_visibility = self.resolve_hero_light_visibility(
            chunks,
            projected.center,
            visit.size,
            emissive,
            surface_depth.representative_depth(),
            lighting.hero,
        );
        self.shade_predecoded_and_splat_with_frame_lighting(
            cam,
            chunks,
            frame_lighting(lighting.fixtures, hero_visibility),
            projected.center,
            visit.size,
            projected.footprint.center[0],
            projected.footprint.center[1],
            half_px,
            surface_depth,
            if emissive {
                mip.emitted_radiance_density
            } else {
                mip.linear_albedo
            },
            mip.baked_rgb,
            emissive.then_some(1.0),
            mip_voxel_type,
            mip.exposed_area,
            visit.crowded_siblings,
        );
    }

    /// Visits occupied octants either near-to-far or in reference index
    /// order. Virtual subdivision reuses one leaf payload at the next depth.
    fn visit_children(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        lighting: ChunkLighting<'_>,
        parent: NodeVisit,
        children: VoxelChildren,
        crowding: u32,
    ) {
        for child in voxel_child_visits(
            parent,
            children,
            crowding,
            cam.pos,
            self.settings.toggles.front_to_back,
        )
        .into_iter()
        .flatten()
        {
            self.traverse_voxel_node(cam, chunks, lighting, child);
        }
    }
}

fn frame_lighting(
    scene_lights: &[LightSource],
    hero_visibility: Option<HeroLightVisibility>,
) -> FrameLighting<'_> {
    match hero_visibility {
        Some(hero) => {
            FrameLighting::with_hero_visibility(scene_lights, hero.light_id, hero.visibility)
        }
        None => FrameLighting::unshadowed(scene_lights),
    }
}
