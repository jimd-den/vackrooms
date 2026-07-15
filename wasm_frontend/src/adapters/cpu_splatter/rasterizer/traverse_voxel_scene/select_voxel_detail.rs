//! Select real-node MIP detail and virtual leaf subdivision.
//!
//! These predicates are the complete LOD policy. Traversal performs the
//! chosen action but does not hide additional quality thresholds.

use super::super::super::settings::CpuRenderSettings;
use super::NodeVisit;
use super::project_voxel_node::ProjectedVoxelNode;

pub(super) fn should_subdivide_leaf(
    settings: &CpuRenderSettings,
    visit: NodeVisit,
    projected: &ProjectedVoxelNode,
) -> bool {
    projected.footprint.enclosing_radius() > settings.max_splat_radius_px
        && visit.size > settings.min_split_size
        && visit.virtual_depth < settings.max_virtual_depth as usize
}

/// OPTIMIZATION (MIP LOD): a subtree fitting in roughly one pixel becomes
/// one prefiltered splat instead of a full descendant walk.
pub(super) fn should_draw_mip_splat(
    settings: &CpuRenderSettings,
    projected: &ProjectedVoxelNode,
) -> bool {
    settings.toggles.mip_lod && projected.footprint.enclosing_radius() < settings.lod_cutoff_px
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::cpu_splatter::rasterizer::project_aabb_footprint::ScreenFootprint;
    use crate::adapters::cpu_splatter::surface_geometry::SurfaceExposure;

    fn visit(size: f32, virtual_depth: usize) -> NodeVisit {
        NodeVisit {
            node_idx: 0,
            min: [0.0; 3],
            size,
            crowded_siblings: 1,
            virtual_leaf: None,
            virtual_depth,
            exposure: SurfaceExposure::ALL_FACES,
        }
    }

    fn near_plane_projection(radius: f32) -> ProjectedVoxelNode {
        ProjectedVoxelNode {
            center: [0.0; 3],
            relative_to_camera: [0.0; 3],
            camera_depth: 0.0,
            footprint: ScreenFootprint {
                center: [64.0, 36.0],
                half_extent: [radius; 2],
                nearest_depth: -1.0,
            },
        }
    }

    #[test]
    fn camera_plane_leaf_subdivides_until_the_explicit_safety_cap() {
        let settings = CpuRenderSettings::default();
        let projected = near_plane_projection(128.0);

        assert!(should_subdivide_leaf(&settings, visit(1.0, 0), &projected));
        assert!(!should_subdivide_leaf(
            &settings,
            visit(1.0, settings.max_virtual_depth as usize),
            &projected,
        ));
    }
}
