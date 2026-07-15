//! Construct occupied child visits in the configured traversal order.

use super::super::super::surface_geometry::exposure_for_octant;
use super::{LeafPayload, NodeVisit};

#[derive(Clone, Copy)]
pub(super) enum VoxelChildren {
    Atlas { base: usize, mask: u32 },
    Virtual(LeafPayload),
}

impl VoxelChildren {
    fn occupied_mask(self) -> u32 {
        match self {
            Self::Atlas { mask, .. } => mask,
            Self::Virtual(_) => 0xFF,
        }
    }

    fn atlas_or_virtual_child(
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

/// Occupied child visits in near-to-far order, or plain atlas order when the
/// optimization is disabled. Empty slots let the caller use one stack-free,
/// fixed-size result without allocation.
pub(super) fn voxel_child_visits(
    parent: NodeVisit,
    children: VoxelChildren,
    crowded_siblings: u32,
    camera_position: [f32; 3],
    front_to_back: bool,
) -> [Option<NodeVisit>; 8] {
    let half_size = parent.size * 0.5;
    let parent_center = [
        parent.min[0] + half_size,
        parent.min[1] + half_size,
        parent.min[2] + half_size,
    ];
    let camera_octant = (camera_position[0] >= parent_center[0]) as u32
        | (((camera_position[1] >= parent_center[1]) as u32) << 1)
        | (((camera_position[2] >= parent_center[2]) as u32) << 2);

    // XOR distances grouped by popcount: camera octant first, opposite last.
    const NEAR_TO_FAR_XOR: [u32; 8] = [0, 1, 2, 4, 3, 5, 6, 7];
    let occupied_mask = children.occupied_mask();
    let mut visits = [None; 8];
    for (index, relative_octant) in NEAR_TO_FAR_XOR.into_iter().enumerate() {
        let octant = if front_to_back {
            camera_octant ^ relative_octant
        } else {
            index as u32
        };
        if occupied_mask & (1 << octant) == 0 {
            continue;
        }

        let min = [
            parent.min[0] + (octant & 1) as f32 * half_size,
            parent.min[1] + ((octant >> 1) & 1) as f32 * half_size,
            parent.min[2] + ((octant >> 2) & 1) as f32 * half_size,
        ];
        let (node_idx, virtual_leaf, virtual_depth) =
            children.atlas_or_virtual_child(octant, parent.virtual_depth);
        visits[index] = Some(NodeVisit {
            node_idx,
            min,
            size: half_size,
            crowded_siblings,
            virtual_leaf,
            virtual_depth,
            exposure: exposure_for_octant(parent.exposure, occupied_mask, octant),
        });
    }
    visits
}
