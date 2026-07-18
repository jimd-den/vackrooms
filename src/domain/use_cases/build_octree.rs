use crate::domain::entities::sparse_voxel_octree::{SparseVoxelOctree, SvoNode};
use crate::domain::entities::voxel_grid::{MATERIAL_COLORS, VOXEL_AIR, VoxelGrid};

const AIR_LEAF: SvoNode = SvoNode::Leaf {
    voxel_type: VOXEL_AIR,
    color: 0,
    light_rgb: [0; 3],
    face_occlusion: 0,
};

/// Use Case to convert a dense VoxelGrid into a collapsed Sparse Voxel Octree.
pub struct BuildOctreeUseCase;

impl BuildOctreeUseCase {
    pub fn new() -> Self {
        Self
    }

    /// Converts a dense VoxelGrid into an SVO of the specified depth.
    pub fn execute(&self, grid: &VoxelGrid, depth: u32, world_size: f32) -> SparseVoxelOctree {
        self.execute_internal::<true>(grid, depth, world_size)
    }

    /// Shared implementation kept generic over the pruning policy so tests can
    /// compare the optimized traversal with the former exhaustive traversal.
    fn execute_internal<const PRUNE_OUTSIDE: bool>(
        &self,
        grid: &VoxelGrid,
        depth: u32,
        world_size: f32,
    ) -> SparseVoxelOctree {
        let mut nodes = Vec::new();
        let size = 1 << depth;

        // Recursive build from root
        let root_node =
            self.build_recursive::<PRUNE_OUTSIDE>(grid, &mut nodes, 0, 0, 0, size, depth);

        // Push the root node to the end of the nodes array
        let root_idx = nodes.len();
        nodes.push(root_node);

        SparseVoxelOctree {
            root: root_idx,
            nodes,
            depth,
            world_size,
        }
    }

    fn build_recursive<const PRUNE_OUTSIDE: bool>(
        &self,
        grid: &VoxelGrid,
        nodes: &mut Vec<SvoNode>,
        x: u32,
        y: u32,
        z: u32,
        size: u32,
        depth: u32,
    ) -> SvoNode {
        // Every coordinate sampled outside the dense grid is canonical air.
        // Recursive cubes only extend in the positive direction, so when the
        // cube's minimum lies past any grid dimension the former exhaustive
        // traversal could only collapse back to this same leaf. It also could
        // not append arena nodes, which makes this short-circuit byte-for-byte
        // equivalent while avoiding all descendant calls.
        if PRUNE_OUTSIDE && self.region_is_outside_grid(grid, x, y, z) {
            return AIR_LEAF;
        }

        if depth == 0 {
            let (v_type, color, ll, fo) = self.get_voxel_attrs(grid, x, y, z);
            return SvoNode::Leaf {
                voxel_type: v_type,
                color,
                light_rgb: ll,
                face_occlusion: fo,
            };
        }

        let child_size = size / 2;
        let mut children = [AIR_LEAF; 8];

        for octant_idx in 0..8 {
            let dx = (octant_idx & 1) * child_size;
            let dy = ((octant_idx >> 1) & 1) * child_size;
            let dz = ((octant_idx >> 2) & 1) * child_size;

            children[octant_idx as usize] = self.build_recursive::<PRUNE_OUTSIDE>(
                grid,
                nodes,
                x + dx,
                y + dy,
                z + dz,
                child_size,
                depth - 1,
            );
        }

        // Check if all 8 children are uniform leaves
        let mut uniform = true;
        let mut first_val = None;
        for child in &children {
            match child {
                SvoNode::Internal { .. } => {
                    uniform = false;
                    break;
                }
                SvoNode::Leaf {
                    voxel_type,
                    color,
                    light_rgb,
                    face_occlusion,
                } => {
                    if let Some((vt, col, ll, fo)) = first_val {
                        if vt != *voxel_type
                            || col != *color
                            || ll != *light_rgb
                            || fo != *face_occlusion
                        {
                            uniform = false;
                            break;
                        }
                    } else {
                        first_val = Some((*voxel_type, *color, *light_rgb, *face_occlusion));
                    }
                }
            }
        }

        if uniform {
            if let Some((vt, col, ll, fo)) = first_val {
                return SvoNode::Leaf {
                    voxel_type: vt,
                    color: col,
                    light_rgb: ll,
                    face_occlusion: fo,
                };
            }
        }

        // Push 8 children contiguously to arena
        let child_base_index = nodes.len() as u32;
        let mut child_mask = 0;

        for octant_idx in 0..8 {
            let child = children[octant_idx as usize];
            match child {
                SvoNode::Leaf { voxel_type, .. } => {
                    if voxel_type != 0 {
                        child_mask |= 1 << octant_idx;
                    }
                }
                SvoNode::Internal { .. } => {
                    child_mask |= 1 << octant_idx;
                }
            }
            nodes.push(child);
        }

        SvoNode::Internal {
            child_base_index,
            child_mask,
        }
    }

    fn region_is_outside_grid(&self, grid: &VoxelGrid, x: u32, y: u32, z: u32) -> bool {
        x as usize >= grid.width() || y as usize >= grid.height() || z as usize >= grid.depth()
    }

    fn get_voxel_attrs(&self, grid: &VoxelGrid, x: u32, y: u32, z: u32) -> (u8, u32, [u8; 3], u8) {
        if x >= grid.width() as u32 || y >= grid.height() as u32 || z >= grid.depth() as u32 {
            return (0, 0, [0; 3], 0); // Out of bounds is air
        }

        let v_id = grid.get(x as usize, y as usize, z as usize);
        if v_id == VOXEL_AIR {
            return (0, 0, [0; 3], 0);
        }

        let ll = grid.get_light_rgb(x as usize, y as usize, z as usize);
        let fo = grid.get_face_occlusion(x as usize, y as usize, z as usize);
        let base_color = MATERIAL_COLORS
            .get(v_id as usize)
            .copied()
            .unwrap_or(0x000000);

        (v_id, base_color, ll, fo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::voxel_grid::{
        FACE_OCCLUDED_NEGATIVE_X, FACE_OCCLUDED_POSITIVE_Y, VOXEL_CEILING, VOXEL_RED_WALL,
        VOXEL_WALL,
    };

    fn traversal_call_count<const PRUNE_OUTSIDE: bool>(
        dimensions: [u32; 3],
        origin: [u32; 3],
        depth: u32,
    ) -> usize {
        if PRUNE_OUTSIDE
            && (origin[0] >= dimensions[0]
                || origin[1] >= dimensions[1]
                || origin[2] >= dimensions[2])
        {
            return 1;
        }
        if depth == 0 {
            return 1;
        }

        let child_size = 1 << (depth - 1);
        let mut calls = 1;
        for octant in 0..8u32 {
            calls += traversal_call_count::<PRUNE_OUTSIDE>(
                dimensions,
                [
                    origin[0] + (octant & 1) * child_size,
                    origin[1] + ((octant >> 1) & 1) * child_size,
                    origin[2] + ((octant >> 2) & 1) * child_size,
                ],
                depth - 1,
            );
        }
        calls
    }

    #[test]
    fn test_empty_grid_collapses_to_single_root_leaf() {
        let grid = VoxelGrid::new(4, 4, 4);
        let builder = BuildOctreeUseCase::new();
        let svo = builder.execute(&grid, 2, 4.0);

        // An entirely empty grid should collapse into a single root node (Leaf containing air)
        assert_eq!(svo.nodes.len(), 1);
        match svo.nodes[svo.root] {
            SvoNode::Leaf { voxel_type, .. } => assert_eq!(voxel_type, 0),
            _ => panic!("Expected root to be a collapsed leaf node"),
        }
    }

    #[test]
    fn test_single_voxel_splits_octree() {
        let mut grid = VoxelGrid::new(4, 4, 4);
        grid.set(0, 0, 0, VOXEL_WALL); // Set wall at origin

        let builder = BuildOctreeUseCase::new();
        let svo = builder.execute(&grid, 2, 4.0); // depth 2

        // Since there is a single wall voxel, the tree must split and cannot be collapsed completely
        assert!(svo.nodes.len() > 1);

        // The root should be an internal node
        match svo.nodes[svo.root] {
            SvoNode::Internal { child_mask, .. } => {
                assert!(child_mask > 0);
            }
            _ => panic!("Expected root to be an internal parent node"),
        }

        // Let's verify we can query the wall voxel at (0,0,0)
        let val = svo.get(0, 0, 0);
        assert!(val.is_some());
        let (v_type, color, _, _) = val.unwrap();
        assert_eq!(v_type, VOXEL_WALL);
        assert_eq!(color, 0xddcc66);
    }

    #[test]
    fn outside_pruning_preserves_non_power_of_two_tree_and_lookups() {
        let mut grid = VoxelGrid::new(5, 3, 6);
        grid.set(0, 0, 0, VOXEL_WALL);
        grid.set_light_rgb(0, 0, 0, [7, 5, 3]);
        grid.set_face_occlusion(0, 0, 0, FACE_OCCLUDED_POSITIVE_Y);
        grid.set(3, 1, 4, VOXEL_CEILING);
        grid.set_light_rgb(3, 1, 4, [2, 4, 6]);
        grid.set(4, 2, 5, VOXEL_RED_WALL);
        grid.set_face_occlusion(4, 2, 5, FACE_OCCLUDED_NEGATIVE_X);

        let builder = BuildOctreeUseCase::new();
        let optimized = builder.execute_internal::<true>(&grid, 3, 8.0);
        let exhaustive = builder.execute_internal::<false>(&grid, 3, 8.0);

        assert_eq!(
            optimized, exhaustive,
            "pruning must not renumber or alter nodes"
        );
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(optimized.get(x, y, z), exhaustive.get(x, y, z));
                }
            }
        }

        let dimensions = [
            grid.width() as u32,
            grid.height() as u32,
            grid.depth() as u32,
        ];
        let optimized_calls = traversal_call_count::<true>(dimensions, [0; 3], 3);
        let exhaustive_calls = traversal_call_count::<false>(dimensions, [0; 3], 3);
        assert_eq!(optimized_calls, 185);
        assert_eq!(exhaustive_calls, 585);
        assert!(
            optimized_calls * 3 <= exhaustive_calls,
            "expected at least a 3x traversal reduction, got {optimized_calls} optimized calls vs {exhaustive_calls} exhaustive calls"
        );
    }
}
