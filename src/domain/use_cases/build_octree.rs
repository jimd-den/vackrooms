use crate::domain::entities::sparse_voxel_octree::{SparseVoxelOctree, SvoNode};
use crate::domain::entities::voxel_grid::{
    VOXEL_AIR, VOXEL_CEILING, VOXEL_FLOOR, VOXEL_GRASS, VOXEL_LIGHT, VOXEL_RED_WALL, VOXEL_TREE,
    VOXEL_WALL, VOXEL_WATER, VOXEL_RED_LIGHT, VoxelGrid,
};

/// Use Case to convert a dense VoxelGrid into a collapsed Sparse Voxel Octree.
pub struct BuildOctreeUseCase;

impl BuildOctreeUseCase {
    pub fn new() -> Self {
        Self
    }

    /// Converts a dense VoxelGrid into an SVO of the specified depth.
    pub fn execute(&self, grid: &VoxelGrid, depth: u32, world_size: f32) -> SparseVoxelOctree {
        let mut nodes = Vec::new();
        let size = 1 << depth;

        // Recursive build from root
        let root_node = self.build_recursive(grid, &mut nodes, 0, 0, 0, size, depth);

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

    fn build_recursive(
        &self,
        grid: &VoxelGrid,
        nodes: &mut Vec<SvoNode>,
        x: u32,
        y: u32,
        z: u32,
        size: u32,
        depth: u32,
    ) -> SvoNode {
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
        let mut children = [SvoNode::Leaf {
            voxel_type: 0,
            color: 0,
            light_rgb: [0; 3],
            face_occlusion: 0,
        }; 8];

        for octant_idx in 0..8 {
            let dx = (octant_idx & 1) * child_size;
            let dy = ((octant_idx >> 1) & 1) * child_size;
            let dz = ((octant_idx >> 2) & 1) * child_size;

            children[octant_idx as usize] =
                self.build_recursive(grid, nodes, x + dx, y + dy, z + dz, child_size, depth - 1);
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
        let base_color = match v_id {
            VOXEL_WALL => 0xddcc66,
            VOXEL_FLOOR => 0x998811,
            VOXEL_CEILING => 0xcccccc,
            VOXEL_LIGHT => 0xffffff,
            VOXEL_RED_LIGHT => 0xff4444,
            VOXEL_RED_WALL => 0x880000,
            VOXEL_GRASS => 0x4f9a3d,
            VOXEL_WATER => 0x3a6fb8,
            VOXEL_TREE => 0x6b4a2f,
            _ => 0x000000,
        };

        (v_id, base_color, ll, fo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
