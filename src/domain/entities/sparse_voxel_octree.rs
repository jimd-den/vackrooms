/// Represents a node inside the Sparse Voxel Octree (SVO).
/// Implements a linearized, GPU-friendly design stored in a flat arena array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvoNode {
    /// Leaf node storing actual voxel attributes
    Leaf {
        voxel_type: u8,
        color: u32,
        light_level: u8,
        face_occlusion: u8,
    },
    /// Internal parent node directing traversal to 8 children
    Internal {
        /// Base index in the arena vector where the 8 children nodes are stored contiguously.
        child_base_index: u32,
        /// Bitmask where bit `i` represents if child `i` is non-empty (1) or empty air (0).
        child_mask: u8,
    },
}

/// The Sparse Voxel Octree aggregate root entity.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVoxelOctree {
    pub root: usize,
    pub nodes: Vec<SvoNode>,
    pub depth: u32,
    pub world_size: f32,
}

impl SparseVoxelOctree {
    /// Creates a new empty Sparse Voxel Octree.
    pub fn new(depth: u32, world_size: f32) -> Self {
        let root_node = SvoNode::Leaf {
            voxel_type: 0,
            color: 0,
            light_level: 0,
            face_occlusion: 0,
        };
        Self {
            root: 0,
            nodes: vec![root_node],
            depth,
            world_size,
        }
    }

    /// Checks if the SVO is entirely empty (air).
    pub fn is_empty(&self) -> bool {
        match self.nodes[self.root] {
            SvoNode::Leaf { voxel_type, .. } => voxel_type == 0,
            SvoNode::Internal { child_mask, .. } => child_mask == 0,
        }
    }

    /// Queries the SVO for voxel attributes at integer coordinates `(x, y, z)`.
    pub fn get(&self, x: u32, y: u32, z: u32) -> Option<(u8, u32, u8, u8)> {
        let max_coord = 1 << self.depth;
        if x >= max_coord || y >= max_coord || z >= max_coord {
            return None;
        }
        self.get_recursive(self.root, x, y, z, self.depth)
    }

    fn get_recursive(
        &self,
        node_idx: usize,
        x: u32,
        y: u32,
        z: u32,
        depth: u32,
    ) -> Option<(u8, u32, u8, u8)> {
        if node_idx >= self.nodes.len() {
            return None;
        }

        match self.nodes[node_idx] {
            SvoNode::Leaf {
                voxel_type,
                color,
                light_level,
                face_occlusion,
            } => Some((voxel_type, color, light_level, face_occlusion)),
            SvoNode::Internal {
                child_base_index,
                child_mask,
            } => {
                if depth == 0 {
                    return None;
                }

                let bit = depth - 1;
                let octant_x = (x >> bit) & 1;
                let octant_y = (y >> bit) & 1;
                let octant_z = (z >> bit) & 1;
                let octant_idx = (octant_z << 2) | (octant_y << 1) | octant_x;

                // Check if the octant is active in the child mask
                if (child_mask & (1 << octant_idx)) == 0 {
                    return Some((0, 0, 0, 0)); // Empty air
                }

                let child_node_idx = child_base_index as usize + octant_idx as usize;
                let mask = (1 << bit) - 1;
                self.get_recursive(child_node_idx, x & mask, y & mask, z & mask, depth - 1)
            }
        }
    }

    /// Sets voxel attributes at coordinate `(x, y, z)`. Splits nodes and collapses uniform subtrees recursively.
    pub fn set(
        &mut self,
        x: u32,
        y: u32,
        z: u32,
        voxel_type: u8,
        color: u32,
        light_level: u8,
        face_occlusion: u8,
    ) {
        let max_coord = 1 << self.depth;
        if x >= max_coord || y >= max_coord || z >= max_coord {
            return;
        }
        self.set_recursive(
            self.root,
            x,
            y,
            z,
            self.depth,
            voxel_type,
            color,
            light_level,
            face_occlusion,
        );
    }

    fn set_recursive(
        &mut self,
        node_idx: usize,
        x: u32,
        y: u32,
        z: u32,
        depth: u32,
        v_type: u8,
        col: u32,
        ll: u8,
        fo: u8,
    ) {
        match self.nodes[node_idx] {
            SvoNode::Leaf {
                voxel_type,
                color,
                light_level,
                face_occlusion,
            } => {
                if voxel_type == v_type && color == col && light_level == ll && face_occlusion == fo
                {
                    return;
                }

                if depth == 0 {
                    self.nodes[node_idx] = SvoNode::Leaf {
                        voxel_type: v_type,
                        color: col,
                        light_level: ll,
                        face_occlusion: fo,
                    };
                    return;
                }

                // Split leaf node
                let child_base_index = self.nodes.len() as u32;
                let child_mask = if voxel_type == 0 { 0 } else { 0xFF };

                for _ in 0..8 {
                    self.nodes.push(SvoNode::Leaf {
                        voxel_type,
                        color,
                        light_level,
                        face_occlusion,
                    });
                }

                self.nodes[node_idx] = SvoNode::Internal {
                    child_base_index,
                    child_mask,
                };

                self.set_recursive(node_idx, x, y, z, depth, v_type, col, ll, fo);
            }
            SvoNode::Internal {
                child_base_index,
                mut child_mask,
            } => {
                let bit = depth - 1;
                let octant_x = (x >> bit) & 1;
                let octant_y = (y >> bit) & 1;
                let octant_z = (z >> bit) & 1;
                let octant_idx = (octant_z << 2) | (octant_y << 1) | octant_x;

                let child_node_idx = child_base_index as usize + octant_idx as usize;

                if v_type != 0 {
                    child_mask |= 1 << octant_idx;
                }

                self.nodes[node_idx] = SvoNode::Internal {
                    child_base_index,
                    child_mask,
                };

                let mask = (1 << bit) - 1;
                self.set_recursive(
                    child_node_idx,
                    x & mask,
                    y & mask,
                    z & mask,
                    depth - 1,
                    v_type,
                    col,
                    ll,
                    fo,
                );

                // Try to collapse
                self.collapse_if_uniform(node_idx);
            }
        }
    }

    fn collapse_if_uniform(&mut self, node_idx: usize) {
        if let SvoNode::Internal {
            child_base_index, ..
        } = self.nodes[node_idx]
        {
            let first_child_idx = child_base_index as usize;
            let mut uniform = true;
            let mut first_val = None;

            for i in 0..8 {
                match self.nodes[first_child_idx + i] {
                    SvoNode::Internal { .. } => {
                        uniform = false;
                        break;
                    }
                    SvoNode::Leaf {
                        voxel_type,
                        color,
                        light_level,
                        face_occlusion,
                    } => {
                        if let Some((vt, col, ll, fo)) = first_val {
                            if vt != voxel_type
                                || col != color
                                || ll != light_level
                                || fo != face_occlusion
                            {
                                uniform = false;
                                break;
                            }
                        } else {
                            first_val = Some((voxel_type, color, light_level, face_occlusion));
                        }
                    }
                }
            }

            if uniform {
                if let Some((vt, col, ll, fo)) = first_val {
                    self.nodes[node_idx] = SvoNode::Leaf {
                        voxel_type: vt,
                        color: col,
                        light_level: ll,
                        face_occlusion: fo,
                    };
                }
            }
        }
    }
}
