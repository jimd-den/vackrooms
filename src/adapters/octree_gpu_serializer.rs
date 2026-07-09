use crate::domain::entities::sparse_voxel_octree::{SparseVoxelOctree, SvoNode};

/// Represents SVO flattened data formatted for direct WebGL2 texture upload.
#[derive(Debug, Clone)]
pub struct OctreeGpuData {
    pub texture_width: u32,
    pub texture_height: u32,
    pub texel_data: Vec<u32>,
}

/// GPU Adapter to flatten the Sparse Voxel Octree into a GPU-compatible texture format.
///
/// DOCUMENTED ENCODING SCHEME (RGBA32UI texture format):
/// Each SVO node occupies exactly one texel (4 x 32-bit unsigned integers: R, G, B, A).
///
/// CHANNEL LAYOUT:
/// - **R (x):** Node Type Flag
///   * `0` = Internal Parent Node
///   * `1` = Leaf Voxel Node
///
/// - **G (y):** Payload 1
///   * If Internal Node (`R == 0`): `child_base_index` (uint32) - Index in the texture
///     of the first child. The remaining 7 children are stored contiguously at indices
///     `child_base_index + 1` through `child_base_index + 7`.
///   * If Leaf Node (`R == 1`): `voxel_type` (uint32) - Voxel type ID (e.g., WALL, FLOOR, etc.).
///
/// - **B (z):** Payload 2
///   * If Internal Node (`R == 0`): `child_mask` (uint32) - 8-bit active children mask.
///     Bit `i` is `1` if child `i` is non-empty, and `0` if child `i` is empty air.
///   * If Leaf Node (`R == 1`): `color` (uint32) - 24-bit packed RGB color (`0xRRGGBB`).
///
/// - **A (w):** Payload 3
///   * If Internal Node (`R == 0`): `0` (Padding).
///   * If Leaf Node (`R == 1`): packed lighting:
///     - bits 0-7:   scalar light level, the max of the RGB channels (0-15)
///     - bits 8-15:  face occlusion mask
///     - bits 16-19: red light channel (0-15)
///     - bits 20-23: green light channel (0-15)
///     - bits 24-27: blue light channel (0-15)
pub struct OctreeGpuSerializer;

impl OctreeGpuSerializer {
    /// Serializes SVO nodes into texture-ready data, padding the last row to maintain grid alignment.
    pub fn serialize_to_gpu_data(octree: &SparseVoxelOctree) -> OctreeGpuData {
        let num_nodes = octree.nodes.len();
        let texture_width = 1024; // Standard GPU texture row size
        let total_texels = ((num_nodes + texture_width - 1) / texture_width) * texture_width;

        let mut texel_data = Vec::with_capacity(total_texels * 4);

        for node in &octree.nodes {
            match *node {
                SvoNode::Leaf {
                    voxel_type,
                    color,
                    light_rgb,
                    face_occlusion,
                } => {
                    let [r, g, b] = light_rgb;
                    let scalar = r.max(g).max(b) as u32;
                    texel_data.push(1); // R
                    texel_data.push(voxel_type as u32); // G
                    texel_data.push(color); // B
                    texel_data.push(
                        scalar
                            | (face_occlusion as u32) << 8
                            | (r as u32 & 0xF) << 16
                            | (g as u32 & 0xF) << 20
                            | (b as u32 & 0xF) << 24,
                    ); // A
                }
                SvoNode::Internal {
                    child_base_index,
                    child_mask,
                } => {
                    texel_data.push(0); // R
                    texel_data.push(child_base_index); // G
                    texel_data.push(child_mask as u32); // B
                    texel_data.push(0); // A
                }
            }
        }

        // Pad the last row with empty leaves to prevent out-of-bound shader fetches
        let padded_nodes_count = total_texels - num_nodes;
        for _ in 0..padded_nodes_count {
            texel_data.push(1); // Leaf
            texel_data.push(0); // Air
            texel_data.push(0); // Color
            texel_data.push(0); // Light Level
        }

        let texture_height = (total_texels / texture_width) as u32;

        OctreeGpuData {
            texture_width: texture_width as u32,
            texture_height,
            texel_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_serialization_padding() {
        let mut octree = SparseVoxelOctree::new(2, 4.0);
        octree.set(0, 0, 0, 1, 0xFF00FF, [15; 3], 0);

        let gpu_data = OctreeGpuSerializer::serialize_to_gpu_data(&octree);

        // Verification tests
        assert_eq!(gpu_data.texture_width, 1024);
        assert_eq!(
            gpu_data.texel_data.len(),
            (gpu_data.texture_width * gpu_data.texture_height * 4) as usize
        );

        // Index 0 maps to texel (0, 0)
        let index = 0;
        let tx = index % 1024;
        let ty = index / 1024;
        assert_eq!(tx, 0);
        assert_eq!(ty, 0);

        // Ensure child_base_index values stay within texture range
        for node in &octree.nodes {
            if let SvoNode::Internal {
                child_base_index, ..
            } = *node
            {
                let max_index = child_base_index + 7;
                assert!(max_index < (gpu_data.texture_width * gpu_data.texture_height));
            }
        }
    }
}
