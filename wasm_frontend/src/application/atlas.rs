//! Merges the per-chunk GPU node arrays into one atlas texel stream.
//!
//! Every chunk's node array is already row-padded to whole 1024-texel rows by
//! the core `OctreeGpuSerializer`, so simple concatenation keeps every chunk
//! row-aligned inside the shared RGBA32UI texture. Each chunk's root index is
//! rebased by its node offset within the merged stream.

use crate::application::ports::ChunkDraw;
use crate::application::streaming::LoadedChunk;

/// The shader's fixed per-frame chunk table size (uniform array length).
pub const MAX_CHUNKS: usize = 25;

pub struct AtlasBuild {
    /// Merged RGBA32UI texel stream (4 u32 per SVO node).
    pub texels: Vec<u32>,
    /// Per-chunk draw table with atlas-rebased root indices.
    pub draws: Vec<ChunkDraw>,
}

pub fn build_atlas<'a, I: IntoIterator<Item = &'a LoadedChunk>>(chunks: I) -> AtlasBuild {
    let mut texels: Vec<u32> = Vec::new();
    let mut draws = Vec::new();

    for chunk in chunks {
        let node_offset = (texels.len() / 4) as u32;
        draws.push(ChunkDraw {
            origin: [chunk.origin.0, 0.0, chunk.origin.1],
            root_index: (node_offset + chunk.payload.root) as i32,
            world_size: chunk.payload.world_size,
        });

        // Literate Documentation:
        // Each chunk's SVO contains local child indices stored as relative pointers in `child_base_index`.
        // Since we are concatenating all chunks into a single large flat texture atlas, we must rebase
        // these pointers by adding the chunk's node_offset within the shared texture.
        // We identify internal nodes by checking if the type flag (first u32, channel R) is 0.
        // If so, we offset the child base index (second u32, channel G) by node_offset.
        let mut chunk_texels = chunk.payload.nodes.clone();
        for i in 0..(chunk_texels.len() / 4) {
            let t = i * 4;
            if chunk_texels[t] == 0 {
                chunk_texels[t + 1] += node_offset;
            }
        }
        texels.extend_from_slice(&chunk_texels);
    }

    debug_assert!(
        draws.len() <= MAX_CHUNKS,
        "chunk table overflows the shader's uniform array ({} > {MAX_CHUNKS})",
        draws.len()
    );

    AtlasBuild { texels, draws }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ports::ChunkPayload;

    fn chunk(origin: (f32, f32), root: u32, node_count: usize) -> LoadedChunk {
        LoadedChunk {
            origin,
            payload: ChunkPayload {
                root,
                nodes: vec![0; node_count * 4],
                world_size: 12.8,
                collision: vec![],
            },
        }
    }

    #[test]
    fn atlas_rebases_roots_by_cumulative_node_count() {
        let chunks = [chunk((0.0, 0.0), 0, 1024), chunk((10.0, 0.0), 3, 2048)];
        let build = build_atlas(chunks.iter());
        assert_eq!(build.draws[0].root_index, 0);
        // Second chunk starts after 1024 nodes; its local root 3 shifts along.
        assert_eq!(build.draws[1].root_index, 1027);
        assert_eq!(build.texels.len(), (1024 + 2048) * 4);
    }

    #[test]
    fn atlas_rebases_internal_nodes_child_base_indices() {
        // Create chunk 0 with 256 nodes, all internal.
        let mut chunk_0_nodes = vec![0; 256 * 4];
        // chunk 0 node 0: internal, child_base = 0
        chunk_0_nodes[0] = 0;
        chunk_0_nodes[1] = 0;

        let mut chunk_1_nodes = vec![0; 512 * 4];
        // chunk 1 node 0: internal, child_base = 8
        chunk_1_nodes[0] = 0;
        chunk_1_nodes[1] = 8;
        // chunk 1 node 1: leaf (type 1)
        chunk_1_nodes[4] = 1;
        chunk_1_nodes[5] = 42; // voxel_type

        let chunks = [
            LoadedChunk {
                origin: (0.0, 0.0),
                payload: ChunkPayload { root: 0, nodes: chunk_0_nodes, world_size: 10.0, collision: vec![] }
            },
            LoadedChunk {
                origin: (10.0, 0.0),
                payload: ChunkPayload { root: 0, nodes: chunk_1_nodes, world_size: 10.0, collision: vec![] }
            }
        ];

        let build = build_atlas(chunks.iter());

        // Chunk 0 node 0: child_base should still be 0 (since offset is 0)
        assert_eq!(build.texels[1], 0);

        // Chunk 1 node 0 (starts at texel index 256):
        // child_base was 8, should be shifted by chunk 0 size (256) -> 264
        assert_eq!(build.texels[256 * 4 + 1], 264);

        // Chunk 1 node 1 (leaf): should NOT be shifted since R == 1
        assert_eq!(build.texels[257 * 4], 1); // R
        assert_eq!(build.texels[257 * 4 + 1], 42); // G (voxel_type)
    }
}
