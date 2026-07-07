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
        texels.extend_from_slice(&chunk.payload.nodes);
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
}
