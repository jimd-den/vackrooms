//! SVO atlas decoding and MIP filtering.
//!
//! The atlas is the GPU-serialized octree shared with every other renderer:
//! 4 `u32` per node (see the core's `OctreeGpuSerializer`). This module owns
//! the two pieces of *data preparation* the splatter needs:
//!
//! * [`decode_node`] — one node's texels → leaf payload or child pointer.
//! * [`build_mips`] — a per-node pyramid of filtered attributes (average
//!   color, max light, occupancy), computed once per atlas upload so the
//!   LOD path can draw a whole subtree as a single splat.

/// Voxel type constants shared with the core serializer.
pub const VOXEL_AIR: u32 = 0;

/// Single CPU-renderer predicate mirroring the domain's emissive palette.
/// Keeping this beside atlas decoding prevents traversal and shading from
/// growing renderer-specific material lists.
pub fn is_emissive(voxel_type: u32) -> bool {
    u8::try_from(voxel_type).ok().is_some_and(|material| {
        vackrooms::domain::entities::voxel_grid::EMISSIVE_MATERIALS.contains(&material)
    })
}

/// A decoded atlas node. Exactly one of the two shapes is meaningful:
/// leaves carry material/color/light, while interiors carry
/// `child_base`/`child_mask`.
#[derive(Debug, Clone, Copy)]
pub struct DecodedNode {
    pub is_leaf: bool,
    pub child_base: usize,
    pub child_mask: u32,
    pub voxel_type: u32,
    /// Leaf albedo in the renderer's 0..255 working range.
    pub color: [f32; 3],
    /// Leaf scalar diffuse-fill level, 0..15.
    pub light: f32,
}

/// Reads node `node_idx` out of the texel stream. Returns `None` when the
/// index runs past the atlas (defensive: a truncated upload must degrade to
/// missing geometry, not out-of-bounds panics).
pub fn decode_node(atlas: &[u32], node_idx: usize) -> Option<DecodedNode> {
    let t = node_idx * 4;
    if t + 3 >= atlas.len() {
        return None;
    }
    let is_leaf = atlas[t] == 1;
    if is_leaf {
        let packed_color = atlas[t + 2];
        Some(DecodedNode {
            is_leaf: true,
            child_base: 0,
            child_mask: 0,
            voxel_type: atlas[t + 1],
            color: [
                ((packed_color >> 16) & 0xFF) as f32,
                ((packed_color >> 8) & 0xFF) as f32,
                (packed_color & 0xFF) as f32,
            ],
            light: (atlas[t + 3] & 0xFF) as f32,
        })
    } else {
        Some(DecodedNode {
            is_leaf: false,
            child_base: atlas[t + 1] as usize,
            child_mask: atlas[t + 2],
            voxel_type: 0,
            color: [0.0; 3],
            light: 0.0,
        })
    }
}

/// Per-node MIP-filtered attributes, rebuilt on every atlas upload.
#[derive(Debug, Clone, Copy, Default)]
pub struct MipNode {
    /// Occupancy-weighted average RGB of the subtree, 0..255 per channel.
    pub color: [f32; 3],
    /// Max diffuse-fill level in the subtree (0..15).
    pub light: f32,
    /// Fraction of the subtree volume that is solid, 0..1.
    pub occupancy: f32,
}

/// Bottom-up MIP filtering over the whole atlas. Node order in the arena is
/// not guaranteed (`SparseVoxelOctree::set` appends children after parents,
/// `BuildOctreeUseCase` pushes the root last), so each subtree is resolved by
/// memoized recursion instead of a single directional sweep. Still O(n): every
/// node is computed exactly once.
pub fn build_mips(atlas: &[u32]) -> Vec<MipNode> {
    let node_count = atlas.len() / 4;
    let mut mips = vec![MipNode::default(); node_count];
    let mut done = vec![false; node_count];
    for i in 0..node_count {
        compute_mip(atlas, i, &mut mips, &mut done);
    }
    mips
}

fn compute_mip(atlas: &[u32], i: usize, mips: &mut [MipNode], done: &mut [bool]) {
    if done[i] {
        return;
    }
    // Marked before descending: a malformed cycle degrades to a zero mip
    // instead of infinite recursion.
    done[i] = true;

    let t = i * 4;
    if atlas[t] == 1 {
        // Leaf.
        let voxel_type = atlas[t + 1];
        if voxel_type != VOXEL_AIR {
            let c = atlas[t + 2];
            mips[i] = MipNode {
                color: [
                    ((c >> 16) & 0xFF) as f32,
                    ((c >> 8) & 0xFF) as f32,
                    (c & 0xFF) as f32,
                ],
                // Bits 0-7 are the scalar level; occlusion and the RGB
                // channels live in the higher bits (see OctreeGpuSerializer).
                light: (atlas[t + 3] & 0xFF) as f32,
                occupancy: 1.0,
            };
        }
    } else {
        let child_base = atlas[t + 1] as usize;
        let child_mask = atlas[t + 2];
        let mut acc = MipNode::default();
        for child in 0..8usize {
            if child_mask & (1 << child) == 0 {
                continue;
            }
            let idx = child_base + child;
            if idx >= mips.len() {
                continue;
            }
            compute_mip(atlas, idx, mips, done);
            let m = mips[idx];
            let w = m.occupancy / 8.0;
            acc.color[0] += m.color[0] * w;
            acc.color[1] += m.color[1] * w;
            acc.color[2] += m.color[2] * w;
            acc.light = acc.light.max(m.light);
            acc.occupancy += w;
        }
        if acc.occupancy > 0.0 {
            let inv = 1.0 / acc.occupancy;
            acc.color[0] *= inv;
            acc.color[1] *= inv;
            acc.color[2] *= inv;
        }
        mips[i] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::is_emissive;
    use vackrooms::domain::entities::voxel_grid::{
        VOXEL_AIR, VOXEL_GLIMMER, VOXEL_LIGHT, VOXEL_RED_LIGHT,
    };

    #[test]
    fn emissive_palette_matches_the_domain() {
        assert!(is_emissive(VOXEL_LIGHT as u32));
        assert!(is_emissive(VOXEL_RED_LIGHT as u32));
        assert!(is_emissive(VOXEL_GLIMMER as u32));
        assert!(!is_emissive(VOXEL_AIR as u32));
        assert!(!is_emissive(u32::MAX));
    }
}
