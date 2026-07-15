//! SVO atlas decoding and MIP filtering.
//!
//! The atlas is the GPU-serialized octree shared with every other renderer:
//! 4 `u32` per node (see the core's `OctreeGpuSerializer`). This module owns
//! the two pieces of *data preparation* the splatter needs:
//!
//! * [`decode_node`] — one node's texels → leaf payload or child pointer.
//! * [`build_mips`] — a per-node pyramid of filtered attributes (average
//!   linear albedo, max light, occupancy), computed once per atlas upload so the
//!   LOD path can draw a whole subtree as a single splat.

use std::ops::Range;

use super::decode_srgb_albedo::decode_packed_albedo;
use super::surface_geometry::SurfaceExposure;

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

/// Authored emitted-radiance scale for one emissive material.
///
/// This is material data, not a color heuristic. Keeping it next to the
/// palette predicate lets traversal carry only the compact material id while
/// the shading boundary receives the exact scalar used by fixture extraction.
pub fn emission_strength(voxel_type: u32) -> Option<f32> {
    vackrooms::domain::entities::voxel_grid::material_emission_strength(
        u8::try_from(voxel_type).ok()?,
    )
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
    /// Leaf albedo in the renderer's 0..255 reference range. Production CPU
    /// traversal uses the upload-prepared linear value instead; the encoded
    /// form remains part of the shared decoder for diagnostic renderers.
    pub color: [f32; 3],
    /// Leaf scalar diffuse-fill level, 0..15.
    pub light: f32,
    /// Quantized colored diffuse-fill channels, 0..15.
    pub baked_rgb: [u8; 3],
    /// Exact authored faces open to air, decoded from the serialized neighbor
    /// occlusion mask.
    pub exposure: SurfaceExposure,
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
        let packed_lighting = atlas[t + 3];
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
            light: (packed_lighting & 0xFF) as f32,
            baked_rgb: decode_baked_rgb(packed_lighting),
            exposure: decode_surface_exposure(packed_lighting),
        })
    } else {
        Some(DecodedNode {
            is_leaf: false,
            child_base: atlas[t + 1] as usize,
            child_mask: atlas[t + 2],
            voxel_type: 0,
            color: [0.0; 3],
            light: 0.0,
            baked_rgb: [0; 3],
            exposure: SurfaceExposure::NONE,
        })
    }
}

/// Per-node MIP-filtered attributes, rebuilt on every atlas upload.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MipNode {
    /// Occupancy-weighted linear albedo of the subtree. Decoding before the
    /// average is both physically correct and removes sRGB powers per splat.
    pub linear_albedo: [f32; 3],
    /// Occupancy-weighted colored diffuse fill (0..15). A maximum would smear
    /// one bright receiver across the entire collapsed subtree.
    pub baked_rgb: [f32; 3],
    /// Fraction of the subtree volume that is solid, 0..1.
    pub occupancy: f32,
    /// Material owning the largest occupied fraction of this subtree.
    pub dominant_voxel_type: u32,
    /// Occupied fraction represented by `dominant_voxel_type`.
    dominant_coverage: f32,
    /// Fraction of the subtree occupied by emissive material.
    pub emissive_coverage: f32,
    /// Emitted radiance averaged over the whole subtree volume. Ordinary LOD
    /// refuses to collapse emissive subtrees; this keeps a bounded hard-cap
    /// fallback visible instead of silently deleting a panel.
    pub emitted_radiance_density: [f32; 3],
    /// Directional exposed surface area prepared from exact leaf masks.
    pub exposed_area: SurfaceExposure,
}

fn decode_baked_rgb(packed_lighting: u32) -> [u8; 3] {
    [
        ((packed_lighting >> 16) & 0xF) as u8,
        ((packed_lighting >> 20) & 0xF) as u8,
        ((packed_lighting >> 24) & 0xF) as u8,
    ]
}

fn decode_surface_exposure(packed_lighting: u32) -> SurfaceExposure {
    SurfaceExposure::from_neighbor_occlusion(((packed_lighting >> 8) & 0x3F) as u8)
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
        compute_mip(atlas, i, &mut mips, &mut done, 0);
    }
    mips
}

/// Rebuilds one independent pooled-slot range without touching other MIPs.
///
/// `done_scratch` is owned by the renderer and reused between stream updates;
/// its size is proportional to the changed slot, never the complete atlas.
pub(crate) fn rebuild_mips_in_node_range(
    atlas: &[u32],
    mips: &mut [MipNode],
    node_range: Range<usize>,
    done_scratch: &mut Vec<bool>,
) -> bool {
    let node_count = atlas.len() / 4;
    if atlas.len() % 4 != 0
        || mips.len() != node_count
        || node_range.is_empty()
        || node_range.end > node_count
    {
        return false;
    }

    mips[node_range.clone()].fill(MipNode::default());
    done_scratch.clear();
    done_scratch.resize(node_range.len(), false);
    for node_index in node_range.clone() {
        compute_mip(atlas, node_index, mips, done_scratch, node_range.start);
    }
    true
}

fn compute_mip(atlas: &[u32], i: usize, mips: &mut [MipNode], done: &mut [bool], done_base: usize) {
    let Some(done_index) = i.checked_sub(done_base).filter(|&index| index < done.len()) else {
        return;
    };
    if done[done_index] {
        return;
    }
    // Marked before descending: a malformed cycle degrades to a zero mip
    // instead of infinite recursion.
    done[done_index] = true;

    let t = i * 4;
    if atlas[t] == 1 {
        // Leaf.
        let voxel_type = atlas[t + 1];
        if voxel_type != VOXEL_AIR {
            let c = atlas[t + 2];
            let packed_lighting = atlas[t + 3];
            let linear_albedo = decode_packed_albedo(c);
            let strength = emission_strength(voxel_type);
            mips[i] = MipNode {
                linear_albedo,
                baked_rgb: decode_baked_rgb(packed_lighting).map(f32::from),
                occupancy: 1.0,
                dominant_voxel_type: voxel_type,
                dominant_coverage: 1.0,
                emissive_coverage: if strength.is_some() { 1.0 } else { 0.0 },
                emitted_radiance_density: strength.map_or([0.0; 3], |scale| {
                    linear_albedo.map(|channel| channel * scale)
                }),
                exposed_area: decode_surface_exposure(packed_lighting),
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
            compute_mip(atlas, idx, mips, done, done_base);
            let m = mips[idx];
            let w = m.occupancy / 8.0;
            acc.linear_albedo[0] += m.linear_albedo[0] * w;
            acc.linear_albedo[1] += m.linear_albedo[1] * w;
            acc.linear_albedo[2] += m.linear_albedo[2] * w;
            acc.baked_rgb[0] += m.baked_rgb[0] * w;
            acc.baked_rgb[1] += m.baked_rgb[1] * w;
            acc.baked_rgb[2] += m.baked_rgb[2] * w;
            acc.emissive_coverage += m.emissive_coverage / 8.0;
            acc.emitted_radiance_density[0] += m.emitted_radiance_density[0] / 8.0;
            acc.emitted_radiance_density[1] += m.emitted_radiance_density[1] / 8.0;
            acc.emitted_radiance_density[2] += m.emitted_radiance_density[2] / 8.0;
            acc.exposed_area
                .add_saturating(m.exposed_area.scaled_to_parent());
            let dominant_coverage = m.dominant_coverage / 8.0;
            if dominant_coverage > acc.dominant_coverage {
                acc.dominant_coverage = dominant_coverage;
                acc.dominant_voxel_type = m.dominant_voxel_type;
            }
            acc.occupancy += w;
        }
        if acc.occupancy > 0.0 {
            let inv = 1.0 / acc.occupancy;
            acc.linear_albedo[0] *= inv;
            acc.linear_albedo[1] *= inv;
            acc.linear_albedo[2] *= inv;
            acc.baked_rgb[0] *= inv;
            acc.baked_rgb[1] *= inv;
            acc.baked_rgb[2] *= inv;
        }
        mips[i] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::{emission_strength, is_emissive};
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
        assert_eq!(emission_strength(VOXEL_LIGHT as u32), Some(10.0));
        assert_eq!(emission_strength(VOXEL_RED_LIGHT as u32), Some(8.0));
        assert_eq!(emission_strength(VOXEL_GLIMMER as u32), Some(0.9));
        assert_eq!(emission_strength(VOXEL_AIR as u32), None);
    }
}
