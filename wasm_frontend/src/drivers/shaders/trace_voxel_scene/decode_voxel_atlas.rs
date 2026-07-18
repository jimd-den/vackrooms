//! Stateless point lookup in the packed SVO atlas.
//!
//! Restarting at the root for each step costs a few integer fetches but
//! removes the fragile mutable stack/pop state that caused cracks and phantom
//! boxes. A masked-out child returns its exact empty octant bounds without a
//! texture fetch; the optimization can then jump over that whole region.

pub const GLSL: &str = r#"
struct AtlasNode {
    bool leaf;
    uint payload;
    uint colorOrMask;
    uint lightWord;
};

struct VoxelLeaf {
    uint material;
    uint color;
    uint lightWord;
    vec3 boundsMin;
    vec3 boundsMax;
};

AtlasNode readAtlasNode(int nodeIndex) {
    ivec2 texel = ivec2(nodeIndex % 1024, nodeIndex / 1024);
    uvec4 packed = texelFetch(uNodeTexture, texel, 0);
    return AtlasNode(packed.x == 1u, packed.y, packed.z, packed.w);
}

VoxelLeaf lookupVoxelLeaf(
    vec3 point,
    int rootIndex,
    float worldSize,
    int svoDepth
) {
    int nodeIndex = rootIndex;
    vec3 boundsMin = vec3(0.0);
    vec3 boundsMax = vec3(worldSize);

    // Generator validation bounds depth; the runtime value only shortens
    // this statically bounded loop for ordinary 6/8-level chunks.
    for (int level = 0; level <= 10; ++level) {
        AtlasNode node = readAtlasNode(nodeIndex);
        if (node.leaf || level >= svoDepth) {
            return VoxelLeaf(node.payload, node.colorOrMask, node.lightWord, boundsMin, boundsMax);
        }

        vec3 center = (boundsMin + boundsMax) * 0.5;
        int childX = point.x >= center.x ? 1 : 0;
        int childY = point.y >= center.y ? 1 : 0;
        int childZ = point.z >= center.z ? 1 : 0;
        int child = (childZ << 2) | (childY << 1) | childX;

        vec3 childMin = vec3(
            childX == 1 ? center.x : boundsMin.x,
            childY == 1 ? center.y : boundsMin.y,
            childZ == 1 ? center.z : boundsMin.z
        );
        vec3 childMax = vec3(
            childX == 1 ? boundsMax.x : center.x,
            childY == 1 ? boundsMax.y : center.y,
            childZ == 1 ? boundsMax.z : center.z
        );

        if ((node.colorOrMask & (1u << uint(child))) == 0u) {
            return VoxelLeaf(0u, 0u, 0u, childMin, childMax);
        }
        nodeIndex = int(node.payload) + child;
        boundsMin = childMin;
        boundsMax = childMax;
    }
    return VoxelLeaf(0u, 0u, 0u, boundsMin, boundsMax);
}
"#;
