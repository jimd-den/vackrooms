//! GLSL ES 3.00 shaders for the fullscreen SVO raymarcher.
//!
//! Ported 1:1 from the proven raw-WebGL2 path of the original JS client.
//! Pipeline per fragment:
//!   1. Reconstruct the camera ray from NDC + yaw/pitch.
//!   2. Slab-test every resident chunk AABB, insertion-sorting hits by entry
//!      distance so the nearest chunk is marched first (early out on hit).
//!   3. March the chunk's SVO with stack-based descent + empty-space skipping:
//!      an empty leaf/octant is exited in a single step via its AABB exit
//!      plane instead of voxel-by-voxel DDA.
//!   4. Shade from the leaf's packed RGB color, BFS light level, face normal
//!      and exponential distance fog.
//!
//! The SVO node atlas lives in a 1024-texel-wide RGBA32UI texture; the texel
//! encoding is documented in the core's `OctreeGpuSerializer`.

/// Fullscreen-triangle-pair vertex shader.
pub const VERTEX_SHADER: &str = r#"#version 300 es
in vec2 position;
out vec2 vUv;
void main() {
    vUv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"#;

pub const FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;
precision highp usampler2D;

in vec2 vUv;
out vec4 fragColor;

uniform vec3 uCameraPosition;
uniform float uYaw;
uniform float uPitch;
uniform float uAspect;

uniform usampler2D uNodeTexture;
uniform int uNumChunks;
uniform vec3 uChunkOrigins[25];
uniform int uChunkRootIndices[25];
uniform float uChunkWorldSizes[25];

struct StackFrame {
    int node_idx;
    vec3 b_min;
    vec3 b_max;
};

struct ChunkHit {
    int idx;
    float t_min;
    float t_max;
};

struct DecodedNode {
    bool isLeaf;
    uint voxelType;
    uint color;
    uint lightLevel;
    uint childBaseIndex;
    uint childMask;
};

ivec2 nodeIndexToTexel(int nodeIndex, int textureWidth) {
    return ivec2(nodeIndex % textureWidth, nodeIndex / textureWidth);
}

DecodedNode decodeNode(usampler2D octreeTex, int nodeIndex, int textureWidth) {
    ivec2 texCoord = nodeIndexToTexel(nodeIndex, textureWidth);
    uvec4 node = texelFetch(octreeTex, texCoord, 0);

    DecodedNode decoded;
    decoded.isLeaf = (node.x == 1u);

    if (decoded.isLeaf) {
        decoded.voxelType = node.y;
        decoded.color = node.z;
        decoded.lightLevel = node.w;
        decoded.childBaseIndex = 0u;
        decoded.childMask = 0u;
    } else {
        decoded.voxelType = 0u;
        decoded.color = 0u;
        decoded.lightLevel = 0u;
        decoded.childBaseIndex = node.y;
        decoded.childMask = node.z;
    }
    return decoded;
}

bool raymarchSVO(
    vec3 ro, vec3 rd,
    int chunk_root_idx,
    float t_entry, float t_exit,
    float world_size,
    out vec4 hitColor, out vec3 hitNormal, out bool isLight, out float hit_t
) {
    float t = t_entry;
    vec3 p = ro + t * rd;

    StackFrame stack[9];
    int stack_ptr = 0;

    int current_node = chunk_root_idx;
    vec3 current_min = vec3(0.0);
    vec3 current_max = vec3(world_size);

    int steps = 0;
    const int MAX_STEPS = 160;

    while (steps < MAX_STEPS) {
        steps++;

        DecodedNode node = decodeNode(uNodeTexture, current_node, 1024);

        if (node.isLeaf) {
            if (node.voxelType != 0u) {
                float r = float((node.color >> 16) & 0xFFu) / 255.0;
                float g = float((node.color >> 8) & 0xFFu) / 255.0;
                float b = float(node.color & 0xFFu) / 255.0;
                float light = float(node.lightLevel) / 15.0;

                hitColor = vec4(r, g, b, light);
                isLight = (node.voxelType == 4u);

                vec3 hit_p = ro + t * rd;
                vec3 center = (current_min + current_max) * 0.5;
                vec3 size = (current_max - current_min) * 0.5;
                vec3 local_p = (hit_p - center) / size;
                vec3 abs_p = abs(local_p);

                if (abs_p.x > abs_p.y && abs_p.x > abs_p.z) {
                    hitNormal = vec3(sign(local_p.x), 0.0, 0.0);
                } else if (abs_p.y > abs_p.x && abs_p.y > abs_p.z) {
                    hitNormal = vec3(0.0, sign(local_p.y), 0.0);
                } else {
                    hitNormal = vec3(0.0, 0.0, sign(local_p.z));
                }
                hit_t = t;
                return true;
            } else {
                // Empty-space skip: jump straight to this leaf's exit plane.
                vec3 t_max_planes = (vec3(
                    rd.x > 0.0 ? current_max.x : current_min.x,
                    rd.y > 0.0 ? current_max.y : current_min.y,
                    rd.z > 0.0 ? current_max.z : current_min.z
                ) - ro) / rd;

                float t_exit_box = min(t_max_planes.x, min(t_max_planes.y, t_max_planes.z));
                t = t_exit_box;
                p = ro + t * rd;
                // Snap every tied axis past its exit plane: leaving an axis
                // exactly ON its plane stalls the march (t stops advancing).
                if (abs(t_exit_box - t_max_planes.x) < 0.0001) p.x = (rd.x > 0.0 ? current_max.x : current_min.x) + (rd.x > 0.0 ? 0.001 : -0.001);
                if (abs(t_exit_box - t_max_planes.y) < 0.0001) p.y = (rd.y > 0.0 ? current_max.y : current_min.y) + (rd.y > 0.0 ? 0.001 : -0.001);
                if (abs(t_exit_box - t_max_planes.z) < 0.0001) p.z = (rd.z > 0.0 ? current_max.z : current_min.z) + (rd.z > 0.0 ? 0.001 : -0.001);

                // Pop EVERY level the ray exited: a skip through a plane
                // shared by several ancestor boxes leaves p outside more
                // than one of them, and descending from a box that no
                // longer contains p corrupts the traversal (wall cracks).
                while (p.x < current_min.x || p.x > current_max.x ||
                       p.y < current_min.y || p.y > current_max.y ||
                       p.z < current_min.z || p.z > current_max.z) {

                    if (stack_ptr == 0) return false;
                    stack_ptr--;
                    current_node = stack[stack_ptr].node_idx;
                    current_min = stack[stack_ptr].b_min;
                    current_max = stack[stack_ptr].b_max;
                }
            }
        } else {
            vec3 center = (current_min + current_max) * 0.5;
            int ox = p.x >= center.x ? 1 : 0;
            int oy = p.y >= center.y ? 1 : 0;
            int oz = p.z >= center.z ? 1 : 0;
            int child_idx = (oz << 2) | (oy << 1) | ox;

            if ((node.childMask & (1u << uint(child_idx))) != 0u) {
                // Descend into the occupied octant.
                if (stack_ptr < 8) {
                    stack[stack_ptr] = StackFrame(current_node, current_min, current_max);
                    stack_ptr++;
                }
                current_min.x = ox == 1 ? center.x : current_min.x;
                current_max.x = ox == 1 ? current_max.x : center.x;
                current_min.y = oy == 1 ? center.y : current_min.y;
                current_max.y = oy == 1 ? current_max.y : center.y;
                current_min.z = oz == 1 ? center.z : current_min.z;
                current_max.z = oz == 1 ? current_max.z : center.z;
                current_node = int(node.childBaseIndex) + child_idx;
            } else {
                // Masked-out (air) octant: skip it in one step.
                vec3 oct_max = vec3(
                    ox == 1 ? current_max.x : center.x,
                    oy == 1 ? current_max.y : center.y,
                    oz == 1 ? current_max.z : center.z
                );
                vec3 oct_min = vec3(
                    ox == 1 ? center.x : current_min.x,
                    oy == 1 ? center.y : current_min.y,
                    oz == 1 ? center.z : current_min.z
                );
                 vec3 t_max_planes = (vec3(
                    rd.x > 0.0 ? oct_max.x : oct_min.x,
                    rd.y > 0.0 ? oct_max.y : oct_min.y,
                    rd.z > 0.0 ? oct_max.z : oct_min.z
                ) - ro) / rd;

                float t_exit_oct = min(t_max_planes.x, min(t_max_planes.y, t_max_planes.z));
                t = t_exit_oct;
                p = ro + t * rd;
                if (abs(t_exit_oct - t_max_planes.x) < 0.0001) p.x = (rd.x > 0.0 ? oct_max.x : oct_min.x) + (rd.x > 0.0 ? 0.001 : -0.001);
                if (abs(t_exit_oct - t_max_planes.y) < 0.0001) p.y = (rd.y > 0.0 ? oct_max.y : oct_min.y) + (rd.y > 0.0 ? 0.001 : -0.001);
                if (abs(t_exit_oct - t_max_planes.z) < 0.0001) p.z = (rd.z > 0.0 ? oct_max.z : oct_min.z) + (rd.z > 0.0 ? 0.001 : -0.001);

                // Same multi-level pop as the empty-leaf skip above.
                while (p.x < current_min.x || p.x > current_max.x ||
                       p.y < current_min.y || p.y > current_max.y ||
                       p.z < current_min.z || p.z > current_max.z) {

                    if (stack_ptr == 0) return false;
                    stack_ptr--;
                    current_node = stack[stack_ptr].node_idx;
                    current_min = stack[stack_ptr].b_min;
                    current_max = stack[stack_ptr].b_max;
                }
            }
        }
    }
    return false;
}

void main() {
    vec2 ndc = vUv * 2.0 - 1.0;
    // 0.767 = tan(75deg FOV / 2)
    vec3 rd_local = normalize(vec3(ndc.x * uAspect * 0.767, ndc.y * 0.767, -1.0));

    float cp = cos(uPitch);
    float sp = sin(uPitch);
    vec3 rd_pitched = vec3(
        rd_local.x,
        rd_local.y * cp - rd_local.z * sp,
        rd_local.y * sp + rd_local.z * cp
    );

    float cy = cos(uYaw);
    float sy = sin(uYaw);
    vec3 rd = vec3(
        rd_pitched.x * cy + rd_pitched.z * sy,
        rd_pitched.y,
        -rd_pitched.x * sy + rd_pitched.z * cy
    );

    vec3 ro = uCameraPosition;

    ChunkHit hits[25];
    int numHits = 0;

    vec3 safe_rd = vec3(
        abs(rd.x) < 1e-4 ? sign(rd.x) * 1e-4 : rd.x,
        abs(rd.y) < 1e-4 ? sign(rd.y) * 1e-4 : rd.y,
        abs(rd.z) < 1e-4 ? sign(rd.z) * 1e-4 : rd.z
    );
    vec3 invRd = 1.0 / safe_rd;

    // Chunk-level "coarse geometry" pass: slab-test each chunk AABB and
    // insertion-sort by entry distance so we march nearest-first.
    for (int i = 0; i < uNumChunks; i++) {
        vec3 local_ro = ro - uChunkOrigins[i];

        vec3 box_min = vec3(0.0);
        vec3 box_max = vec3(uChunkWorldSizes[i]);

        vec3 t1 = (box_min - local_ro) * invRd;
        vec3 t2 = (box_max - local_ro) * invRd;
        vec3 t_min_p = min(t1, t2);
        vec3 t_max_p = max(t1, t2);

        float t_entry = max(t_min_p.x, max(t_min_p.y, t_min_p.z));
        float t_exit = min(t_max_p.x, min(t_max_p.y, t_max_p.z));

        if (t_entry < t_exit && t_exit > 0.0) {
            float real_entry = max(t_entry, 0.0);
            int insert_pos = numHits;
            while (insert_pos > 0 && hits[insert_pos - 1].t_min > real_entry) {
                hits[insert_pos] = hits[insert_pos - 1];
                insert_pos--;
            }
            hits[insert_pos] = ChunkHit(i, real_entry, t_exit);
            numHits++;
        }
    }

    vec4 finalColor = vec4(0.0);
    vec3 finalNormal = vec3(0.0);
    bool hitSolid = false;
    bool hitIsLight = false;
    float closest_t = 1e6;
    int hit_chunk_idx = -1;

    for (int k = 0; k < numHits; k++) {
        int i = hits[k].idx;
        vec3 local_ro = ro - uChunkOrigins[i];

        vec4 col;
        vec3 norm;
        bool isL;
        float h_t;
        // Literate Documentation:
        // Pass safe_rd instead of the raw direction vector rd to prevent division-by-zero and
        // resulting NaN states in SVO traversal when the view direction aligns with any coordinate axis.
        if (raymarchSVO(local_ro, safe_rd, uChunkRootIndices[i], hits[k].t_min, hits[k].t_max,
                        uChunkWorldSizes[i], col, norm, isL, h_t)) {
            finalColor = col;
            finalNormal = norm;
            hitSolid = true;
            hitIsLight = isL;
            closest_t = h_t;
            hit_chunk_idx = i;
            break;
        }
    }

    if (hitSolid) {
        vec3 albedo = hitIsLight ? vec3(1.0) : finalColor.rgb;
        float staticLight = finalColor.a;
        
        vec3 N = finalNormal;
        vec3 V = -rd; // View direction pointing back to camera
        
        // Emissive light source: render as constant white
        if (hitIsLight) {
            fragColor = vec4(vec3(1.5), 1.0);
            return;
        }
        
        // 1. Natural Ambient & Better Shadows
        float voxel_scale = uChunkWorldSizes[hit_chunk_idx] > 20.0 ? 0.1 : 0.2;
        vec3 local_hit_p = (ro - uChunkOrigins[hit_chunk_idx]) + closest_t * rd;
        
        float distToFloor = local_hit_p.y - voxel_scale;
        float distToCeiling = (3.0 - voxel_scale) - local_hit_p.y;
        
        // Soft contact shadow AO in corners where walls meet floor/ceiling
        float edgeAO = 1.0;
        if (abs(N.y) < 0.5) { // Only apply to vertical walls
            edgeAO = smoothstep(0.0, 0.4, distToFloor) * smoothstep(0.0, 0.4, distToCeiling);
            edgeAO = mix(0.55, 1.0, edgeAO);
        }
        
        // Brighter and warmer ambient colors for a natural Backrooms look
        vec3 groundColor = vec3(0.24, 0.21, 0.16);
        vec3 skyColor = vec3(0.52, 0.48, 0.40);
        vec3 ambient = mix(groundColor, skyColor, N.y * 0.5 + 0.5) * (0.45 + 0.55 * staticLight) * edgeAO;
        
        // 2. Static Room/Ceiling Lights
        vec3 staticL = vec3(0.0, 1.0, 0.0); // Directional light from ceiling pointing down
        float staticDiffuse = max(dot(N, staticL), 0.0) * staticLight * 0.8;
        vec3 staticDiffuseContrib = staticDiffuse * albedo * vec3(1.0, 0.98, 0.90);
        
        // Specular reflections - active on walls and ceiling, but disabled on carpet floor
        float specularMask = 1.0 - smoothstep(0.3, 0.7, N.y);
        vec3 staticH = normalize(staticL + V);
        float staticSpecular = pow(max(dot(N, staticH), 0.0), 24.0) * staticLight * 0.18 * specularMask;
        vec3 staticSpecularContrib = staticSpecular * vec3(1.0, 0.98, 0.90);
        
        vec3 staticContrib = (staticDiffuseContrib + staticSpecularContrib) * edgeAO;
        
        // Total Lighting (No Flashlight)
        vec3 litColor = albedo * ambient + staticContrib;
        
        // 3. Fog and Vignette
        float dist = closest_t;
        float fogFactor = exp(-0.02 * dist);
        
        float vignette = vUv.x * vUv.y * (1.0 - vUv.x) * (1.0 - vUv.y);
        vignette = clamp(pow(16.0 * vignette, 0.25), 0.0, 1.0);
        
        fragColor = vec4(mix(vec3(0.0), litColor, fogFactor * vignette), 1.0);
    } else {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
"#;
