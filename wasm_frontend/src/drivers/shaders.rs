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
// Camera basis vectors, precomputed once per frame on the CPU so no
// per-pixel trigonometry is needed to build the ray.
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uAspect;
// tan(FOV/2); 0.767 = 75 degree default.
uniform float uFovTan;

uniform float uFaceWeightTop;
uniform float uFaceWeightBottom;
uniform float uFaceWeightX;
uniform float uFaceWeightZ;

uniform int uFlashlightEnabled;

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
    out vec4 hitColor, out vec3 hitNormal, out bool isLight, out uint hitVoxelType, out float hit_t, out uint hitLightWord
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
                float light = float(node.lightLevel & 0xFFu) / 15.0;

                hitColor = vec4(r, g, b, light);
                isLight = (node.voxelType == 4u || node.voxelType == 9u);
                hitVoxelType = node.voxelType;
                // Full packed light word: scalar (0-7), occlusion mask
                // (8-15), RGB flood-fill channels (16-27).
                hitLightWord = node.lightLevel;

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
    vec3 rd = normalize(
        uCamRight * (ndc.x * uAspect * uFovTan) +
        uCamUp * (ndc.y * uFovTan) +
        uCamForward
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
    uint finalVoxelType = 0u;
    uint finalLightWord = 0u;
    float closest_t = 1e6;
    int hit_chunk_idx = -1;

    for (int k = 0; k < numHits; k++) {
        int i = hits[k].idx;
        vec3 local_ro = ro - uChunkOrigins[i];

        vec4 col;
        vec3 norm;
        bool isL;
        uint h_voxel_type;
        float h_t;
        uint h_light_word;
        // Literate Documentation:
        // Pass safe_rd instead of the raw direction vector rd to prevent division-by-zero and
        // resulting NaN states in SVO traversal when the view direction aligns with any coordinate axis.
        if (raymarchSVO(local_ro, safe_rd, uChunkRootIndices[i], hits[k].t_min, hits[k].t_max,
                        uChunkWorldSizes[i], col, norm, isL, h_voxel_type, h_t, h_light_word)) {
            finalColor = col;
            finalNormal = norm;
            hitSolid = true;
            hitIsLight = isL;
            finalVoxelType = h_voxel_type;
            finalLightWord = h_light_word;
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
        
        if (hitIsLight) {
            fragColor = vec4(finalColor.rgb * 1.5, 1.0);
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
        float bfs_term = 0.4 + 0.8 * staticLight; // Boosted base brightness
        vec3 vSq = V * V;
        float y_val = V.y > 0.0 ? uFaceWeightTop : uFaceWeightBottom;
        float face_shading_scalar = vSq.x * uFaceWeightX + vSq.y * y_val + vSq.z * uFaceWeightZ;

        // Decode per-face occlusion from the 6-bit mask
        uint finalOcclusion = (finalLightWord >> 8u) & 0x3Fu;
        bool isOccluded = false;
        if (N.x > 0.5) { isOccluded = ((finalOcclusion & 1u) != 0u); }
        else if (N.x < -0.5) { isOccluded = ((finalOcclusion & 2u) != 0u); }
        else if (N.y > 0.5) { isOccluded = ((finalOcclusion & 4u) != 0u); }
        else if (N.y < -0.5) { isOccluded = ((finalOcclusion & 8u) != 0u); }
        else if (N.z > 0.5) { isOccluded = ((finalOcclusion & 16u) != 0u); }
        else if (N.z < -0.5) { isOccluded = ((finalOcclusion & 32u) != 0u); }

        // Colored flood-fill light: the BFS propagates three channels, so
        // the local light *chroma* comes straight from the atlas instead of
        // a binary red/white flag. Falls back to a warm white where unlit.
        vec3 lightRgb = vec3(
            float((finalLightWord >> 16u) & 0xFu),
            float((finalLightWord >> 20u) & 0xFu),
            float((finalLightWord >> 24u) & 0xFu)
        ) * (1.0 / 15.0);
        float maxChannel = max(lightRgb.r, max(lightRgb.g, lightRgb.b));
        vec3 lightTint = maxChannel > 0.001 ? lightRgb / maxChannel : vec3(1.0, 0.98, 0.90);

        float face_occlusion_factor = isOccluded ? 0.35 : 1.0;

        vec3 groundColor = vec3(0.24, 0.21, 0.16);
        vec3 skyColor = vec3(0.52, 0.48, 0.40);
        vec3 ambient = mix(groundColor, skyColor, N.y * 0.5 + 0.5) * (bfs_term * face_shading_scalar * face_occlusion_factor) * edgeAO;
        // Let strong colored light bleed subtly into the ambient term.
        ambient *= mix(vec3(1.0), lightTint, 0.5 * staticLight);
        
        // 2. Beautiful Faux-Directional Room/Ceiling Lights
        // Assume a point light is roughly at the center of the nearest 10x10 cell (3 world units)
        vec2 cell_center = floor(local_hit_p.xz / 3.0) * 3.0 + 1.5;
        vec3 light_pos = vec3(cell_center.x, 3.6, cell_center.y);
        vec3 auto_L = normalize(light_pos - local_hit_p);
        
        // Wrapped diffuse allows vertical walls to catch ambient overhead light realistically
        float wrappedNdotL = dot(N, auto_L) * 0.5 + 0.5;
        float staticDiffuse = wrappedNdotL * staticLight * 0.9;
        vec3 staticDiffuseContrib = staticDiffuse * albedo * lightTint;
        
        // Per-voxel material lookup (Fixed mappings)
        float roughness = 0.9;
        float specularStrength = 0.0;
        if (finalVoxelType == 1u) { // VOXEL_WALL
            roughness = 0.85;
            specularStrength = 0.02;
        } else if (finalVoxelType == 2u) { // VOXEL_FLOOR (carpet)
            roughness = 1.0;
            specularStrength = 0.0; // Completely diffuse
        } else if (finalVoxelType == 3u) { // VOXEL_CEILING
            roughness = 0.5;
            specularStrength = 0.15;
        } else if (finalVoxelType == 5u) { // VOXEL_RED_WALL
            roughness = 0.7;
            specularStrength = 0.08;
        } else if (finalVoxelType == 8u) { // VOXEL_TREE / PILLAR
            roughness = 0.4;
            specularStrength = 0.25;
        }

        // Blinn-Phong specular term modulated by per-voxel roughness
        float shininess = mix(256.0, 4.0, roughness);
        vec3 staticH = normalize(auto_L + V);
        float staticSpecular = pow(max(dot(N, staticH), 0.0), shininess) * staticLight * specularStrength;
        vec3 staticSpecularContrib = staticSpecular * lightTint;
        
        vec3 staticContrib = (staticDiffuseContrib + staticSpecularContrib) * edgeAO;
        
        // Total Lighting
        vec3 litColor = albedo * ambient + staticContrib;
        
        if (uFlashlightEnabled == 1) {
            // Flashlight points forward from the camera (in world space, the view direction is roughly V = -rd).
            // L is the vector from the surface point to the light.
            // Since the flashlight is attached to the camera, L is essentially V.
            vec3 flashL = V;
            
            // Attenuation based on distance (closest_t is ray length)
            float dist = closest_t;
            float atten = 1.0 / (1.0 + 0.1 * dist + 0.05 * dist * dist);
            
            // Spotlight effect: how close is the fragment to the center of the screen?
            // The camera forward vector in world space is roughly the center ray, but we don't have it directly.
            // However, we can approximate the spotlight falloff by comparing 'rd' to the center ray...
            // Actually, a simpler headlamp effect is just to use N dot V.
            float flashNdotL = max(dot(N, flashL), 0.0);
            vec3 flashDiffuse = albedo * flashNdotL * atten * 1.5;
            
            // Specular for flashlight
            vec3 flashH = normalize(flashL + V);
            float flashSpecular = pow(max(dot(N, flashH), 0.0), shininess) * atten * specularStrength * 1.5;
            
            litColor += (flashDiffuse + flashSpecular) * vec3(1.0, 0.95, 0.9);
        }
        
        // 3. Fog and Vignette
        float dist = closest_t;
        float fogFactor = exp(-0.02 * dist);

        float vignette = vUv.x * vUv.y * (1.0 - vUv.x) * (1.0 - vUv.y);
        vignette = clamp(pow(16.0 * vignette, 0.25), 0.0, 1.0);

        vec3 outColor = mix(vec3(0.0), litColor, fogFactor * vignette);
        // Interleaved-gradient-noise dither: breaks up the 8-bit banding
        // that fog/vignette gradients produce, which is especially visible
        // at the reduced internal resolutions low-spec machines render at.
        // Two fracts and a dot — far cheaper than a sin-hash.
        float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        outColor += (ign - 0.5) * (1.0 / 128.0);
        fragColor = vec4(outColor, 1.0);
    } else {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
"#;

/// Indexed-surface shaders for the default renderer. They intentionally use
/// only ordinary WebGL2 vertex/index buffers: raster depth testing replaces
/// the fullscreen SVO traversal while the SVO remains available for collision
/// and debug/reference queries.
pub const SURFACE_VERTEX_SHADER: &str = r#"#version 300 es
precision highp float;

in vec3 aPosition;
in float aNormalAxis;
in float aMaterial;
in float aLight;
in float aAo;

uniform mat4 uProjection;
uniform mat4 uView;
uniform vec3 uChunkOrigin;

out vec3 vWorldPosition;
out vec3 vNormal;
flat out float vMaterial;
flat out float vLight;
flat out float vAo;

vec3 normalForAxis(float axis) {
    if (axis < 0.5) return vec3(0.0, 1.0, 0.0);
    if (axis < 1.5) return vec3(0.0, -1.0, 0.0);
    if (axis < 2.5) return vec3(0.0, 0.0, -1.0);
    if (axis < 3.5) return vec3(0.0, 0.0, 1.0);
    if (axis < 4.5) return vec3(1.0, 0.0, 0.0);
    return vec3(-1.0, 0.0, 0.0);
}

void main() {
    vec3 world = uChunkOrigin + aPosition * (1.0 / 1024.0);
    vWorldPosition = world;
    vNormal = normalForAxis(aNormalAxis);
    vMaterial = aMaterial;
    vLight = aLight;
    vAo = aAo;
    gl_Position = uProjection * uView * vec4(world, 1.0);
}
"#;

pub const SURFACE_FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;

in vec3 vWorldPosition;
in vec3 vNormal;
flat in float vMaterial;
flat in float vLight;
flat in float vAo;

uniform vec3 uCameraPosition;
uniform int uFlashlightEnabled;

out vec4 fragColor;

vec3 materialColor(float material) {
    if (material < 1.5) return vec3(0.87, 0.80, 0.40); // wall
    if (material < 2.5) return vec3(0.60, 0.53, 0.07); // floor
    if (material < 3.5) return vec3(0.80, 0.78, 0.67); // ceiling
    if (material < 4.5) return vec3(1.00, 0.97, 0.78); // fluorescent
    if (material < 5.5) return vec3(0.53, 0.00, 0.00); // red wall
    if (material < 6.5) return vec3(0.31, 0.60, 0.24); // grass
    if (material < 7.5) return vec3(0.18, 0.42, 0.72); // water
    if (material < 8.5) return vec3(0.42, 0.29, 0.18); // tree
    return vec3(1.0, 0.16, 0.10);                      // red light
}

bool emissive(float material) {
    return abs(material - 4.0) < 0.1 || abs(material - 9.0) < 0.1;
}

void main() {
    vec3 albedo = materialColor(vMaterial);
    vec3 N = normalize(vNormal);
    vec3 toCamera = uCameraPosition - vWorldPosition;
    float distanceToCamera = length(toCamera);
    vec3 V = toCamera / max(distanceToCamera, 0.001);

    float baked = clamp(vLight / 15.0, 0.0, 1.0);
    float faceResponse = N.y > 0.5 ? 1.0 : (N.y < -0.5 ? 0.74 : 0.86);
    float ao = mix(1.0, 0.58, clamp(vAo, 0.0, 1.0));
    vec3 indirect = vec3(0.09, 0.075, 0.035)
        + vec3(1.0, 0.95, 0.72) * (0.08 + 0.92 * baked);

    float flashlight = 0.0;
    if (uFlashlightEnabled == 1) {
        // The flashlight follows the view sufficiently closely for a cheap
        // stabilizing fill light; no shadow map or extra geometry pass.
        float facing = max(dot(N, V), 0.0);
        flashlight = facing * facing * (1.0 - smoothstep(2.0, 24.0, distanceToCamera)) * 1.35;
    }

    vec3 color;
    if (emissive(vMaterial)) {
        color = albedo * (1.7 + 0.35 * baked);
    } else {
        color = albedo * (indirect * faceResponse * ao + vec3(flashlight));
    }

    float fog = exp(-distanceToCamera * 0.040);
    color = mix(vec3(0.065, 0.055, 0.025), color, fog);
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    fragColor = vec4(color, 1.0);
}
"#;
