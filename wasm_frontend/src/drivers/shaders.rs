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

layout(location = 0) in vec3 aPosition;
in float aNormalAxis;
in float aMaterial;
in float aStaticIndirect;
in float aAo;

uniform mat4 uProjection;
uniform mat4 uView;
uniform vec3 uChunkOrigin;

out vec3 vWorldPosition;
out vec3 vNormal;
flat out float vMaterial;
flat out float vStaticIndirect;
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
    vStaticIndirect = aStaticIndirect;
    vAo = aAo;
    gl_Position = uProjection * uView * vec4(world, 1.0);
}
"#;

pub const SURFACE_FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;
precision highp sampler3D;

in vec3 vWorldPosition;
in vec3 vNormal;
flat in float vMaterial;
flat in float vStaticIndirect;
flat in float vAo;

uniform vec3 uCameraPosition;
uniform int uFlashlightEnabled;
uniform sampler3D uLightVolume;
uniform vec3 uChunkOrigin;
uniform vec3 uChunkSize;

uniform int uLightCount;
uniform vec3 uLightPositions[4];
uniform vec3 uLightColors[4];
uniform vec4 uLightParams[4];

uniform sampler2D uShadowMap;
uniform mat4 uLightViewProjection;
uniform int uShadowedLightIndex;

// Dropped-flare cores: xyz = world position, w = intensity (pre-flickered).
uniform int uCoreCount;
uniform vec4 uCores[4];
uniform vec3 uCoreColors[4];

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
    if (material < 9.5) return vec3(1.00, 0.27, 0.20);  // red light
    if (material < 10.5) return vec3(0.85, 0.82, 0.75); // pale arch wall
    if (material < 11.5) return vec3(0.54, 0.50, 0.36); // rough damaged wall
    if (material < 12.5) return vec3(0.76, 0.72, 0.42); // dry shallow carpet
    if (material < 13.5) return vec3(0.42, 0.37, 0.13); // deep wet carpet
    if (material < 14.5) return vec3(0.48, 0.29, 0.15); // sticky red carpet
    if (material < 15.5) return vec3(0.18, 0.16, 0.13); // dark pooled fluid
    return vec3(0.62, 0.77, 0.91);                      // cool glimmer
}

float pcf4(sampler2D shadowMap, vec2 uv, float compareDepth) {
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    float shadow = 0.0;
    
    vec2 offsets[4] = vec2[](
        vec2(-0.5, -0.5), vec2(0.5, -0.5),
        vec2(-0.5,  0.5), vec2(0.5,  0.5)
    );
    
    for(int i = 0; i < 4; i++) {
        float pcfDepth = texture(shadowMap, uv + offsets[i] * texelSize).r;
        shadow += (compareDepth > pcfDepth) ? 0.0 : 1.0;
    }
    return shadow / 4.0;
}

float ign(vec2 p) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(p, magic.xy)));
}

float shadowVisibility(vec3 worldPos, vec3 normal, vec3 lightDir, vec2 halfSize) {
    // Interleaved gradient noise for area sampling
    float noise = ign(gl_FragCoord.xy) * 6.283185;
    vec2 offset = vec2(cos(noise), sin(noise)) * halfSize;
    
    // Simulate moving the light by moving the receiver in the opposite direction
    vec3 jitteredPos = worldPos + vec3(-offset.x, 0.0, -offset.y);
    
    vec4 lightClip = uLightViewProjection * vec4(jitteredPos + normal * 0.025, 1.0);
    vec3 p = lightClip.xyz / lightClip.w;
    vec2 uv = p.xy * 0.5 + 0.5;
    float receiverDepth = p.z * 0.5 + 0.5;

    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || receiverDepth >= 1.0) {
        return 1.0;
    }

    float bias = max(0.003 * (1.0 - dot(normal, lightDir)), 0.0008);
    return pcf4(uShadowMap, uv, receiverDepth - bias);
}

bool emissive(float material) {
    return abs(material - 4.0) < 0.1 || abs(material - 9.0) < 0.1
        || abs(material - 16.0) < 0.1;
}

float hash3D(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

float getOrderedDither() {
    int x = int(gl_FragCoord.x) % 4;
    int y = int(gl_FragCoord.y) % 4;
    int index = x + y * 4;
    float threshold = 0.0;
    if (index == 0) threshold = 0.0625;
    else if (index == 1) threshold = 0.5625;
    else if (index == 2) threshold = 0.1875;
    else if (index == 3) threshold = 0.6875;
    else if (index == 4) threshold = 0.8125;
    else if (index == 5) threshold = 0.3125;
    else if (index == 6) threshold = 0.9375;
    else if (index == 7) threshold = 0.4375;
    else if (index == 8) threshold = 0.25;
    else if (index == 9) threshold = 0.75;
    else if (index == 10) threshold = 0.125;
    else if (index == 11) threshold = 0.625;
    else if (index == 12) threshold = 0.875;
    else if (index == 13) threshold = 0.375;
    else if (index == 14) threshold = 0.95;
    else if (index == 15) threshold = 0.45;
    return threshold - 0.5;
}

vec3 quantize5(vec3 lightColor, float dither) {
    vec3 bands = lightColor * 4.0;
    bands.r = floor(bands.r + dither * 0.4 + 0.5) / 4.0;
    bands.g = floor(bands.g + dither * 0.4 + 0.5) / 4.0;
    bands.b = floor(bands.b + dither * 0.4 + 0.5) / 4.0;
    return max(bands, 0.0);
}

float getGrime(vec3 pos) {
    float n = sin(pos.x * 0.15) * cos(pos.z * 0.15) * sin(pos.y * 0.3);
    n += 0.35 * sin(pos.x * 0.8 + pos.y * 0.5) * cos(pos.z * 0.8);
    return mix(0.85, 1.0, clamp(n * 0.5 + 0.5, 0.0, 1.0));
}

void main() {
    vec3 albedo = materialColor(vMaterial);
    vec3 N = normalize(vNormal);
    vec3 toCamera = uCameraPosition - vWorldPosition;
    float distanceToCamera = length(toCamera);
    vec3 V = toCamera / max(distanceToCamera, 0.001);

    vec3 uvw = (vWorldPosition - uChunkOrigin) / uChunkSize;
    vec3 irradiance = texture(uLightVolume, clamp(uvw, 0.0, 1.0)).rgb;
    irradiance = max(irradiance, vec3(0.16, 0.13, 0.07));
    irradiance = mix(vec3(0.18, 0.15, 0.08), irradiance, 0.25);    
    
    // Upward-facing surfaces brightest, downward surfaces notably darker, side walls vary subtly by cardinal direction
    float faceResponse = 0.8;
    if (N.y > 0.5) {
        faceResponse = 1.0;
    } else if (N.y < -0.5) {
        faceResponse = 0.35;
    } else {
        faceResponse = 0.65 + 0.1 * N.x + 0.05 * N.z;
    }
    
    // Strengthen voxel AO, but clamp so it is never pure black
    float ao = mix(1.0, 0.22, clamp(vAo * 1.3, 0.0, 1.0));
    
    float roughness = 0.8;
    if (vMaterial > 1.5 && vMaterial < 2.5) {
        roughness = 0.4; // Pillars
    } else if (vMaterial > 0.5 && vMaterial < 1.5) {
        roughness = 0.6; // Floor
    }

    // Yellow-green ambient and olive shadows
    vec3 ambientUp = vec3(0.35, 0.38, 0.22) * irradiance;
    vec3 ambientDown = vec3(0.14, 0.15, 0.08) * irradiance;
    vec3 ambient = mix(ambientDown, ambientUp, N.y * 0.5 + 0.5) * ao * faceResponse;

    float flashlight = 0.0;
    if (uFlashlightEnabled == 1) {
        float facing = max(dot(N, V), 0.0);
        flashlight = facing * facing * (1.0 - smoothstep(2.0, 24.0, distanceToCamera)) * 1.5;
    }

    vec3 color;
    if (emissive(vMaterial)) {
        // Restrained emissive fixtures
        color = albedo * 1.5;
    } else {
        vec3 directDiffuse = vec3(0.0);
        vec3 directSpecular = vec3(0.0);
        
        for (int i = 0; i < 4; ++i) {
            if (i >= uLightCount) break;
            
            vec3 toLight = uLightPositions[i] - vWorldPosition;
            float d2 = dot(toLight, toLight);
            float range = uLightParams[i].x;
            float range2 = range * range;

            if (d2 >= range2) continue;

            float d = sqrt(d2);
            vec3 L = toLight / max(d, 0.001);
            
            float ndotl = max(dot(N, L), 0.0);
            
            // Specular (Blinn-Phong)
            vec3 H = normalize(L + V);
            float ndoth = max(dot(N, H), 0.0);
            float specPower = exp2(10.0 * (1.0 - roughness) + 1.0);
            float specular = pow(ndoth, specPower) * (1.0 - roughness) * 0.5;

            // Inverse square falloff
            float falloff = pow(max(1.0 - d2 / range2, 0.0), 2.0) / (1.0 + 0.05 * d2);

            float visible = 1.0;
            if (i == uShadowedLightIndex) {
                visible = shadowVisibility(vWorldPosition, N, L, uLightParams[i].zw);
            }

            // Subtle deterministic world-space fluorescent flicker/noise to direct light only
            float lightHash = hash3D(uLightPositions[i] * 10.0);
            float eyeHum = hash3D(vWorldPosition + uCameraPosition * 0.01);
            float directMod = (1.0 + 0.05 * (lightHash - 0.5)) * (1.0 + 0.02 * (eyeHum - 0.5));

            // Cream highlights under fixtures
            vec3 lightColor = mix(uLightColors[i], vec3(1.0, 0.96, 0.85), 0.3);

            directDiffuse += lightColor * uLightParams[i].y * falloff * visible * ndotl * directMod;
            directSpecular += lightColor * uLightParams[i].y * falloff * visible * specular * directMod;
        }

        // Quantize indirect (ambient) + direct diffuse into 5 bands with ordered dithering
        float ditherVal = getOrderedDither();
        vec3 totalDiffuseLight = quantize5(ambient + directDiffuse, ditherVal);

        // Low-frequency grime/stain modulation on albedo
        float grime = getGrime(vWorldPosition);
        vec3 modulatedAlbedo = albedo * grime;

        color = totalDiffuseLight * modulatedAlbedo + directSpecular + modulatedAlbedo * vec3(flashlight);
    }

    // Fog with onset distance - pushed further back for large open spaces like the Atrium
    float fogOnset = 15.0;
    float fogDist = max(0.0, distanceToCamera - fogOnset);
    float fogDensity = 0.028;
    float heightFactor = 1.0 + 0.35 * smoothstep(0.0, 3.4, vWorldPosition.y);
    
    // Ensure fog reaches exactly 1.0 before the chunk ungenerated boundary
    float fogAmount = 1.0 - exp(-fogDist * fogDensity * heightFactor);
    fogAmount = max(fogAmount, smoothstep(38.0, 50.0, distanceToCamera));

    // Volumetric scattering (glow) 
    vec3 glow = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        if (i >= uLightCount) break;
        vec3 toLight = uLightPositions[i] - uCameraPosition;
        float dl = length(toLight);
        vec3 L = toLight / max(dl, 0.001);
        
        float cosTheta = max(dot(L, V), 0.0);
        float phase = pow(cosTheta, 12.0) * 0.6 + pow(cosTheta, 4.0) * 0.15;
        
        float depthMask = smoothstep(dl - 2.0, dl + 2.0, distanceToCamera);
        float attenuation = smoothstep(45.0, 0.0, dl);
        
        glow += uLightColors[i] * uLightParams[i].y * phase * attenuation * depthMask * 2.5;
    }

    // Slight distance desaturation and contrast compression before fog, retaining bright emissive ceiling fixtures
    if (!emissive(vMaterial)) {
        float desatFactor = clamp(distanceToCamera * 0.015, 0.0, 0.55);
        float gray = dot(color, vec3(0.299, 0.587, 0.114));
        color = mix(color, vec3(gray), desatFactor);
        color = mix(color, vec3(0.22, 0.18, 0.12), desatFactor * 0.35); // pull toward a warm middle gray
    }

    // Baseline fog color
    vec3 baselineFogColor = vec3(0.15, 0.125, 0.055);
    
    // Height-based fog color: warmer/darker near the floor, sickly-green/dimmer near the ceiling
    float hNorm = clamp(vWorldPosition.y / 5.0, 0.0, 1.0);
    vec3 heightFogColor = mix(
        vec3(0.12, 0.09, 0.035), // warmer/darker near the floor
        vec3(0.14, 0.15, 0.06), // sickly-green/dimmer near the ceiling
        hNorm
    );
    
    // Seamless transition back to baseline fog color at far boundary
    float farFade = smoothstep(35.0, 50.0, distanceToCamera);
    vec3 currentFogColor = mix(heightFogColor, baselineFogColor, farFade);
    
    // Also fade the volumetric glow to 0 at the far clip so it doesn't cause a gap
    currentFogColor += glow * (1.0 - farFade);
    
    // Blend geometry color toward currentFogColor
    color = mix(color, currentFogColor, clamp(fogAmount, 0.0, 1.0));
    
    // Add a subtle ambient glow to the near-field (so it isn't completely dry and flat up close)
    color += glow * 0.12 * exp(-distanceToCamera * 0.05);

    // Flare cores: small additive glow sprites, occluded by any surface
    // nearer than the flare along this fragment's view ray — no x-ray dots.
    {
        vec3 toFrag = vWorldPosition - uCameraPosition;
        float fragDist = length(toFrag);
        vec3 rd = toFrag / max(fragDist, 1e-4);
        for (int i = 0; i < uCoreCount; i++) {
            vec3 toLight = uCores[i].xyz - uCameraPosition;
            float along = dot(toLight, rd);
            if (along <= 0.05 || along >= fragDist) continue;
            float perp = length(toLight - rd * along);
            float coreSize = 0.06 + along * 0.004;
            float core = 1.0 - smoothstep(coreSize * 0.4, coreSize, perp);
            color += uCoreColors[i] * uCores[i].w * core * 1.4;
        }
    }

    // Simple tone mapping and gamma
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    
    fragColor = vec4(color, 1.0);
}
"#;

/// Instanced face-splat shaders. One instance is one axis-aligned surface
/// rectangle (see `PackedFaceInstance`); the vertex shader rebuilds the quad
/// from center + extents + axis via `gl_VertexID` (4-vertex TRIANGLE_STRIP)
/// and evaluates ALL lighting once per face, flat — the voxel look, and the
/// reason the fragment shader shrinks to fog + grain + dither.
pub const SPLAT_VERTEX_SHADER: &str = r#"#version 300 es
precision highp float;
precision highp sampler3D;

in vec3 aFacePos;      // face center, chunk-local fixed point (1/1024 u)
in vec2 aFaceExtents;  // cells along the face's U/V axes
in vec4 aFaceMeta;     // normal_axis, material, baked_light (0-15), ao
in float aFaceFlags;   // bit 0: emissive

uniform mat4 uProjection;
uniform mat4 uView;
uniform vec3 uChunkOrigin;
uniform vec3 uChunkSize;
uniform float uVoxelScale;
uniform vec3 uCameraPosition;
uniform int uFlashlightEnabled;
uniform sampler3D uLightVolume;

uniform int uLightCount;
uniform vec3 uLightPositions[4];
uniform vec3 uLightColors[4];
uniform vec4 uLightParams[4];

uniform sampler2D uShadowMap;
uniform mat4 uLightViewProjection;
uniform int uShadowedLightIndex;
uniform int uShadowTaps;

out vec3 vWorldPos;
flat out vec3 vColor;
flat out vec3 vNormal;
flat out float vEmissive;
flat out float vGrainAmp;
flat out float vBeamPattern;

vec3 normalForAxis(float axis) {
    if (axis < 0.5) return vec3(0.0, 1.0, 0.0);
    if (axis < 1.5) return vec3(0.0, -1.0, 0.0);
    if (axis < 2.5) return vec3(0.0, 0.0, -1.0);
    if (axis < 3.5) return vec3(0.0, 0.0, 1.0);
    if (axis < 4.5) return vec3(1.0, 0.0, 0.0);
    return vec3(-1.0, 0.0, 0.0);
}

// In-plane basis with U x V = outward normal, so the TRIANGLE_STRIP corner
// order (-1,-1)(1,-1)(-1,1)(1,1) is counter-clockwise from the front.
void faceBasis(float axis, out vec3 u, out vec3 v) {
    if (axis < 0.5)      { u = vec3(1.0, 0.0, 0.0); v = vec3(0.0, 0.0, -1.0); }
    else if (axis < 1.5) { u = vec3(1.0, 0.0, 0.0); v = vec3(0.0, 0.0, 1.0); }
    else if (axis < 2.5) { u = vec3(1.0, 0.0, 0.0); v = vec3(0.0, -1.0, 0.0); }
    else if (axis < 3.5) { u = vec3(1.0, 0.0, 0.0); v = vec3(0.0, 1.0, 0.0); }
    else if (axis < 4.5) { u = vec3(0.0, 1.0, 0.0); v = vec3(0.0, 0.0, 1.0); }
    else                 { u = vec3(0.0, 1.0, 0.0); v = vec3(0.0, 0.0, -1.0); }
}

vec3 materialColor(float material) {
    if (material < 1.5) return vec3(0.87, 0.80, 0.40); // wall
    if (material < 2.5) return vec3(0.60, 0.53, 0.07); // floor
    if (material < 3.5) return vec3(0.80, 0.78, 0.67); // ceiling
    if (material < 4.5) return vec3(1.00, 0.97, 0.78); // fluorescent
    if (material < 5.5) return vec3(0.53, 0.00, 0.00); // red wall
    if (material < 6.5) return vec3(0.31, 0.60, 0.24); // grass
    if (material < 7.5) return vec3(0.18, 0.42, 0.72); // water
    if (material < 8.5) return vec3(0.42, 0.29, 0.18); // tree
    if (material < 9.5) return vec3(1.00, 0.27, 0.20);  // red light
    if (material < 10.5) return vec3(0.85, 0.82, 0.75); // pale arch wall
    if (material < 11.5) return vec3(0.54, 0.50, 0.36); // rough damaged wall
    if (material < 12.5) return vec3(0.76, 0.72, 0.42); // dry shallow carpet
    if (material < 13.5) return vec3(0.42, 0.37, 0.13); // deep wet carpet
    if (material < 14.5) return vec3(0.48, 0.29, 0.15); // sticky red carpet
    if (material < 15.5) return vec3(0.18, 0.16, 0.13); // dark pooled fluid
    return vec3(0.62, 0.77, 0.91);                      // cool glimmer
}

float hash3D(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

// Face-center shadow visibility. Vertex shaders have no derivatives or
// gl_FragCoord, so this uses a fixed symmetric tap pattern; tap count is a
// spec-profile knob (1 on Pi-class, 4 on high).
float shadowVisibility(vec3 worldPos, vec3 normal, vec3 lightDir) {
    vec4 lightClip = uLightViewProjection * vec4(worldPos + normal * 0.025, 1.0);
    vec3 p = lightClip.xyz / lightClip.w;
    vec2 uv = p.xy * 0.5 + 0.5;
    float receiverDepth = p.z * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || receiverDepth >= 1.0) {
        return 1.0;
    }
    float bias = max(0.003 * (1.0 - dot(normal, lightDir)), 0.0008);
    float compare = receiverDepth - bias;
    if (uShadowTaps <= 1) {
        float d = textureLod(uShadowMap, uv, 0.0).r;
        return (compare > d) ? 0.0 : 1.0;
    }
    vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));
    vec2 offsets[4] = vec2[](
        vec2(-0.5, -0.5), vec2(0.5, -0.5),
        vec2(-0.5,  0.5), vec2(0.5,  0.5)
    );
    float shadow = 0.0;
    for (int i = 0; i < 4; i++) {
        float d = textureLod(uShadowMap, uv + offsets[i] * texelSize, 0.0).r;
        shadow += (compare > d) ? 0.0 : 1.0;
    }
    return shadow * 0.25;
}

// Quantize the light magnitude while preserving its RGB ratio. Quantizing
// channels independently makes warm yellow light jump toward neutral grey.
vec3 quantizeLighting(vec3 lightColor) {
    float peak = max(max(lightColor.r, lightColor.g), lightColor.b);
    if (peak < 0.0001) {
        return vec3(0.08, 0.07, 0.035);
    }
    float band = max(floor(peak * 6.0 + 0.5) / 6.0, 0.08);
    return lightColor * (band / peak);
}

void main() {
    vec2 corner = vec2(
        (gl_VertexID == 1 || gl_VertexID == 3) ? 1.0 : -1.0,
        (gl_VertexID >= 2) ? 1.0 : -1.0
    );
    vec3 N = normalForAxis(aFaceMeta.x);
    vec3 U, V;
    faceBasis(aFaceMeta.x, U, V);

    vec3 center = uChunkOrigin + aFacePos * (1.0 / 1024.0);
    float halfU = aFaceExtents.x * uVoxelScale * 0.5;
    float halfV = aFaceExtents.y * uVoxelScale * 0.5;
    vec3 world = center + corner.x * halfU * U + corner.y * halfV * V;
    vWorldPos = world;
    vNormal = N;
    gl_Position = uProjection * uView * vec4(world, 1.0);

    vec3 albedo = materialColor(aFaceMeta.y);
    bool emissive = aFaceFlags > 0.5;
    vEmissive = emissive ? 1.0 : 0.0;
    // Per-voxel grain amplitude: subtle on walls/floors, none on ceilings
    // (a bright ceiling plane turns grain into salt-and-pepper noise) or
    // emissive fixtures.
    bool isCeiling = aFaceMeta.y > 2.5 && aFaceMeta.y < 3.5;
    vGrainAmp = (emissive || isCeiling) ? 0.0 : 0.02;
    // Coffer beams live in shading, not geometry: downward ceiling faces
    // get a world-locked beam-grid darkening in the fragment shader.
    vBeamPattern = (isCeiling && aFaceMeta.x > 0.5 && aFaceMeta.x < 1.5) ? 1.0 : 0.0;
    if (emissive) {
        vColor = albedo * 1.5;
        return;
    }

    vec3 toCamera = uCameraPosition - center;
    float distanceToCamera = length(toCamera);
    vec3 Vdir = toCamera / max(distanceToCamera, 0.001);

    // Sample the baked light volume half a voxel into the air so a face
    // never reads the dark interior of its own wall.
    vec3 samplePos = center + N * (0.5 * uVoxelScale);
    vec3 uvw = (samplePos - uChunkOrigin) / uChunkSize;
    vec3 irradiance = textureLod(uLightVolume, clamp(uvw, 0.0, 1.0), 0.0).rgb;
    // The compact face record carries the scalar bake too. It is a robust
    // floor at chunk borders and preserves the CPU splatter's light bands,
    // while the 3D volume supplies the actual warm/red chroma.
    float baked = aFaceMeta.z * (1.0 / 15.0);
    vec3 bakedWarm = vec3(baked, baked * 0.94, baked * 0.72);
    irradiance = max(irradiance, bakedWarm);
    irradiance = max(irradiance, vec3(0.035, 0.03, 0.015));

    float faceResponse = 0.8;
    if (N.y > 0.5) {
        faceResponse = 1.0;
    } else if (N.y < -0.5) {
        faceResponse = 0.35;
    } else {
        faceResponse = 0.65 + 0.1 * N.x + 0.05 * N.z;
    }

    float ao = mix(1.0, 0.22, clamp(aFaceMeta.w * 1.3, 0.0, 1.0));

    vec3 ambientUp = vec3(1.05, 1.0, 0.72) * irradiance;
    vec3 ambientDown = vec3(0.58, 0.52, 0.30) * irradiance;
    vec3 ambient = mix(ambientDown, ambientUp, N.y * 0.5 + 0.5) * ao * faceResponse;

    vec3 directDiffuse = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        if (i >= uLightCount) break;
        vec3 toLight = uLightPositions[i] - center;
        float d2 = dot(toLight, toLight);
        float range = uLightParams[i].x;
        float range2 = range * range;
        if (d2 >= range2) continue;

        float d = sqrt(d2);
        vec3 L = toLight / max(d, 0.001);
        float ndotl = max(dot(N, L), 0.0);
        float falloff = pow(max(1.0 - d2 / range2, 0.0), 2.0) / (1.0 + 0.05 * d2);

        float visible = 1.0;
        if (i == uShadowedLightIndex) {
            visible = shadowVisibility(center, N, L);
        }

        float lightHash = hash3D(uLightPositions[i] * 10.0);
        float directMod = 1.0 + 0.05 * (lightHash - 0.5);
        vec3 lightColor = mix(uLightColors[i], vec3(1.0, 0.96, 0.85), 0.3);
        directDiffuse += lightColor * uLightParams[i].y * falloff * visible * ndotl * directMod;
    }

    // Warm-white flashlight, kept INSIDE the light sum: the material color
    // is the immutable base and every light multiplies it, so the flashlight
    // brightens yellow walls toward brighter yellow, never neutral grey.
    vec3 flashlight = vec3(0.0);
    if (uFlashlightEnabled == 1) {
        float facing = max(dot(N, Vdir), 0.0);
        float amount = facing * facing * (1.0 - smoothstep(2.0, 24.0, distanceToCamera)) * 1.5;
        flashlight = vec3(1.0, 0.96, 0.88) * amount;
    }

    vec3 lighting = quantizeLighting(ambient + directDiffuse + flashlight);
    vColor = albedo * lighting;
}
"#;

/// The splat fragment shader is deliberately tiny (the Pi win): fog,
/// voxel-lattice albedo grain, ordered dither, tone map. All lighting
/// arrived flat from the vertex shader.
pub const SPLAT_FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;

in vec3 vWorldPos;
flat in vec3 vColor;
flat in vec3 vNormal;
flat in float vEmissive;
flat in float vGrainAmp;
flat in float vBeamPattern;

uniform vec3 uCameraPosition;
uniform float uVoxelScale;

// Dropped-flare cores: xyz = world position, w = intensity (pre-flickered).
uniform int uCoreCount;
uniform vec4 uCores[4];
uniform vec3 uCoreColors[4];

out vec4 fragColor;

float hash3D(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

float ign(vec2 p) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(p, magic.xy)));
}

void main() {
    vec3 color = vColor;
    float distanceToCamera = length(uCameraPosition - vWorldPos);

    if (vEmissive < 0.5) {
        // World-locked per-voxel albedo grain: sample half a voxel inside
        // the surface so faces lying exactly on lattice planes hash stably.
        // Amplitude is per-face (zero on ceilings) so large bright planes
        // read as solid construction, not noise.
        vec3 lattice = floor((vWorldPos - vNormal * (0.5 * uVoxelScale)) / uVoxelScale);
        float grain = 1.0 + (hash3D(lattice) - 0.5) * 2.0 * vGrainAmp;
        color *= grain;

        // Coffer beam grid on ceilings (matches the generator's
        // COFFER_PERIOD): intentional construction lines, zero geometry.
        if (vBeamPattern > 0.5) {
            bool onBeam = mod(vWorldPos.x, 2.8) < 0.22 || mod(vWorldPos.z, 2.8) < 0.22;
            if (onBeam) {
                color *= 0.86;
            }
        }

    }

    // Warm haze nearby, fading all the way to black before the draw limit.
    // The framebuffer is cleared to the same black, so a not-yet-generated
    // chunk reads as void instead of a brown rectangle.
    float fogOnset = 14.0;
    float fogDist = max(0.0, distanceToCamera - fogOnset);
    float fogDensity = 0.032;
    float heightFactor = 1.0 + 0.35 * smoothstep(0.0, 3.4, vWorldPos.y);
    float fogAmount = 1.0 - exp(-fogDist * fogDensity * heightFactor);
    fogAmount = max(fogAmount, smoothstep(34.0, 48.0, distanceToCamera));

    float hNorm = clamp(vWorldPos.y / 5.0, 0.0, 1.0);
    vec3 heightFogColor = mix(
        vec3(0.055, 0.040, 0.014),
        vec3(0.055, 0.060, 0.020),
        hNorm
    );
    float farFade = smoothstep(28.0, 48.0, distanceToCamera);
    vec3 currentFogColor = mix(heightFogColor, vec3(0.0), farFade);
    color = mix(color, currentFogColor, clamp(fogAmount, 0.0, 1.0));

    // Flare cores (see the surface shader for the occlusion rule).
    {
        vec3 toFrag = vWorldPos - uCameraPosition;
        float fragDist = length(toFrag);
        vec3 rd = toFrag / max(fragDist, 1e-4);
        for (int i = 0; i < uCoreCount; i++) {
            vec3 toLight = uCores[i].xyz - uCameraPosition;
            float along = dot(toLight, rd);
            if (along <= 0.05 || along >= fragDist) continue;
            float perp = length(toLight - rd * along);
            float coreSize = 0.06 + along * 0.004;
            float core = 1.0 - smoothstep(coreSize * 0.4, coreSize, perp);
            color += uCoreColors[i] * uCores[i].w * core * 1.4;
        }
    }

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    color += (ign(gl_FragCoord.xy) - 0.5) * (1.0 / 128.0) * (1.0 - fogAmount);

    fragColor = vec4(color, 1.0);
}
"#;

pub const SHADOW_VERTEX_SHADER: &str = r#"#version 300 es
layout(location = 0) in vec3 aPosition;
uniform mat4 uLightViewProjection;
uniform vec3 uChunkOrigin;

void main() {
    vec3 worldPosition = uChunkOrigin + aPosition * (1.0 / 1024.0);
    gl_Position = uLightViewProjection * vec4(worldPosition, 1.0);
}
"#;

pub const SHADOW_FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;
void main() {
    // Depth is automatically written to the depth buffer.
}
"#;
