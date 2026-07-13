//! Fullscreen SVO raymarcher program (`?renderer=raymarch`, debug path).
//!
//! Pipeline per fragment:
//!   1. Reconstruct the camera ray from NDC + the precomputed camera basis.
//!   2. Slab-test every resident chunk AABB, insertion-sorting hits by entry
//!      distance so the nearest chunk is marched first (early out on hit).
//!   3. March the chunk's SVO with stack-based descent + empty-space
//!      skipping: an empty leaf/octant is exited in a single step via its
//!      AABB exit plane instead of voxel-by-voxel DDA.
//!   4. Shade from the leaf's packed RGB color, BFS light word, face normal,
//!      the flashlight cone, dynamic lights, and exponential distance fog.
//!
//! The SVO node atlas lives in a 1024-texel-wide RGBA32UI texture; the texel
//! encoding is documented in the core's `OctreeGpuSerializer`.

use super::chunks;

/// Fullscreen-triangle-pair vertex shader.
pub const VERTEX_SHADER: &str = r#"#version 300 es
in vec2 position;
out vec2 vUv;
void main() {
    vUv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"#;

/// Preamble: version, precision, varyings, uniforms.
const FRAGMENT_HEADER: &str = r#"#version 300 es
precision highp float;
precision highp int;
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
// OPTIMIZATION (rt_f2b): sort chunk intersections nearest-first and stop at
// the first solid hit. Off = visit input order and compare every hit.
uniform int uFrontToBackEnabled;
// Optimization/effect toggle: banding-noise dither (see RenderToggles).
uniform int uDitherEnabled;

// Level atmosphere: outdoors (grassland) the miss/fog colors switch to a
// bright sky and the ambient response scales up. Indoors these are unused.
uniform int uOutdoor;
uniform vec3 uSkyColor;
uniform vec3 uFogColor;
uniform float uAmbientScale;

// Per-frame dynamic lights (dropped flares): xyz = position, w = radius /
// intensity. Same contract as the surface/splat/CPU paths.
uniform int uDynamicLightCount;
uniform vec4 uDynamicPosRadius[4];
uniform vec4 uDynamicColorIntensity[4];

uniform usampler2D uNodeTexture;
uniform int uNumChunks;
uniform vec3 uChunkOrigins[25];
uniform int uChunkRootIndices[25];
uniform float uChunkWorldSizes[25];
"#;

/// SVO traversal + shading body (everything after the shared chunks).
const FRAGMENT_BODY: &str = r#"
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
                // Keep the emissive material set in parity with the domain
                // palette: fluorescent (4), red light (9), glimmer (16).
                isLight = (node.voxelType == 4u || node.voxelType == 9u || node.voxelType == 16u);
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
                // TRAVERSAL: empty-space skip — jump straight to this
                // leaf's exit plane instead of stepping voxel by voxel.
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
                // TRAVERSAL: masked-out (air) octant skipped in one step.
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

    // Clamp near-zero direction components: the slab test divides by them,
    // and a NaN here corrupts the whole traversal.
    vec3 safe_rd = vec3(
        abs(rd.x) < 1e-4 ? (rd.x < 0.0 ? -1e-4 : 1e-4) : rd.x,
        abs(rd.y) < 1e-4 ? (rd.y < 0.0 ? -1e-4 : 1e-4) : rd.y,
        abs(rd.z) < 1e-4 ? (rd.z < 0.0 ? -1e-4 : 1e-4) : rd.z
    );
    vec3 invRd = 1.0 / safe_rd;

    // TRAVERSAL: chunk-level coarse pass — slab-test each chunk AABB before
    // entering its SVO.
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
            // OPTIMIZATION (rt_f2b): insertion-sort the small fixed hit list.
            // The disabled reference path retains input order.
            int insert_pos = numHits;
            if (uFrontToBackEnabled == 1) {
                while (insert_pos > 0 && hits[insert_pos - 1].t_min > real_entry) {
                    hits[insert_pos] = hits[insert_pos - 1];
                    insert_pos--;
                }
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
        if (raymarchSVO(local_ro, safe_rd, uChunkRootIndices[i], hits[k].t_min, hits[k].t_max,
                        uChunkWorldSizes[i], col, norm, isL, h_voxel_type, h_t, h_light_word)) {
            if (uFrontToBackEnabled == 1 || h_t < closest_t) {
                finalColor = col;
                finalNormal = norm;
                hitSolid = true;
                hitIsLight = isL;
                finalVoxelType = h_voxel_type;
                finalLightWord = h_light_word;
                closest_t = h_t;
                hit_chunk_idx = i;
            }
            if (uFrontToBackEnabled == 1) break;
        }
    }

    if (hitSolid) {
        vec3 albedo = hitIsLight ? vec3(1.0) : finalColor.rgb;
        float staticLight = finalColor.a;

        vec3 N = finalNormal;
        vec3 V = -rd; // View direction pointing back to camera

        vec3 litColor;
        if (hitIsLight) {
            // Emissive materials skip diffuse lighting, but still pass
            // through the common fog/vignette/dither stage below so distant
            // fixtures cannot pierce the level atmosphere.
            litColor = finalColor.rgb * 1.5;
        } else {
        // 1. Natural ambient and contact shadows.
        float voxel_scale = uChunkWorldSizes[hit_chunk_idx] > 20.0 ? 0.1 : 0.2;
        vec3 local_hit_p = (ro - uChunkOrigins[hit_chunk_idx]) + closest_t * rd;

        float distToFloor = local_hit_p.y - voxel_scale;
        float distToCeiling = (3.0 - voxel_scale) - local_hit_p.y;

        // Soft contact-shadow AO in corners where walls meet floor/ceiling.
        float edgeAO = 1.0;
        if (abs(N.y) < 0.5) { // Only apply to vertical walls
            edgeAO = smoothstep(0.0, 0.4, distToFloor) * smoothstep(0.0, 0.4, distToCeiling);
            edgeAO = mix(0.55, 1.0, edgeAO);
        }

        // Brighter, warmer ambient for a natural Backrooms look.
        float bfs_term = 0.4 + 0.8 * staticLight;
        vec3 vSq = V * V;
        float y_val = V.y > 0.0 ? uFaceWeightTop : uFaceWeightBottom;
        float face_shading_scalar = vSq.x * uFaceWeightX + vSq.y * y_val + vSq.z * uFaceWeightZ;

        // Decode per-face occlusion from the 6-bit mask.
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
        if (uOutdoor == 1) {
            // Daylight: cool bright hemisphere instead of sickly interior.
            groundColor = vec3(0.38, 0.40, 0.36);
            skyColor = vec3(0.72, 0.78, 0.86);
        }
        vec3 ambient = mix(groundColor, skyColor, N.y * 0.5 + 0.5) * (bfs_term * face_shading_scalar * face_occlusion_factor) * edgeAO * uAmbientScale;
        // Let strong colored light bleed subtly into the ambient term.
        ambient *= mix(vec3(1.0), lightTint, 0.5 * staticLight);

        // 2. Faux-directional room lights: assume a fixture at the center
        // of the nearest 3-unit ceiling cell.
        vec2 cell_center = floor(local_hit_p.xz / 3.0) * 3.0 + 1.5;
        vec3 light_pos = vec3(cell_center.x, 3.6, cell_center.y);
        vec3 auto_L = normalize(light_pos - local_hit_p);

        // Wrapped diffuse lets vertical walls catch overhead light.
        float wrappedNdotL = dot(N, auto_L) * 0.5 + 0.5;
        float staticDiffuse = wrappedNdotL * staticLight * 0.9;
        vec3 staticDiffuseContrib = staticDiffuse * albedo * lightTint;

        // Per-voxel material response (fixed mappings).
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

        // Blinn-Phong specular modulated by per-voxel roughness.
        float shininess = mix(256.0, 4.0, roughness);
        vec3 staticH = normalize(auto_L + V);
        float staticSpecular = pow(max(dot(N, staticH), 0.0), shininess) * staticLight * specularStrength;
        vec3 staticSpecularContrib = staticSpecular * lightTint;

        vec3 staticContrib = (staticDiffuseContrib + staticSpecularContrib) * edgeAO;

        litColor = albedo * ambient + staticContrib;

        // 3. Flashlight: a true spotlight cone around the camera forward
        // axis (shared spec with the CPU splatter — see SPOT_* above).
        // This replaces the old N-dot-V "headlamp" approximation that lit
        // everything facing the camera regardless of screen position.
        if (uFlashlightEnabled == 1) {
            vec3 world_hit = ro + closest_t * rd;
            litColor += albedo * spotBeam(uCameraPosition, uCamForward, world_hit, N);
        }

        // Dynamic lights (dropped flares): local point lights with quadratic
        // falloff, matching the other render paths.
        {
            vec3 world_hit = ro + closest_t * rd;
            for (int i = 0; i < 4; i++) {
                if (i >= uDynamicLightCount) break;
                vec3 toL = uDynamicPosRadius[i].xyz - world_hit;
                float d2 = dot(toL, toL);
                float r = uDynamicPosRadius[i].w;
                float r2 = r * r;
                if (d2 >= r2) continue;
                float d = sqrt(d2);
                vec3 L = toL / max(d, 1e-4);
                float ndotl = max(dot(N, L), 0.0);
                float atten = pow(max(1.0 - d2 / r2, 0.0), 2.0);
                litColor += albedo * uDynamicColorIntensity[i].rgb
                    * (uDynamicColorIntensity[i].a * atten * ndotl);
            }
        }
        }

        // 4. Fog and vignette.
        float dist = closest_t;
        float fogFactor = exp(-0.02 * dist);

        float vignette = vUv.x * vUv.y * (1.0 - vUv.x) * (1.0 - vUv.y);
        vignette = clamp(pow(16.0 * vignette, 0.25), 0.0, 1.0);

        vec3 fogTarget = uOutdoor == 1 ? uFogColor : vec3(0.0);
        vec3 outColor = mix(fogTarget, litColor, fogFactor * vignette);
        // Interleaved-gradient-noise dither breaks up the 8-bit banding of
        // fog/vignette gradients (toggle: rt_dither).
        outColor += (ign(gl_FragCoord.xy) - 0.5) * (1.0 / 128.0) * float(uDitherEnabled);
        fragColor = vec4(outColor, 1.0);
    } else {
        fragColor = vec4(uOutdoor == 1 ? uSkyColor : vec3(0.0), 1.0);
    }
}
"#;

/// Assembles the fragment shader from the preamble, shared chunks, and the
/// traversal/shading body. Runs once at program link time.
pub fn fragment_source() -> String {
    let mut source = String::with_capacity(
        FRAGMENT_HEADER.len()
            + chunks::NOISE_GLSL.len()
            + chunks::SPOT_CONE_GLSL.len()
            + FRAGMENT_BODY.len(),
    );
    source.push_str(FRAGMENT_HEADER);
    source.push_str(chunks::NOISE_GLSL);
    source.push_str(chunks::SPOT_CONE_GLSL);
    source.push_str(FRAGMENT_BODY);
    source
}
