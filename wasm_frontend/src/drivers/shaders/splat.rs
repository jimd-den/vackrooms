//! Instanced face-splat program — the GPU microvoxel path.
//!
//! One instance is one axis-aligned surface rectangle (`PackedFaceInstance`);
//! the vertex shader rebuilds the quad from center + extents + axis via
//! `gl_VertexID` (4-vertex TRIANGLE_STRIP) and evaluates ALL lighting once
//! per face, flat — the voxel look, and the reason the fragment shader
//! shrinks to fog + grain + dither (the Pi win).

use super::chunks;

/// Preamble: version, attributes, uniforms, varyings.
const VERTEX_HEADER: &str = r#"#version 300 es
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
// Camera forward axis for the flashlight cone.
uniform vec3 uCamForward;
uniform int uFlashlightEnabled;
uniform sampler3D uLightVolume;

uniform int uLightCount;
uniform vec3 uLightPositions[8];
uniform vec3 uLightColors[8];
uniform vec4 uLightParams[8];

uniform sampler2D uShadowMap;
uniform mat4 uLightViewProjection;
uniform int uShadowedLightIndex;
uniform int uShadowTaps;

// Level atmosphere (see the surface shader).
uniform int uOutdoor;
uniform float uAmbientScale;

out vec3 vWorldPos;
flat out vec3 vColor;
flat out vec3 vNormal;
flat out float vEmissive;
flat out float vGrainAmp;
flat out float vBeamPattern;
"#;

/// Quad reconstruction + per-face lighting body.
const VERTEX_BODY: &str = r#"
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
    vec3 roomAmbient = vec3(0.16, 0.145, 0.075);
    vec3 bouncedLight = irradiance * 0.75;
    irradiance = max(roomAmbient, bouncedLight);
    if (uOutdoor == 1) {
        irradiance = max(irradiance, vec3(0.30, 0.32, 0.36));
    }

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
    if (uOutdoor == 1) {
        ambientUp = vec3(0.98, 1.02, 1.08) * irradiance;
        ambientDown = vec3(0.56, 0.58, 0.55) * irradiance;
    }
    vec3 ambient = mix(ambientDown, ambientUp, N.y * 0.5 + 0.5) * ao * faceResponse * uAmbientScale;

    vec3 directDiffuse = vec3(0.0);
    for (int i = 0; i < 8; ++i) {
        if (i >= uLightCount) break;
        vec3 toLight = uLightPositions[i] - center;
        float d2 = dot(toLight, toLight);
        float range = uLightParams[i].x;
        float range2 = range * range;
        if (d2 >= range2) continue;

        float d = sqrt(d2);
        vec3 L = toLight / max(d, 0.001);
        float ndotl = max(dot(N, L), 0.0);
        float x = clamp(d2 / range2, 0.0, 1.0);
        float falloff = (1.0 - x) * (1.0 - x);

        float visible = 1.0;
        if (i == uShadowedLightIndex) {
            visible = shadowVisibility(center, N, L);
        }

        float lightHash = hash3D(uLightPositions[i] * 10.0);
        float directMod = 1.0 + 0.05 * (lightHash - 0.5);
        vec3 lightColor = mix(uLightColors[i], vec3(1.0, 0.96, 0.85), 0.3);
        directDiffuse += lightColor * uLightParams[i].y * falloff * visible * ndotl * directMod;
    }

    // Flashlight: the shared spotlight cone (spec in the CPU splatter),
    // evaluated once per face at its center — flat like all splat lighting.
    // Kept INSIDE the light sum: the material color is the immutable base
    // and every light multiplies it, so the flashlight brightens yellow
    // walls toward brighter yellow, never neutral grey.
    vec3 flashlight = vec3(0.0);
    if (uFlashlightEnabled == 1) {
        flashlight = spotBeam(uCameraPosition, uCamForward, center, N);
    }

    vec3 lighting = quantizeLighting(ambient + directDiffuse + flashlight);
    vColor = albedo * lighting;
}
"#;

/// Fragment preamble.
const FRAGMENT_HEADER: &str = r#"#version 300 es
precision highp float;
// Fragment ints default to mediump; the vertex stage declares uOutdoor at
// its (highp) default, and linking demands the precisions match.
precision highp int;

in vec3 vWorldPos;
flat in vec3 vColor;
flat in vec3 vNormal;
flat in float vEmissive;
flat in float vGrainAmp;
flat in float vBeamPattern;

uniform vec3 uCameraPosition;
uniform float uVoxelScale;
// Optimization/effect toggle: grain + banding dither (RenderToggles).
uniform int uDitherEnabled;

// Dropped-flare cores: xyz = world position, w = intensity (pre-flickered).
uniform int uCoreCount;
uniform vec4 uCores[4];
uniform vec3 uCoreColors[4];

// Level atmosphere: outdoors the fog fades to a bright sky haze instead of
// black (the framebuffer is cleared to the matching sky color).
uniform int uOutdoor;
uniform vec3 uFogColor;

out vec4 fragColor;
"#;

/// The splat fragment body is deliberately tiny (the Pi win): fog,
/// voxel-lattice albedo grain, ordered dither, tone map. All lighting
/// arrived flat from the vertex shader.
const FRAGMENT_BODY: &str = r#"
void main() {
    vec3 color = vColor;
    float distanceToCamera = length(uCameraPosition - vWorldPos);

    if (vEmissive < 0.5) {
        // World-locked per-voxel albedo grain: sample half a voxel inside
        // the surface so faces lying exactly on lattice planes hash stably.
        // Amplitude is per-face (zero on ceilings) so large bright planes
        // read as solid construction, not noise.
        vec3 lattice = floor((vWorldPos - vNormal * (0.5 * uVoxelScale)) / uVoxelScale);
        float grain = 1.0 + (hash3D(lattice) - 0.5) * 2.0 * vGrainAmp * float(uDitherEnabled);
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
    if (uOutdoor == 1) {
        currentFogColor = uFogColor;
    }
    color = mix(color, currentFogColor, clamp(fogAmount, 0.0, 1.0));

    // Flare cores (shared chunk; occluded by nearer surfaces).
    color += flareCores(vWorldPos, uCameraPosition);

    color = toneMap(color);
    color += (ign(gl_FragCoord.xy) - 0.5) * (1.0 / 128.0) * (1.0 - fogAmount) * float(uDitherEnabled);

    fragColor = vec4(color, 1.0);
}
"#;

/// Assembles the vertex shader (quad rebuild + all lighting, flat).
pub fn vertex_source() -> String {
    let mut source = String::with_capacity(
        VERTEX_HEADER.len()
            + chunks::MATERIAL_COLOR_GLSL.len()
            + chunks::NOISE_GLSL.len()
            + chunks::SPOT_CONE_GLSL.len()
            + VERTEX_BODY.len(),
    );
    source.push_str(VERTEX_HEADER);
    source.push_str(chunks::MATERIAL_COLOR_GLSL);
    source.push_str(chunks::NOISE_GLSL);
    source.push_str(chunks::SPOT_CONE_GLSL);
    source.push_str(VERTEX_BODY);
    source
}

/// Assembles the (deliberately tiny) fragment shader.
pub fn fragment_source() -> String {
    let mut source = String::with_capacity(
        FRAGMENT_HEADER.len()
            + chunks::NOISE_GLSL.len()
            + chunks::TONE_MAP_GLSL.len()
            + chunks::FLARE_CORES_GLSL.len()
            + FRAGMENT_BODY.len(),
    );
    source.push_str(FRAGMENT_HEADER);
    source.push_str(chunks::NOISE_GLSL);
    source.push_str(chunks::TONE_MAP_GLSL);
    source.push_str(chunks::FLARE_CORES_GLSL);
    source.push_str(FRAGMENT_BODY);
    source
}
