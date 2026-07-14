//! Indexed-surface program — the default renderer's shaders.
//!
//! They intentionally use only ordinary WebGL2 vertex/index buffers: raster
//! depth testing replaces the fullscreen SVO traversal while the SVO remains
//! available for collision and debug/reference queries. Lighting runs per
//! fragment: baked 3D light volume + up to four selected point lights (one
//! shadow-mapped hero light) + the flashlight cone + flare cores + fog.

use super::chunks;

pub const VERTEX_SHADER: &str = r#"#version 300 es
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

/// Preamble: version, precision, varyings, uniforms.
const FRAGMENT_HEADER: &str = r#"#version 300 es
precision highp float;
precision highp sampler3D;

in vec3 vWorldPosition;
in vec3 vNormal;
flat in float vMaterial;
flat in float vStaticIndirect;
flat in float vAo;

uniform vec3 uCameraPosition;
// Camera forward axis for the flashlight cone.
uniform vec3 uCamForward;
uniform int uFlashlightEnabled;
// Optimization/effect toggle: ordered-dither light banding (RenderToggles).
uniform int uDitherEnabled;
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

// Level atmosphere: outdoors (grassland) fog turns into bright sky haze and
// the ambient response scales up. Indoors these are unused.
uniform int uOutdoor;
uniform vec3 uFogColor;
uniform float uAmbientScale;

out vec4 fragColor;
"#;

/// Lighting/fog body (everything after the shared chunks).
const FRAGMENT_BODY: &str = r#"
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
    // The packed face record carries the scalar BFS bake as a seam-safe
    // fallback when a linearly filtered 3D sample straddles a chunk edge.
    // Consuming it here also makes `aStaticIndirect` part of the linked
    // program instead of letting the GLSL linker optimize the attribute out.
    float baked = vStaticIndirect * (1.0 / 15.0);
    vec3 bakedWarm = vec3(baked, baked * 0.94, baked * 0.72);
    irradiance = max(irradiance, bakedWarm);
    vec3 roomAmbient = vec3(0.16, 0.145, 0.075);
    vec3 bouncedLight = irradiance * 0.75;
    irradiance = max(roomAmbient, bouncedLight);
    if (uOutdoor == 1) {
        // Daylight floor: the sky layer's BFS light dominates outdoors; do
        // not crush it toward the dim interior baseline.
        irradiance = max(irradiance, vec3(0.30, 0.32, 0.36));
    }

    // Upward-facing surfaces brightest, downward surfaces notably darker,
    // side walls vary subtly by cardinal direction.
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

    // Yellow-green ambient and olive shadows (cool neutral hemisphere
    // outdoors so the grassland reads as daylight, not office fluorescence).
    vec3 ambientUp   = vec3(0.16, 0.17, 0.09) * irradiance;
    vec3 ambientDown = vec3(0.035, 0.030, 0.012) * irradiance;
    if (uOutdoor == 1) {
        ambientUp = vec3(0.62, 0.66, 0.72) * irradiance;
        ambientDown = vec3(0.34, 0.36, 0.33) * irradiance;
    }
    vec3 ambient = mix(ambientDown, ambientUp, N.y * 0.5 + 0.5) * ao * faceResponse * uAmbientScale;

    // Flashlight: a true spotlight cone around the camera forward axis
    // (shared spec with the CPU splatter). Replaces the old N-dot-V
    // "headlamp" that lit every camera-facing surface regardless of where
    // the player pointed.
    vec3 flashlight = vec3(0.0);
    if (uFlashlightEnabled == 1) {
        flashlight = spotBeam(uCameraPosition, uCamForward, vWorldPosition, N);
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
            float x = clamp(d2 / range2, 0.0, 1.0);
            float falloff = (1.0 - x) * (1.0 - x);

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

        // Quantize indirect (ambient) + direct diffuse into 5 bands with
        // ordered dithering (rt_dither disables the dither offset only, so
        // the banded art style survives the toggle).
        float ditherVal = uDitherEnabled == 1 ? getOrderedDither() : 0.0;
        vec3 totalDiffuseLight = quantize5(ambient + directDiffuse, ditherVal);

        // Low-frequency grime/stain modulation on albedo
        float grime = getGrime(vWorldPosition);
        vec3 modulatedAlbedo = albedo * grime;

        // Keep the flashlight as its own additive radiance term. Diffuse and
        // direct specular retain the material response; the beam tint itself
        // must not disappear into a dark/grimy albedo.
        color = totalDiffuseLight * modulatedAlbedo
              + directSpecular * modulatedAlbedo
              + flashlight;
    }

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

    // Fog with onset distance
    float fogOnset = 22.0;
    float fogDist = max(0.0, distanceToCamera - fogOnset);
    float fogDensity = 0.010;

    // Keep distant geometry readable; streaming should not force full opacity.
    float fogAmount = min(1.0 - exp(-fogDist * fogDensity), 0.82);

    float hNorm = clamp(vWorldPosition.y / 3.4, 0.0, 1.0);
    vec3 fogColor = mix(
        vec3(0.018, 0.014, 0.006),  // floor: nearly black warm brown
        vec3(0.026, 0.030, 0.010),  // ceiling: faint sickly olive
        hNorm
    );
    if (uOutdoor == 1) {
        fogColor = uFogColor;
    }

    // Separate emissive glow and flares to retain visibility through fog
    vec3 emissiveGlow = glow + flareCores(vWorldPosition, uCameraPosition);

    // Blend geometry color toward fogColor
    vec3 foggedSurface = mix(color, fogColor, fogAmount);

    // Add emissive terms, reduced but not destroyed by fog.
    color = foggedSurface + emissiveGlow * (1.0 - fogAmount * 0.55);

    // Add a subtle ambient glow to the near-field (so it isn't completely dry and flat up close)
    color += glow * 0.12 * exp(-distanceToCamera * 0.05);

    color = toneMap(color);

    fragColor = vec4(color, 1.0);
}
"#;

/// Assembles the fragment shader from the preamble, shared chunks, and the
/// lighting body. Runs once at program link time.
pub fn fragment_source() -> String {
    let mut source = String::with_capacity(
        FRAGMENT_HEADER.len()
            + chunks::MATERIAL_COLOR_GLSL.len()
            + chunks::NOISE_GLSL.len()
            + chunks::SPOT_CONE_GLSL.len()
            + chunks::TONE_MAP_GLSL.len()
            + chunks::FLARE_CORES_GLSL.len()
            + FRAGMENT_BODY.len(),
    );
    source.push_str(FRAGMENT_HEADER);
    source.push_str(chunks::MATERIAL_COLOR_GLSL);
    source.push_str(chunks::NOISE_GLSL);
    source.push_str(chunks::SPOT_CONE_GLSL);
    source.push_str(chunks::TONE_MAP_GLSL);
    source.push_str(chunks::FLARE_CORES_GLSL);
    source.push_str(FRAGMENT_BODY);
    source
}
