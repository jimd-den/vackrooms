//! Compose material emission, irradiance, fog, and display encoding.

pub const GLSL: &str = r#"
bool isEmissiveMaterial(float material) {
    return abs(material - 4.0) < 0.1
        || abs(material - 9.0) < 0.1
        || abs(material - 16.0) < 0.1;
}

bool isDownwardEmittingFace(vec3 normal) {
    return normal.y < -0.5;
}

vec3 emittedRadiance(float material, vec3 albedo, vec3 normal) {
    if (!isDownwardEmittingFace(normal)) return vec3(0.0);
    if (abs(material - 4.0) < 0.1) return albedo * 10.0;
    if (abs(material - 9.0) < 0.1) return albedo * 8.0;
    if (abs(material - 16.0) < 0.1) return albedo * 0.9;
    return vec3(0.0);
}

float shadowVisibility(vec3 worldPosition, vec3 normal, vec3 surfaceToLight) {
    vec4 clip = uLightViewProjection * vec4(worldPosition + normal * 0.02, 1.0);
    vec3 projected = clip.xyz / max(abs(clip.w), 1e-8);
    vec2 uv = projected.xy * 0.5 + 0.5;
    float receiverDepth = projected.z * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))
        || receiverDepth <= 0.0 || receiverDepth >= 1.0) {
        return 1.0;
    }

    vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0));
    float bias = max(0.0015 * (1.0 - dot(normal, surfaceToLight)), 0.0005);
    float visible = 0.0;
    vec2 offsets[4] = vec2[](
        vec2(-0.5, -0.5), vec2(0.5, -0.5),
        vec2(-0.5, 0.5), vec2(0.5, 0.5)
    );
    for (int tap = 0; tap < 4; ++tap) {
        float storedDepth = texture(uShadowMap, uv + offsets[tap] * texel).r;
        visible += receiverDepth - bias <= storedDepth ? 1.0 : 0.0;
    }
    return visible * 0.25;
}

vec3 ambientIrradiance() {
    if (uOutdoor == 1) return vec3(0.32, 0.38, 0.48) * uAmbientScale;
    return vec3(0.045, 0.040, 0.024) * uAmbientScale;
}

void main() {
    vec3 normal = normalize(vNormal);
    vec3 albedo = srgbToLinear(materialColor(vMaterial));
    float distanceToCamera = length(uCameraPosition - vWorldPosition);

    vec3 radiance;
    if (isEmissiveMaterial(vMaterial) && isDownwardEmittingFace(normal)) {
        radiance = emittedRadiance(vMaterial, albedo, normal);
    } else {
        float ambientOcclusion = mix(1.0, 0.62, clamp(vAo, 0.0, 1.0));
        vec3 ambient = ambientIrradiance();
        // The voxel field is deliberately a restrained diffuse-fill term.
        // It never substitutes for the analytic fixtures below.
        vec3 cachedStatic = sampleStaticIrradiance(vWorldPosition, normal) * 0.12;
        vec3 analyticDirect = vec3(0.0);

        for (int lightIndex = 0; lightIndex < uLightCount; ++lightIndex) {
            SceneLight light = readSceneLight(uLightFirst + lightIndex);
            vec3 delta = light.position - vWorldPosition;
            vec3 surfaceToLight = normalize(delta);
            float visibility = lightIndex == uShadowedLightIndex
                ? shadowVisibility(vWorldPosition, normal, surfaceToLight)
                : 1.0;
            analyticDirect += evaluateSceneLight(
                vWorldPosition,
                normal,
                light.position,
                light.halfSize,
                light.color,
                light.range,
                light.intensity,
                light.kind
            ) * visibility;
        }
        for (int lightIndex = 0; lightIndex < 4; ++lightIndex) {
            if (lightIndex >= uDynamicLightCount) break;
            analyticDirect += evaluatePointLight(
                vWorldPosition,
                normal,
                uDynamicPosRadius[lightIndex].xyz,
                uDynamicColorIntensity[lightIndex].rgb,
                uDynamicPosRadius[lightIndex].w,
                uDynamicColorIntensity[lightIndex].a
            );
        }

        // Screen-space/voxel AO is only a model of indirect sky/room
        // visibility. Applying it to analytic or already-occluded cached
        // fixture light darkens corners twice and violates the light model.
        radiance = albedo * (
            ambient * ambientOcclusion + cachedStatic + analyticDirect
        ) * (1.0 / PI);
        if (uFlashlightEnabled == 1) {
            radiance += albedo * spotBeam(
                uCameraPosition, uCamForward, vWorldPosition, normal
            );
        }
    }

    radiance = applyDistanceFog(
        radiance, uFogColor, distanceToCamera, uFogStart, uFogDensity
    );
    // Core sprites lie between the camera and the receiver. Attenuate each
    // at its own along-ray distance rather than fogging it as if it were on
    // the receiver surface.
    radiance += flareCoresThroughMedium(
        vWorldPosition, uCameraPosition, uFogStart, uFogDensity
    );

    vec3 displayColor = encodeDisplayColor(radiance);
    if (uDitherEnabled == 1) {
        displayColor += (ign(gl_FragCoord.xy) - 0.5) * (1.0 / 255.0);
    }
    fragColor = vec4(clamp(displayColor, 0.0, 1.0), 1.0);
}
"#;
