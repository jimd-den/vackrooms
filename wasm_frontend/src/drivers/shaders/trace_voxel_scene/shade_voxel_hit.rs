//! Turn one selected voxel hit into linear scene radiance.

pub const GLSL: &str = r#"
bool isEmissiveVoxel(uint material) {
    return material == 4u || material == 9u || material == 16u;
}

vec3 decodePackedColor(uint packed) {
    return vec3(
        float((packed >> 16u) & 0xffu),
        float((packed >> 8u) & 0xffu),
        float(packed & 0xffu)
    ) * (1.0 / 255.0);
}

bool isDownwardEmittingFace(vec3 normal) {
    return normal.y < -0.5;
}

vec3 emittedVoxelRadiance(uint material, vec3 albedo, vec3 normal) {
    if (!isDownwardEmittingFace(normal)) return vec3(0.0);
    if (material == 4u) return albedo * 10.0;
    if (material == 9u) return albedo * 8.0;
    if (material == 16u) return albedo * 0.9;
    return vec3(0.0);
}

vec3 shadeVoxel(
    VoxelHit hit,
    vec3 worldPosition,
    float receiverVoxelSize,
    int receiverChunkIndex
) {
    vec3 albedo = srgbToLinear(decodePackedColor(hit.color));
    if (isEmissiveVoxel(hit.material) && isDownwardEmittingFace(hit.normal)) {
        return emittedVoxelRadiance(hit.material, albedo, hit.normal);
    }

    vec3 ambient = (uOutdoor == 1
        ? vec3(0.32, 0.38, 0.48)
        : vec3(0.045, 0.040, 0.024)) * uAmbientScale;
    vec3 analyticDirect = vec3(0.0);

    // Analytic fixtures are the sole static-light authority. A single packed
    // RGB value cannot describe the six incident face directions, so this
    // correctness path deliberately does not consume the optional surface
    // diffuse field.
    int lightFirst = uChunkLightFirst[receiverChunkIndex];
    int lightCount = uChunkLightCounts[receiverChunkIndex];
    for (int lightIndex = 0; lightIndex < lightCount; ++lightIndex) {
        SceneLight light = readSceneLight(lightFirst + lightIndex);
        analyticDirect += evaluateVisibleSceneLight(
            worldPosition, hit.normal, light, receiverVoxelSize
        );
    }
    for (int lightIndex = 0; lightIndex < 4; ++lightIndex) {
        if (lightIndex >= uDynamicLightCount) break;
        analyticDirect += evaluateVisiblePointLight(
            worldPosition,
            hit.normal,
            uDynamicPosRadius[lightIndex].xyz,
            uDynamicColorIntensity[lightIndex].rgb,
            uDynamicPosRadius[lightIndex].w,
            uDynamicColorIntensity[lightIndex].a,
            receiverVoxelSize
        );
    }

    vec3 radiance = albedo * (
        ambient + analyticDirect
    ) * (1.0 / PI);
    if (uFlashlightEnabled == 1) {
        radiance += albedo * spotBeam(
            uCameraPosition, uCamForward, worldPosition, hit.normal
        );
    }
    return radiance;
}
"#;
