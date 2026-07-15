//! Compose emission, ambient fill, cached fill, fixtures, and flashlight.

pub const GLSL: &str = r#"
vec3 ambientIrradiance() {
    if (uOutdoor == 1) return vec3(0.32, 0.38, 0.48) * uAmbientScale;
    return vec3(0.045, 0.040, 0.024) * uAmbientScale;
}

vec3 composeSurfaceLighting(vec3 normal, vec3 albedo) {
    if (isEmissiveMaterial(vMaterial) && isDownwardEmittingFace(normal)) {
        return emittedRadiance(vMaterial, albedo, normal);
    }

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
    vec3 radiance = albedo * (
        ambient * ambientOcclusion + cachedStatic + analyticDirect
    ) * (1.0 / PI);
    if (uFlashlightEnabled == 1) {
        radiance += albedo * spotBeam(
            uCameraPosition, uCamForward, vWorldPosition, normal
        );
    }
    return radiance;
}
"#;
