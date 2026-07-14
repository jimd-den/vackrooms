//! Compose material emission, irradiance, fog, and display encoding.

pub const GLSL: &str = r#"
bool isEmissiveMaterial(float material) {
    return abs(material - 4.0) < 0.1
        || abs(material - 9.0) < 0.1
        || abs(material - 16.0) < 0.1;
}

vec3 emittedRadiance(float material, vec3 albedo) {
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
    if (isEmissiveMaterial(vMaterial)) {
        radiance = emittedRadiance(vMaterial, albedo);
    } else {
        float ambientOcclusion = mix(1.0, 0.62, clamp(vAo, 0.0, 1.0));
        vec3 irradiance = ambientIrradiance()
            + sampleStaticIrradiance(vWorldPosition, normal) * 0.34;

        for (int lightIndex = 0; lightIndex < 8; ++lightIndex) {
            if (lightIndex >= uLightCount) break;
            vec3 delta = uLightPositions[lightIndex] - vWorldPosition;
            vec3 surfaceToLight = normalize(delta);
            float visibility = lightIndex == uShadowedLightIndex
                ? shadowVisibility(vWorldPosition, normal, surfaceToLight)
                : 1.0;
            irradiance += evaluateSceneLight(
                vWorldPosition,
                normal,
                uLightPositions[lightIndex],
                uLightParams[lightIndex].zw,
                uLightColors[lightIndex],
                uLightParams[lightIndex].x,
                uLightParams[lightIndex].y,
                uLightKinds[lightIndex]
            ) * visibility;
        }

        radiance = albedo * (irradiance * ambientOcclusion) * (1.0 / PI);
        if (uFlashlightEnabled == 1) {
            radiance += albedo * spotBeam(
                uCameraPosition, uCamForward, vWorldPosition, normal
            );
        }
    }

    radiance += flareCores(vWorldPosition, uCameraPosition);
    radiance = applyDistanceFog(
        radiance, uFogColor, distanceToCamera, uFogStart, uFogDensity
    );

    vec3 displayColor = encodeDisplayColor(radiance);
    if (uDitherEnabled == 1) {
        displayColor += (ign(gl_FragCoord.xy) - 0.5) * (1.0 / 255.0);
    }
    fragColor = vec4(clamp(displayColor, 0.0, 1.0), 1.0);
}
"#;
