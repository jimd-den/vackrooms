//! Select the nearest scene hit, shade it in linear space, then apply fog.

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

vec3 emittedVoxelRadiance(uint material, vec3 albedo) {
    if (material == 4u) return albedo * 10.0;
    if (material == 9u) return albedo * 8.0;
    if (material == 16u) return albedo * 0.9;
    return vec3(0.0);
}

float packedFaceVisibility(uint lightWord, vec3 normal) {
    uint mask = (lightWord >> 8u) & 0x3fu;
    bool occluded = normal.x > 0.5 ? (mask & 1u) != 0u
        : normal.x < -0.5 ? (mask & 2u) != 0u
        : normal.y > 0.5 ? (mask & 4u) != 0u
        : normal.y < -0.5 ? (mask & 8u) != 0u
        : normal.z > 0.5 ? (mask & 16u) != 0u
        : (mask & 32u) != 0u;
    return occluded ? 0.62 : 1.0;
}

vec3 packedIrradiance(uint lightWord) {
    if (uBakedLightingEnabled == 0) return vec3(0.0);
    return vec3(
        float((lightWord >> 16u) & 0x0fu),
        float((lightWord >> 20u) & 0x0fu),
        float((lightWord >> 24u) & 0x0fu)
    ) * (1.0 / 15.0);
}

vec3 shadeVoxel(
    VoxelHit hit,
    vec3 worldPosition,
    vec3 rayDirection
) {
    vec3 albedo = srgbToLinear(decodePackedColor(hit.color));
    if (isEmissiveVoxel(hit.material)) {
        return emittedVoxelRadiance(hit.material, albedo);
    }

    vec3 irradiance = (uOutdoor == 1
        ? vec3(0.32, 0.38, 0.48)
        : vec3(0.045, 0.040, 0.024)) * uAmbientScale;
    irradiance += packedIrradiance(hit.lightWord) * 0.34;

    // The quantized bake is a cache of static lighting, not an additional
    // light source. The diagnostic path evaluates static fixtures
    // analytically; the cache path replaces that loop and cannot double it.
    if (uBakedLightingEnabled == 0) {
        for (int lightIndex = 0; lightIndex < 8; ++lightIndex) {
            if (lightIndex >= uLightCount) break;
            irradiance += evaluateSceneLight(
                worldPosition,
                hit.normal,
                uLightPositions[lightIndex],
                uLightParams[lightIndex].zw,
                uLightColors[lightIndex],
                uLightParams[lightIndex].x,
                uLightParams[lightIndex].y,
                uLightKinds[lightIndex]
            );
        }
    }
    for (int lightIndex = 0; lightIndex < 4; ++lightIndex) {
        if (lightIndex >= uDynamicLightCount) break;
        irradiance += evaluatePointLight(
            worldPosition,
            hit.normal,
            uDynamicPosRadius[lightIndex].xyz,
            uDynamicColorIntensity[lightIndex].rgb,
            uDynamicPosRadius[lightIndex].w,
            uDynamicColorIntensity[lightIndex].a
        );
    }

    vec3 radiance = albedo * irradiance
        * packedFaceVisibility(hit.lightWord, hit.normal) * (1.0 / PI);
    if (uFlashlightEnabled == 1) {
        radiance += albedo * spotBeam(
            uCameraPosition, uCamForward, worldPosition, hit.normal
        );
    }
    return radiance;
}

void main() {
    vec2 ndc = vUv * 2.0 - 1.0;
    vec3 rayDirection = normalize(
        uCamRight * (ndc.x * uAspect * uFovTan)
        + uCamUp * (ndc.y * uFovTan)
        + uCamForward
    );

    ChunkInterval intervals[25];
    int intervalCount = 0;
    for (int chunkIndex = 0; chunkIndex < uNumChunks; ++chunkIndex) {
        vec3 localOrigin = uCameraPosition - uChunkOrigins[chunkIndex];
        RayBoxHit box = intersectBox(
            localOrigin,
            rayDirection,
            vec3(0.0),
            vec3(uChunkWorldSizes[chunkIndex])
        );
        if (!box.hit) continue;

        int insertion = intervalCount;
        if (uFrontToBackEnabled == 1) {
            while (insertion > 0 && intervals[insertion - 1].entry > box.entry) {
                intervals[insertion] = intervals[insertion - 1];
                insertion--;
            }
        }
        intervals[insertion] = ChunkInterval(chunkIndex, box.entry, box.exit, box.entryNormal);
        intervalCount++;
    }

    VoxelHit closest = VoxelHit(false, TRACE_INFINITY, vec3(0.0), 0u, 0u, 0u);
    for (int intervalIndex = 0; intervalIndex < intervalCount; ++intervalIndex) {
        ChunkInterval interval = intervals[intervalIndex];
        // Sorted traversal may stop only after every remaining AABB begins
        // beyond the nearest solid hit. Breaking after the first hit is wrong
        // because power-of-two padded chunk boxes overlap.
        if (uFrontToBackEnabled == 1 && interval.entry >= closest.distance) break;

        int chunkIndex = interval.chunkIndex;
        vec3 localOrigin = uCameraPosition - uChunkOrigins[chunkIndex];
        RayBoxHit box = RayBoxHit(true, interval.entry, interval.exit, interval.entryNormal);
        VoxelHit candidate = traceChunk(
            localOrigin,
            rayDirection,
            uChunkRootIndices[chunkIndex],
            uChunkWorldSizes[chunkIndex],
            uChunkVoxelSizes[chunkIndex],
            uChunkDepths[chunkIndex],
            box
        );
        if (candidate.hit && candidate.distance < closest.distance) closest = candidate;
    }

    if (!closest.hit) {
        vec3 missRadiance = uOutdoor == 1 ? uSkyColor : uFogColor;
        fragColor = vec4(encodeDisplayColor(missRadiance), 1.0);
        return;
    }

    vec3 worldPosition = uCameraPosition + rayDirection * closest.distance;
    vec3 radiance = shadeVoxel(closest, worldPosition, rayDirection);
    radiance = applyDistanceFog(
        radiance, uFogColor, closest.distance, uFogStart, uFogDensity
    );
    vec3 displayColor = encodeDisplayColor(radiance);
    if (uDitherEnabled == 1) {
        displayColor += (ign(gl_FragCoord.xy) - 0.5) * (1.0 / 255.0);
    }
    fragColor = vec4(clamp(displayColor, 0.0, 1.0), 1.0);
}
"#;
