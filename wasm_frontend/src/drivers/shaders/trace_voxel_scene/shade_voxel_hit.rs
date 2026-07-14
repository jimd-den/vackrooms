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
    int closestChunkIndex = -1;
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
        if (candidate.hit && candidate.distance < closest.distance) {
            closest = candidate;
            closestChunkIndex = chunkIndex;
        }
    }

    if (!closest.hit) {
        vec3 missRadiance = uOutdoor == 1 ? uSkyColor : uFogColor;
        fragColor = vec4(encodeDisplayColor(missRadiance), 1.0);
        return;
    }

    vec3 worldPosition = uCameraPosition + rayDirection * closest.distance;
    float receiverVoxelSize = uChunkVoxelSizes[closestChunkIndex];
    vec3 radiance = shadeVoxel(
        closest, worldPosition, receiverVoxelSize, closestChunkIndex
    );
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
