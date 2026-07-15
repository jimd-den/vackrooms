//! Occlusion queries from a visible voxel face to analytic emitter samples.
//!
//! The primary-ray hit and every shadow segment use the same chunk slabs,
//! stateless SVO lookup, and selected stepping policy. Rectangle visibility
//! is evaluated at the same four Gauss samples as its irradiance integral;
//! a center-only visibility term would leak around partial blockers.

pub const GLSL: &str = r#"
bool directSampleIsVisible(
    vec3 surfacePosition,
    vec3 surfaceNormal,
    vec3 samplePosition,
    float receiverVoxelSize
) {
    if (uDirectVisibilityEnabled == 0) return true;

    float bias = max(receiverVoxelSize * 1e-3, 1e-5);
    vec3 segmentOrigin = surfacePosition + surfaceNormal * bias;
    vec3 segment = samplePosition - segmentOrigin;
    float segmentLength = length(segment);
    if (!(segmentLength > bias * 2.0)) return true;
    vec3 direction = segment / segmentLength;
    float blockerLimit = segmentLength - bias * 2.0;

    for (int chunkIndex = 0; chunkIndex < uNumChunks; ++chunkIndex) {
        vec3 localOrigin = segmentOrigin - uChunkOrigins[chunkIndex];
        RayBoxHit box = intersectBox(
            localOrigin,
            direction,
            vec3(0.0),
            vec3(uChunkWorldSizes[chunkIndex])
        );
        if (!box.hit || box.entry >= blockerLimit) continue;
        box.exit = min(box.exit, blockerLimit);
        VoxelHit blocker = traceChunk(
            localOrigin,
            direction,
            uChunkRootIndices[chunkIndex],
            uChunkWorldSizes[chunkIndex],
            uChunkVoxelSizes[chunkIndex],
            uChunkDepths[chunkIndex],
            box
        );
        if (blocker.hit && blocker.distance < blockerLimit) return false;
    }
    return true;
}

vec3 evaluateVisiblePointLight(
    vec3 surfacePosition,
    vec3 normal,
    vec3 lightPosition,
    vec3 lightColor,
    float range,
    float intensity,
    float receiverVoxelSize
) {
    vec3 delta = lightPosition - surfacePosition;
    float distanceSquared = dot(delta, delta);
    vec3 surfaceToLight = delta * inversesqrt(max(distanceSquared, 1e-8));
    float weight = finiteRangeInverseSquare(distanceSquared, range)
        * receiverCosine(normal, surfaceToLight);
    if (!(weight > 0.0)
        || !directSampleIsVisible(
            surfacePosition, normal, lightPosition, receiverVoxelSize
        )) {
        return vec3(0.0);
    }
    return max(lightColor, vec3(0.0)) * max(intensity, 0.0) * weight;
}

vec3 evaluateVisibleSceneLight(
    vec3 surfacePosition,
    vec3 normal,
    SceneLight light,
    float receiverVoxelSize
) {
    if (light.kind == LIGHT_POINT) {
        return evaluateVisiblePointLight(
            surfacePosition, normal, light.position, light.color,
            light.range, light.intensity, receiverVoxelSize
        );
    }

    float area = 4.0 * max(light.halfSize.x, 0.0) * max(light.halfSize.y, 0.0);
    if (!(area > 0.0)) return vec3(0.0);
    const float GAUSS = 0.5773502691896258;
    vec2 offsets[4] = vec2[](
        vec2(-GAUSS, -GAUSS), vec2(GAUSS, -GAUSS),
        vec2(-GAUSS,  GAUSS), vec2(GAUSS,  GAUSS)
    );
    float integral = 0.0;
    for (int sampleIndex = 0; sampleIndex < 4; ++sampleIndex) {
        vec2 offset = offsets[sampleIndex] * light.halfSize;
        vec3 samplePosition = light.position + vec3(offset.x, 0.0, offset.y);
        vec3 delta = samplePosition - surfacePosition;
        float distanceSquared = dot(delta, delta);
        vec3 surfaceToLight = delta * inversesqrt(max(distanceSquared, 1e-8));
        float weight = finiteRangeInverseSquare(distanceSquared, light.range)
            * receiverCosine(normal, surfaceToLight)
            * emitterCosine(light.kind, surfaceToLight);
        if (weight > 0.0
            && directSampleIsVisible(
                surfacePosition, normal, samplePosition, receiverVoxelSize
            )) {
            integral += weight;
        }
    }
    return max(light.color, vec3(0.0)) * max(light.intensity, 0.0)
        * area * (integral * 0.25);
}
"#;
