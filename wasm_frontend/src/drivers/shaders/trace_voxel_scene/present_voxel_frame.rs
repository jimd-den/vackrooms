//! Resolve a miss or carry a shaded voxel through fog into display space.

pub const GLSL: &str = r#"
void main() {
    vec3 rayDirection = constructCameraRay();

    ChunkInterval intervals[25];
    int intervalCount = collectOrderedChunkIntervals(rayDirection, intervals);

    int closestChunkIndex;
    VoxelHit closest = selectNearestVoxelHit(
        rayDirection, intervals, intervalCount, closestChunkIndex
    );

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
