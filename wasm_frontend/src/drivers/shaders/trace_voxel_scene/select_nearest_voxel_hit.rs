//! Trace every candidate chunk interval and retain the nearest solid voxel.

pub const GLSL: &str = r#"
VoxelHit selectNearestVoxelHit(
    vec3 rayDirection,
    ChunkInterval intervals[25],
    int intervalCount,
    out int closestChunkIndex
) {
    VoxelHit closest = VoxelHit(
        false, TRACE_INFINITY, vec3(0.0), 0u, 0u, 0u
    );
    closestChunkIndex = -1;
    for (int intervalIndex = 0; intervalIndex < intervalCount; ++intervalIndex) {
        ChunkInterval interval = intervals[intervalIndex];
        // Sorted traversal may stop only after every remaining AABB begins
        // beyond the nearest solid hit. Breaking after the first hit is wrong
        // because power-of-two padded chunk boxes overlap.
        if (uFrontToBackEnabled == 1 && interval.entry >= closest.distance) break;

        int chunkIndex = interval.chunkIndex;
        vec3 localOrigin = uCameraPosition - uChunkOrigins[chunkIndex];
        RayBoxHit box = RayBoxHit(
            true, interval.entry, interval.exit, interval.entryNormal
        );
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
    return closest;
}
"#;
