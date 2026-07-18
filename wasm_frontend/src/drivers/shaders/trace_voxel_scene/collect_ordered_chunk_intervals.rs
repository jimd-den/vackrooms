//! Intersect the camera ray with resident chunks and optionally order entries.

pub const GLSL: &str = r#"
int collectOrderedChunkIntervals(
    vec3 rayDirection,
    out ChunkInterval intervals[25]
) {
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
        intervals[insertion] = ChunkInterval(
            chunkIndex, box.entry, box.exit, box.entryNormal
        );
        intervalCount++;
    }
    return intervalCount;
}
"#;
