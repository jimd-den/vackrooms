//! Exact ray/box math, diagnostic DDA, and optional empty-leaf stepping.

pub const GLSL: &str = r#"
const float TRACE_INFINITY = 1e30;
const float TRACE_MIN_TIE_EPSILON = 1e-7;

struct RayBoxHit {
    bool hit;
    float entry;
    float exit;
    vec3 entryNormal;
};

struct ChunkInterval {
    int chunkIndex;
    float entry;
    float exit;
    vec3 entryNormal;
};

struct VoxelHit {
    bool hit;
    float distance;
    vec3 normal;
    uint material;
    uint color;
    uint lightWord;
};

float inverseDirection(float direction) {
    if (abs(direction) < 1e-20) {
        return direction < 0.0 ? -TRACE_INFINITY : TRACE_INFINITY;
    }
    return 1.0 / direction;
}

float distanceTieTolerance(float distance) {
    return max(abs(distance) * 1e-6, TRACE_MIN_TIE_EPSILON);
}

float traversalSampleBias(float voxelSize) {
    // This moves classification points off an exactly shared plane; it must
    // remain smaller than nearby, genuinely distinct boundary crossings.
    // At the 0.2 m reference grid, 1e-4 skipped a 13 micrometre
    // interval between a wall and ceiling and changed the reported face.
    return max(voxelSize * 1e-5, TRACE_MIN_TIE_EPSILON);
}

RayBoxHit intersectBox(vec3 rayOrigin, vec3 rayDirection, vec3 boundsMin, vec3 boundsMax) {
    // Parallel axes are handled explicitly so a boundary value never forms
    // the undefined product 0 * infinity.
    if ((abs(rayDirection.x) < 1e-20 && (rayOrigin.x < boundsMin.x || rayOrigin.x > boundsMax.x))
        || (abs(rayDirection.y) < 1e-20 && (rayOrigin.y < boundsMin.y || rayOrigin.y > boundsMax.y))
        || (abs(rayDirection.z) < 1e-20 && (rayOrigin.z < boundsMin.z || rayOrigin.z > boundsMax.z))) {
        return RayBoxHit(false, 0.0, 0.0, vec3(0.0));
    }

    vec3 inverseRay = vec3(
        inverseDirection(rayDirection.x),
        inverseDirection(rayDirection.y),
        inverseDirection(rayDirection.z)
    );
    vec3 first = (boundsMin - rayOrigin) * inverseRay;
    vec3 second = (boundsMax - rayOrigin) * inverseRay;
    vec3 nearTimes = min(first, second);
    vec3 farTimes = max(first, second);
    if (abs(rayDirection.x) < 1e-20) { nearTimes.x = -TRACE_INFINITY; farTimes.x = TRACE_INFINITY; }
    if (abs(rayDirection.y) < 1e-20) { nearTimes.y = -TRACE_INFINITY; farTimes.y = TRACE_INFINITY; }
    if (abs(rayDirection.z) < 1e-20) { nearTimes.z = -TRACE_INFINITY; farTimes.z = TRACE_INFINITY; }

    float entry = max(nearTimes.x, max(nearTimes.y, nearTimes.z));
    float exit = min(farTimes.x, min(farTimes.y, farTimes.z));
    if (entry > exit || exit < 0.0) {
        return RayBoxHit(false, entry, exit, vec3(0.0));
    }

    vec3 normal;
    if (entry <= 0.0) {
        vec3 absoluteRay = abs(rayDirection);
        if (absoluteRay.x >= absoluteRay.y && absoluteRay.x >= absoluteRay.z) normal = vec3(-sign(rayDirection.x), 0.0, 0.0);
        else if (absoluteRay.y >= absoluteRay.z) normal = vec3(0.0, -sign(rayDirection.y), 0.0);
        else normal = vec3(0.0, 0.0, -sign(rayDirection.z));
    } else if (nearTimes.x >= nearTimes.y && nearTimes.x >= nearTimes.z) {
        normal = vec3(rayDirection.x > 0.0 ? -1.0 : 1.0, 0.0, 0.0);
    } else if (nearTimes.y >= nearTimes.z) {
        normal = vec3(0.0, rayDirection.y > 0.0 ? -1.0 : 1.0, 0.0);
    } else {
        normal = vec3(0.0, 0.0, rayDirection.z > 0.0 ? -1.0 : 1.0);
    }
    return RayBoxHit(true, max(entry, 0.0), exit, normal);
}

float exitDistanceFromBox(
    vec3 rayOrigin,
    vec3 rayDirection,
    vec3 boundsMin,
    vec3 boundsMax,
    out vec3 nextNormal
) {
    vec3 exitTimes = vec3(TRACE_INFINITY);
    if (abs(rayDirection.x) >= 1e-20) exitTimes.x = ((rayDirection.x > 0.0 ? boundsMax.x : boundsMin.x) - rayOrigin.x) / rayDirection.x;
    if (abs(rayDirection.y) >= 1e-20) exitTimes.y = ((rayDirection.y > 0.0 ? boundsMax.y : boundsMin.y) - rayOrigin.y) / rayDirection.y;
    if (abs(rayDirection.z) >= 1e-20) exitTimes.z = ((rayDirection.z > 0.0 ? boundsMax.z : boundsMin.z) - rayOrigin.z) / rayDirection.z;

    float nearest = min(exitTimes.x, min(exitTimes.y, exitTimes.z));
    // Traversal bias and tie tolerance solve different problems. The former
    // moves a sample into the next cell; using that comparatively large value
    // here incorrectly merges distinct, very close wall/ceiling crossings.
    float tolerance = distanceTieTolerance(nearest);
    // Use the same tolerant X/Y/Z precedence as finest-cell DDA. Without
    // this, a mathematically tied edge ray can acquire a different face
    // normal solely because one path jumped across a larger empty leaf.
    if (abs(exitTimes.x - nearest) <= tolerance) {
        nextNormal = vec3(rayDirection.x > 0.0 ? -1.0 : 1.0, 0.0, 0.0);
        return nearest;
    }
    if (abs(exitTimes.y - nearest) <= tolerance) {
        nextNormal = vec3(0.0, rayDirection.y > 0.0 ? -1.0 : 1.0, 0.0);
        return nearest;
    }
    nextNormal = vec3(0.0, 0.0, rayDirection.z > 0.0 ? -1.0 : 1.0);
    return nearest;
}

VoxelHit traceChunkDda(
    vec3 rayOrigin,
    vec3 rayDirection,
    int rootIndex,
    float worldSize,
    float voxelSize,
    int svoDepth,
    RayBoxHit chunkBox
);

VoxelHit traceChunkSkippingEmptyLeaves(
    vec3 rayOrigin,
    vec3 rayDirection,
    int rootIndex,
    float worldSize,
    float voxelSize,
    int svoDepth,
    RayBoxHit chunkBox
) {
    float distance = chunkBox.entry;
    vec3 normal = chunkBox.entryNormal;
    float sampleBias = traversalSampleBias(voxelSize);
    float refinementEntry = chunkBox.entry;
    vec3 refinementEntryNormal = chunkBox.entryNormal;

    for (int stepIndex = 0; stepIndex < 768; ++stepIndex) {
        if (distance > chunkBox.exit) break;
        vec3 samplePoint = rayOrigin + rayDirection * min(distance + sampleBias, chunkBox.exit);
        VoxelLeaf leaf = lookupVoxelLeaf(samplePoint, rootIndex, worldSize, svoDepth);
        if (leaf.material != 0u) {
            // The coarse walk proves emptiness only. Resolve the final
            // voxel-scale crossing with the diagnostic algorithm so both
            // policies produce one canonical distance, material, and normal.
            RayBoxHit refinementBox = RayBoxHit(
                true, refinementEntry, chunkBox.exit, refinementEntryNormal
            );
            return traceChunkDda(
                rayOrigin,
                rayDirection,
                rootIndex,
                worldSize,
                voxelSize,
                svoDepth,
                refinementBox
            );
        }

        vec3 crossedNormal;
        float nextDistance = exitDistanceFromBox(
            rayOrigin, rayDirection, leaf.boundsMin, leaf.boundsMax, crossedNormal
        );
        if (nextDistance <= distance + sampleBias * 0.25) {
            nextDistance = distance + sampleBias;
        }
        // A normalized ray moves at most one voxel per axis over this short
        // interval. Retaining it lets the exact resolver reconstruct every
        // potentially competing face without replaying the skipped octant.
        refinementEntry = max(distance, nextDistance - voxelSize);
        refinementEntryNormal = normal;
        distance = nextDistance;
        normal = crossedNormal;
    }
    return VoxelHit(false, 0.0, vec3(0.0), 0u, 0u, 0u);
}

VoxelHit traceChunkDda(
    vec3 rayOrigin,
    vec3 rayDirection,
    int rootIndex,
    float worldSize,
    float voxelSize,
    int svoDepth,
    RayBoxHit chunkBox
) {
    float sampleBias = traversalSampleBias(voxelSize);
    float distance = chunkBox.entry;
    vec3 samplePoint = rayOrigin + rayDirection * min(distance + sampleBias, chunkBox.exit);
    vec3 cell = floor(samplePoint / voxelSize);
    vec3 directionStep = sign(rayDirection);
    vec3 normal = chunkBox.entryNormal;

    for (int stepIndex = 0; stepIndex < 768; ++stepIndex) {
        if (distance > chunkBox.exit) break;
        samplePoint = rayOrigin + rayDirection * min(distance + sampleBias, chunkBox.exit);
        VoxelLeaf leaf = lookupVoxelLeaf(samplePoint, rootIndex, worldSize, svoDepth);
        if (leaf.material != 0u) {
            return VoxelHit(true, distance, normal, leaf.material, leaf.color, leaf.lightWord);
        }

        // Recompute boundary times from the canonical integer cell each step.
        // Repeated `nextTime += deltaTime` accumulates enough rounding error
        // to reverse two distinct wall/ceiling crossings near a long ray.
        vec3 nextBoundary = (cell + max(directionStep, vec3(0.0))) * voxelSize;
        vec3 nextTimes = vec3(TRACE_INFINITY);
        if (abs(rayDirection.x) >= 1e-20) nextTimes.x = (nextBoundary.x - rayOrigin.x) / rayDirection.x;
        if (abs(rayDirection.y) >= 1e-20) nextTimes.y = (nextBoundary.y - rayOrigin.y) / rayDirection.y;
        if (abs(rayDirection.z) >= 1e-20) nextTimes.z = (nextBoundary.z - rayOrigin.z) / rayDirection.z;

        float nextDistance = min(nextTimes.x, min(nextTimes.y, nextTimes.z));
        if (nextDistance > chunkBox.exit) break;
        // Step every tied axis so a ray through an edge/corner cannot stall.
        // This is a floating-point equality tolerance, not the sampling bias.
        // Keeping them separate preserves the mathematically nearer face when
        // two boundaries are close but not actually coincident.
        float tieEpsilon = distanceTieTolerance(nextDistance);
        bool crossX = abs(nextTimes.x - nextDistance) <= tieEpsilon;
        bool crossY = abs(nextTimes.y - nextDistance) <= tieEpsilon;
        bool crossZ = abs(nextTimes.z - nextDistance) <= tieEpsilon;
        if (crossX) cell.x += directionStep.x;
        if (crossY) cell.y += directionStep.y;
        if (crossZ) cell.z += directionStep.z;
        if (crossX) normal = vec3(-directionStep.x, 0.0, 0.0);
        else if (crossY) normal = vec3(0.0, -directionStep.y, 0.0);
        else normal = vec3(0.0, 0.0, -directionStep.z);
        distance = nextDistance;
    }
    return VoxelHit(false, 0.0, vec3(0.0), 0u, 0u, 0u);
}

VoxelHit traceChunk(
    vec3 rayOrigin,
    vec3 rayDirection,
    int rootIndex,
    float worldSize,
    float voxelSize,
    int svoDepth,
    RayBoxHit chunkBox
) {
    if (uEmptySpaceSkipEnabled == 1) {
        return traceChunkSkippingEmptyLeaves(
            rayOrigin, rayDirection, rootIndex, worldSize, voxelSize, svoDepth, chunkBox
        );
    }
    return traceChunkDda(
        rayOrigin, rayDirection, rootIndex, worldSize, voxelSize, svoDepth, chunkBox
    );
}
"#;
