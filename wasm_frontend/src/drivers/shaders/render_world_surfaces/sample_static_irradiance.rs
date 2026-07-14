//! Sample the baked field in the air cell adjacent to a visible face.
//!
//! Texture texels represent voxel centers. Sampling on the geometric face
//! plane blends the solid receiver with air and was the source of false
//! floor patches. A half-voxel normal offset lands exactly at the adjacent
//! air-cell center for every axis-aligned face.

pub const GLSL: &str = r#"
vec3 sampleStaticIrradiance(vec3 worldPosition, vec3 normal) {
    if (uBakedLightingEnabled == 0) return vec3(0.0);

    ivec3 dimensions = textureSize(uLightVolume, 0);
    if (any(lessThanEqual(dimensions, ivec3(0)))) {
        float scalarFallback = clamp(vStaticIndirect * (1.0 / 15.0), 0.0, 1.0);
        return vec3(scalarFallback);
    }

    vec3 worldExtent = vec3(dimensions) * uVoxelSize;
    vec3 adjacentAirCenter = worldPosition + normal * (0.5 * uVoxelSize);
    vec3 uvw = (adjacentAirCenter - uChunkOrigin) / worldExtent;
    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
        return vec3(0.0);
    }
    return texture(uLightVolume, uvw).rgb;
}
"#;
