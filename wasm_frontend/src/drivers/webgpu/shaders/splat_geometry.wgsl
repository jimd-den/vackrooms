// Compact face expansion shared by visible and caster passes.

struct PackedFace {
    xy: u32,
    z_extents: u32,
    surface: u32,
    // low 24: domain sRGB color; high 8: authored emission over 0..=10.
    visual: u32,
};

@group(1) @binding(1) var<storage, read> packed_faces: array<PackedFace>;

fn face_u(axis: u32) -> vec3<f32> {
    if axis <= 3u { return vec3<f32>(1.0, 0.0, 0.0); }
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn face_v(axis: u32) -> vec3<f32> {
    switch axis {
        case 0u: { return vec3<f32>(0.0, 0.0, -1.0); }
        case 1u: { return vec3<f32>(0.0, 0.0, 1.0); }
        case 2u: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 3u: { return vec3<f32>(0.0, 1.0, 0.0); }
        case 4u: { return vec3<f32>(0.0, 0.0, 1.0); }
        default: { return vec3<f32>(0.0, 0.0, -1.0); }
    }
}

fn splat_face_center(packed: PackedFace) -> vec3<f32> {
    let center_local = vec3<f32>(
        f32(packed.xy & 65535u),
        f32(packed.xy >> 16u),
        f32(packed.z_extents & 65535u)
    ) * (1.0 / 1024.0);
    return chunk.origin_world_size.xyz + center_local;
}

fn splat_face_world_position(packed: PackedFace, corner_index: u32) -> vec3<f32> {
    let extent_u = f32((packed.z_extents >> 16u) & 255u);
    let extent_v = f32(packed.z_extents >> 24u);
    let axis = packed.surface & 255u;
    let corner = vec2<f32>(
        select(-1.0, 1.0, corner_index == 1u || corner_index == 3u),
        select(-1.0, 1.0, corner_index >= 2u)
    );
    return splat_face_center(packed)
        + face_u(axis) * corner.x * extent_u * chunk.bounds_voxel_size.w * 0.5
        + face_v(axis) * corner.y * extent_v * chunk.bounds_voxel_size.w * 0.5;
}
