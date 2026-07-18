// Compact indexed-surface position contract shared by visible and caster
// passes. Material/light fields remain available to the visible program.

struct PackedSurfaceVertex {
    xy: u32,
    z_normal_material: u32,
    light_ao: u32,
    // low 24: domain sRGB color; high 8: authored emission over 0..=10.
    visual: u32,
};

@group(1) @binding(1) var<storage, read> packed_vertices: array<PackedSurfaceVertex>;

fn surface_world_position(packed: PackedSurfaceVertex) -> vec3<f32> {
    let local = vec3<f32>(
        f32(packed.xy & 65535u),
        f32(packed.xy >> 16u),
        f32(packed.z_normal_material & 65535u)
    ) * (1.0 / 1024.0);
    return chunk.origin_world_size.xyz + local;
}
