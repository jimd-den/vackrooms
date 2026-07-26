// Instanced face-splat strategy. One compact 16-byte record becomes a quad
// in the vertex stage; face geometry is generated entirely on the GPU.

struct SplatVertex {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) visual: u32,
    @location(3) @interpolate(flat) material: u32,
    @location(4) @interpolate(flat) baked: f32,
    @location(5) @interpolate(flat) ao: f32,
};

@vertex
fn splat_vertex(
    @builtin(vertex_index) corner_index: u32,
    @builtin(instance_index) face_index: u32,
) -> SplatVertex {
    let packed = packed_faces[face_index];
    let axis = packed.surface & 255u;
    let material = (packed.surface >> 8u) & 255u;
    let baked = f32((packed.surface >> 16u) & 255u) / 15.0;
    let ao = 1.0 - f32((packed.surface >> 24u) & 1u);
    let normal = normal_for_axis(axis);
    let world = splat_face_world_position(packed, corner_index);

    var output: SplatVertex;
    output.clip_position = frame.view_projection * vec4<f32>(world, 1.0);
    output.world = world;
    output.normal = normal;
    output.visual = packed.visual;
    output.material = material;
    output.baked = baked;
    output.ao = ao;
    return output;
}

@fragment
fn splat_fragment(input: SplatVertex) -> @location(0) vec4<f32> {
    let albedo = apply_material_pattern(
        input.material,
        packed_srgb_to_linear(input.visual),
        input.world,
        input.normal,
        chunk.bounds_voxel_size.w
    );
    let emission = f32(input.visual >> 24u) / 25.5;
    let emits = emission > 0.0 && input.normal.y < -0.5;
    let radiance = select(
        raster_surface_radiance(
            input.world,
            input.normal,
            albedo,
            input.baked,
            input.ao,
            chunk.draw.x,
            chunk.draw.y
        ),
        albedo * emission,
        emits
    );
    return present_radiance(radiance, input.world, input.clip_position.xy);
}
