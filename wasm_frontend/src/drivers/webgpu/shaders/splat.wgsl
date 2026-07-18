// Instanced face-splat strategy. One compact 16-byte record becomes a quad
// in the vertex stage; face geometry is generated entirely on the GPU.

struct SplatVertex {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) radiance: vec3<f32>,
};

@vertex
fn splat_vertex(
    @builtin(vertex_index) corner_index: u32,
    @builtin(instance_index) face_index: u32,
) -> SplatVertex {
    let packed = packed_faces[face_index];
    let axis = packed.surface & 255u;
    let baked = f32((packed.surface >> 16u) & 255u) / 15.0;
    let ao = 1.0 - f32(packed.surface >> 24u) / 255.0;
    let normal = normal_for_axis(axis);
    let center = splat_face_center(packed);
    let world = splat_face_world_position(packed, corner_index);
    let albedo = packed_srgb_to_linear(packed.visual);
    let emission = f32(packed.visual >> 24u) / 25.5;
    let emits = emission > 0.0 && normal.y < -0.5;

    var output: SplatVertex;
    output.clip_position = frame.view_projection * vec4<f32>(world, 1.0);
    output.world = world;
    output.normal = normal;
    output.radiance = select(
        raster_surface_radiance(center, normal, albedo, baked, ao, chunk.draw.x, chunk.draw.y),
        albedo * emission,
        emits
    );
    return output;
}

@fragment
fn splat_fragment(input: SplatVertex) -> @location(0) vec4<f32> {
    return present_radiance(input.radiance, input.world, input.clip_position.xy);
}
