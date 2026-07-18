// Indexed greedy-surface strategy. Packed vertices are pulled from a storage
// buffer and decoded on the GPU, avoiding a CPU-side expanded vertex copy.

struct SurfaceVertex {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) visual: u32,
    @location(3) @interpolate(flat) baked: f32,
    @location(4) @interpolate(flat) ao: f32,
};

@vertex
fn surface_vertex(@builtin(vertex_index) vertex_index: u32) -> SurfaceVertex {
    let packed = packed_vertices[vertex_index];
    let normal_axis = (packed.z_normal_material >> 16u) & 255u;
    let world = surface_world_position(packed);

    var output: SurfaceVertex;
    output.clip_position = frame.view_projection * vec4<f32>(world, 1.0);
    output.world = world;
    output.normal = normal_for_axis(normal_axis);
    output.visual = packed.visual;
    output.baked = f32(packed.light_ao & 255u) / 15.0;
    output.ao = 1.0 - f32((packed.light_ao >> 8u) & 255u) / 255.0;
    return output;
}

@fragment
fn surface_fragment(input: SurfaceVertex) -> @location(0) vec4<f32> {
    let albedo = packed_srgb_to_linear(input.visual);
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
