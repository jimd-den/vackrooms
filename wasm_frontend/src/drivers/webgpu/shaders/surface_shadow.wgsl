// Depth caster for the indexed Surface strategy. Positions are pulled from
// the same compact vertex buffer used by its visible pass.

@vertex
fn surface_shadow_vertex(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let packed = packed_vertices[vertex_index];
    let world = surface_world_position(packed);
    return hero_shadow.light_view_projection * vec4<f32>(world, 1.0);
}
