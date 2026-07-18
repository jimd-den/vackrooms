// Depth caster for the face-only Splat strategy. No indexed mesh exists:
// each packed face expands to the same four-vertex strip as the visible pass.

@vertex
fn splat_shadow_vertex(
    @builtin(vertex_index) corner_index: u32,
    @builtin(instance_index) face_index: u32,
) -> @builtin(position) vec4<f32> {
    let packed = packed_faces[face_index];
    let world = splat_face_world_position(packed, corner_index);
    return hero_shadow.light_view_projection * vec4<f32>(world, 1.0);
}
