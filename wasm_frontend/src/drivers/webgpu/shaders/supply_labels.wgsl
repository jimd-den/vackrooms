// Supply-item label overlay. A single point record expands into a
// camera-facing quad in the vertex stage, so labels require no transient
// vertex or index allocation.

struct SupplySprite {
    // xyz: world center; w: stacked atlas row.
    position_atlas_row: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> supply_sprites: array<SupplySprite>;
@group(1) @binding(1) var supply_atlas: texture_2d<f32>;
@group(1) @binding(2) var supply_sampler: sampler;

struct SupplyVertex {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) atlas_row: f32,
    @location(2) world: vec3<f32>,
};

fn supply_corner(index: u32) -> vec2<f32> {
    switch index {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>( 1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0, -1.0); }
        case 4u: { return vec2<f32>( 1.0,  1.0); }
        default: { return vec2<f32>(-1.0, 1.0); }
    }
}

@vertex
fn supply_label_vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) sprite_index: u32,
) -> SupplyVertex {
    let sprite = supply_sprites[sprite_index];
    let corner = supply_corner(vertex_index);
    let world = sprite.position_atlas_row.xyz
        + frame.camera_right.xyz * (corner.x * 0.26)
        + vec3<f32>(0.0, corner.y * 0.195, 0.0);

    var output: SupplyVertex;
    output.clip_position = frame.view_projection * vec4<f32>(world, 1.0);
    output.uv = vec2<f32>(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    output.atlas_row = sprite.position_atlas_row.w;
    output.world = world;
    return output;
}

@fragment
fn supply_label_fragment(input: SupplyVertex) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(input.uv.x, (input.uv.y + input.atlas_row) * 0.5);
    let label = textureSample(supply_atlas, supply_sampler, uv);
    if label.a < 0.5 {
        discard;
    }

    let view_distance = length(input.world - frame.camera_position.xyz);
    let fog_distance = max(view_distance - frame.viewport_fov_start.w, 0.0);
    let transmittance = exp(-max(frame.fog_color_density.w, 0.0) * fog_distance);
    let radiance = mix(frame.fog_color_density.xyz, label.rgb * 0.92, transmittance);
    return vec4<f32>(display_color(radiance), 1.0);
}
