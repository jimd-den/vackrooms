// The CPU reference keeps its platform-free software rasterizer. WebGPU owns
// presentation: the completed RGBA buffer is uploaded once and sampled by a
// fullscreen triangle, replacing the browser-only Canvas2D blit.

@group(0) @binding(0) var cpu_frame: texture_2d<f32>;
@group(0) @binding(1) var cpu_sampler: sampler;

struct PresentVertex {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn cpu_present_vertex(@builtin(vertex_index) vertex_index: u32) -> PresentVertex {
    var output: PresentVertex;
    let x = select(-1.0, 3.0, vertex_index == 1u);
    let y = select(-1.0, 3.0, vertex_index == 2u);
    output.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return output;
}

@fragment
fn cpu_present_fragment(input: PresentVertex) -> @location(0) vec4<f32> {
    return textureSample(cpu_frame, cpu_sampler, input.uv);
}
