// CPU/WGSL contract shared by caster and receiver pipelines.

struct HeroShadowUniforms {
    light_view_projection: mat4x4<f32>,
    // xy: inverse map extent; z: minimum bias; w: slope-scaled bias.
    sampling: vec4<f32>,
    // x: enabled; y: absolute GPU-light index; z: taps; w: map extent.
    options: vec4<u32>,
};
