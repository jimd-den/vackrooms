// Per-chunk binding shared by visible and depth-only raster programs.

struct ChunkUniforms {
    origin_world_size: vec4<f32>,
    bounds_voxel_size: vec4<f32>,
    // x: first clustered light, y: light count, z/w reserved.
    draw: vec4<u32>,
};

@group(1) @binding(0) var<uniform> chunk: ChunkUniforms;
