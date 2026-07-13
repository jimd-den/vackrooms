//! Hero-light shadow-map program: depth-only render of chunk meshes from
//! the light's view. Both the surface and splat renderers use it — the
//! splat path casts from the retained greedy mesh so caster and visible
//! splats always agree (same grids, same LOD).

pub const VERTEX_SHADER: &str = r#"#version 300 es
layout(location = 0) in vec3 aPosition;
uniform mat4 uLightViewProjection;
uniform vec3 uChunkOrigin;

void main() {
    vec3 worldPosition = uChunkOrigin + aPosition * (1.0 / 1024.0);
    gl_Position = uLightViewProjection * vec4(worldPosition, 1.0);
}
"#;

pub const FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;
void main() {
    // Depth is automatically written to the depth buffer.
}
"#;
