//! Reconstruct chunk-local packed vertices as world-space surfaces.

pub const VERTEX_SHADER: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
in float aNormalAxis;
in float aMaterial;
in float aStaticIndirect;
in float aAo;

uniform mat4 uProjection;
uniform mat4 uView;
uniform vec3 uChunkOrigin;

out vec3 vWorldPosition;
out vec3 vNormal;
flat out float vMaterial;
flat out float vStaticIndirect;
flat out float vAo;

vec3 normalForAxis(float axis) {
    if (axis < 0.5) return vec3(0.0, 1.0, 0.0);
    if (axis < 1.5) return vec3(0.0, -1.0, 0.0);
    if (axis < 2.5) return vec3(0.0, 0.0, -1.0);
    if (axis < 3.5) return vec3(0.0, 0.0, 1.0);
    if (axis < 4.5) return vec3(1.0, 0.0, 0.0);
    return vec3(-1.0, 0.0, 0.0);
}

void main() {
    vec3 worldPosition = uChunkOrigin + aPosition * (1.0 / 1024.0);
    vWorldPosition = worldPosition;
    vNormal = normalForAxis(aNormalAxis);
    vMaterial = aMaterial;
    vStaticIndirect = aStaticIndirect;
    vAo = aAo;
    gl_Position = uProjection * uView * vec4(worldPosition, 1.0);
}
"#;
