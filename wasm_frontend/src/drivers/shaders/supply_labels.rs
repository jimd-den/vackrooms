//! Billboard pass for supply-item product labels.
//!
//! Each sprite is one camera-facing (cylindrical: yaw only, world-up locked)
//! textured quad expanded entirely in the vertex shader from `gl_VertexID`,
//! so the pass needs no vertex buffers at all. The atlas stacks one product
//! logo per row; alpha cutout keeps the pass depth-tested and order-free.

use super::chunks;

/// Matches `application::ports::MAX_SUPPLY_SPRITES`.
pub const MAX_SPRITES: usize = 16;
/// Label quad half-extents in world units (matches the 4:3 label art).
pub const HALF_WIDTH: f32 = 0.26;
pub const HALF_HEIGHT: f32 = 0.195;
/// Atlas rows available to `SupplySprite::atlas_row`.
pub const ATLAS_ROWS: f32 = 2.0;

pub const VERTEX_SHADER: &str = r#"#version 300 es
precision highp float;

uniform mat4 uProjection;
uniform mat4 uView;
// xyz = world center of the label, w = atlas row.
uniform vec4 uSprites[16];
// Ground-plane camera right: the quad turns with the player's yaw and
// never tilts, like a stand-up card.
uniform vec3 uCamRight;
uniform vec2 uHalfSize;

out vec2 vUv;
flat out float vRow;
out vec3 vWorld;

const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

void main() {
    int sprite = gl_VertexID / 6;
    vec2 corner = CORNERS[gl_VertexID % 6];
    vec4 s = uSprites[sprite];
    vec3 world = s.xyz
        + uCamRight * (corner.x * uHalfSize.x)
        + vec3(0.0, 1.0, 0.0) * (corner.y * uHalfSize.y);
    vUv = vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    vRow = s.w;
    vWorld = world;
    gl_Position = uProjection * uView * vec4(world, 1.0);
}
"#;

pub fn fragment_source() -> String {
    let mut source = String::from(
        r#"#version 300 es
precision highp float;

in vec2 vUv;
flat in float vRow;
in vec3 vWorld;

uniform sampler2D uAtlas;
uniform vec3 uCameraPosition;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uFogStart;

out vec4 outColor;
"#,
    );
    source.push_str(chunks::TONE_MAP_GLSL);
    source.push_str(
        r#"
void main() {
    vec2 uv = vec2(vUv.x, (vUv.y + vRow) / 2.0);
    vec4 label = texture(uAtlas, uv);
    if (label.a < 0.5) discard;

    // Printed matter under ambient interior light, then the shared fog.
    vec3 color = label.rgb * 0.92;
    float dist = length(vWorld - uCameraPosition);
    float fog = 1.0 - exp(-uFogDensity * max(dist - uFogStart, 0.0));
    color = mix(color, uFogColor, clamp(fog, 0.0, 1.0));
    outColor = vec4(toneMap(color), 1.0);
}
"#,
    );
    source
}
