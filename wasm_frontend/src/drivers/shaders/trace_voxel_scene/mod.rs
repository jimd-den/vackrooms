//! Fullscreen voxel-scene tracing program.
//!
//! The correctness path is exact finest-cell DDA. The optional empty-space
//! optimization swaps only the stepping policy: both paths use the same
//! stateless SVO lookup, hit record, lighting, fog, and display transform.

mod decode_voxel_atlas;
mod intersect_voxel_scene;
mod shade_voxel_hit;
mod trace_direct_light_visibility;

use super::{
    apply_distance_fog, chunks, encode_display_color, evaluate_scene_lighting, sample_scene_lights,
};

pub const VERTEX_SHADER: &str = r#"#version 300 es
in vec2 position;
out vec2 vUv;
void main() {
    vUv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"#;

const FRAGMENT_INTERFACE: &str = r#"#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;

in vec2 vUv;
out vec4 fragColor;

uniform vec3 uCameraPosition;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uAspect;
uniform float uFovTan;

uniform int uFlashlightEnabled;
uniform int uFrontToBackEnabled;
uniform int uEmptySpaceSkipEnabled;
uniform int uDirectVisibilityEnabled;
uniform int uDitherEnabled;

uniform int uOutdoor;
uniform vec3 uSkyColor;
uniform vec3 uFogColor;
uniform float uAmbientScale;
uniform float uFogDensity;
uniform float uFogStart;

uniform sampler2D uSceneLightTexture;
uniform int uSceneLightTextureWidth;

uniform int uDynamicLightCount;
uniform vec4 uDynamicPosRadius[4];
uniform vec4 uDynamicColorIntensity[4];

uniform usampler2D uNodeTexture;
uniform int uNumChunks;
uniform vec3 uChunkOrigins[25];
uniform int uChunkRootIndices[25];
uniform float uChunkWorldSizes[25];
uniform float uChunkVoxelSizes[25];
uniform int uChunkDepths[25];
uniform int uChunkLightFirst[25];
uniform int uChunkLightCounts[25];
"#;

pub fn fragment_source() -> String {
    [
        FRAGMENT_INTERFACE,
        chunks::NOISE_GLSL,
        chunks::SPOT_CONE_GLSL,
        encode_display_color::GLSL,
        evaluate_scene_lighting::GLSL,
        sample_scene_lights::GLSL,
        apply_distance_fog::GLSL,
        decode_voxel_atlas::GLSL,
        intersect_voxel_scene::GLSL,
        trace_direct_light_visibility::GLSL,
        shade_voxel_hit::GLSL,
    ]
    .concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_and_reference_traversal_are_explicitly_separate() {
        let source = fragment_source();
        assert!(source.contains("traceChunkDda"));
        assert!(source.contains("traceChunkSkippingEmptyLeaves"));
        assert!(source.contains("RayBoxHit refinementBox"));
        assert!(source.contains("uEmptySpaceSkipEnabled"));
        assert!(source.contains("directSampleIsVisible"));
        assert!(!source.contains("worldSize > 20.0"));
        assert!(!source.contains("vec2 cell_center"));
    }

    #[test]
    fn fog_is_not_multiplied_by_a_vignette() {
        let source = fragment_source();
        assert!(source.contains("applyDistanceFog"));
        assert!(source.contains("isDownwardEmittingFace"));
        assert!(!source.contains("vignette"));
    }
}
