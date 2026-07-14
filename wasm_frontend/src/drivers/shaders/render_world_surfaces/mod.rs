//! Indexed world-surface program.
//!
//! The module name states the use case; the files below state the successive
//! mathematical responsibilities. Shader assembly is the only string work
//! and happens once, when WebGL links the program.

mod reconstruct_surface;
mod sample_static_irradiance;
mod shade_visible_surface;

use super::{apply_distance_fog, chunks, encode_display_color, evaluate_scene_lighting};

pub use reconstruct_surface::VERTEX_SHADER;

const FRAGMENT_INTERFACE: &str = r#"#version 300 es
precision highp float;
precision highp sampler3D;

in vec3 vWorldPosition;
in vec3 vNormal;
flat in float vMaterial;
flat in float vStaticIndirect;
flat in float vAo;

uniform vec3 uCameraPosition;
uniform vec3 uCamForward;
uniform int uFlashlightEnabled;
uniform int uDitherEnabled;
uniform int uBakedLightingEnabled;

uniform sampler3D uLightVolume;
uniform vec3 uChunkOrigin;
uniform float uVoxelSize;

uniform int uLightCount;
uniform vec3 uLightPositions[8];
uniform vec3 uLightColors[8];
// radius, intensity, half-size X, half-size Z
uniform vec4 uLightParams[8];
uniform int uLightKinds[8];

uniform sampler2D uShadowMap;
uniform mat4 uLightViewProjection;
uniform int uShadowedLightIndex;

uniform int uCoreCount;
uniform vec4 uCores[4];
uniform vec3 uCoreColors[4];

uniform int uOutdoor;
uniform vec3 uFogColor;
uniform float uAmbientScale;
uniform float uFogDensity;
uniform float uFogStart;

out vec4 fragColor;
"#;

/// Builds one readable GLSL translation unit from purpose-sized sections.
pub fn fragment_source() -> String {
    [
        FRAGMENT_INTERFACE,
        chunks::MATERIAL_COLOR_GLSL,
        chunks::NOISE_GLSL,
        chunks::SPOT_CONE_GLSL,
        chunks::FLARE_CORES_GLSL,
        encode_display_color::GLSL,
        evaluate_scene_lighting::GLSL,
        apply_distance_fog::GLSL,
        sample_static_irradiance::GLSL,
        shade_visible_surface::GLSL,
    ]
    .concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_has_one_explicit_stage_for_each_rendering_responsibility() {
        let source = fragment_source();
        for symbol in [
            "sampleStaticIrradiance",
            "evaluateRectangleLight",
            "applyDistanceFog",
            "encodeDisplayColor",
        ] {
            assert!(source.contains(symbol), "missing shader section {symbol}");
        }
        assert!(!source.contains("quantize5"));
        assert!(!source.contains("heightFactor"));
    }
}
