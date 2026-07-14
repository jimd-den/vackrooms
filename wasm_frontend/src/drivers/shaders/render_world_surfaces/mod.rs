//! Indexed world-surface program.
//!
//! The module name states the use case; the files below state the successive
//! mathematical responsibilities. Shader assembly is the only string work
//! and happens once, when WebGL links the program.

mod compose_surface_lighting;
mod hero_shadow_visibility;
mod present_surface_frame;
mod reconstruct_surface;
mod sample_static_irradiance;
mod surface_material_semantics;

use super::{
    apply_distance_fog, chunks, encode_display_color, evaluate_scene_lighting, sample_scene_lights,
};

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
uniform vec3 uLightVolumeOrigin;
uniform float uVoxelSize;

uniform int uLightCount;
uniform int uLightFirst;
uniform sampler2D uSceneLightTexture;
uniform int uSceneLightTextureWidth;

uniform int uDynamicLightCount;
uniform vec4 uDynamicPosRadius[4];
uniform vec4 uDynamicColorIntensity[4];

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
        sample_scene_lights::GLSL,
        apply_distance_fog::GLSL,
        sample_static_irradiance::GLSL,
        surface_material_semantics::GLSL,
        hero_shadow_visibility::GLSL,
        compose_surface_lighting::GLSL,
        present_surface_frame::GLSL,
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
        assert!(source.contains("isDownwardEmittingFace"));
    }

    #[test]
    fn surface_pipeline_is_assembled_from_named_responsibilities() {
        let source = fragment_source();
        let stages = [
            "bool isEmissiveMaterial(",
            "float shadowVisibility(",
            "vec3 composeSurfaceLighting(",
            "void main()",
        ];

        let mut previous = 0;
        for stage in stages {
            let position = source.find(stage).expect("missing shader responsibility");
            assert!(
                position >= previous,
                "shader responsibilities are out of order"
            );
            previous = position;
        }
        assert_eq!(source.matches("void main()").count(), 1);
    }
}
