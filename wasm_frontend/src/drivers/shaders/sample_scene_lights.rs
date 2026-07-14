//! Decode the complete analytic-fixture list from its RGBA32F texture.

pub const GLSL: &str = r#"
struct SceneLight {
    vec3 position;
    int kind;
    vec3 color;
    float intensity;
    float range;
    vec2 halfSize;
};

vec4 sceneLightTexel(int flatIndex) {
    return texelFetch(
        uSceneLightTexture,
        ivec2(flatIndex % uSceneLightTextureWidth, flatIndex / uSceneLightTextureWidth),
        0
    );
}

SceneLight readSceneLight(int lightIndex) {
    int firstTexel = lightIndex * 3;
    vec4 positionAndKind = sceneLightTexel(firstTexel);
    vec4 colorAndIntensity = sceneLightTexel(firstTexel + 1);
    vec4 rangeAndSize = sceneLightTexel(firstTexel + 2);
    return SceneLight(
        positionAndKind.xyz,
        int(positionAndKind.w + 0.5),
        colorAndIntensity.rgb,
        colorAndIntensity.a,
        rangeAndSize.x,
        rangeAndSize.yz
    );
}
"#;
