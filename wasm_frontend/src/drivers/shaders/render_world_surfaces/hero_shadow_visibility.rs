//! Sample the optional hero-light shadow map for one visible surface point.

pub const GLSL: &str = r#"
float shadowVisibility(vec3 worldPosition, vec3 normal, vec3 surfaceToLight) {
    vec4 clip = uLightViewProjection * vec4(worldPosition + normal * 0.02, 1.0);
    vec3 projected = clip.xyz / max(abs(clip.w), 1e-8);
    vec2 uv = projected.xy * 0.5 + 0.5;
    float receiverDepth = projected.z * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))
        || receiverDepth <= 0.0 || receiverDepth >= 1.0) {
        return 1.0;
    }

    vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0));
    float bias = max(0.0015 * (1.0 - dot(normal, surfaceToLight)), 0.0005);
    float visible = 0.0;
    vec2 offsets[4] = vec2[](
        vec2(-0.5, -0.5), vec2(0.5, -0.5),
        vec2(-0.5, 0.5), vec2(0.5, 0.5)
    );
    for (int tap = 0; tap < 4; ++tap) {
        float storedDepth = texture(uShadowMap, uv + offsets[tap] * texel).r;
        visible += receiverDepth - bias <= storedDepth ? 1.0 : 0.0;
    }
    return visible * 0.25;
}
"#;
