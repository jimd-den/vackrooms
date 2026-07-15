//! Construct one normalized world-space camera ray from the fullscreen UV.

pub const GLSL: &str = r#"
vec3 constructCameraRay() {
    vec2 ndc = vUv * 2.0 - 1.0;
    return normalize(
        uCamRight * (ndc.x * uAspect * uFovTan)
        + uCamUp * (ndc.y * uFovTan)
        + uCamForward
    );
}
"#;
