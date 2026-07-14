//! Carry surface radiance through fog and flare into encoded display output.

pub const GLSL: &str = r#"
void main() {
    vec3 normal = normalize(vNormal);
    vec3 albedo = srgbToLinear(materialColor(vMaterial));
    float distanceToCamera = length(uCameraPosition - vWorldPosition);
    vec3 radiance = composeSurfaceLighting(normal, albedo);

    radiance = applyDistanceFog(
        radiance, uFogColor, distanceToCamera, uFogStart, uFogDensity
    );
    // Core sprites lie between the camera and the receiver. Attenuate each
    // at its own along-ray distance rather than fogging it as if it were on
    // the receiver surface.
    radiance += flareCoresThroughMedium(
        vWorldPosition, uCameraPosition, uFogStart, uFogDensity
    );

    vec3 displayColor = encodeDisplayColor(radiance);
    if (uDitherEnabled == 1) {
        displayColor += (ign(gl_FragCoord.xy) - 0.5) * (1.0 / 255.0);
    }
    fragColor = vec4(clamp(displayColor, 0.0, 1.0), 1.0);
}
"#;
