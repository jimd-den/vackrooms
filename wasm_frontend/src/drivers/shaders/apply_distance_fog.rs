//! Homogeneous distance fog shared by every new GPU path.
//!
//! This is deliberately one equation, not an art-effect stack. For an
//! extinction coefficient `sigma` and travelled distance `d`, Beer–Lambert
//! transmittance is `T = exp(-sigma * d)`. Medium radiance fills exactly the
//! energy removed from the surface term.

pub const GLSL: &str = r#"
float fogTransmittance(float distanceToCamera, float fogStart, float sigmaExtinction) {
    float distanceInMedium = max(distanceToCamera - max(fogStart, 0.0), 0.0);
    return exp(-max(sigmaExtinction, 0.0) * distanceInMedium);
}

vec3 applyDistanceFog(
    vec3 surfaceRadiance,
    vec3 mediumRadiance,
    float distanceToCamera,
    float fogStart,
    float sigmaExtinction
) {
    float transmittance = fogTransmittance(distanceToCamera, fogStart, sigmaExtinction);
    return surfaceRadiance * transmittance + mediumRadiance * (1.0 - transmittance);
}
"#;
