//! Direct-light equations shared by the surface and voxel-hit shaders.
//!
//! Rectangular emitters use deterministic 2×2 Gauss quadrature over their
//! physical area. Point lights use the same finite-range inverse-square law
//! without the area integral. This is intentionally compact and diagnosable;
//! shadow visibility is an explicit input owned by the caller.

pub const GLSL: &str = r#"
const float PI = 3.14159265358979323846;
const int LIGHT_POINT = 0;
const int LIGHT_CEILING_PANEL = 1;
const int LIGHT_STRIP = 2;
const int LIGHT_EMERGENCY = 3;

float finiteRangeInverseSquare(float distanceSquared, float range) {
    if (!(range > 0.0) || !(distanceSquared >= 0.0)) return 0.0;
    float rangeSquared = range * range;
    float normalizedSquared = distanceSquared / rangeSquared;
    float window = max(1.0 - normalizedSquared * normalizedSquared, 0.0);
    return (window * window) / max(distanceSquared, 0.04);
}

float receiverCosine(vec3 normal, vec3 surfaceToLight) {
    return max(dot(normal, surfaceToLight), 0.0);
}

float emitterCosine(int kind, vec3 surfaceToLight) {
    if (kind == LIGHT_POINT) return 1.0;
    // Rectangular fixtures lie in XZ and point down. `surfaceToLight.y` is
    // positive exactly when the receiver is in their emitting hemisphere.
    return max(surfaceToLight.y, 0.0);
}

vec3 evaluatePointLight(
    vec3 surfacePosition,
    vec3 normal,
    vec3 lightPosition,
    vec3 lightColor,
    float range,
    float intensity
) {
    vec3 delta = lightPosition - surfacePosition;
    float distanceSquared = dot(delta, delta);
    vec3 surfaceToLight = delta * inversesqrt(max(distanceSquared, 1e-8));
    float geometry = receiverCosine(normal, surfaceToLight);
    return max(lightColor, vec3(0.0)) * max(intensity, 0.0)
        * finiteRangeInverseSquare(distanceSquared, range) * geometry;
}

vec3 evaluateRectangleLight(
    vec3 surfacePosition,
    vec3 normal,
    vec3 lightPosition,
    vec2 halfSize,
    vec3 lightColor,
    float range,
    float intensity,
    int kind
) {
    const float GAUSS = 0.5773502691896258;
    vec2 offsets[4] = vec2[](
        vec2(-GAUSS, -GAUSS), vec2(GAUSS, -GAUSS),
        vec2(-GAUSS,  GAUSS), vec2(GAUSS,  GAUSS)
    );
    vec3 integral = vec3(0.0);
    for (int sampleIndex = 0; sampleIndex < 4; ++sampleIndex) {
        vec2 offset = offsets[sampleIndex] * halfSize;
        vec3 samplePosition = lightPosition + vec3(offset.x, 0.0, offset.y);
        vec3 delta = samplePosition - surfacePosition;
        float distanceSquared = dot(delta, delta);
        vec3 surfaceToLight = delta * inversesqrt(max(distanceSquared, 1e-8));
        float geometry = receiverCosine(normal, surfaceToLight)
            * emitterCosine(kind, surfaceToLight);
        integral += finiteRangeInverseSquare(distanceSquared, range) * geometry;
    }
    float area = 4.0 * max(halfSize.x, 0.0) * max(halfSize.y, 0.0);
    if (!(area > 0.0)) return vec3(0.0);
    return max(lightColor, vec3(0.0)) * max(intensity, 0.0)
        * area * (integral * 0.25);
}

vec3 evaluateSceneLight(
    vec3 surfacePosition,
    vec3 normal,
    vec3 lightPosition,
    vec2 halfSize,
    vec3 lightColor,
    float range,
    float intensity,
    int kind
) {
    if (kind == LIGHT_POINT) {
        return evaluatePointLight(
            surfacePosition, normal, lightPosition, lightColor, range, intensity
        );
    }
    return evaluateRectangleLight(
        surfacePosition, normal, lightPosition, halfSize,
        lightColor, range, intensity, kind
    );
}
"#;
