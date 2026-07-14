//! One linear-HDR to canvas conversion for surfaces and raymarching.

pub const GLSL: &str = r#"
vec3 srgbToLinear(vec3 encoded) {
    vec3 lowPart = encoded / 12.92;
    vec3 highPart = pow((encoded + 0.055) / 1.055, vec3(2.4));
    return mix(lowPart, highPart, step(vec3(0.04045), encoded));
}

vec3 linearToSrgb(vec3 linearColor) {
    linearColor = max(linearColor, vec3(0.0));
    vec3 lowPart = linearColor * 12.92;
    vec3 highPart = 1.055 * pow(linearColor, vec3(1.0 / 2.4)) - 0.055;
    return mix(lowPart, highPart, step(vec3(0.0031308), linearColor));
}

vec3 encodeDisplayColor(vec3 radiance) {
    // A monotone, channel-independent Reinhard curve. Lighting and fog are
    // composed before this point in linear space.
    vec3 mapped = max(radiance, vec3(0.0)) / (vec3(1.0) + max(radiance, vec3(0.0)));
    return linearToSrgb(mapped);
}
"#;
