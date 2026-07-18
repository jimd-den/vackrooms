//! Define which visible material faces emit and their linear radiance.

pub const GLSL: &str = r#"
bool isEmissiveMaterial(float material) {
    return abs(material - 4.0) < 0.1
        || abs(material - 9.0) < 0.1
        || abs(material - 16.0) < 0.1;
}

bool isDownwardEmittingFace(vec3 normal) {
    return normal.y < -0.5;
}

vec3 emittedRadiance(float material, vec3 albedo, vec3 normal) {
    if (!isDownwardEmittingFace(normal)) return vec3(0.0);
    if (abs(material - 4.0) < 0.1) return albedo * 10.0;
    if (abs(material - 9.0) < 0.1) return albedo * 8.0;
    if (abs(material - 16.0) < 0.1) return albedo * 0.9;
    return vec3(0.0);
}
"#;
