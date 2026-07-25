//! Shared GLSL chunks — snippets spliced into more than one shader so the
//! engine has exactly one definition of each cross-cutting effect.
//!
//! Shader sources are assembled at startup by `format!` concatenation (a
//! few string copies once per program link — nothing per frame). Each chunk
//! is a self-contained set of GLSL functions/consts with no `uniform`
//! declarations, so any stage can include it.

use std::fmt::Write;

use vackrooms::adapters::material_palette::MATERIAL_VISUALS;

/// The flashlight spotlight cone — **GLSL mirror of the single spec in
/// `adapters::cpu_splatter::flashlight`** (inner/outer angle, finite-range
/// inverse-square transport, lamp offset, and tint). A tuning change there
/// must be mirrored here.
///
/// Shape: smoothstep between an inner cone (full strength, 11°) and an
/// outer cone (zero, 24°) around the camera forward axis. Transport uses a
/// compact-support inverse-square law and a strict Lambert receiver cosine.
pub const SPOT_CONE_GLSL: &str = r#"
// --- flashlight cone (spec: adapters/cpu_splatter/flashlight.rs) ---
const float SPOT_INNER_COS = 0.9816272; // cos(11 deg)
const float SPOT_OUTER_COS = 0.9135455; // cos(24 deg)
const float SPOT_RANGE_END = 14.0;
const float SPOT_INTENSITY = 32.0;
const float SPOT_MIN_DISTANCE = 0.25;
const float SPOT_INV_PI = 0.3183098861837907;
const vec3  SPOT_TINT = vec3(1.0, 0.91, 0.72);

// The hand-held lamp sits slightly forward of and below the eye.
vec3 spotLampPos(vec3 camPos, vec3 camForward) {
    return camPos + camForward * 0.18 - vec3(0.0, 0.10, 0.0);
}

// Angular falloff: 1 inside the inner cone, 0 outside the outer.
float spotCone(vec3 beam, vec3 camForward) {
    return smoothstep(SPOT_OUTER_COS, SPOT_INNER_COS, dot(beam, camForward));
}

// Finite-range inverse-square transport. The compact window and its first
// derivative both reach zero at RANGE_END, so the beam has no hard rim.
float spotAttenuation(float dist) {
    if (!(dist >= 0.0) || dist >= SPOT_RANGE_END) return 0.0;
    float normalizedSquared = (dist * dist) / (SPOT_RANGE_END * SPOT_RANGE_END);
    float window = max(1.0 - normalizedSquared * normalizedSquared, 0.0);
    float minimumSquared = SPOT_MIN_DISTANCE * SPOT_MIN_DISTANCE;
    return (window * window) / max(dist * dist, minimumSquared);
}

// Lambertian reflected-radiance factor at a surface point with normal N.
// The CPU contract evaluates irradiance first and multiplies albedo / PI;
// folding 1/PI here preserves that exact composition at GPU call sites.
vec3 spotBeam(vec3 camPos, vec3 camForward, vec3 surfacePos, vec3 N) {
    vec3 lamp = spotLampPos(camPos, camForward);
    vec3 toSurf = surfacePos - lamp;
    float dist = length(toSurf);
    vec3 beam = toSurf / max(dist, 1e-4);
    float cone = spotCone(beam, camForward);
    float attenuation = spotAttenuation(dist);
    float facing = max(-dot(beam, N), 0.0);
    return SPOT_TINT * (SPOT_INTENSITY * SPOT_INV_PI * cone * attenuation * facing);
}
"#;

/// Builds the palette lookup shared by both WebGL rasterizers. Shader source
/// is assembled once at startup, so deriving it here removes a hand-maintained
/// GLSL mirror without adding frame-time work.
pub fn material_color_glsl() -> String {
    let mut source = String::from("vec3 materialColor(float material) {\n");
    for (material, visual) in MATERIAL_VISUALS.iter().enumerate() {
        let red = (visual.color >> 16) & 0xff;
        let green = (visual.color >> 8) & 0xff;
        let blue = visual.color & 0xff;
        writeln!(
            source,
            "    if (material < {:.1}) return vec3({red}.0, {green}.0, {blue}.0) / 255.0;",
            material as f32 + 0.5,
        )
        .expect("writing to a String cannot fail");
    }
    source.push_str("    return vec3(0.0);\n}\n");
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_glsl_contains_every_palette_entry() {
        let glsl = material_color_glsl();
        for visual in MATERIAL_VISUALS {
            let red = (visual.color >> 16) & 0xff;
            let green = (visual.color >> 8) & 0xff;
            let blue = visual.color & 0xff;
            assert!(glsl.contains(&format!("vec3({red}.0, {green}.0, {blue}.0)")));
        }
    }
}

/// World-hash and interleaved-gradient-noise helpers.
///
/// OPTIMIZATION: IGN is two `fract`s and a `dot` — far cheaper than a
/// sin-hash — and is used to break up 8-bit banding in fog gradients,
/// which is especially visible at reduced internal resolutions.
pub const NOISE_GLSL: &str = r#"
float hash3D(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

float ign(vec2 p) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(p, magic.xy)));
}
"#;

/// Reinhard tone map + gamma, shared by the surface and splat fragments.
pub const TONE_MAP_GLSL: &str = r#"
vec3 toneMap(vec3 color) {
    color = color / (color + vec3(1.0));
    return pow(color, vec3(1.0 / 2.2));
}
"#;

/// Flare-core glow sprites: additive embers occluded by any surface nearer
/// than the flare along the fragment's view ray — no x-ray dots. Shared by
/// the surface and splat fragments; requires the `uCoreCount` / `uCores` /
/// `uCoreColors` uniforms to be declared before this chunk is spliced in
/// (both programs declare them identically).
pub const FLARE_CORES_GLSL: &str = r#"
vec3 flareCoresThroughMedium(
    vec3 worldPos,
    vec3 camPos,
    float fogStart,
    float sigmaExtinction
) {
    vec3 toFrag = worldPos - camPos;
    float fragDist = length(toFrag);
    vec3 rd = toFrag / max(fragDist, 1e-4);
    vec3 sum = vec3(0.0);
    for (int i = 0; i < uCoreCount; i++) {
        vec3 toLight = uCores[i].xyz - camPos;
        float along = dot(toLight, rd);
        if (along <= 0.05 || along >= fragDist) continue;
        float perp = length(toLight - rd * along);
        float coreSize = 0.06 + along * 0.004;
        float core = 1.0 - smoothstep(coreSize * 0.4, coreSize, perp);
        float mediumDistance = max(along - max(fogStart, 0.0), 0.0);
        float transmittance = exp(-max(sigmaExtinction, 0.0) * mediumDistance);
        sum += uCoreColors[i] * uCores[i].w * core * 1.4 * transmittance;
    }
    return sum;
}

vec3 flareCores(vec3 worldPos, vec3 camPos) {
    return flareCoresThroughMedium(worldPos, camPos, 0.0, 0.0);
}
"#;
