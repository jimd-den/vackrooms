//! Shared GLSL chunks — snippets spliced into more than one shader so the
//! engine has exactly one definition of each cross-cutting effect.
//!
//! Shader sources are assembled at startup by `format!` concatenation (a
//! few string copies once per program link — nothing per frame). Each chunk
//! is a self-contained set of GLSL functions/consts with no `uniform`
//! declarations, so any stage can include it.

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

/// Material palette lookup, shared by the surface and splat paths so both
/// rasterizers color a wall identically.
pub const MATERIAL_COLOR_GLSL: &str = r#"
vec3 materialColor(float material) {
    // Exact bytes from domain::voxel_grid::MATERIAL_COLORS. Keeping the
    // integer numerators visible makes palette drift reviewable.
    if (material < 1.5) return vec3(221.0, 204.0, 102.0) / 255.0;
    if (material < 2.5) return vec3(153.0, 136.0,  17.0) / 255.0;
    if (material < 3.5) return vec3(204.0, 204.0, 204.0) / 255.0;
    if (material < 4.5) return vec3(255.0, 248.0, 214.0) / 255.0;
    if (material < 5.5) return vec3(136.0,   0.0,   0.0) / 255.0;
    if (material < 6.5) return vec3( 79.0, 154.0,  61.0) / 255.0;
    if (material < 7.5) return vec3( 58.0, 111.0, 184.0) / 255.0;
    if (material < 8.5) return vec3(107.0,  74.0,  47.0) / 255.0;
    if (material < 9.5) return vec3(255.0,  68.0,  51.0) / 255.0;
    if (material < 10.5) return vec3(216.0, 210.0, 192.0) / 255.0;
    if (material < 11.5) return vec3(138.0, 127.0,  92.0) / 255.0;
    if (material < 12.5) return vec3(194.0, 183.0, 107.0) / 255.0;
    if (material < 13.5) return vec3(107.0,  94.0,  34.0) / 255.0;
    if (material < 14.5) return vec3(122.0,  74.0,  38.0) / 255.0;
    if (material < 15.5) return vec3( 46.0,  42.0,  34.0) / 255.0;
    return vec3(159.0, 196.0, 232.0) / 255.0;
}
"#;

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
