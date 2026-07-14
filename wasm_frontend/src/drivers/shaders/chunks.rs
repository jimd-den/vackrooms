//! Shared GLSL chunks — snippets spliced into more than one shader so the
//! engine has exactly one definition of each cross-cutting effect.
//!
//! Shader sources are assembled at startup by `format!` concatenation (a
//! few string copies once per program link — nothing per frame). Each chunk
//! is a self-contained set of GLSL functions/consts with no `uniform`
//! declarations, so any stage can include it.

/// The flashlight spotlight cone — **GLSL mirror of the single spec in
/// `adapters::cpu_splatter::flashlight`** (inner/outer angle, range, facing
/// floor, lamp offset, tint). A tuning change there must be mirrored here.
///
/// Shape: smoothstep between an inner cone (full strength, 11°) and an
/// outer cone (zero, 24°) around the camera forward axis, times a range
/// fade (full to 3 u, gone at 14 u), times a floored Lambert facing term.
pub const SPOT_CONE_GLSL: &str = r#"
// --- flashlight cone (spec: adapters/cpu_splatter/flashlight.rs) ---
const float SPOT_INNER_COS = 0.9816272; // cos(11 deg)
const float SPOT_OUTER_COS = 0.9135455; // cos(24 deg)
const float SPOT_RANGE_FULL = 3.0;
const float SPOT_RANGE_END = 14.0;
const float SPOT_FACING_FLOOR = 0.08;
const vec3  SPOT_TINT = vec3(1.0, 0.96, 0.85);

// The hand-held lamp sits slightly forward of and below the eye.
vec3 spotLampPos(vec3 camPos, vec3 camForward) {
    return camPos + camForward * 0.18 - vec3(0.0, 0.10, 0.0);
}

// Angular falloff: 1 inside the inner cone, 0 outside the outer.
float spotCone(vec3 beam, vec3 camForward) {
    return smoothstep(SPOT_OUTER_COS, SPOT_INNER_COS, dot(beam, camForward));
}

// Distance falloff: full out to RANGE_FULL, zero at RANGE_END.
float spotRange(float dist) {
    return 1.0 - smoothstep(SPOT_RANGE_FULL, SPOT_RANGE_END, dist);
}

// Full beam response at a surface point with normal N. This returns the light
// term; each renderer owns its final material/radiance composition.
vec3 spotBeam(vec3 camPos, vec3 camForward, vec3 surfacePos, vec3 N) {
    vec3 lamp = spotLampPos(camPos, camForward);
    vec3 toSurf = surfacePos - lamp;
    float dist = length(toSurf);
    vec3 beam = toSurf / max(dist, 1e-4);
    float cone = spotCone(beam, camForward);
    float range = spotRange(dist);
    float facing = max(-dot(beam, N), SPOT_FACING_FLOOR);
    return SPOT_TINT * (cone * range * facing);
}
"#;

/// Material palette lookup, shared by the surface and splat paths so both
/// rasterizers color a wall identically.
pub const MATERIAL_COLOR_GLSL: &str = r#"
vec3 materialColor(float material) {
    if (material < 1.5) return vec3(0.87, 0.80, 0.40); // wall
    if (material < 2.5) return vec3(0.60, 0.53, 0.07); // floor
    if (material < 3.5) return vec3(0.80, 0.78, 0.67); // ceiling
    if (material < 4.5) return vec3(1.00, 0.97, 0.78); // fluorescent
    if (material < 5.5) return vec3(0.53, 0.00, 0.00); // red wall
    if (material < 6.5) return vec3(0.31, 0.60, 0.24); // grass
    if (material < 7.5) return vec3(0.18, 0.42, 0.72); // water
    if (material < 8.5) return vec3(0.42, 0.29, 0.18); // tree
    if (material < 9.5) return vec3(1.00, 0.27, 0.20);  // red light
    if (material < 10.5) return vec3(0.85, 0.82, 0.75); // pale arch wall
    if (material < 11.5) return vec3(0.54, 0.50, 0.36); // rough damaged wall
    if (material < 12.5) return vec3(0.76, 0.72, 0.42); // dry shallow carpet
    if (material < 13.5) return vec3(0.42, 0.37, 0.13); // deep wet carpet
    if (material < 14.5) return vec3(0.48, 0.29, 0.15); // sticky red carpet
    if (material < 15.5) return vec3(0.18, 0.16, 0.13); // dark pooled fluid
    return vec3(0.62, 0.77, 0.91);                      // cool glimmer
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
vec3 flareCores(vec3 worldPos, vec3 camPos) {
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
        sum += uCoreColors[i] * uCores[i].w * core * 1.4;
    }
    return sum;
}
"#;
