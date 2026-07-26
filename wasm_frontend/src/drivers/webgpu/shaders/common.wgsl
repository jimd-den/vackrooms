// Shared radiometric vocabulary for all WebGPU render strategies.
//
// Bind group 0 is deliberately identical for every pipeline. A frame is a
// snapshot of plain application data; shaders never read browser globals.

struct FrameUniforms {
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_forward: vec4<f32>,
    fog_color_density: vec4<f32>,
    sky_color_ambient: vec4<f32>,
    viewport_fov_start: vec4<f32>,
    // x: uploaded light count, y: flashlight, z: dither, w: trace budget.
    render_options: vec4<u32>,
    // x: outdoor environment, y: cached diffuse-fill enabled, z/w reserved.
    lighting_options: vec4<u32>,
};

struct GpuLight {
    position_radius: vec4<f32>,
    color_intensity: vec4<f32>,
    // xy: rectangle half-size, z: LightKind, w: enabled (0/1).
    half_size_kind: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;

fn normal_for_axis(axis: u32) -> vec3<f32> {
    switch axis {
        case 0u: { return vec3<f32>(0.0, 1.0, 0.0); }
        case 1u: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 0.0, -1.0); }
        case 3u: { return vec3<f32>(0.0, 0.0, 1.0); }
        case 4u: { return vec3<f32>(1.0, 0.0, 0.0); }
        default: { return vec3<f32>(-1.0, 0.0, 0.0); }
    }
}

fn srgb_channel_to_linear(encoded: f32) -> f32 {
    if encoded <= 0.04045 {
        return encoded / 12.92;
    }
    return pow((encoded + 0.055) / 1.055, 2.4);
}

fn packed_srgb_to_linear(packed: u32) -> vec3<f32> {
    let encoded = vec3<f32>(
        f32((packed >> 16u) & 255u),
        f32((packed >> 8u) & 255u),
        f32(packed & 255u)
    ) / 255.0;
    return vec3<f32>(
        srgb_channel_to_linear(encoded.x),
        srgb_channel_to_linear(encoded.y),
        srgb_channel_to_linear(encoded.z)
    );
}

fn hash_screen(pixel: vec2<f32>) -> f32 {
    return fract(sin(dot(pixel, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn material_hash(cell: vec2<f32>) -> f32 {
    return fract(sin(dot(cell, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn material_noise(point: vec2<f32>) -> f32 {
    let cell = floor(point);
    let f = fract(point);
    let u = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let a = material_hash(cell);
    let b = material_hash(cell + vec2<f32>(1.0, 0.0));
    let c = material_hash(cell + vec2<f32>(0.0, 1.0));
    let d = material_hash(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn apply_material_pattern(
    material: u32,
    albedo: vec3<f32>,
    world: vec3<f32>,
    normal: vec3<f32>,
    sample_footprint: f32,
) -> vec3<f32> {
    let carpet = material == 2u || material == 12u || material == 13u
        || material == 14u || material == 25u;
    if carpet && normal.y > 0.5 {
        let broad = material_noise(world.xz * 0.18);
        let fibers = material_noise(world.xz * 1.15);
        let detail = clamp(1.15 - sample_footprint * 0.32, 0.20, 1.0);
        let wear = 0.88 + 0.18 * broad + (fibers - 0.5) * 0.055 * detail;
        return albedo * wear;
    }
    let wallpaper = material == 1u || material == 24u;
    if wallpaper && abs(normal.y) < 0.5 {
        let along = select(world.x, world.z, abs(normal.x) > 0.5);
        let repeat = fract(along / 1.35);
        let center = abs(repeat - 0.5);
        let stem = 1.0 - smoothstep(0.035, 0.085, center);
        let vine = 0.5 + 0.5 * sin(world.y * 3.1 + sin(along * 2.4) * 0.8);
        let medallion = pow(max(cos((repeat - 0.5) * 6.2831853), 0.0), 6.0)
            * pow(max(cos((world.y - 0.35) * 3.1415926), 0.0), 4.0);
        let age = select(0.0, 1.0, material == 24u);
        let stain = material_noise(vec2<f32>(along * 0.09, world.y * 0.16));
        let motif = stem * (0.45 + 0.55 * vine) + medallion * 0.55;
        let factor = 1.025 - motif * (0.12 + age * 0.035) - age * stain * 0.07;
        return albedo * factor;
    }
    return albedo;
}

const PI: f32 = 3.14159265358979323846;
const LIGHT_POINT: u32 = 0u;
const MINIMUM_SCENE_LIGHT_DISTANCE: f32 = 0.2;
const INDOOR_AMBIENT_IRRADIANCE: vec3<f32> = vec3<f32>(0.045, 0.040, 0.024);
const OUTDOOR_AMBIENT_IRRADIANCE: vec3<f32> = vec3<f32>(0.32, 0.38, 0.48);
const CACHED_DIFFUSE_FILL_GAIN: f32 = 0.12;

// Mirrors application::rendering::FLASHLIGHT_SPEC; a Rust source-contract
// test prevents either side from changing independently.
const FLASHLIGHT_INNER_COSINE: f32 = 0.981627166;
const FLASHLIGHT_OUTER_COSINE: f32 = 0.913545430;
const FLASHLIGHT_RANGE: f32 = 14.0;
const FLASHLIGHT_INTENSITY: f32 = 32.0;
const FLASHLIGHT_MINIMUM_DISTANCE: f32 = 0.25;
const FLASHLIGHT_FORWARD_OFFSET: f32 = 0.18;
const FLASHLIGHT_DOWNWARD_OFFSET: f32 = 0.10;
const FLASHLIGHT_TINT: vec3<f32> = vec3<f32>(1.0, 0.91, 0.72);

fn flashlight_irradiance(world: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    if frame.render_options.y == 0u {
        return vec3<f32>(0.0);
    }
    let lamp_position = frame.camera_position.xyz
        + frame.camera_forward.xyz * FLASHLIGHT_FORWARD_OFFSET
        - vec3<f32>(0.0, FLASHLIGHT_DOWNWARD_OFFSET, 0.0);
    let lamp_to_receiver = world - lamp_position;
    let distance_squared = dot(lamp_to_receiver, lamp_to_receiver);
    if distance_squared <= 1e-8 {
        return vec3<f32>(0.0);
    }
    let direction = lamp_to_receiver / sqrt(distance_squared);
    let cone = smoothstep(
        FLASHLIGHT_OUTER_COSINE,
        FLASHLIGHT_INNER_COSINE,
        dot(direction, frame.camera_forward.xyz)
    );
    let receiver = receiver_cosine(normal, -direction);
    let attenuation = finite_range_inverse_square(
        distance_squared,
        FLASHLIGHT_RANGE,
        FLASHLIGHT_MINIMUM_DISTANCE
    );
    return FLASHLIGHT_TINT * FLASHLIGHT_INTENSITY * cone * receiver * attenuation;
}

fn finite_range_inverse_square(
    distance_squared: f32,
    range: f32,
    minimum_distance: f32,
) -> f32 {
    if range <= 0.0 || minimum_distance <= 0.0 || distance_squared < 0.0 {
        return 0.0;
    }
    let range_squared = range * range;
    let normalized_squared = distance_squared / range_squared;
    let window = max(1.0 - normalized_squared * normalized_squared, 0.0);
    return (window * window)
        / max(distance_squared, minimum_distance * minimum_distance);
}

fn receiver_cosine(normal: vec3<f32>, surface_to_light: vec3<f32>) -> f32 {
    return max(dot(normal, surface_to_light), 0.0);
}

fn emitter_cosine(kind: u32, surface_to_light: vec3<f32>) -> f32 {
    if kind == LIGHT_POINT {
        return 1.0;
    }
    // Rectangle fixtures lie in XZ and point down. A receiver below the
    // fixture has a positive receiver-to-emitter Y direction.
    return max(surface_to_light.y, 0.0);
}

/// Radiance contribution of one point-like endpoint before visibility.
/// Ray visibility can multiply this result by an exact finite-segment test
/// without reimplementing attenuation or receiver geometry.
fn point_light_sample_irradiance(
    world: vec3<f32>,
    normal: vec3<f32>,
    light: GpuLight,
    sample_position: vec3<f32>,
) -> vec3<f32> {
    let delta = sample_position - world;
    let distance_squared = dot(delta, delta);
    let surface_to_light = delta / sqrt(max(distance_squared, 1e-8));
    let geometry = receiver_cosine(normal, surface_to_light);
    return max(light.color_intensity.xyz, vec3<f32>(0.0))
        * max(light.color_intensity.w, 0.0)
        * finite_range_inverse_square(
            distance_squared,
            light.position_radius.w,
            MINIMUM_SCENE_LIGHT_DISTANCE
        )
        * geometry;
}

fn evaluate_point_light(
    world: vec3<f32>,
    normal: vec3<f32>,
    light: GpuLight,
) -> vec3<f32> {
    return point_light_sample_irradiance(
        world,
        normal,
        light,
        light.position_radius.xyz
    );
}

/// Position of one deterministic 2x2 Gauss endpoint on an XZ emitter.
fn rectangle_gauss_sample_position(light: GpuLight, sample_index: u32) -> vec3<f32> {
    let gauss = 0.5773502691896258;
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-gauss, -gauss),
        vec2<f32>( gauss, -gauss),
        vec2<f32>(-gauss,  gauss),
        vec2<f32>( gauss,  gauss)
    );
    let offset = offsets[min(sample_index, 3u)]
        * max(light.half_size_kind.xy, vec2<f32>(0.0));
    return light.position_radius.xyz + vec3<f32>(offset.x, 0.0, offset.y);
}

/// Scalar geometry/attenuation term for one rectangle quadrature endpoint.
fn rectangle_light_sample_weight(
    world: vec3<f32>,
    normal: vec3<f32>,
    light: GpuLight,
    kind: u32,
    sample_position: vec3<f32>,
) -> f32 {
    let delta = sample_position - world;
    let distance_squared = dot(delta, delta);
    let surface_to_light = delta / sqrt(max(distance_squared, 1e-8));
    return finite_range_inverse_square(
        distance_squared,
        light.position_radius.w,
        MINIMUM_SCENE_LIGHT_DISTANCE
    )
        * receiver_cosine(normal, surface_to_light)
        * emitter_cosine(kind, surface_to_light);
}

/// Converts a (possibly visibility-weighted) quadrature integral into RGB.
fn rectangle_light_irradiance_from_integral(
    light: GpuLight,
    integral: f32,
) -> vec3<f32> {
    let half_size = max(light.half_size_kind.xy, vec2<f32>(0.0));
    let area = 4.0 * half_size.x * half_size.y;
    if area <= 0.0 {
        return vec3<f32>(0.0);
    }
    return max(light.color_intensity.xyz, vec3<f32>(0.0))
        * max(light.color_intensity.w, 0.0)
        * area
        * integral
        * 0.25;
}

fn evaluate_rectangle_light(
    world: vec3<f32>,
    normal: vec3<f32>,
    light: GpuLight,
    kind: u32,
) -> vec3<f32> {
    let half_size = max(light.half_size_kind.xy, vec2<f32>(0.0));
    let area = 4.0 * half_size.x * half_size.y;
    if area <= 0.0 {
        return vec3<f32>(0.0);
    }

    var integral = 0.0;
    for (var sample_index = 0u; sample_index < 4u; sample_index += 1u) {
        let sample_position = rectangle_gauss_sample_position(light, sample_index);
        integral += rectangle_light_sample_weight(
            world,
            normal,
            light,
            kind,
            sample_position
        );
    }
    return rectangle_light_irradiance_from_integral(light, integral);
}

fn evaluate_scene_light(
    world: vec3<f32>,
    normal: vec3<f32>,
    light: GpuLight,
) -> vec3<f32> {
    let kind = u32(max(light.half_size_kind.z, 0.0) + 0.5);
    if kind == LIGHT_POINT {
        return evaluate_point_light(world, normal, light);
    }
    return evaluate_rectangle_light(world, normal, light, kind);
}

fn ambient_irradiance() -> vec3<f32> {
    let base = select(
        INDOOR_AMBIENT_IRRADIANCE,
        OUTDOOR_AMBIENT_IRRADIANCE,
        frame.lighting_options.x != 0u
    );
    return base * max(frame.sky_color_ambient.w, 0.0);
}

fn cached_static_irradiance(baked: f32) -> vec3<f32> {
    if frame.lighting_options.y == 0u {
        return vec3<f32>(0.0);
    }
    return vec3<f32>(clamp(baked, 0.0, 1.0) * CACHED_DIFFUSE_FILL_GAIN);
}

fn surface_radiance_from_analytic_direct(
    world: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    baked: f32,
    ao: f32,
    analytic_direct: vec3<f32>,
) -> vec3<f32> {
    let ambient_visibility = mix(0.62, 1.0, clamp(ao, 0.0, 1.0));
    let ambient = ambient_irradiance() * ambient_visibility;
    let cached_static = cached_static_irradiance(baked);
    let flashlight = flashlight_irradiance(world, normal);
    let irradiance = ambient + cached_static + analytic_direct + flashlight;
    return albedo * irradiance * (1.0 / PI);
}

fn surface_radiance(
    world: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    baked: f32,
    ao: f32,
    light_first: u32,
    light_count: u32,
) -> vec3<f32> {
    var analytic_direct = vec3<f32>(0.0);
    let safe_count = min(light_count, frame.render_options.x - min(light_first, frame.render_options.x));
    for (var local = 0u; local < safe_count; local += 1u) {
        let light = lights[light_first + local];
        if light.half_size_kind.w < 0.5 {
            continue;
        }
        analytic_direct += evaluate_scene_light(world, normal, light);
    }
    return surface_radiance_from_analytic_direct(
        world,
        normal,
        albedo,
        baked,
        ao,
        analytic_direct
    );
}

fn display_color(linear_radiance: vec3<f32>) -> vec3<f32> {
    let mapped = max(linear_radiance, vec3<f32>(0.0))
        / (vec3<f32>(1.0) + max(linear_radiance, vec3<f32>(0.0)));
    let low = mapped * 12.92;
    let high = 1.055 * pow(mapped, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(high, low, mapped <= vec3<f32>(0.0031308));
}

fn present_radiance(
    linear_radiance: vec3<f32>,
    world: vec3<f32>,
    pixel: vec2<f32>,
) -> vec4<f32> {
    let view_distance = length(world - frame.camera_position.xyz);
    let fog_distance = max(view_distance - frame.viewport_fov_start.w, 0.0);
    let transmittance = exp(-max(frame.fog_color_density.w, 0.0) * fog_distance);
    var composed = mix(frame.fog_color_density.xyz, linear_radiance, transmittance);
    var encoded = display_color(composed);
    if frame.render_options.z != 0u {
        encoded += vec3<f32>((hash_screen(pixel) - 0.5) / 128.0);
    }
    return vec4<f32>(clamp(encoded, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
