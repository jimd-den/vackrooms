// Receiver-side hero shadow sampling shared by Surface and Splat.

@group(2) @binding(0) var<uniform> hero_shadow: HeroShadowUniforms;
@group(2) @binding(1) var hero_shadow_depth: texture_depth_2d;
@group(2) @binding(2) var hero_shadow_sampler: sampler_comparison;

fn hero_shadow_visibility(
    world: vec3<f32>,
    normal: vec3<f32>,
    surface_to_light: vec3<f32>,
) -> f32 {
    if hero_shadow.options.x == 0u {
        return 1.0;
    }
    let clip = hero_shadow.light_view_projection
        * vec4<f32>(world + normal * 0.0005, 1.0);
    let projected = clip.xyz / max(abs(clip.w), 1e-8);
    // WebGPU texture coordinates start at the top-left; NDC Y points up.
    let uv = vec2<f32>(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
    let receiver_depth = projected.z;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))
        || receiver_depth <= 0.0 || receiver_depth >= 1.0 {
        return 1.0;
    }

    let slope = 1.0 - clamp(dot(normal, surface_to_light), 0.0, 1.0);
    let bias = max(hero_shadow.sampling.z, hero_shadow.sampling.w * slope);
    let reference_depth = receiver_depth - bias;
    if hero_shadow.options.z <= 1u {
        return textureSampleCompareLevel(
            hero_shadow_depth,
            hero_shadow_sampler,
            uv,
            reference_depth
        );
    }

    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5,  0.5)
    );
    var visibility = 0.0;
    for (var tap = 0u; tap < 4u; tap += 1u) {
        visibility += textureSampleCompareLevel(
            hero_shadow_depth,
            hero_shadow_sampler,
            uv + offsets[tap] * hero_shadow.sampling.xy,
            reference_depth
        );
    }
    return visibility * 0.25;
}

fn raster_surface_radiance(
    world: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    baked: f32,
    ao: f32,
    light_first: u32,
    light_count: u32,
) -> vec3<f32> {
    var analytic_direct = vec3<f32>(0.0);
    let safe_count = min(
        light_count,
        frame.render_options.x - min(light_first, frame.render_options.x)
    );
    for (var local = 0u; local < safe_count; local += 1u) {
        let light_index = light_first + local;
        let light = lights[light_index];
        if light.half_size_kind.w < 0.5 {
            continue;
        }
        var contribution = evaluate_scene_light(world, normal, light);
        if hero_shadow.options.x != 0u && light_index == hero_shadow.options.y {
            let delta = light.position_radius.xyz - world;
            let surface_to_light = delta / sqrt(max(dot(delta, delta), 1e-8));
            contribution *= hero_shadow_visibility(world, normal, surface_to_light);
        }
        analytic_direct += contribution;
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
