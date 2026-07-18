// SVO raymarch strategy. Nodes remain in the canonical four-u32 encoding;
// point lookup is stateless, and optional empty-leaf stepping changes only
// how far an empty sample advances.

struct RayChunk {
    origin_world_size: vec4<f32>,
    voxel_depth: vec4<f32>,
    // x: root node index in the merged atlas; y: stable source ordinal.
    indices: vec4<u32>,
};

struct RayScene {
    chunk_count: u32,
    policy_flags: u32,
    _reserved: vec2<u32>,
};

const RAY_POLICY_EMPTY_SPACE_SKIP: u32 = 1u;
const RAY_POLICY_FRONT_TO_BACK: u32 = 2u;
const RAY_POLICY_DIRECT_LIGHT_VISIBILITY: u32 = 4u;
const TRACE_INFINITY: f32 = 1e30;
const TRACE_MIN_TIE_EPSILON: f32 = 1e-7;
const DIRECT_VISIBILITY_TRACE_BUDGET: u32 = 4096u;

@group(1) @binding(0) var<storage, read> atlas_words: array<u32>;
@group(1) @binding(1) var<storage, read> ray_chunks: array<RayChunk>;
@group(1) @binding(2) var<uniform> ray_scene: RayScene;

struct FullscreenVertex {
    @builtin(position) clip_position: vec4<f32>,
};

struct BoxHit {
    hit: bool,
    entry: f32,
    exit: f32,
    normal: vec3<f32>,
};

struct VoxelLeaf {
    material: u32,
    color: u32,
    light_word: u32,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
};

struct VoxelHit {
    hit: bool,
    distance: f32,
    normal: vec3<f32>,
    material: u32,
    color: u32,
    light_word: u32,
    voxel_size: f32,
};

struct VisibilityTrace {
    blocked: bool,
    steps: u32,
};

@vertex
fn raymarch_vertex(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertex {
    var output: FullscreenVertex;
    let x = select(-1.0, 3.0, vertex_index == 1u);
    let y = select(-1.0, 3.0, vertex_index == 2u);
    output.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return output;
}

fn safe_inverse(direction: f32) -> f32 {
    if abs(direction) < 1e-20 {
        return select(TRACE_INFINITY, -TRACE_INFINITY, direction < 0.0);
    }
    return 1.0 / direction;
}

fn distance_tie_tolerance(distance: f32) -> f32 {
    return max(abs(distance) * 1e-6, TRACE_MIN_TIE_EPSILON);
}

fn traversal_sample_bias(voxel_size: f32) -> f32 {
    return max(voxel_size * 1e-5, TRACE_MIN_TIE_EPSILON);
}

fn ray_policy_enabled(flag: u32) -> bool {
    return (ray_scene.policy_flags & flag) != 0u;
}

fn box_intersection(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
) -> BoxHit {
    if (abs(ray_direction.x) < 1e-20 && (ray_origin.x < bounds_min.x || ray_origin.x > bounds_max.x))
        || (abs(ray_direction.y) < 1e-20 && (ray_origin.y < bounds_min.y || ray_origin.y > bounds_max.y))
        || (abs(ray_direction.z) < 1e-20 && (ray_origin.z < bounds_min.z || ray_origin.z > bounds_max.z)) {
        return BoxHit(false, 0.0, 0.0, vec3<f32>(0.0));
    }
    let inverse = vec3<f32>(
        safe_inverse(ray_direction.x),
        safe_inverse(ray_direction.y),
        safe_inverse(ray_direction.z)
    );
    var near_times = min((bounds_min - ray_origin) * inverse, (bounds_max - ray_origin) * inverse);
    var far_times = max((bounds_min - ray_origin) * inverse, (bounds_max - ray_origin) * inverse);
    if abs(ray_direction.x) < 1e-20 { near_times.x = -1e30; far_times.x = 1e30; }
    if abs(ray_direction.y) < 1e-20 { near_times.y = -1e30; far_times.y = 1e30; }
    if abs(ray_direction.z) < 1e-20 { near_times.z = -1e30; far_times.z = 1e30; }
    let entry = max(near_times.x, max(near_times.y, near_times.z));
    let exit = min(far_times.x, min(far_times.y, far_times.z));
    if entry > exit || exit < 0.0 {
        return BoxHit(false, entry, exit, vec3<f32>(0.0));
    }
    var normal = vec3<f32>(0.0);
    if entry <= 0.0 {
        let absolute = abs(ray_direction);
        if absolute.x >= absolute.y && absolute.x >= absolute.z {
            normal = vec3<f32>(-sign(ray_direction.x), 0.0, 0.0);
        } else if absolute.y >= absolute.z {
            normal = vec3<f32>(0.0, -sign(ray_direction.y), 0.0);
        } else {
            normal = vec3<f32>(0.0, 0.0, -sign(ray_direction.z));
        }
    } else if near_times.x >= near_times.y && near_times.x >= near_times.z {
        normal = vec3<f32>(select(1.0, -1.0, ray_direction.x > 0.0), 0.0, 0.0);
    } else if near_times.y >= near_times.z {
        normal = vec3<f32>(0.0, select(1.0, -1.0, ray_direction.y > 0.0), 0.0);
    } else {
        normal = vec3<f32>(0.0, 0.0, select(1.0, -1.0, ray_direction.z > 0.0));
    }
    return BoxHit(true, max(entry, 0.0), exit, normal);
}

fn lookup_leaf(point: vec3<f32>, chunk: RayChunk) -> VoxelLeaf {
    var node_index = chunk.indices.x;
    var bounds_min = vec3<f32>(0.0);
    var bounds_max = vec3<f32>(chunk.origin_world_size.w);
    let depth = u32(chunk.voxel_depth.y);
    for (var level = 0u; level <= 10u; level += 1u) {
        let word = node_index * 4u;
        let node_type = atlas_words[word];
        let payload = atlas_words[word + 1u];
        let color_or_mask = atlas_words[word + 2u];
        let light_word = atlas_words[word + 3u];
        if node_type == 1u || level >= depth {
            return VoxelLeaf(payload, color_or_mask, light_word, bounds_min, bounds_max);
        }
        let center = (bounds_min + bounds_max) * 0.5;
        let child_x = select(0u, 1u, point.x >= center.x);
        let child_y = select(0u, 1u, point.y >= center.y);
        let child_z = select(0u, 1u, point.z >= center.z);
        let child = (child_z << 2u) | (child_y << 1u) | child_x;
        let child_min = vec3<f32>(
            select(bounds_min.x, center.x, child_x == 1u),
            select(bounds_min.y, center.y, child_y == 1u),
            select(bounds_min.z, center.z, child_z == 1u)
        );
        let child_max = vec3<f32>(
            select(center.x, bounds_max.x, child_x == 1u),
            select(center.y, bounds_max.y, child_y == 1u),
            select(center.z, bounds_max.z, child_z == 1u)
        );
        if (color_or_mask & (1u << child)) == 0u {
            return VoxelLeaf(0u, 0u, 0u, child_min, child_max);
        }
        node_index = payload + child;
        bounds_min = child_min;
        bounds_max = child_max;
    }
    return VoxelLeaf(0u, 0u, 0u, bounds_min, bounds_max);
}

fn next_leaf_exit(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
) -> vec4<f32> {
    var times = vec3<f32>(TRACE_INFINITY);
    if abs(ray_direction.x) >= 1e-20 {
        times.x = (select(bounds_min.x, bounds_max.x, ray_direction.x > 0.0) - ray_origin.x) / ray_direction.x;
    }
    if abs(ray_direction.y) >= 1e-20 {
        times.y = (select(bounds_min.y, bounds_max.y, ray_direction.y > 0.0) - ray_origin.y) / ray_direction.y;
    }
    if abs(ray_direction.z) >= 1e-20 {
        times.z = (select(bounds_min.z, bounds_max.z, ray_direction.z > 0.0) - ray_origin.z) / ray_direction.z;
    }
    let nearest = min(times.x, min(times.y, times.z));
    let tolerance = distance_tie_tolerance(nearest);
    // Match finest-DDA X/Y/Z precedence for mathematically tied exits. The
    // traversal bias is intentionally not reused as an equality tolerance.
    if abs(times.x - nearest) <= tolerance {
        return vec4<f32>(nearest, -sign(ray_direction.x), 0.0, 0.0);
    }
    if abs(times.y - nearest) <= tolerance {
        return vec4<f32>(nearest, 0.0, -sign(ray_direction.y), 0.0);
    }
    return vec4<f32>(nearest, 0.0, 0.0, -sign(ray_direction.z));
}

fn trace_chunk_dda(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    chunk: RayChunk,
    interval: BoxHit,
) -> VoxelHit {
    let voxel_size = chunk.voxel_depth.x;
    let bias = traversal_sample_bias(voxel_size);
    var distance = interval.entry;
    var normal = interval.normal;
    var sample = ray_origin + ray_direction * min(distance + bias, interval.exit);
    var cell = floor(sample / voxel_size);
    let direction_step = sign(ray_direction);
    let budget = clamp(frame.render_options.w, 1u, 4096u);
    for (var step = 0u; step < 4096u; step += 1u) {
        if step >= budget || distance > interval.exit {
            break;
        }
        sample = ray_origin + ray_direction * min(distance + bias, interval.exit);
        let leaf = lookup_leaf(sample, chunk);
        if leaf.material != 0u {
            return VoxelHit(
                true,
                distance,
                normal,
                leaf.material,
                leaf.color,
                leaf.light_word,
                voxel_size
            );
        }

        let next_boundary = (cell + max(direction_step, vec3<f32>(0.0))) * voxel_size;
        var next_times = vec3<f32>(TRACE_INFINITY);
        if abs(ray_direction.x) >= 1e-20 { next_times.x = (next_boundary.x - ray_origin.x) / ray_direction.x; }
        if abs(ray_direction.y) >= 1e-20 { next_times.y = (next_boundary.y - ray_origin.y) / ray_direction.y; }
        if abs(ray_direction.z) >= 1e-20 { next_times.z = (next_boundary.z - ray_origin.z) / ray_direction.z; }
        let next_distance = min(next_times.x, min(next_times.y, next_times.z));
        if next_distance > interval.exit { break; }
        let tolerance = distance_tie_tolerance(next_distance);
        let cross_x = abs(next_times.x - next_distance) <= tolerance;
        let cross_y = abs(next_times.y - next_distance) <= tolerance;
        let cross_z = abs(next_times.z - next_distance) <= tolerance;
        if cross_x { cell.x += direction_step.x; }
        if cross_y { cell.y += direction_step.y; }
        if cross_z { cell.z += direction_step.z; }
        if cross_x {
            normal = vec3<f32>(-direction_step.x, 0.0, 0.0);
        } else if cross_y {
            normal = vec3<f32>(0.0, -direction_step.y, 0.0);
        } else {
            normal = vec3<f32>(0.0, 0.0, -direction_step.z);
        }
        distance = next_distance;
    }
    return VoxelHit(false, 0.0, vec3<f32>(0.0), 0u, 0u, 0u, 0.0);
}

fn trace_chunk_skipping_empty_leaves(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    chunk: RayChunk,
    interval: BoxHit,
) -> VoxelHit {
    let voxel_size = chunk.voxel_depth.x;
    let bias = traversal_sample_bias(voxel_size);
    var distance = interval.entry;
    var normal = interval.normal;
    var refinement_entry = interval.entry;
    var refinement_entry_normal = interval.normal;
    let budget = clamp(frame.render_options.w, 1u, 4096u);

    for (var step = 0u; step < 4096u; step += 1u) {
        if step >= budget || distance > interval.exit {
            break;
        }
        let sample = ray_origin + ray_direction * min(distance + bias, interval.exit);
        let leaf = lookup_leaf(sample, chunk);
        if leaf.material != 0u {
            // The coarse walk proves emptiness only. Re-run the canonical
            // finest-cell DDA over the final short interval so skip on/off
            // returns one distance, material, and normal contract.
            let refinement_interval = BoxHit(
                true,
                refinement_entry,
                interval.exit,
                refinement_entry_normal
            );
            return trace_chunk_dda(
                ray_origin,
                ray_direction,
                chunk,
                refinement_interval
            );
        }

        let exit = next_leaf_exit(
            ray_origin,
            ray_direction,
            leaf.bounds_min,
            leaf.bounds_max
        );
        var next_distance = exit.x;
        if next_distance <= distance + bias * 0.25 {
            next_distance = distance + bias;
        }
        // A normalized ray travels at most one finest voxel per axis over
        // this retained interval, which is enough for DDA to reconstruct all
        // potentially competing boundary crossings.
        refinement_entry = max(distance, next_distance - voxel_size);
        refinement_entry_normal = normal;
        distance = next_distance;
        normal = exit.yzw;
    }
    return VoxelHit(false, 0.0, vec3<f32>(0.0), 0u, 0u, 0u, 0.0);
}

fn trace_chunk(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    chunk: RayChunk,
    interval: BoxHit,
) -> VoxelHit {
    if ray_policy_enabled(RAY_POLICY_EMPTY_SPACE_SKIP) {
        return trace_chunk_skipping_empty_leaves(
            ray_origin,
            ray_direction,
            chunk,
            interval
        );
    }
    return trace_chunk_dda(ray_origin, ray_direction, chunk, interval);
}

fn direct_visibility_bias(voxel_size: f32) -> f32 {
    return max(voxel_size * 1e-3, 1e-5);
}

// Secondary visibility has its own hard budget. It deliberately does not
// consult the primary-ray trace budget or empty-space policy toggle.
fn trace_visibility_chunk_dda(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    chunk: RayChunk,
    interval: BoxHit,
    step_budget: u32,
) -> VisibilityTrace {
    let voxel_size = chunk.voxel_depth.x;
    let bias = traversal_sample_bias(voxel_size);
    var distance = interval.entry;
    var sample = ray_origin + ray_direction * min(distance + bias, interval.exit);
    var cell = floor(sample / voxel_size);
    let direction_step = sign(ray_direction);
    var steps = 0u;

    for (var iteration = 0u; iteration < DIRECT_VISIBILITY_TRACE_BUDGET; iteration += 1u) {
        if distance > interval.exit {
            return VisibilityTrace(false, steps);
        }
        if steps >= step_budget {
            // Exhaustion is conservatively opaque, avoiding a light leak.
            return VisibilityTrace(true, steps);
        }
        steps += 1u;
        sample = ray_origin + ray_direction * min(distance + bias, interval.exit);
        let leaf = lookup_leaf(sample, chunk);
        if leaf.material != 0u {
            return VisibilityTrace(true, steps);
        }

        let next_boundary = (cell + max(direction_step, vec3<f32>(0.0))) * voxel_size;
        var next_times = vec3<f32>(TRACE_INFINITY);
        if abs(ray_direction.x) >= 1e-20 { next_times.x = (next_boundary.x - ray_origin.x) / ray_direction.x; }
        if abs(ray_direction.y) >= 1e-20 { next_times.y = (next_boundary.y - ray_origin.y) / ray_direction.y; }
        if abs(ray_direction.z) >= 1e-20 { next_times.z = (next_boundary.z - ray_origin.z) / ray_direction.z; }
        let next_distance = min(next_times.x, min(next_times.y, next_times.z));
        if next_distance > interval.exit {
            return VisibilityTrace(false, steps);
        }
        let tolerance = distance_tie_tolerance(next_distance);
        let cross_x = abs(next_times.x - next_distance) <= tolerance;
        let cross_y = abs(next_times.y - next_distance) <= tolerance;
        let cross_z = abs(next_times.z - next_distance) <= tolerance;
        if cross_x { cell.x += direction_step.x; }
        if cross_y { cell.y += direction_step.y; }
        if cross_z { cell.z += direction_step.z; }
        distance = next_distance;
    }
    return VisibilityTrace(true, steps);
}

// Coarse leaves prove only that an interval is empty. A non-empty leaf is
// resolved by the same finest-cell DDA contract as primary traversal, while
// both phases share one caller-supplied secondary-ray budget.
fn trace_visibility_chunk_skipping_empty_leaves(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    chunk: RayChunk,
    interval: BoxHit,
    step_budget: u32,
) -> VisibilityTrace {
    let voxel_size = chunk.voxel_depth.x;
    let bias = traversal_sample_bias(voxel_size);
    var distance = interval.entry;
    var normal = interval.normal;
    var refinement_entry = interval.entry;
    var refinement_entry_normal = interval.normal;
    var steps = 0u;

    for (var iteration = 0u; iteration < DIRECT_VISIBILITY_TRACE_BUDGET; iteration += 1u) {
        if distance > interval.exit {
            return VisibilityTrace(false, steps);
        }
        if steps >= step_budget {
            return VisibilityTrace(true, steps);
        }
        steps += 1u;
        let sample = ray_origin + ray_direction * min(distance + bias, interval.exit);
        let leaf = lookup_leaf(sample, chunk);
        if leaf.material != 0u {
            let refinement_interval = BoxHit(
                true,
                refinement_entry,
                interval.exit,
                refinement_entry_normal
            );
            let refined = trace_visibility_chunk_dda(
                ray_origin,
                ray_direction,
                chunk,
                refinement_interval,
                step_budget - steps
            );
            return VisibilityTrace(refined.blocked, steps + refined.steps);
        }

        let exit = next_leaf_exit(
            ray_origin,
            ray_direction,
            leaf.bounds_min,
            leaf.bounds_max
        );
        var next_distance = exit.x;
        if next_distance <= distance + bias * 0.25 {
            next_distance = distance + bias;
        }
        refinement_entry = max(distance, next_distance - voxel_size);
        refinement_entry_normal = normal;
        distance = next_distance;
        normal = exit.yzw;
    }
    return VisibilityTrace(true, steps);
}

fn finite_segment_visible(
    receiver_world: vec3<f32>,
    receiver_normal: vec3<f32>,
    receiver_voxel_size: f32,
    endpoint_world: vec3<f32>,
) -> bool {
    let endpoint_bias = direct_visibility_bias(receiver_voxel_size);
    let biased_origin = receiver_world + receiver_normal * endpoint_bias;
    let segment = endpoint_world - biased_origin;
    let endpoint_distance = length(segment);
    if endpoint_distance <= endpoint_bias * 2.0 {
        return true;
    }
    let ray_direction = segment / endpoint_distance;
    // Do not classify the light fixture's own endpoint as an occluder.
    let maximum_distance = endpoint_distance - endpoint_bias;
    var remaining_steps = DIRECT_VISIBILITY_TRACE_BUDGET;
    let chunk_count = min(ray_scene.chunk_count, 25u);

    for (var index = 0u; index < chunk_count; index += 1u) {
        if remaining_steps == 0u {
            return false;
        }
        let chunk = ray_chunks[index];
        let local_origin = biased_origin - chunk.origin_world_size.xyz;
        let slab = box_intersection(
            local_origin,
            ray_direction,
            vec3<f32>(0.0),
            vec3<f32>(chunk.origin_world_size.w)
        );
        let finite_exit = min(slab.exit, maximum_distance);
        if !slab.hit || finite_exit <= slab.entry {
            continue;
        }
        let interval = BoxHit(true, slab.entry, finite_exit, slab.normal);
        let trace = trace_visibility_chunk_skipping_empty_leaves(
            local_origin,
            ray_direction,
            chunk,
            interval,
            remaining_steps
        );
        remaining_steps -= min(trace.steps, remaining_steps);
        if trace.blocked {
            return false;
        }
    }
    return true;
}

fn has_positive_rgb(value: vec3<f32>) -> bool {
    return max(value.x, max(value.y, value.z)) > 0.0;
}

fn rectangle_visibility_sample_count() -> u32 {
    return select(1u, 4u, ray_scene._reserved.x == 4u);
}

fn evaluate_visible_scene_light(
    world: vec3<f32>,
    normal: vec3<f32>,
    receiver_voxel_size: f32,
    light: GpuLight,
) -> vec3<f32> {
    let kind = u32(max(light.half_size_kind.z, 0.0) + 0.5);
    if kind == LIGHT_POINT {
        let endpoint = light.position_radius.xyz;
        let unoccluded = point_light_sample_irradiance(world, normal, light, endpoint);
        if !has_positive_rgb(unoccluded) {
            return vec3<f32>(0.0);
        }
        if finite_segment_visible(world, normal, receiver_voxel_size, endpoint) {
            return unoccluded;
        }
        return vec3<f32>(0.0);
    }

    if rectangle_visibility_sample_count() == 4u {
        var visible_integral = 0.0;
        for (var sample_index = 0u; sample_index < 4u; sample_index += 1u) {
            let endpoint = rectangle_gauss_sample_position(light, sample_index);
            let sample_weight = rectangle_light_sample_weight(
                world,
                normal,
                light,
                kind,
                endpoint
            );
            let sample_irradiance = rectangle_light_irradiance_from_integral(
                light,
                sample_weight
            );
            if !has_positive_rgb(sample_irradiance) {
                continue;
            }
            if finite_segment_visible(world, normal, receiver_voxel_size, endpoint) {
                visible_integral += sample_weight;
            }
        }
        return rectangle_light_irradiance_from_integral(light, visible_integral);
    }

    // Low quality retains the existing four-point unoccluded quadrature and
    // gates its integral with one finite segment to the rectangle center.
    let unoccluded = evaluate_rectangle_light(world, normal, light, kind);
    if !has_positive_rgb(unoccluded) {
        return vec3<f32>(0.0);
    }
    if finite_segment_visible(
        world,
        normal,
        receiver_voxel_size,
        light.position_radius.xyz
    ) {
        return unoccluded;
    }
    return vec3<f32>(0.0);
}

fn raymarch_analytic_direct(
    world: vec3<f32>,
    normal: vec3<f32>,
    receiver_voxel_size: f32,
) -> vec3<f32> {
    var analytic_direct = vec3<f32>(0.0);
    for (var index = 0u; index < frame.render_options.x; index += 1u) {
        let light = lights[index];
        if light.half_size_kind.w < 0.5 {
            continue;
        }
        analytic_direct += evaluate_visible_scene_light(
            world,
            normal,
            receiver_voxel_size,
            light
        );
    }
    return analytic_direct;
}

fn camera_ray(pixel: vec2<f32>) -> vec3<f32> {
    let viewport = max(frame.viewport_fov_start.xy, vec2<f32>(1.0));
    let normalized = pixel / viewport;
    let aspect = viewport.x / viewport.y;
    let screen = vec2<f32>(
        (normalized.x * 2.0 - 1.0) * aspect,
        1.0 - normalized.y * 2.0
    ) * frame.viewport_fov_start.z;
    return normalize(
        frame.camera_forward.xyz
        + frame.camera_right.xyz * screen.x
        + frame.camera_up.xyz * screen.y
    );
}

@fragment
fn raymarch_fragment(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let direction = camera_ray(pixel.xy);
    var nearest = VoxelHit(false, 1e30, vec3<f32>(0.0), 0u, 0u, 0u, 0.0);
    var nearest_source_index = 0xffffffffu;
    let chunk_count = min(ray_scene.chunk_count, 25u);
    for (var index = 0u; index < chunk_count; index += 1u) {
        let chunk = ray_chunks[index];
        let local_origin = frame.camera_position.xyz - chunk.origin_world_size.xyz;
        let interval = box_intersection(
            local_origin,
            direction,
            vec3<f32>(0.0),
            vec3<f32>(chunk.origin_world_size.w)
        );
        if !interval.hit || interval.entry > nearest.distance { continue; }
        let hit = trace_chunk(local_origin, direction, chunk, interval);
        let stable_tie = hit.distance == nearest.distance
            && chunk.indices.y < nearest_source_index;
        if hit.hit && (hit.distance < nearest.distance || stable_tie) {
            nearest = hit;
            nearest_source_index = chunk.indices.y;
        }
    }

    if !nearest.hit {
        return vec4<f32>(display_color(frame.sky_color_ambient.xyz), 1.0);
    }
    let world = frame.camera_position.xyz + direction * nearest.distance;
    let albedo = packed_srgb_to_linear(nearest.color);
    let emission = authored_emission_strength(nearest.material);
    let emits = emission > 0.0 && nearest.normal.y < -0.5;
    var radiance = vec3<f32>(0.0);
    if emits {
        radiance = albedo * emission;
    } else if ray_policy_enabled(RAY_POLICY_DIRECT_LIGHT_VISIBILITY) {
        let analytic_direct = raymarch_analytic_direct(
            world,
            nearest.normal,
            nearest.voxel_size
        );
        radiance = surface_radiance_from_analytic_direct(
            world,
            nearest.normal,
            albedo,
            0.0,
            1.0,
            analytic_direct
        );
    } else {
        radiance = surface_radiance(
            world,
            nearest.normal,
            albedo,
            0.0,
            1.0,
            0u,
            frame.render_options.x
        );
    }
    return present_radiance(radiance, world, pixel.xy);
}
