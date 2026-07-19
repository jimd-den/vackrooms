//! Path-traced reference of randomly generated scenes, written as PNG.
//!
//! Ground truth for renderer testing: generates a random world with the
//! production `LocalChunkSource`, then Monte-Carlo path traces the spawn
//! neighborhood through the same serialized SVO and canonical `trace_svo`
//! traversal the CPU renderer uses. Ceiling fixtures are the only emitters
//! (their authored emission strengths), every other material is a diffuse
//! reflector of its palette albedo — so global illumination, soft shadows,
//! and color bleed in the output are physically plausible for the exact
//! same scene data the real-time paths draw.
//!
//! ```sh
//! cargo run -p wasm_frontend --example path_trace --release
//! SEED=7 SPP=512 SIZE=960x540 cargo run -p wasm_frontend --example path_trace --release
//! ```
//!
//! Env knobs: `SEED` (default random — a new scene every run), `RADIUS`
//! (chunk rings around spawn: 1 → 3x3 chunks, 2 → 5x5, …; default 1),
//! `SIZE` (`WxH`, default 640x360), `SPP` (samples per pixel, default 128),
//! `BOUNCES` (default 6), `EXPOSURE` (default 1.4). Output lands in
//! `target/path-traces/seed-<seed>.png`.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use std::collections::HashMap;

use rayon::prelude::*;
use vackrooms::domain::entities::voxel_grid::material_color_f32;
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::spawn_point;
use wasm_frontend::adapters::cpu_splatter::atlas::{emission_strength, is_emissive};
use wasm_frontend::adapters::cpu_splatter::trace_svo;
use wasm_frontend::adapters::local_chunk_source::LocalChunkSource;
use wasm_frontend::application::atlas::{AtlasPool, payload_rows};
use wasm_frontend::application::ports::{ChunkDraw, ChunkSourcePort, LightSource};
use wasm_frontend::application::streaming::chunk_key;

const MAX_TRACE_DISTANCE: f32 = 300.0;
const FOV_TAN: f32 = 0.767_327;

fn main() {
    let seed: u32 = std::env::var("SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(random_seed);
    let (width, height) = parse_size();
    let radius: i32 = env_number("RADIUS", 1).clamp(0, 8) as i32;
    let spp: u32 = env_number("SPP", 128);
    let bounces: u32 = env_number("BOUNCES", 6);
    let exposure: f32 = std::env::var("EXPOSURE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1.4);

    let scene = load_scene(seed, radius);
    eprintln!(
        "scene: seed {seed}, {} chunks (radius {radius}), {} lights, {} atlas words | {width}x{height}, {spp} spp, {bounces} bounces",
        scene.chunks.len(),
        scene.lights.len(),
        scene.atlas.len(),
    );

    let start = Instant::now();
    let camera = Camera::at_spawn(seed, width, height);
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    let rows_done = std::sync::atomic::AtomicUsize::new(0);
    let report_every = (height as usize / 50).max(1);
    pixels
        .par_chunks_mut((width * 3) as usize)
        .enumerate()
        .for_each(|(row, out)| {
            for column in 0..width as usize {
                let mut accumulated = [0.0f32; 3];
                for sample in 0..spp {
                    let mut rng =
                        Pcg32::new(((row as u64) << 40) ^ ((column as u64) << 20) ^ sample as u64, seed as u64);
                    let ray = camera.primary_ray(column as f32, row as f32, &mut rng);
                    let radiance = trace_path(&scene, ray, bounces, &mut rng);
                    for axis in 0..3 {
                        accumulated[axis] += radiance[axis];
                    }
                }
                for axis in 0..3 {
                    let mean = accumulated[axis] / spp as f32;
                    // Filmic-ish exposure curve, then gamma for the PNG.
                    let mapped = 1.0 - (-mean * exposure).exp();
                    out[column * 3 + axis] = (mapped.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                }
            }
            // Row-level progress on one rewritten terminal line: percent,
            // rows, elapsed, and a remaining-time estimate.
            let done = rows_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % report_every == 0 || done == height as usize {
                let elapsed = start.elapsed().as_secs_f32();
                let fraction = done as f32 / height as f32;
                let remaining = elapsed / fraction - elapsed;
                eprint!(
                    "\rtracing {:>3.0}%  ({done}/{} rows, {elapsed:.0}s elapsed, ~{remaining:.0}s left)   ",
                    fraction * 100.0,
                    height,
                );
            }
        });
    eprintln!("\ntraced in {:.1}s", start.elapsed().as_secs_f32());

    let out_dir = PathBuf::from("target/path-traces");
    fs::create_dir_all(&out_dir).expect("create output directory");
    let path = out_dir.join(format!("seed-{seed}.png"));
    image::RgbImage::from_raw(width, height, pixels)
        .expect("pixel buffer matches dimensions")
        .save(&path)
        .expect("write png");
    println!("{}", path.display());
}

// ---------------------------------------------------------------------------
// Scene: the production spawn neighborhood, atlas-assembled like streaming.
// ---------------------------------------------------------------------------

struct Scene {
    atlas: Vec<u32>,
    chunks: Vec<ChunkDraw>,
    /// Renderer-neutral analytic fixtures exported by generation; the
    /// next-event estimator samples their panel rectangles directly.
    lights: Vec<LightSource>,
    voxel_size: f32,
}

fn load_scene(seed: u32, radius: i32) -> Scene {
    let config = GeneratorConfig::low_spec();
    let spawn = spawn_point(seed);
    let origin_x = (spawn.x / config.chunk_size).floor() * config.chunk_size;
    let origin_z = (spawn.z / config.chunk_size).floor() * config.chunk_size;
    let source = LocalChunkSource::new(SimpleNoiseProvider::new(), seed, config);

    let mut payloads = Vec::new();
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            let x = origin_x + dx as f32 * config.chunk_size;
            let z = origin_z + dz as f32 * config.chunk_size;
            payloads.push((x, z, source.load(x, z, 0, 0)));
        }
    }
    let max_rows = payloads
        .iter()
        .map(|(_, _, payload)| payload_rows(payload))
        .max()
        .expect("spawn neighborhood is never empty");
    let mut pool = AtlasPool::new();
    pool.ensure_layout(payloads.len(), max_rows);

    let mut atlas = Vec::new();
    let mut chunks = Vec::new();
    let mut voxel_size = config.voxel_scale;
    let mut unique_lights = HashMap::new();
    for (x, z, payload) in &payloads {
        let slot = pool.assign(chunk_key(*x, *z)).expect("atlas sized for all");
        let offset = slot * pool.slot_nodes();
        atlas.extend(pool.rebased_block(slot, payload));
        voxel_size = payload.voxel_size;
        chunks.push(ChunkDraw {
            origin: [*x, 0.0, *z],
            root_index: (offset + payload.root as usize) as i32,
            world_size: payload.world_size,
            voxel_size: payload.voxel_size,
            svo_depth: payload.svo_depth,
        });
        for light in &payload.lights {
            if light.enabled {
                unique_lights.entry(light.id).or_insert(*light);
            }
        }
    }
    Scene {
        atlas,
        chunks,
        lights: unique_lights.into_values().collect(),
        voxel_size,
    }
}

// ---------------------------------------------------------------------------
// Path tracing over the SVO.
// ---------------------------------------------------------------------------

struct Ray {
    origin: [f32; 3],
    direction: [f32; 3],
}

/// Full path-tracing estimator: at every diffuse vertex, next-event
/// estimation samples one fixture rectangle through a shadow ray; the
/// bounce ray then carries indirect light. Emitter hits contribute only on
/// camera rays (NEE already accounts for them at later vertices), which
/// keeps the split unbiased without double counting.
fn trace_path(scene: &Scene, mut ray: Ray, bounces: u32, rng: &mut Pcg32) -> [f32; 3] {
    let mut radiance = [0.0f32; 3];
    let mut throughput = [1.0f32; 3];
    for bounce in 0..bounces {
        let Some(hit) = trace_svo(
            &scene.atlas,
            &scene.chunks,
            ray.origin,
            ray.direction,
            MAX_TRACE_DISTANCE,
            true,
        ) else {
            break; // The Backrooms have no sky.
        };
        let voxel = hit.voxel_type;
        let color = material_color_f32(voxel as u8);
        if is_emissive(voxel) {
            if bounce == 0 {
                let strength = emission_strength(voxel).unwrap_or(1.0);
                for axis in 0..3 {
                    radiance[axis] += throughput[axis] * color[axis] * strength;
                }
            }
            break; // Fixtures emit; they do not bounce.
        }

        let hit_point = [
            ray.origin[0] + ray.direction[0] * hit.t,
            ray.origin[1] + ray.direction[1] * hit.t,
            ray.origin[2] + ray.direction[2] * hit.t,
        ];
        let normal = face_normal(hit_point, ray.direction, scene.voxel_size);
        let epsilon = scene.voxel_size * 0.05;
        let surface_point = [
            hit_point[0] + normal[0] * epsilon,
            hit_point[1] + normal[1] * epsilon,
            hit_point[2] + normal[2] * epsilon,
        ];

        // Next-event estimation toward one sampled fixture.
        let direct = sample_direct_light(scene, surface_point, normal, rng);
        for axis in 0..3 {
            radiance[axis] += throughput[axis] * color[axis] * direct[axis];
        }

        for axis in 0..3 {
            throughput[axis] *= color[axis];
        }
        // Russian roulette once the throughput has earned it.
        if bounce >= 3 {
            let survive = throughput[0]
                .max(throughput[1])
                .max(throughput[2])
                .clamp(0.05, 0.95);
            if rng.uniform() > survive {
                break;
            }
            for axis in 0..3 {
                throughput[axis] /= survive;
            }
        }
        ray = Ray {
            origin: surface_point,
            direction: cosine_hemisphere(normal, rng),
        };
    }
    radiance
}

/// Estimates direct irradiance (already divided by π for the Lambertian
/// BRDF) at a surface point by sampling one fixture rectangle uniformly.
/// Fixture panels hang from ceilings and emit downward (-Y).
fn sample_direct_light(
    scene: &Scene,
    surface_point: [f32; 3],
    normal: [f32; 3],
    rng: &mut Pcg32,
) -> [f32; 3] {
    if scene.lights.is_empty() {
        return [0.0; 3];
    }
    let picked = (rng.uniform() * scene.lights.len() as f32) as usize;
    let light = &scene.lights[picked.min(scene.lights.len() - 1)];
    let half = [light.half_size[0].max(0.01), light.half_size[1].max(0.01)];
    let sample_point = [
        light.position[0] + (rng.uniform() * 2.0 - 1.0) * half[0],
        light.position[1],
        light.position[2] + (rng.uniform() * 2.0 - 1.0) * half[1],
    ];
    let to_light = [
        sample_point[0] - surface_point[0],
        sample_point[1] - surface_point[1],
        sample_point[2] - surface_point[2],
    ];
    let distance_sq = to_light[0] * to_light[0] + to_light[1] * to_light[1] + to_light[2] * to_light[2];
    let distance = distance_sq.sqrt().max(1e-4);
    let direction = [
        to_light[0] / distance,
        to_light[1] / distance,
        to_light[2] / distance,
    ];
    let cos_surface = direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2];
    // Panels radiate downward; the receiver must also face the panel.
    let cos_light = direction[1].max(0.0);
    if cos_surface <= 0.0 || cos_light <= 0.0 {
        return [0.0; 3];
    }
    // Shadow ray. The panel's own emissive voxels sit at the endpoint, so a
    // near-endpoint emissive hit counts as arrival, not occlusion.
    if let Some(obstruction) = trace_svo(
        &scene.atlas,
        &scene.chunks,
        surface_point,
        direction,
        distance,
        true,
    ) {
        let reached_panel = is_emissive(obstruction.voxel_type)
            && obstruction.t > distance - scene.voxel_size * 4.0;
        if !reached_panel {
            return [0.0; 3];
        }
    }
    let area = 4.0 * half[0] * half[1];
    let pdf = 1.0 / (area * scene.lights.len() as f32);
    let geometry = cos_surface * cos_light / distance_sq;
    let scale = light.intensity * geometry / (std::f32::consts::PI * pdf);
    [
        light.color[0] * scale,
        light.color[1] * scale,
        light.color[2] * scale,
    ]
}

/// Which axis face of the voxel grid the hit crossed: the axis whose
/// coordinate sits nearest a voxel boundary. Exact for face hits; edges and
/// corners (measure zero) pick either adjacent face, which is harmless in
/// Monte-Carlo estimation.
fn face_normal(point: [f32; 3], direction: [f32; 3], voxel_size: f32) -> [f32; 3] {
    let mut best_axis = 0;
    let mut best_distance = f32::MAX;
    for axis in 0..3 {
        let cell = point[axis] / voxel_size;
        let fractional = cell - cell.floor();
        let distance = fractional.min(1.0 - fractional);
        if distance < best_distance {
            best_distance = distance;
            best_axis = axis;
        }
    }
    let mut normal = [0.0f32; 3];
    normal[best_axis] = -direction[best_axis].signum();
    normal
}

fn cosine_hemisphere(normal: [f32; 3], rng: &mut Pcg32) -> [f32; 3] {
    // Orthonormal basis around the normal (axis-aligned normals make this
    // trivial, but stay general for safety).
    let tangent = if normal[0].abs() < 0.9 {
        normalize(cross(normal, [1.0, 0.0, 0.0]))
    } else {
        normalize(cross(normal, [0.0, 1.0, 0.0]))
    };
    let bitangent = cross(normal, tangent);
    let u1 = rng.uniform();
    let u2 = rng.uniform();
    let r = u1.sqrt();
    let phi = std::f32::consts::TAU * u2;
    let (x, y) = (r * phi.cos(), r * phi.sin());
    let z = (1.0 - u1).max(0.0).sqrt();
    normalize([
        tangent[0] * x + bitangent[0] * y + normal[0] * z,
        tangent[1] * x + bitangent[1] * y + normal[1] * z,
        tangent[2] * x + bitangent[2] * y + normal[2] * z,
    ])
}

// ---------------------------------------------------------------------------
// Camera at the spawn corridor, matching the real-time boot view.
// ---------------------------------------------------------------------------

struct Camera {
    position: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
    width: f32,
    height: f32,
    aspect: f32,
}

impl Camera {
    fn at_spawn(seed: u32, width: u32, height: u32) -> Self {
        let spawn = spawn_point(seed);
        let yaw = -std::f32::consts::FRAC_PI_2;
        let forward = [yaw.sin(), 0.0, -yaw.cos()];
        let right = [-forward[2], 0.0, forward[0]];
        Self {
            position: [spawn.x, 1.7, spawn.z],
            right,
            up: [0.0, 1.0, 0.0],
            forward,
            width: width as f32,
            height: height as f32,
            aspect: width as f32 / height as f32,
        }
    }

    fn primary_ray(&self, column: f32, row: f32, rng: &mut Pcg32) -> Ray {
        let u = ((column + rng.uniform()) / self.width * 2.0 - 1.0) * FOV_TAN * self.aspect;
        let v = (1.0 - (row + rng.uniform()) / self.height * 2.0) * FOV_TAN;
        Ray {
            origin: self.position,
            direction: normalize([
                self.right[0] * u + self.up[0] * v + self.forward[0],
                self.right[1] * u + self.up[1] * v + self.forward[1],
                self.right[2] * u + self.up[2] * v + self.forward[2],
            ]),
        }
    }
}

// ---------------------------------------------------------------------------
// Small numerics: PCG32 sampling and vector helpers.
// ---------------------------------------------------------------------------

/// Minimal PCG32 (O'Neill); deterministic per (pixel, sample, seed).
struct Pcg32 {
    state: u64,
    increment: u64,
}

impl Pcg32 {
    fn new(stream: u64, seed: u64) -> Self {
        let mut rng = Self {
            state: seed.wrapping_add(0x853c_49e6_748f_ea9b),
            increment: (stream << 1) | 1,
        };
        rng.next();
        rng.next();
        rng
    }

    fn next(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.increment);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rotation = (old >> 59) as u32;
        xorshifted.rotate_right(rotation)
    }

    fn uniform(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn random_seed() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish() as u32
}

fn env_number(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn parse_size() -> (u32, u32) {
    std::env::var("SIZE")
        .ok()
        .and_then(|value| {
            let (w, h) = value.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((640, 360))
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / length, v[1] / length, v[2] / length]
}
