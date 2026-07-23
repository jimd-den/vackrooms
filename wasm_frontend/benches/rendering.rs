//! Size-parameterized benchmarks for the three rendering-side hot paths
//! flagged in the audit: SVO ray traversal, the CPU rasterizer's full-frame
//! cost, and the brute-force collision query.
//!
//! `cargo bench -p wasm_frontend --bench rendering`.

use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::spawn_point;
use wasm_frontend::adapters::cpu_splatter::rasterizer::SoftwareRasterizer;
use wasm_frontend::adapters::cpu_splatter::settings::CpuQualityPreset;
use wasm_frontend::adapters::cpu_splatter::trace_svo;
use wasm_frontend::adapters::local_chunk_source::LocalChunkSource;
use wasm_frontend::application::atlas::{AtlasPool, payload_rows};
use wasm_frontend::application::collision::{Aabb, CollisionWorld};
use wasm_frontend::application::ports::{ChunkDraw, ChunkSourcePort, FrameParams, RendererPort};
use wasm_frontend::application::streaming::chunk_key;

const SEED: u32 = 42;

struct Scene {
    atlas: Vec<u32>,
    chunks: Vec<ChunkDraw>,
    frame: FrameParams,
    collision_boxes: Vec<Aabb>,
}

/// Same assembly as `examples/path_trace.rs`: the production
/// generate -> atlas-pool -> `ChunkDraw` pipeline, at a chosen chunk radius
/// and voxel scale, so bench inputs are real generated worlds, not
/// hand-built stand-ins.
fn load_scene(radius: i32, voxel_scale: f32) -> Scene {
    let mut config = GeneratorConfig::low_spec();
    config.voxel_scale = voxel_scale;
    let spawn = spawn_point(SEED);
    let origin_x = (spawn.x / config.chunk_size).floor() * config.chunk_size;
    let origin_z = (spawn.z / config.chunk_size).floor() * config.chunk_size;
    let source = LocalChunkSource::new(SimpleNoiseProvider::new(), SEED, config);

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
        .expect("neighborhood is never empty");
    let mut pool = AtlasPool::new();
    pool.ensure_layout(payloads.len(), max_rows);

    let mut atlas = Vec::new();
    let mut chunks = Vec::new();
    let mut unique_lights = HashMap::new();
    let mut collision_boxes = Vec::new();
    for (x, z, payload) in &payloads {
        let slot = pool.assign(chunk_key(*x, *z)).expect("atlas sized for all");
        let offset = slot * pool.slot_nodes();
        atlas.extend(pool.rebased_block(slot, payload));
        chunks.push(ChunkDraw {
            origin: [*x, 0.0, *z],
            root_index: (offset + payload.root as usize) as i32,
            world_size: payload.world_size,
            voxel_size: payload.voxel_size,
            svo_depth: payload.svo_depth,
        });
        for light in &payload.lights {
            unique_lights.entry(light.id).or_insert(*light);
        }
        collision_boxes.extend(payload.collision.iter().copied());
    }
    let frame = FrameParams {
        camera_pos: [spawn.x, 1.7, spawn.z],
        yaw: -std::f32::consts::FRAC_PI_2,
        scene_lights: unique_lights.into_values().collect(),
        ..FrameParams::default()
    };
    Scene {
        atlas,
        chunks,
        frame,
        collision_boxes,
    }
}

/// A fixed 32x18 ray fan spanning a ~75-degree FOV from the scene's camera,
/// reused identically across every benchmark iteration and every scene size.
fn ray_fan(frame: &FrameParams) -> Vec<[f32; 3]> {
    let yaw = frame.yaw;
    let forward = [yaw.sin(), 0.0, -yaw.cos()];
    let right = [-forward[2], 0.0, forward[0]];
    let up = [0.0, 1.0, 0.0];
    let fov_tan = 0.767_327f32;
    let mut rays = Vec::with_capacity(32 * 18);
    for row in 0..18 {
        for column in 0..32 {
            let u = (column as f32 / 31.0 * 2.0 - 1.0) * fov_tan * (32.0 / 18.0);
            let v = (1.0 - row as f32 / 17.0 * 2.0) * fov_tan;
            let direction = [
                right[0] * u + up[0] * v + forward[0],
                right[1] * u + up[1] * v + forward[1],
                right[2] * u + up[2] * v + forward[2],
            ];
            let length = (direction[0] * direction[0]
                + direction[1] * direction[1]
                + direction[2] * direction[2])
                .sqrt()
                .max(1e-8);
            rays.push([
                direction[0] / length,
                direction[1] / length,
                direction[2] / length,
            ]);
        }
    }
    rays
}

fn trace_svo_by_chunk_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_svo_by_chunk_count");
    for radius in [0i32, 1, 2, 3] {
        let scene = load_scene(radius, 0.2);
        let rays = ray_fan(&scene.frame);
        let chunk_count = scene.chunks.len();
        group.bench_with_input(
            BenchmarkId::new("chunks", chunk_count),
            &(scene, rays),
            |b, (scene, rays)| {
                b.iter(|| {
                    for direction in rays {
                        black_box(trace_svo(
                            &scene.atlas,
                            &scene.chunks,
                            scene.frame.camera_pos,
                            *direction,
                            300.0,
                            true,
                        ));
                    }
                });
            },
        );
    }
    group.finish();
}

fn trace_svo_by_voxel_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_svo_by_voxel_size");
    for voxel_scale in [0.4f32, 0.2, 0.1] {
        let scene = load_scene(1, voxel_scale);
        let rays = ray_fan(&scene.frame);
        group.bench_with_input(
            BenchmarkId::new("voxel_scale", voxel_scale),
            &(scene, rays),
            |b, (scene, rays)| {
                b.iter(|| {
                    for direction in rays {
                        black_box(trace_svo(
                            &scene.atlas,
                            &scene.chunks,
                            scene.frame.camera_pos,
                            *direction,
                            300.0,
                            true,
                        ));
                    }
                });
            },
        );
    }
    group.finish();
}

fn cpu_rasterizer_full_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_rasterizer_full_frame");
    for radius in [0i32, 1, 2] {
        let scene = load_scene(radius, 0.2);
        let mut renderer = SoftwareRasterizer::new(384, 216);
        renderer.settings = CpuQualityPreset::Balanced.settings();
        renderer.upload_atlas(&scene.atlas);
        let chunk_count = scene.chunks.len();
        group.bench_with_input(
            BenchmarkId::new("chunks", chunk_count),
            &scene,
            |b, scene| {
                b.iter(|| {
                    renderer.draw(&scene.frame, &scene.chunks);
                    black_box(renderer.framebuffer());
                });
            },
        );
    }
    group.finish();
}

fn synthetic_collision_boxes(count: usize) -> Vec<Aabb> {
    (0..count)
        .map(|i| {
            let x = (i % 64) as f32 * 0.6;
            let z = (i / 64) as f32 * 0.6;
            Aabb::new([x, 0.0, z], [x + 0.5, 2.6, z + 0.5])
        })
        .collect()
}

fn collision_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("collision_query");
    for count in [100usize, 1_000, 10_000] {
        let mut world = CollisionWorld::new();
        world.rebuild(&synthetic_collision_boxes(count));
        // A point near the middle of the synthetic grid: representative of
        // an ordinary query that must scan a meaningful fraction of boxes
        // before resolving (worst case for a linear, unindexed scan).
        let probe = [19.2, 1.0, 19.2];
        group.bench_with_input(BenchmarkId::new("boxes", count), &world, |b, world| {
            b.iter(|| black_box(world.collides(black_box(probe))));
        });
    }
    // Realistic anchor: collision boxes actually produced by generation for
    // a 3x3 chunk neighborhood, queried at the spawn point.
    let scene = load_scene(1, 0.2);
    let mut world = CollisionWorld::new();
    world.rebuild(&scene.collision_boxes);
    let box_count = world.len();
    group.bench_with_input(
        BenchmarkId::new("boxes", format!("{box_count}_generated")),
        &(world, scene.frame.camera_pos),
        |b, (world, probe)| {
            b.iter(|| black_box(world.collides(black_box(*probe))));
        },
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = trace_svo_by_chunk_count, trace_svo_by_voxel_size, cpu_rasterizer_full_frame, collision_query
}
criterion_main!(benches);
