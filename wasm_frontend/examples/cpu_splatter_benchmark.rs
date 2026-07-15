use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::spawn_point;
use wasm_frontend::adapters::cpu_splatter::rasterizer::SoftwareRasterizer;
use wasm_frontend::adapters::cpu_splatter::settings::CpuQualityPreset;
use wasm_frontend::adapters::local_chunk_source::LocalChunkSource;
use wasm_frontend::application::atlas::{payload_rows, AtlasPool};
use wasm_frontend::application::ports::{ChunkDraw, ChunkSourcePort, FrameParams, RendererPort};
use wasm_frontend::application::streaming::chunk_key;

fn main() {
    let config = GeneratorConfig::low_spec();
    let spawn = spawn_point(42);
    let origin_x = (spawn.x / config.chunk_size).floor() * config.chunk_size;
    let origin_z = (spawn.z / config.chunk_size).floor() * config.chunk_size;
    let source = LocalChunkSource::new(SimpleNoiseProvider::new(), 42, config);
    let mut payloads = Vec::new();
    for dz in -1..=1 {
        for dx in -1..=1 {
            let x = origin_x + dx as f32 * config.chunk_size;
            let z = origin_z + dz as f32 * config.chunk_size;
            payloads.push((x, z, source.load(x, z, 0, 0)));
        }
    }
    let max_rows = payloads
        .iter()
        .map(|(_, _, p)| payload_rows(p))
        .max()
        .unwrap();
    let mut pool = AtlasPool::new();
    pool.ensure_layout(payloads.len(), max_rows);
    let mut atlas = Vec::new();
    let mut chunks = Vec::new();
    let mut unique_lights = HashMap::new();
    for (x, z, payload) in &payloads {
        let key = chunk_key(*x, *z);
        let slot = pool.assign(key).unwrap();
        let offset = slot * pool.slot_nodes();
        atlas.extend(pool.rebased_block(slot, payload));
        chunks.push(ChunkDraw {
            origin: [*x, 0.0, *z],
            root_index: (offset + payload.root as usize) as i32,
            world_size: payload.world_size,
            voxel_size: payload.voxel_size,
            svo_depth: payload.svo_depth,
        });
        for light in &payload.surface.lights {
            unique_lights.entry(light.id).or_insert(*light);
        }
    }
    let mut frame = FrameParams {
        camera_pos: [spawn.x, 1.7, spawn.z],
        yaw: -std::f32::consts::FRAC_PI_2,
        scene_lights: unique_lights.into_values().collect(),
        ..FrameParams::default()
    };

    for (label, preset, width, height, flashlight) in [
        (
            "performance",
            CpuQualityPreset::Performance,
            192,
            108,
            false,
        ),
        ("balanced", CpuQualityPreset::Balanced, 384, 216, false),
        (
            "balanced+flashlight",
            CpuQualityPreset::Balanced,
            384,
            216,
            true,
        ),
    ] {
        frame.flashlight = flashlight;
        let mut renderer = SoftwareRasterizer::new(width, height);
        renderer.settings = preset.settings();
        renderer.upload_atlas(&atlas);
        for _ in 0..5 {
            renderer.draw(&frame, &chunks);
        }

        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            renderer.draw(black_box(&frame), black_box(&chunks));
            black_box(renderer.framebuffer());
        }
        let elapsed = start.elapsed();
        let stats = renderer.telemetry();
        println!(
            "{label}: {:.3} ms/frame, nodes={}/{}, splats={}, writes={}/{}, limited={}",
            elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            stats.visited_nodes,
            stats.node_visit_limit,
            stats.splat_count,
            stats.pixel_writes,
            stats.pixel_write_limit,
            stats.budget_exhausted,
        );
    }
}
