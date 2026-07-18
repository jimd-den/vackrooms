//! Native Vulkan test build: renders the real spawn neighborhood through
//! every production WebGPU pipeline and writes one PNG per renderer.
//!
//! This is the browserless way to eyeball the actual game image — same
//! generation (`LocalChunkSource` → renderer artifacts), same pipelines the
//! browser composition root drives, but on wgpu's Vulkan backend:
//!
//! ```sh
//! cargo run -p wasm_frontend --example vulkan_snapshot --release
//! ```
//!
//! Output lands in `target/vulkan-snapshots/*.png`. Environment knobs:
//! `SEED` (default 42), `SNAPSHOT_SIZE` (`WxH`, default 960x540).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;

use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::spawn_point;
use wasm_frontend::adapters::cpu_splatter::{CpuRenderSettings, CpuShadowMode};
use wasm_frontend::adapters::local_chunk_source::LocalChunkSource;
use wasm_frontend::application::atlas::{AtlasPool, payload_rows};
use wasm_frontend::application::ports::{
    ChunkDraw, ChunkPayload, ChunkSourcePort, Environment, FrameParams, SurfaceChunk,
};
use wasm_frontend::application::render_settings::RenderToggles;
use wasm_frontend::application::streaming::chunk_key;
use wasm_frontend::drivers::webgpu::frame_resources::FrameResources;
use wasm_frontend::drivers::webgpu::gpu_types::{GpuFrameUniforms, collect_frame_lights};
use wasm_frontend::drivers::webgpu::pipelines::{
    CpuPresentPipeline, RaymarchPipeline, RaymarchRuntimeOptions, SplatPipeline, SurfacePipeline,
};

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const FOV_TAN: f32 = 0.767_327;
const MAX_DRAW_DISTANCE: f32 = 48.0;
const TRACE_BUDGET: u32 = 512;
const FACE_BUDGET: u32 = 262_144;

struct LoadedWorld {
    payloads: Vec<(f32, f32, ChunkPayload)>,
    chunks: Vec<ChunkDraw>,
    atlas: Vec<u32>,
    frame: FrameParams,
}

fn main() {
    let seed: u32 = std::env::var("SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(42);
    let (width, height) = snapshot_size();

    let (device, queue) = create_vulkan_device();
    let world = load_spawn_world(seed);
    eprintln!(
        "world: {} chunks, {} lights, {} atlas words (seed {seed})",
        world.chunks.len(),
        world.frame.scene_lights.len(),
        world.atlas.len(),
    );

    let toggles = RenderToggles::default();
    let lights = collect_frame_lights(&world.frame);
    let uniforms = GpuFrameUniforms::from_frame(
        &world.frame,
        width,
        height,
        FOV_TAN,
        MAX_DRAW_DISTANCE,
        TRACE_BUDGET,
        lights.len(),
        toggles,
    );
    let mut frame_resources = FrameResources::new(&device);
    frame_resources.write(&device, &queue, &uniforms, &lights);

    let out_dir = PathBuf::from("target/vulkan-snapshots");
    fs::create_dir_all(&out_dir).expect("create snapshot directory");

    // Surface: indexed meshes, one upload per chunk.
    let mut surface = SurfacePipeline::new(
        &device,
        TARGET_FORMAT,
        frame_resources.layout(),
        MAX_DRAW_DISTANCE,
    );
    let surface_chunks: Vec<SurfaceChunk> = world
        .payloads
        .iter()
        .map(|(x, z, payload)| SurfaceChunk {
            key: chunk_key(*x, *z),
            origin: [*x, 0.0, *z],
            mesh: &payload.surface,
        })
        .collect();
    surface.upload(&device, &surface_chunks);
    let pixels = render_offscreen(&device, &queue, width, height, true, |encoder, color, depth| {
        surface.draw(
            &queue,
            encoder,
            color,
            depth.expect("surface pass renders with depth"),
            &frame_resources,
            &world.frame,
            toggles,
            world.frame.scene_lights.len() as u32,
        );
    });
    write_png(&out_dir, "surface", width, height, &pixels);

    // Splat: compact face pages expanded on the GPU.
    let mut splat = SplatPipeline::new(
        &device,
        TARGET_FORMAT,
        frame_resources.layout(),
        MAX_DRAW_DISTANCE,
        FACE_BUDGET,
    );
    splat.upload(&device, &surface_chunks);
    let pixels = render_offscreen(&device, &queue, width, height, true, |encoder, color, depth| {
        splat.draw(
            &queue,
            encoder,
            color,
            depth.expect("splat pass renders with depth"),
            &frame_resources,
            &world.frame,
            toggles,
            world.frame.scene_lights.len() as u32,
        );
    });
    write_png(&out_dir, "splat", width, height, &pixels);

    // Raymarch: canonical SVO words traced in the fragment shader.
    let mut raymarch = RaymarchPipeline::new(
        &device,
        TARGET_FORMAT,
        frame_resources.layout(),
        world.chunks.len(),
    );
    raymarch.configure(RaymarchRuntimeOptions::new(MAX_DRAW_DISTANCE, true, 4));
    raymarch.upload_atlas(&device, &world.atlas);
    let pixels = render_offscreen(&device, &queue, width, height, false, |encoder, color, _| {
        raymarch.draw(
            &queue,
            encoder,
            color,
            &frame_resources,
            &world.frame,
            &world.chunks,
            toggles,
        );
    });
    write_png(&out_dir, "raymarch", width, height, &pixels);

    // CPU splatter, presented through the production WebGPU texture pass.
    let mut cpu_present = CpuPresentPipeline::new(&device, TARGET_FORMAT, width, height);
    cpu_present.upload_atlas(&world.atlas);
    let cpu_settings = CpuRenderSettings {
        max_draw_distance: MAX_DRAW_DISTANCE,
        shadows: CpuShadowMode::Off,
        fov_tan: FOV_TAN,
        toggles,
        ..CpuRenderSettings::default()
    };
    let pixels = render_offscreen(&device, &queue, width, height, false, |encoder, color, _| {
        cpu_present.draw(
            &queue,
            encoder,
            color,
            &world.frame,
            &world.chunks,
            cpu_settings,
        );
    });
    write_png(&out_dir, "cpu", width, height, &pixels);

    eprintln!("snapshots written to {}", out_dir.display());
}

fn snapshot_size() -> (u32, u32) {
    let requested = std::env::var("SNAPSHOT_SIZE").unwrap_or_default();
    let parsed = requested
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)));
    let (width, height): (u32, u32) = parsed.unwrap_or((960, 540));
    // Readback rows must be 256-byte aligned; keep it simple by requiring
    // the width itself to align (RGBA8 => 64-pixel multiples).
    assert!(
        (width * 4) % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0,
        "SNAPSHOT_SIZE width must be a multiple of 64 pixels"
    );
    (width.max(64), height.max(1))
}

fn create_vulkan_device() -> (wgpu::Device, wgpu::Queue) {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .unwrap_or_else(|error| {
        panic!("no Vulkan adapter; install Mesa (Lavapipe) or expose a Vulkan GPU: {error}")
    });
    let info = adapter.get_info();
    eprintln!(
        "Vulkan adapter: {} ({}, {})",
        info.name, info.driver, info.driver_info
    );
    let limits = wgpu::Limits::default().or_worse_values_from(&adapter.limits());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vackrooms.vulkan-snapshot"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("failed to create the Vulkan device: {error}"));
    (device, queue)
}

/// Generates the 3x3 chunk neighborhood around the spawn corridor with the
/// same source the browser workers use, and assembles the shared SVO atlas
/// exactly like the streaming engine does.
fn load_spawn_world(seed: u32) -> LoadedWorld {
    let config = GeneratorConfig::low_spec();
    let spawn = spawn_point(seed);
    let origin_x = (spawn.x / config.chunk_size).floor() * config.chunk_size;
    let origin_z = (spawn.z / config.chunk_size).floor() * config.chunk_size;
    let source = LocalChunkSource::new(SimpleNoiseProvider::new(), seed, config);

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
        .map(|(_, _, payload)| payload_rows(payload))
        .max()
        .expect("spawn neighborhood is never empty");
    let mut pool = AtlasPool::new();
    pool.ensure_layout(payloads.len(), max_rows);

    let mut atlas = Vec::new();
    let mut chunks = Vec::new();
    let mut unique_lights = HashMap::new();
    for (x, z, payload) in &payloads {
        let key = chunk_key(*x, *z);
        let slot = pool.assign(key).expect("atlas sized for every chunk");
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
            let quantized = [
                (light.position[0] * 16.0) as i32,
                (light.position[1] * 16.0) as i32,
                (light.position[2] * 16.0) as i32,
            ];
            unique_lights.entry(quantized).or_insert(*light);
        }
    }

    let frame = FrameParams {
        camera_pos: [spawn.x, 1.7, spawn.z],
        yaw: -std::f32::consts::FRAC_PI_2,
        pitch: 0.0,
        scene_lights: unique_lights.into_values().collect(),
        environment: Environment::interior(),
        ..FrameParams::default()
    };

    LoadedWorld {
        payloads,
        chunks,
        atlas,
        frame,
    }
}

fn render_offscreen(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    with_depth: bool,
    draw: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, Option<&wgpu::TextureView>),
) -> Vec<u8> {
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vackrooms.snapshot-color"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color.create_view(&Default::default());
    let depth_view = with_depth.then(|| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("vackrooms.snapshot-depth"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default())
    });

    let bytes_per_row = width * 4;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vackrooms.snapshot-readback"),
        size: u64::from(bytes_per_row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    draw(&mut encoder, &color_view, depth_view.as_ref());
    encoder.copy_texture_to_buffer(
        color.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        extent,
    );
    queue.submit([encoder.finish()]);

    let (mapped_tx, mapped_rx) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = mapped_tx.send(result);
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll for snapshot readback");
    mapped_rx
        .recv()
        .expect("map_async callback")
        .expect("snapshot readback mapping");
    let pixels = readback
        .slice(..)
        .get_mapped_range()
        .expect("snapshot readback range")
        .to_vec();
    readback.unmap();
    pixels
}

fn write_png(dir: &std::path::Path, name: &str, width: u32, height: u32, rgba: &[u8]) {
    let path = dir.join(format!("{name}.png"));
    let image = image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .expect("snapshot buffer matches dimensions");
    image.save(&path).expect("write snapshot png");
    eprintln!("wrote {}", path.display());
}
