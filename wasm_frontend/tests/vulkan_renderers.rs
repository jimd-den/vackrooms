//! Headless contracts for the production WebGPU pipelines.
//!
//! This target is feature-gated because requesting Vulkan is an explicit
//! environment contract, not something an ordinary portable `cargo test`
//! should silently skip. CI exposes only Mesa Lavapipe and enables validation
//! layers; a developer invoking this target without Vulkan gets a direct
//! adapter-creation failure rather than another backend.

use std::sync::mpsc;

use vackrooms::domain::entities::voxel_grid::VoxelGrid;
use wasm_frontend::adapters::cpu_splatter::{CpuRenderSettings, CpuShadowMode};
use wasm_frontend::adapters::surface_mesh::build_surface_artifacts;
use wasm_frontend::application::ports::{
    Environment, FrameParams, RenderArtifactNeeds, SupplySprite, SurfaceChunk, SurfaceMeshPayload,
};
use wasm_frontend::application::render_settings::RenderToggles;
use wasm_frontend::drivers::webgpu::config::ShadowMapConfig;
use wasm_frontend::drivers::webgpu::frame_resources::FrameResources;
use wasm_frontend::drivers::webgpu::gpu_types::{GpuFrameUniforms, collect_frame_lights};
use wasm_frontend::drivers::webgpu::pipelines::{
    CpuPresentPipeline, HeroShadowOptions, RaymarchPipeline, RaymarchRuntimeOptions, SplatPipeline,
    SupplyLabelPipeline, SurfacePipeline,
};
use wasm_frontend::reference::{RenderSceneSnapshot, RoomScene, build_render_scene};

const WIDTH: u32 = 48;
const HEIGHT: u32 = 32;
const RGBA_BYTES_PER_ROW: u32 = WIDTH * 4;
const COPY_BYTES_PER_ROW: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const FOV_TAN: f32 = 0.767_327;

struct VulkanContext {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct Fixture {
    surface_mesh: SurfaceMeshPayload,
    splat_mesh: SurfaceMeshPayload,
    scene: RenderSceneSnapshot,
    frame: FrameParams,
    toggles: RenderToggles,
}

struct ProductionPipelines {
    frame: FrameResources,
    surface: SurfacePipeline,
    splat: SplatPipeline,
    raymarch: RaymarchPipeline,
    cpu_present: CpuPresentPipeline,
    supply_labels: SupplyLabelPipeline,
}

#[test]
fn every_production_pipeline_renders_offscreen_on_vulkan() {
    assert_eq!(COPY_BYTES_PER_ROW, 256);
    assert!(RGBA_BYTES_PER_ROW < COPY_BYTES_PER_ROW);

    let gpu = create_vulkan_context();
    let fixture = deterministic_fixture();
    let mut pipelines = create_production_pipelines(&gpu);
    upload_fixture(&gpu, &fixture, &mut pipelines);

    let surface = render_offscreen(&gpu, "surface", |encoder, target, depth| {
        pipelines.surface.draw(
            &gpu.queue,
            encoder,
            target,
            depth,
            &pipelines.frame,
            &fixture.frame,
            fixture.toggles,
            fixture.frame.scene_lights.len() as u32,
        );
    });
    assert_non_clear_variation("surface", &surface);

    let surface_with_label = render_offscreen(
        &gpu,
        "surface-with-supply-label",
        |encoder, target, depth| {
            pipelines.surface.draw(
                &gpu.queue,
                encoder,
                target,
                depth,
                &pipelines.frame,
                &fixture.frame,
                fixture.toggles,
                fixture.frame.scene_lights.len() as u32,
            );
            pipelines.supply_labels.draw(
                &gpu.queue,
                encoder,
                target,
                depth,
                &pipelines.frame,
                &fixture.frame,
            );
        },
    );
    assert_non_clear_variation("surface-with-supply-label", &surface_with_label);
    assert_ne!(
        surface, surface_with_label,
        "the production GPU-expanded supply-label pass must affect visible pixels"
    );

    let splat = render_offscreen(&gpu, "splat", |encoder, target, depth| {
        pipelines.splat.draw(
            &gpu.queue,
            encoder,
            target,
            depth,
            &pipelines.frame,
            &fixture.frame,
            fixture.toggles,
            fixture.frame.scene_lights.len() as u32,
        );
    });
    assert_non_clear_variation("splat", &splat);

    let raymarch = render_offscreen(&gpu, "raymarch", |encoder, target, _depth| {
        pipelines.raymarch.draw(
            &gpu.queue,
            encoder,
            target,
            &pipelines.frame,
            &fixture.frame,
            &fixture.scene.chunks,
            fixture.toggles,
        );
    });
    assert_non_clear_variation("raymarch", &raymarch);

    let mut reference_toggles = fixture.toggles;
    reference_toggles.empty_space_skip = false;
    let raymarch_reference =
        render_offscreen(&gpu, "raymarch-reference-dda", |encoder, target, _depth| {
            pipelines.raymarch.draw(
                &gpu.queue,
                encoder,
                target,
                &pipelines.frame,
                &fixture.frame,
                &fixture.scene.chunks,
                reference_toggles,
            );
        });
    assert_eq!(
        raymarch, raymarch_reference,
        "empty-leaf skipping must refine to the canonical finest-DDA image"
    );

    let mut visibility_toggles = fixture.toggles;
    visibility_toggles.shadow_pass = true;
    let raymarch_visibility = render_offscreen(
        &gpu,
        "raymarch-direct-visibility",
        |encoder, target, _depth| {
            pipelines.raymarch.draw(
                &gpu.queue,
                encoder,
                target,
                &pipelines.frame,
                &fixture.frame,
                &fixture.scene.chunks,
                visibility_toggles,
            );
        },
    );
    assert_non_clear_variation("raymarch-direct-visibility", &raymarch_visibility);

    exercise_raster_shadow_profiles(&gpu, &fixture, &mut pipelines);

    let mut cpu_settings = CpuRenderSettings {
        max_draw_distance: 16.0,
        shadows: CpuShadowMode::Off,
        fov_tan: FOV_TAN,
        toggles: fixture.toggles,
        ..CpuRenderSettings::default()
    };
    cpu_settings.toggles.empty_space_skip = true;
    let cpu_present = render_offscreen(&gpu, "cpu-present", |encoder, target, _depth| {
        pipelines.cpu_present.draw(
            &gpu.queue,
            encoder,
            target,
            &fixture.frame,
            &fixture.scene.chunks,
            cpu_settings,
        );
    });
    assert_non_clear_variation("cpu-present", &cpu_present);
}

fn create_vulkan_context() -> VulkanContext {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    descriptor.flags =
        wgpu::InstanceFlags::debugging() | wgpu::InstanceFlags::STRICT_WEBGPU_COMPLIANCE;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .unwrap_or_else(|error| {
        panic!(
            "Vulkan renderer tests require a Vulkan 1.2+ adapter; install Mesa Lavapipe or expose a Vulkan device: {error}"
        )
    });
    let info = adapter.get_info();
    assert_eq!(
        info.backend,
        wgpu::Backend::Vulkan,
        "test harness must never fall back from Vulkan; selected {info:?}"
    );
    eprintln!(
        "Vulkan test adapter: {} ({}, {})",
        info.name, info.driver, info.driver_info
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vackrooms.vulkan-test-device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::defaults(),
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("failed to create the Vulkan test device: {error}"));

    VulkanContext {
        _instance: instance,
        device,
        queue,
    }
}

fn deterministic_fixture() -> Fixture {
    let room = RoomScene::single_ceiling_fixture();
    let voxel_size = room.voxel_size();
    let halo = add_lateral_air_halo(room.voxels());
    let surface_mesh =
        build_surface_artifacts(&halo, voxel_size, 0, 1, RenderArtifactNeeds::SURFACE);
    let splat_mesh = build_surface_artifacts(&halo, voxel_size, 0, 1, RenderArtifactNeeds::SPLAT);
    let scene = build_render_scene(room);
    let frame = FrameParams {
        camera_pos: [2.25, 1.5, 3.25],
        yaw: 0.0,
        pitch: 0.1,
        scene_lights: scene.scene_lights.clone(),
        environment: Environment::interior(),
        supply_sprites: vec![SupplySprite {
            position: [2.25, 1.5, 2.6],
            atlas_row: 0,
        }],
        ..FrameParams::default()
    };
    let toggles = RenderToggles {
        hierarchical_z: false,
        front_to_back: true,
        empty_space_skip: true,
        mip_lod: false,
        flashlight_occlusion: false,
        deferred_shading: false,
        ambient_occlusion: false,
        shadow_pass: false,
        cell_culling: false,
        face_budget: false,
        distance_cull: false,
        dither: false,
        baked_lighting: false,
    };
    Fixture {
        surface_mesh,
        splat_mesh,
        scene,
        frame,
        toggles,
    }
}

fn add_lateral_air_halo(source: &VoxelGrid) -> VoxelGrid {
    let mut halo = VoxelGrid::new(source.width() + 2, source.height(), source.depth() + 2);
    for z in 0..source.depth() {
        for y in 0..source.height() {
            for x in 0..source.width() {
                halo.set(x + 1, y, z + 1, source.get(x, y, z));
                halo.set_light_rgb(x + 1, y, z + 1, source.get_light_rgb(x, y, z));
            }
        }
    }
    halo
}

fn create_production_pipelines(gpu: &VulkanContext) -> ProductionPipelines {
    let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let frame = FrameResources::new(&gpu.device);
    let surface = SurfacePipeline::new(&gpu.device, TARGET_FORMAT, frame.layout(), 16.0);
    let splat = SplatPipeline::new(&gpu.device, TARGET_FORMAT, frame.layout(), 16.0, 16_384);
    let mut raymarch = RaymarchPipeline::new(&gpu.device, TARGET_FORMAT, frame.layout(), 1);
    raymarch.configure(RaymarchRuntimeOptions::new(16.0, true, 4));
    let cpu_present = CpuPresentPipeline::new(&gpu.device, TARGET_FORMAT, WIDTH, HEIGHT);
    let supply_labels = SupplyLabelPipeline::new(&gpu.device, TARGET_FORMAT, frame.layout());
    assert_validation_clean(&gpu.device, validation, "production pipeline creation");
    ProductionPipelines {
        frame,
        surface,
        splat,
        raymarch,
        cpu_present,
        supply_labels,
    }
}

fn upload_fixture(gpu: &VulkanContext, fixture: &Fixture, pipelines: &mut ProductionPipelines) {
    let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let lights = collect_frame_lights(&fixture.frame);
    let frame = GpuFrameUniforms::from_frame(
        &fixture.frame,
        WIDTH,
        HEIGHT,
        FOV_TAN,
        16.0,
        512,
        lights.len(),
        fixture.toggles,
    );
    pipelines
        .frame
        .write(&gpu.device, &gpu.queue, &frame, &lights);
    let surface_chunk = SurfaceChunk {
        key: (0, 0),
        origin: [0.0; 3],
        mesh: &fixture.surface_mesh,
    };
    let splat_chunk = SurfaceChunk {
        key: (0, 0),
        origin: [0.0; 3],
        mesh: &fixture.splat_mesh,
    };
    pipelines.surface.upload(&gpu.device, &[surface_chunk]);
    pipelines.splat.upload(&gpu.device, &[splat_chunk]);
    pipelines
        .raymarch
        .upload_atlas(&gpu.device, &fixture.scene.atlas);
    pipelines.cpu_present.upload_atlas(&fixture.scene.atlas);
    let label_atlas = [
        236, 214, 147, 255, 236, 214, 147, 255, 236, 214, 147, 255, 236, 214, 147, 255, 196, 76,
        51, 255, 196, 76, 51, 255, 196, 76, 51, 255, 196, 76, 51, 255,
    ];
    pipelines
        .supply_labels
        .upload_atlas(&gpu.device, &gpu.queue, &label_atlas, 2, 4);
    gpu.queue.submit([]);
    assert_validation_clean(&gpu.device, validation, "fixture upload");
}

fn exercise_raster_shadow_profiles(
    gpu: &VulkanContext,
    fixture: &Fixture,
    pipelines: &mut ProductionPipelines,
) {
    let mut toggles = fixture.toggles;
    toggles.shadow_pass = true;

    let surface_low = render_offscreen(gpu, "surface-shadow-low", |encoder, target, depth| {
        pipelines.surface.draw(
            &gpu.queue,
            encoder,
            target,
            depth,
            &pipelines.frame,
            &fixture.frame,
            toggles,
            fixture.frame.scene_lights.len() as u32,
        );
    });
    assert_non_clear_variation("surface-shadow-low", &surface_low);

    let splat_low = render_offscreen(gpu, "splat-shadow-low", |encoder, target, depth| {
        pipelines.splat.draw(
            &gpu.queue,
            encoder,
            target,
            depth,
            &pipelines.frame,
            &fixture.frame,
            toggles,
            fixture.frame.scene_lights.len() as u32,
        );
    });
    assert_non_clear_variation("splat-shadow-low", &splat_low);

    let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let high_options = HeroShadowOptions::new(ShadowMapConfig::high(), true);
    let mut surface_high = SurfacePipeline::new_with_shadow_options(
        &gpu.device,
        TARGET_FORMAT,
        pipelines.frame.layout(),
        16.0,
        high_options,
    );
    let mut splat_high = SplatPipeline::new_with_shadow_options(
        &gpu.device,
        TARGET_FORMAT,
        pipelines.frame.layout(),
        16.0,
        16_384,
        high_options,
    );
    upload_raster_fixture(gpu, fixture, &mut surface_high, &mut splat_high);
    assert_validation_clean(
        &gpu.device,
        validation,
        "high shadow pipeline creation and upload",
    );

    let surface_high_pixels =
        render_offscreen(gpu, "surface-shadow-high", |encoder, target, depth| {
            surface_high.draw(
                &gpu.queue,
                encoder,
                target,
                depth,
                &pipelines.frame,
                &fixture.frame,
                toggles,
                fixture.frame.scene_lights.len() as u32,
            );
        });
    assert_non_clear_variation("surface-shadow-high", &surface_high_pixels);

    let splat_high_pixels = render_offscreen(gpu, "splat-shadow-high", |encoder, target, depth| {
        splat_high.draw(
            &gpu.queue,
            encoder,
            target,
            depth,
            &pipelines.frame,
            &fixture.frame,
            toggles,
            fixture.frame.scene_lights.len() as u32,
        );
    });
    assert_non_clear_variation("splat-shadow-high", &splat_high_pixels);
}

fn upload_raster_fixture(
    gpu: &VulkanContext,
    fixture: &Fixture,
    surface: &mut SurfacePipeline,
    splat: &mut SplatPipeline,
) {
    surface.upload(
        &gpu.device,
        &[SurfaceChunk {
            key: (0, 0),
            origin: [0.0; 3],
            mesh: &fixture.surface_mesh,
        }],
    );
    splat.upload(
        &gpu.device,
        &[SurfaceChunk {
            key: (0, 0),
            origin: [0.0; 3],
            mesh: &fixture.splat_mesh,
        }],
    );
}

fn render_offscreen(
    gpu: &VulkanContext,
    label: &str,
    draw: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, &wgpu::TextureView),
) -> Vec<u8> {
    let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let extent = wgpu::Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };
    let color = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vackrooms.vulkan-test-color"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vackrooms.vulkan-test-depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vackrooms.vulkan-test-readback"),
        size: u64::from(COPY_BYTES_PER_ROW * HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    draw(&mut encoder, &color_view, &depth_view);
    encoder.copy_texture_to_buffer(
        color.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(COPY_BYTES_PER_ROW),
                rows_per_image: Some(HEIGHT),
            },
        },
        extent,
    );
    gpu.queue.submit([encoder.finish()]);
    let validation_result = validation.pop();

    let (mapped_tx, mapped_rx) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            mapped_tx.send(result).expect("map receiver remains alive");
        });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap_or_else(|error| panic!("{label}: device polling failed: {error}"));
    mapped_rx
        .recv()
        .expect("Vulkan map callback must run")
        .unwrap_or_else(|error| panic!("{label}: readback mapping failed: {error}"));

    let pixels = {
        let padded = readback
            .slice(..)
            .get_mapped_range()
            .unwrap_or_else(|error| panic!("{label}: mapped range is unavailable: {error}"));
        let mut pixels = Vec::with_capacity((RGBA_BYTES_PER_ROW * HEIGHT) as usize);
        for row in padded.chunks_exact(COPY_BYTES_PER_ROW as usize) {
            pixels.extend_from_slice(&row[..RGBA_BYTES_PER_ROW as usize]);
        }
        pixels
    };
    readback.unmap();
    if let Some(error) = pollster::block_on(validation_result) {
        panic!("{label}: WebGPU/Vulkan validation error: {error}");
    }
    pixels
}

fn assert_validation_clean(device: &wgpu::Device, validation: wgpu::ErrorScopeGuard, label: &str) {
    let result = validation.pop();
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap_or_else(|error| panic!("{label}: device polling failed: {error}"));
    if let Some(error) = pollster::block_on(result) {
        panic!("{label}: WebGPU/Vulkan validation error: {error}");
    }
}

fn assert_non_clear_variation(label: &str, rgba: &[u8]) {
    assert_eq!(rgba.len(), (WIDTH * HEIGHT * 4) as usize);
    let first = &rgba[..3];
    let mut different_from_clear = 0usize;
    let mut min_luminance = u16::MAX;
    let mut max_luminance = 0u16;
    for pixel in rgba.chunks_exact(4) {
        assert_eq!(pixel[3], 255, "{label} emitted non-opaque pixels");
        different_from_clear += usize::from(&pixel[..3] != first);
        let luminance = u16::from(pixel[0]) + u16::from(pixel[1]) + u16::from(pixel[2]);
        min_luminance = min_luminance.min(luminance);
        max_luminance = max_luminance.max(luminance);
    }
    assert!(
        different_from_clear > 8,
        "{label} remained at its clear color: first={first:?}, changed={different_from_clear}"
    );
    assert!(
        max_luminance.saturating_sub(min_luminance) > 6,
        "{label} produced no meaningful scene contrast: min={min_luminance}, max={max_luminance}"
    );
}
