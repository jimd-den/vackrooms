use std::path::Path;
use wasm_frontend::adapters::cpu_splatter::settings::{CpuRenderSettings, CpuShadowMode};
use wasm_frontend::application::ports::{ChunkDraw, ChunkSourcePort, Environment};
use wasm_frontend::application::render_settings::RenderToggles;
use wasm_frontend::core::domain::room::{CameraSpec, RoomFixture, RoomScene};
use wasm_frontend::core::ports::artifact_sink::ArtifactSinkPort;
use wasm_frontend::core::ports::reference_renderer::{
    ReferenceRenderSettings, ReferenceRendererPort, RenderSceneSnapshot,
};
use wasm_frontend::drivers::cpu_reference_renderer::CpuReferenceRenderer;
use wasm_frontend::drivers::fs_artifact_sink::FsArtifactSink;
use wasm_frontend::drivers::raymarch_reference_renderer::RaymarchReferenceRenderer;

use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use wasm_frontend::adapters::local_chunk_source::LocalChunkSource;
use wasm_frontend::application::atlas::{AtlasPool, payload_rows};
use wasm_frontend::application::streaming::{LoadedChunk, chunk_key};

pub struct SingleCeilingFixture;

impl RoomFixture for SingleCeilingFixture {
    fn id(&self) -> &'static str {
        "single_ceiling_fixture"
    }

    fn build(&self) -> RoomScene {
        RoomScene {}
    }

    fn camera(&self) -> CameraSpec {
        CameraSpec {
            position: [5.0, 1.7, 5.0], // Look from center of chunk
            yaw: 0.0,
            pitch: -0.2, // Tilt slightly down to see floor
            fov_degrees: 90.0,
        }
    }

    fn environment(&self) -> Environment {
        Environment::interior()
    }
}

fn build_fixture_scene(_fixture: &impl RoomFixture) -> RenderSceneSnapshot {
    let seed = 42;
    let config = GeneratorConfig::low_spec();
    let source = LocalChunkSource::new(SimpleNoiseProvider::new(), seed, config);

    // Load a 3x3 grid of chunks around origin to ensure we have geometry enclosing the camera
    let mut loaded_chunks = Vec::new();
    for dz in -1..=1 {
        for dx in -1..=1 {
            let ox = dx as f32 * 10.0;
            let oz = dz as f32 * 10.0;
            let payload = source.load(ox, oz, 0, 0);
            loaded_chunks.push(LoadedChunk::new(
                (ox, oz),
                0,
                vackrooms::domain::entities::anomaly::RealitySnapshot::empty(),
                payload,
            ));
        }
    }

    // Build unified SVO atlas using AtlasPool
    let mut pool = AtlasPool::new();
    let rows = loaded_chunks
        .iter()
        .map(|c| payload_rows(&c.payload))
        .max()
        .unwrap_or(1)
        .max(1);
    pool.ensure_layout(loaded_chunks.len(), rows);

    let mut draws = Vec::new();
    for c in &loaded_chunks {
        let key = chunk_key(c.origin.0, c.origin.1);
        pool.assign(key).expect("pool sized for all chunks");
        let offset = pool.node_offset_of(key).unwrap();
        draws.push(ChunkDraw {
            origin: [c.origin.0, 0.0, c.origin.1],
            root_index: (offset + c.payload.root as usize) as i32,
            world_size: c.payload.world_size,
            voxel_size: c.payload.voxel_size,
            svo_depth: c.payload.svo_depth,
        });
    }

    let atlas = pool.full_texels(|k| {
        loaded_chunks
            .iter()
            .find(|c| chunk_key(c.origin.0, c.origin.1) == k)
            .map(|c| &c.payload)
    });

    RenderSceneSnapshot {
        atlas,
        chunks: draws,
    }
}

fn reference_settings(fixture: &impl RoomFixture) -> ReferenceRenderSettings {
    ReferenceRenderSettings {
        width: 1920,
        height: 1080,
        camera: fixture.camera(),
        environment: fixture.environment(),
        fixed_frame: 0,
        toggles: RenderToggles {
            hierarchical_z: false,
            front_to_back: true,
            empty_space_skip: true,
            mip_lod: true,
            flashlight_occlusion: false,
            shadow_pass: false,
            cell_culling: false,
            face_budget: false,
            distance_cull: true,
            dither: false,
            baked_lighting: true,
            gpu_timer: false,
        },
        cpu: CpuRenderSettings {
            internal_scale: 1.0,
            shadows: CpuShadowMode::Off,
            ..CpuRenderSettings::default()
        },
    }
}

#[test]
fn generate_baseline_images() {
    let fixture = SingleCeilingFixture;
    let scene = build_fixture_scene(&fixture);
    let settings = reference_settings(&fixture);
    let sink = FsArtifactSink;

    let out_dir = Path::new("tests/golden");
    std::fs::create_dir_all(out_dir).unwrap();

    // 1. Render CPU Splatter
    let mut cpu_renderer = CpuReferenceRenderer::new();
    let cpu_image = cpu_renderer.render(&scene, &settings);
    sink.write_png(&out_dir.join("single_ceiling_fixture_cpu.png"), &cpu_image)
        .unwrap();

    // 2. Render CPU Raymarcher
    let mut raymarch_renderer = RaymarchReferenceRenderer::new();
    let raymarch_image = raymarch_renderer.render(&scene, &settings);
    sink.write_png(
        &out_dir.join("single_ceiling_fixture_raymarch.png"),
        &raymarch_image,
    )
    .unwrap();

    assert!(out_dir.join("single_ceiling_fixture_cpu.png").exists());
    assert!(out_dir.join("single_ceiling_fixture_raymarch.png").exists());
}
