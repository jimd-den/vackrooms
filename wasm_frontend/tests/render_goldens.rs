//! Fast, in-memory renderer regression for one deliberately tiny room.
//!
//! The PNGs under `tests/golden/` are preserved as developer artifacts, but
//! ordinary `cargo test` never writes them. This test checks the scene and
//! rendered image semantics directly, avoiding machine-specific pixel churn.

use vackrooms::domain::entities::voxel_grid::{
    EMISSIVE_MATERIALS, VOXEL_AIR, VOXEL_CEILING, VOXEL_LIGHT,
};
use wasm_frontend::adapters::cpu_splatter::settings::{CpuRenderSettings, CpuShadowMode};
use wasm_frontend::application::ports::Environment;
use wasm_frontend::application::render_settings::RenderToggles;
use wasm_frontend::core::domain::room::{CameraSpec, RoomFixture, RoomScene};
use wasm_frontend::core::ports::reference_renderer::{ReferenceRenderSettings, RenderedImage};
use wasm_frontend::core::usecases::build_render_scene::build_render_scene;
use wasm_frontend::core::usecases::render_reference::render_reference;
use wasm_frontend::drivers::cpu_reference_renderer::CpuReferenceRenderer;
use wasm_frontend::drivers::raymarch_reference_renderer::RaymarchReferenceRenderer;

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;

struct SingleCeilingFixture;

impl RoomFixture for SingleCeilingFixture {
    fn id(&self) -> &'static str {
        "single_ceiling_fixture"
    }

    fn build(&self) -> RoomScene {
        RoomScene::single_ceiling_fixture()
    }

    fn camera(&self) -> CameraSpec {
        CameraSpec {
            position: [2.25, 1.5, 3.25],
            yaw: 0.0,
            pitch: 0.1,
            fov_degrees: 75.0,
        }
    }

    fn environment(&self) -> Environment {
        Environment::interior()
    }
}

fn reference_settings(fixture: &impl RoomFixture) -> ReferenceRenderSettings {
    ReferenceRenderSettings {
        width: WIDTH,
        height: HEIGHT,
        camera: fixture.camera(),
        environment: fixture.environment(),
        toggles: RenderToggles {
            hierarchical_z: false,
            front_to_back: false,
            empty_space_skip: false,
            mip_lod: false,
            flashlight_occlusion: false,
            shadow_pass: false,
            cell_culling: false,
            face_budget: false,
            distance_cull: false,
            dither: false,
            baked_lighting: true,
            gpu_timer: false,
        },
        cpu: CpuRenderSettings {
            max_splat_half_px: 4.0,
            fog_density: 0.0,
            max_draw_distance: 16.0,
            shadows: CpuShadowMode::Off,
            ..CpuRenderSettings::default()
        },
    }
}

#[test]
fn fixture_has_one_exposed_ceiling_panel_and_no_floor_emissive() {
    let fixture = SingleCeilingFixture;
    assert_eq!(fixture.id(), "single_ceiling_fixture");
    assert_fixture_geometry(&fixture.build());
}

#[test]
fn reference_renderers_produce_deterministic_lit_room_images() {
    let fixture = SingleCeilingFixture;
    let scene = build_render_scene(fixture.build());
    let settings = reference_settings(&fixture);

    assert_eq!(scene.chunks.len(), 1);
    assert!(!scene.atlas.is_empty());
    assert_eq!(scene.chunks[0].world_size, 4.0);
    assert_eq!(scene.chunks[0].voxel_size, 0.5);
    assert_eq!(scene.chunks[0].svo_depth, 3);

    let cpu = render_reference(&scene, &settings, CpuReferenceRenderer::new());
    let cpu_repeat = render_reference(&scene, &settings, CpuReferenceRenderer::new());
    assert_eq!(
        cpu, cpu_repeat,
        "CPU reference rendering must be repeatable"
    );
    assert_lit_room("CPU splatter", &cpu);

    let raymarch = render_reference(&scene, &settings, RaymarchReferenceRenderer::new());
    let raymarch_repeat = render_reference(&scene, &settings, RaymarchReferenceRenderer::new());
    assert_eq!(
        raymarch, raymarch_repeat,
        "raymarch reference rendering must be repeatable"
    );
    assert_lit_room("CPU raymarch", &raymarch);
}

fn assert_fixture_geometry(scene: &RoomScene) {
    let voxels = scene.voxels();
    let mut emissive = Vec::new();
    for y in 0..voxels.height() {
        for z in 0..voxels.depth() {
            for x in 0..voxels.width() {
                if EMISSIVE_MATERIALS.contains(&voxels.get(x, y, z)) {
                    emissive.push((x, y, z));
                }
            }
        }
    }

    assert_eq!(emissive.len(), 1, "fixture must contain one emitter");
    let (x, y, z) = emissive[0];
    assert_eq!(voxels.get(x, y, z), VOXEL_LIGHT);
    assert!(y > 0);
    assert_eq!(
        voxels.get(x, y - 1, z),
        VOXEL_AIR,
        "panel underside must be exposed to room air"
    );
    assert_eq!(voxels.get(x - 1, y, z), VOXEL_CEILING);
    assert_eq!(voxels.get(x + 1, y, z), VOXEL_CEILING);
    assert_eq!(voxels.get(x, y, z - 1), VOXEL_CEILING);
    assert_eq!(voxels.get(x, y, z + 1), VOXEL_CEILING);

    for z in 0..voxels.depth() {
        for x in 0..voxels.width() {
            assert!(
                !EMISSIVE_MATERIALS.contains(&voxels.get(x, 0, z)),
                "floor voxel ({x}, 0, {z}) must not emit"
            );
        }
    }
}

fn assert_lit_room(renderer: &str, image: &RenderedImage) {
    assert_eq!((image.width, image.height), (WIDTH, HEIGHT));
    assert_eq!(image.rgba.len(), (WIDTH * HEIGHT * 4) as usize);

    let mut visible_pixels = 0usize;
    let mut fixture_pixels = 0usize;
    for (index, pixel) in image.rgba.chunks_exact(4).enumerate() {
        assert_eq!(pixel[3], 255, "{renderer} produced non-opaque output");
        if pixel[..3] != [0, 0, 0] {
            visible_pixels += 1;
        }
        let x = index % image.width as usize;
        let y = index / image.width as usize;
        let in_panel_window = x >= image.width as usize / 3
            && x <= image.width as usize * 2 / 3
            && y < image.height as usize / 3;
        if in_panel_window && pixel[0] >= 240 && pixel[1] >= 225 && pixel[2] >= 190 {
            fixture_pixels += 1;
        }
    }

    let pixel_count = (WIDTH * HEIGHT) as usize;
    assert!(
        visible_pixels > pixel_count / 20,
        "{renderer} failed to show the baked-lit room: {visible_pixels}/{pixel_count} visible pixels"
    );
    assert!(
        fixture_pixels > 0,
        "{renderer} failed to show the exposed fluorescent panel"
    );
}
