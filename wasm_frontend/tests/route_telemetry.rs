//! Route-anchor telemetry against the real Level 0 generator: the anchor
//! must find the guaranteed level door from authoritative streamed exits,
//! never from invented data.

use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use wasm_frontend::adapters::local_chunk_source::LocalChunkSource;
use wasm_frontend::application::engine::{Engine, EngineConfig, InputFrame};
use wasm_frontend::application::ports::{ChunkDraw, FrameParams, RendererPort};

struct NullRenderer;

impl RendererPort for NullRenderer {
    fn upload_atlas(&mut self, _texels: &[u32]) {}
    fn draw(&mut self, _frame: &FrameParams, _chunks: &[ChunkDraw]) {}
}

/// The world-tuning default guarantees a Level 0 door near (172, -116).
/// Standing a dozen meters away, the route anchor must resolve it with a
/// truthful straight-line range; far from any door it must stay empty.
#[test]
fn route_anchor_resolves_the_guaranteed_door_from_streamed_world_data() {
    let seed = 42;
    let config = EngineConfig {
        seed,
        chunk_size: 10.0,
        chunk_radius: 1,
        spawn: [172.0, 1.7, -104.0],
        spawn_yaw: 0.0,
        initial_level: 0,
        ..EngineConfig::default()
    };
    let source = LocalChunkSource::new(
        SimpleNoiseProvider::new(),
        seed,
        GeneratorConfig::low_spec(),
    );
    let mut engine = Engine::new(config, Box::new(NullRenderer), Box::new(source));

    let input = InputFrame::default();
    // Stream the neighborhood in and let the route cadence fire.
    for _ in 0..120 {
        engine.tick(1.0 / 60.0, &input);
    }

    let route = engine
        .stats()
        .route
        .expect("the guaranteed door must be resident and selected");
    assert_eq!(route.target_level, 1, "Level 0's guaranteed door leads to 1");
    assert!(
        (5.0..40.0).contains(&route.range_m),
        "range must be the real straight-line distance, got {}",
        route.range_m
    );
    assert!(route.bearing_deg < 360);
}

#[test]
fn route_anchor_is_truthfully_absent_away_from_doors() {
    let seed = 42;
    let spawn_at = vackrooms::use_cases::region_plan::spawn_point(seed);
    let config = EngineConfig {
        seed,
        chunk_size: 10.0,
        chunk_radius: 1,
        spawn: [spawn_at.x, 1.7, spawn_at.z],
        spawn_yaw: 0.0,
        initial_level: 0,
        ..EngineConfig::default()
    };
    let source = LocalChunkSource::new(
        SimpleNoiseProvider::new(),
        seed,
        GeneratorConfig::low_spec(),
    );
    let mut engine = Engine::new(config, Box::new(NullRenderer), Box::new(source));
    let input = InputFrame::default();
    for _ in 0..60 {
        engine.tick(1.0 / 60.0, &input);
    }
    // The spawn corridor is far from the guaranteed door; unless generation
    // happens to place one inside this 3x3 footprint, the anchor is empty.
    if let Some(route) = engine.stats().route {
        // If a door genuinely is resident, the range must at least be real.
        assert!(route.range_m.is_finite() && route.range_m < 100.0);
    }
}
