//! The frame-loop orchestrator: the application's central use case.
//!
//! `Engine::tick` is called once per animation frame by the browser driver
//! with plain data in (`InputFrame`, dt) and drives everything else through
//! ports: chunk loading via [`ChunkSourcePort`], drawing via [`RendererPort`].
//! No browser types appear anywhere in this file, so the whole game loop is
//! natively unit-testable.

use crate::application::atlas::{build_atlas, MAX_CHUNKS};
use crate::application::collision::CollisionWorld;
use crate::application::player::{MoveIntent, Player};
use crate::application::ports::{ChunkDraw, ChunkSourcePort, FrameParams, RendererPort};
use crate::application::streaming::{chunk_key, ChunkStore, LoadedChunk, StreamingPolicy};

/// Static configuration chosen by the composition root.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub seed: u32,
    /// World-space chunk side length (must match the generator config).
    pub chunk_size: f32,
    /// Streaming radius in chunks (1 = 3x3, 2 = 5x5).
    pub chunk_radius: i32,
    /// Player spawn (eye position).
    pub spawn: [f32; 3],
    /// Chunks generated per tick; kept at 1 so generation never blows the
    /// frame budget (wasm is single-threaded — time-slicing replaces the
    /// background worker a native engine would use).
    pub max_loads_per_tick: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            chunk_size: 10.0,
            chunk_radius: 1,
            spawn: [5.0, 1.7, 5.0],
            max_loads_per_tick: 1,
        }
    }
}

/// One frame's worth of user input, already translated to domain terms by
/// the input adapter.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputFrame {
    pub intent: MoveIntent,
    pub look_dx: f32,
    pub look_dy: f32,
    /// Whether pointer lock is engaged; movement is frozen otherwise.
    pub locked: bool,
}

/// Snapshot for the HUD presenter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudStats {
    pub resident_chunks: usize,
    pub atlas_nodes: usize,
    pub collision_boxes: usize,
    pub resolution_scale: f32,
    /// True once the spawn chunk is resident (drives the loading overlay).
    pub ready: bool,
}

/// Adaptive internal-resolution governor. Low-spec machines get fewer rays
/// per frame by rendering at a reduced backing-store resolution; the driver
/// applies the scale to the canvas. Hysteresis + cooldown prevent flapping.
#[derive(Debug, Clone, Copy)]
pub struct PerfGovernor {
    ema_frame_ms: f32,
    scale: f32,
    cooldown_frames: u32,
}

impl PerfGovernor {
    const SCALES: [f32; 4] = [0.5, 0.65, 0.8, 1.0];

    pub fn new() -> Self {
        Self { ema_frame_ms: 16.0, scale: 0.8, cooldown_frames: 0 }
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn update(&mut self, frame_ms: f32) {
        self.ema_frame_ms = self.ema_frame_ms * 0.95 + frame_ms.clamp(0.0, 100.0) * 0.05;
        if self.cooldown_frames > 0 {
            self.cooldown_frames -= 1;
            return;
        }
        let idx = Self::SCALES.iter().position(|&s| s == self.scale).unwrap_or(2);
        if self.ema_frame_ms > 33.0 && idx > 0 {
            // Sustained under ~30 fps: drop one resolution step.
            self.scale = Self::SCALES[idx - 1];
            self.cooldown_frames = 120;
        } else if self.ema_frame_ms < 15.0 && idx + 1 < Self::SCALES.len() {
            // Sustained over ~66 fps: try one step up.
            self.scale = Self::SCALES[idx + 1];
            self.cooldown_frames = 240;
        }
    }
}

pub struct Engine {
    config: EngineConfig,
    player: Player,
    policy: StreamingPolicy,
    store: ChunkStore,
    world: CollisionWorld,
    draws: Vec<ChunkDraw>,
    atlas_nodes: usize,
    governor: PerfGovernor,
    renderer: Box<dyn RendererPort>,
    source: Box<dyn ChunkSourcePort>,
}

impl Engine {
    pub fn new(
        config: EngineConfig,
        renderer: Box<dyn RendererPort>,
        source: Box<dyn ChunkSourcePort>,
    ) -> Self {
        let radius = config.chunk_radius.clamp(0, 2); // 5x5 max = shader table size
        Self {
            player: Player::new(config.spawn),
            policy: StreamingPolicy { chunk_size: config.chunk_size, radius },
            store: ChunkStore::new(),
            world: CollisionWorld::new(),
            draws: Vec::new(),
            atlas_nodes: 0,
            governor: PerfGovernor::new(),
            renderer,
            source,
            config,
        }
    }

    /// Advances the simulation one frame and issues the draw.
    pub fn tick(&mut self, dt_seconds: f32, input: &InputFrame) {
        // Clamp dt so a background-tab hitch can't teleport the player.
        let dt = dt_seconds.clamp(0.0, 0.1);
        self.governor.update(dt_seconds * 1000.0);

        if input.locked {
            self.player.apply_look(input.look_dx, input.look_dy);
            self.player.step(dt, &input.intent, &self.world);
        }

        self.stream_chunks();

        let frame = FrameParams {
            camera_pos: self.player.position,
            yaw: self.player.yaw,
            pitch: self.player.pitch,
        };
        self.renderer.draw(&frame, &self.draws);
    }

    /// Loads at most `max_loads_per_tick` missing chunks (nearest first),
    /// evicts out-of-range ones, and rebuilds the atlas + collision world
    /// only when the resident set actually changed.
    fn stream_chunks(&mut self) {
        let desired = self
            .policy
            .desired_origins(self.player.position[0], self.player.position[2]);
        let keep: Vec<_> = desired.iter().map(|&(x, z)| chunk_key(x, z)).collect();

        let mut changed = self.store.retain_keys(&keep);

        let mut loads = 0;
        for &(ox, oz) in &desired {
            if loads >= self.config.max_loads_per_tick {
                break;
            }
            let key = chunk_key(ox, oz);
            if !self.store.contains(key) {
                let payload = self.source.load(ox, oz);
                self.store.insert(key, LoadedChunk { origin: (ox, oz), payload });
                loads += 1;
                changed = true;
            }
        }

        if changed {
            let build = build_atlas(self.store.iter_ordered());
            self.atlas_nodes = build.texels.len() / 4;
            self.draws = build.draws;
            self.draws.truncate(MAX_CHUNKS);
            self.renderer.upload_atlas(&build.texels);

            let boxes: Vec<_> = self.store.all_collision_boxes().copied().collect();
            self.world.rebuild(boxes.iter());
        }
    }

    pub fn player(&self) -> &Player {
        &self.player
    }

    pub fn collision_world(&self) -> &CollisionWorld {
        &self.world
    }

    pub fn stats(&self) -> HudStats {
        let spawn_key = chunk_key(
            (self.config.spawn[0] / self.config.chunk_size).floor() * self.config.chunk_size,
            (self.config.spawn[2] / self.config.chunk_size).floor() * self.config.chunk_size,
        );
        HudStats {
            resident_chunks: self.store.len(),
            atlas_nodes: self.atlas_nodes,
            collision_boxes: self.world.len(),
            resolution_scale: self.governor.scale(),
            ready: self.store.contains(spawn_key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::collision::Aabb;
    use crate::application::ports::ChunkPayload;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct RecordingRenderer {
        uploads: Rc<RefCell<Vec<usize>>>,
        draws: Rc<RefCell<Vec<usize>>>,
    }

    impl RendererPort for RecordingRenderer {
        fn upload_atlas(&mut self, texels: &[u32]) {
            self.uploads.borrow_mut().push(texels.len());
        }
        fn draw(&mut self, _frame: &FrameParams, chunks: &[ChunkDraw]) {
            self.draws.borrow_mut().push(chunks.len());
        }
    }

    struct FlatChunkSource;

    impl ChunkSourcePort for FlatChunkSource {
        fn load(&self, origin_x: f32, _origin_z: f32) -> ChunkPayload {
            ChunkPayload {
                root: 0,
                nodes: [1u32, 0, 0, 0].repeat(1024), // one padded row of air leaves
                world_size: 12.8,
                collision: vec![Aabb::new(
                    [origin_x, 0.0, 0.0],
                    [origin_x + 0.2, 3.0, 0.2],
                )],
            }
        }
    }

    fn engine_with_recorder() -> (Engine, Rc<RefCell<Vec<usize>>>, Rc<RefCell<Vec<usize>>>) {
        let renderer = RecordingRenderer::default();
        let uploads = renderer.uploads.clone();
        let draws = renderer.draws.clone();
        let engine = Engine::new(
            EngineConfig::default(),
            Box::new(renderer),
            Box::new(FlatChunkSource),
        );
        (engine, uploads, draws)
    }

    #[test]
    fn streams_one_chunk_per_tick_until_radius_filled() {
        let (mut engine, uploads, draws) = engine_with_recorder();
        let input = InputFrame::default();
        for _ in 0..9 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.stats().resident_chunks, 9);
        // Atlas re-uploaded on every tick that loaded a chunk.
        assert_eq!(uploads.borrow().len(), 9);
        // Every tick drew exactly the resident chunk count.
        assert_eq!(*draws.borrow().last().unwrap(), 9);
        assert!(engine.stats().ready);
    }

    #[test]
    fn steady_state_does_not_reupload_atlas() {
        let (mut engine, uploads, _) = engine_with_recorder();
        let input = InputFrame::default();
        for _ in 0..20 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(uploads.borrow().len(), 9, "no uploads once resident set is stable");
    }

    #[test]
    fn collision_world_tracks_resident_chunks() {
        let (mut engine, _, _) = engine_with_recorder();
        let input = InputFrame::default();
        for _ in 0..9 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.collision_world().len(), 9);
    }

    #[test]
    fn governor_drops_resolution_under_sustained_load() {
        let mut governor = PerfGovernor::new();
        for _ in 0..200 {
            governor.update(50.0); // 20 fps
        }
        assert!(governor.scale() < 0.8);
    }
}
