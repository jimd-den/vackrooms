//! The frame-loop orchestrator: the application's central use case.
//!
//! `Engine::tick` is called once per animation frame by the browser driver
//! with plain data in (`InputFrame`, dt) and drives everything else through
//! ports: chunk loading via [`ChunkSourcePort`], drawing via [`RendererPort`].
//! No browser types appear anywhere in this file, so the whole game loop is
//! natively unit-testable.

use crate::application::atlas::{MAX_CHUNKS, build_atlas};
use crate::application::collision::CollisionWorld;
use crate::application::player::{MoveIntent, Player};
use crate::application::ports::{ChunkDraw, ChunkSourcePort, FrameParams, RendererPort};
use crate::application::streaming::{ChunkStore, LoadedChunk, StreamingPolicy, chunk_key};

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
    /// Chunks generated per tick. 2 balances streaming latency against
    /// frame hitches: one chunk costs ~15 ms native (more in wasm), so
    /// higher budgets stall the frame visibly.
    pub max_loads_per_tick: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            chunk_size: 10.0,
            chunk_radius: 1,
            spawn: [5.0, 1.7, 5.0],
            max_loads_per_tick: 2,
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
    pub flashlight: bool,
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
        Self {
            ema_frame_ms: 16.0,
            scale: 0.8,
            cooldown_frames: 0,
        }
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
        let idx = Self::SCALES
            .iter()
            .position(|&s| s == self.scale)
            .unwrap_or(2);
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

/// Backrooms level ids the engine can noclip between.
const LEVEL_BACKROOMS: u32 = 0;
const LEVEL_GRASSLAND: u32 = 34;
/// Keep pushing into a wall this long before a noclip roll happens...
const NOCLIP_PUSH_SECONDS: f32 = 1.2;
/// ...then one roll per second of continued pushing, at this probability.
const NOCLIP_CHANCE: f32 = 0.2;

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
    /// Active Backrooms level; chunks are requested for this level.
    level: u32,
    /// How long the player has been pushing into a wall without moving.
    push_seconds: f32,
    /// Seconds until the next noclip roll is allowed.
    noclip_cooldown: f32,
    /// xorshift state for the noclip dice.
    rng: u64,
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
            policy: StreamingPolicy {
                chunk_size: config.chunk_size,
                radius,
            },
            store: ChunkStore::new(),
            world: CollisionWorld::new(),
            draws: Vec::new(),
            atlas_nodes: 0,
            governor: PerfGovernor::new(),
            renderer,
            source,
            level: LEVEL_BACKROOMS,
            push_seconds: 0.0,
            noclip_cooldown: 0.0,
            rng: (config.seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            config,
        }
    }

    fn rng_next01(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Backrooms lore made mechanical: holding a walk into solid wall long
    /// enough occasionally phases the player through reality.
    fn update_noclip(&mut self, dt: f32, input: &InputFrame, old_pos: [f32; 3]) {
        self.noclip_cooldown = (self.noclip_cooldown - dt).max(0.0);

        let i = input.intent;
        let pushing = input.locked && (i.forward || i.backward || i.left || i.right);
        let dx = self.player.position[0] - old_pos[0];
        let dz = self.player.position[2] - old_pos[2];
        let pinned = pushing && (dx * dx + dz * dz) < 0.002 * 0.002;

        if !pinned {
            self.push_seconds = 0.0;
            return;
        }
        self.push_seconds += dt;
        if self.push_seconds < NOCLIP_PUSH_SECONDS || self.noclip_cooldown > 0.0 {
            return;
        }
        self.noclip_cooldown = 1.0;
        if self.rng_next01() < NOCLIP_CHANCE {
            self.noclip();
        }
    }

    /// Phase between Level 0 and the grassland: drop every resident chunk so
    /// the streamer rebuilds the world from the new level's generator.
    fn noclip(&mut self) {
        self.level = if self.level == LEVEL_BACKROOMS {
            LEVEL_GRASSLAND
        } else {
            LEVEL_BACKROOMS
        };
        self.store.retain_keys(&[]);
        self.world.rebuild([].iter());
        self.draws.clear();
        self.push_seconds = 0.0;
        // Into the grassland you phase in place; the way back drops you at
        // the spawn clearing so you can't rematerialize inside a wall.
        if self.level == LEVEL_BACKROOMS {
            self.player.position = self.config.spawn;
        }
    }

    /// The active Backrooms level id.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Advances the simulation one frame and issues the draw.
    pub fn tick(&mut self, dt_seconds: f32, input: &InputFrame) {
        // Clamp dt so a background-tab hitch can't teleport the player.
        let dt = dt_seconds.clamp(0.0, 0.1);
        self.governor.update(dt_seconds * 1000.0);

        let old_pos = self.player.position;
        if input.locked {
            self.player.apply_look(input.look_dx, input.look_dy);
            self.player.step(dt, &input.intent, &self.world);
        }
        self.update_noclip(dt, input, old_pos);

        self.stream_chunks();

        let frame = FrameParams {
            camera_pos: self.player.position,
            yaw: self.player.yaw,
            pitch: self.player.pitch,
            flashlight: input.flashlight,
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
                let payload = self.source.load(ox, oz, self.level);
                self.store.insert(
                    key,
                    LoadedChunk {
                        origin: (ox, oz),
                        payload,
                    },
                );
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
        fn load(&self, origin_x: f32, _origin_z: f32, _level: u32) -> ChunkPayload {
            ChunkPayload {
                root: 0,
                nodes: [1u32, 0, 0, 0].repeat(1024), // one padded row of air leaves
                world_size: 12.8,
                collision: vec![Aabb::new([origin_x, 0.0, 0.0], [origin_x + 0.2, 3.0, 0.2])],
            }
        }
    }

    /// Chunk source that traps the player: one huge collision box around the
    /// spawn, and it records which level each load was for.
    struct TrappingChunkSource {
        levels: Rc<RefCell<Vec<u32>>>,
    }

    impl ChunkSourcePort for TrappingChunkSource {
        fn load(&self, _x: f32, _z: f32, level: u32) -> ChunkPayload {
            self.levels.borrow_mut().push(level);
            ChunkPayload {
                root: 0,
                nodes: [1u32, 0, 0, 0].repeat(1024),
                world_size: 12.8,
                collision: vec![Aabb::new([-100.0, 0.0, -100.0], [100.0, 3.0, 100.0])],
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
    fn streams_multiple_chunks_per_tick_up_to_max_loads() {
        let (mut engine, uploads, _draws) = engine_with_recorder();
        let input = InputFrame::default();

        // 9 chunks total are desired (radius 1). With max_loads_per_tick=2:
        // 2+2+2+2+1 over five ticks.
        for expected in [2usize, 4, 6, 8, 9] {
            engine.tick(1.0 / 60.0, &input);
            assert_eq!(engine.stats().resident_chunks, expected);
        }
        assert_eq!(uploads.borrow().len(), 5);

        // Subsequent ticks should not load anything or re-upload.
        for _ in 0..5 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.stats().resident_chunks, 9);
        assert_eq!(uploads.borrow().len(), 5);
        assert!(engine.stats().ready);
    }

    #[test]
    fn steady_state_does_not_reupload_atlas() {
        let (mut engine, uploads, _) = engine_with_recorder();
        let input = InputFrame::default();
        for _ in 0..20 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(
            uploads.borrow().len(),
            5,
            "no uploads once resident set is stable"
        );
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
    fn pushing_into_a_wall_eventually_noclips_to_the_grassland() {
        let levels = Rc::new(RefCell::new(Vec::new()));
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(TrappingChunkSource {
                levels: levels.clone(),
            }),
        );
        assert_eq!(engine.level(), 0);

        // Hold forward into the wall. 20% per roll, one roll per second of
        // pushing: 120 simulated seconds without a switch is ~1e-11 likely.
        let input = InputFrame {
            intent: crate::application::player::MoveIntent {
                forward: true,
                ..Default::default()
            },
            locked: true,
            ..Default::default()
        };
        let mut switched_at = None;
        for tick in 0..(120 * 60) {
            engine.tick(1.0 / 60.0, &input);
            if engine.level() != 0 {
                switched_at = Some(tick);
                break;
            }
        }
        assert_eq!(
            engine.level(),
            34,
            "never noclipped (switched_at={switched_at:?})"
        );
        // New chunks must be requested for the grassland level.
        for _ in 0..5 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert!(
            levels.borrow().iter().any(|&l| l == 34),
            "chunk source never asked for level 34: {:?}",
            levels.borrow()
        );
    }

    #[test]
    fn walking_freely_never_noclips() {
        let (mut engine, _, _) = engine_with_recorder();
        let input = InputFrame {
            intent: crate::application::player::MoveIntent {
                forward: true,
                ..Default::default()
            },
            locked: true,
            ..Default::default()
        };
        for _ in 0..(60 * 60) {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.level(), 0, "noclip fired without a wall");
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
