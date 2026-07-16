//! The frame-loop orchestrator: the application's central use case.
//!
//! `Engine::tick` is called once per animation frame by the browser driver
//! with plain data in (`InputFrame`, dt) and drives everything else through
//! ports: chunk loading via [`ChunkSourcePort`], drawing via [`RendererPort`].
//! No browser types appear anywhere in this file, so the whole game loop is
//! natively unit-testable.

use std::collections::{HashMap, HashSet};

use crate::application::atlas::{AtlasPool, MAX_CHUNKS, payload_rows};
use crate::application::collision::{CollisionWorld, player_aabb};
use crate::application::player::{MoveIntent, Player};
use crate::application::ports::{
    ChunkDraw, ChunkRequest, ChunkSourcePort, CompletedChunk, DynamicLight, Environment,
    FrameParams, MAX_DYNAMIC_LIGHTS, RendererPort, SurfaceChunk,
};
use crate::application::prepare_frame_lighting::select_scene_lights;
use crate::application::streaming::{
    ChunkKey, ChunkStore, LoadedChunk, StreamingPolicy, chunk_key,
};
use crate::application::thermal;
use vackrooms::domain::entities::anomaly::{
    AnomalyKind, FABRIC_DRIFT_CELL, RealitySnapshot, TraversalGateKind, WorldBounds,
    fabric_drift_cell_of,
};
use vackrooms::domain::entities::player_vitals::PlayerVitals;
use vackrooms::domain::entities::supplies::SupplyKind;

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
    /// Initial yaw, radians. Set by the composition root so the player wakes
    /// up looking *down* the main corridor, not at a wall.
    pub spawn_yaw: f32,
    /// Full-resolution chunk loads per tick. 2 balances streaming latency
    /// against frame hitches: one fine chunk costs ~15 ms native (more in
    /// wasm), so higher budgets stall the frame visibly. Coarse loads cost
    /// a fraction of this budget (see `FINE_LOAD_COST`).
    pub max_loads_per_tick: usize,
    /// Screen-space-error proxy: chunks whose nearest point is farther than
    /// this from the player stay at the coarse LOD; nearer chunks refine to
    /// full resolution. A coarse voxel at this distance projects to roughly
    /// the same pixels as a fine voxel at half of it.
    pub fine_distance: f32,
    /// Level to boot into (0 = Backrooms, 34 = grassland). Exposed as the
    /// `?level=` debug query so any level is reachable in any renderer
    /// without waiting on a noclip roll.
    pub initial_level: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            chunk_size: 10.0,
            chunk_radius: 1,
            spawn: [5.0, 1.7, 5.0],
            spawn_yaw: 0.0,
            max_loads_per_tick: 2,
            fine_distance: 15.0,
            initial_level: LEVEL_BACKROOMS,
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
    /// Edge-triggered flare drop for this frame (G key / touch button).
    pub drop_flare: bool,
}

/// Snapshot for the HUD presenter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudStats {
    pub resident_chunks: usize,
    /// Chunks resident at full resolution; the rest are coarse LOD.
    pub fine_chunks: usize,
    pub atlas_nodes: usize,
    pub collision_boxes: usize,
    pub resolution_scale: f32,
    /// True once the spawn chunk is resident (drives the loading overlay).
    pub ready: bool,
    /// Session pedometer: resolved horizontal movement in world units
    /// (= meters), collision already applied, relocations excluded.
    pub distance_m: f32,
    /// Survival vitals, normalized 0..=1 for the HUD bars.
    pub hydration: f32,
    pub satiety: f32,
    pub condition: f32,
    /// Carried supplies (auto-consumed when the matching vital runs low).
    pub almond_bottles: u32,
    pub rations: u32,
    /// Smoothed ambient temperature at the player, °C.
    pub ambient_c: f32,
    /// Times the wanderer has succumbed this session.
    pub deaths: u32,
    /// The active Backrooms level id.
    pub level: u32,
}

/// Adaptive internal-resolution governor. Low-spec machines get fewer rays
/// per frame by rendering at a reduced backing-store resolution; the driver
/// applies the scale to the canvas. Hysteresis + cooldown prevent flapping.
#[derive(Debug, Clone, Copy)]
pub struct PerfGovernor {
    ema_frame_ms: f32,
    scale: f32,
    cooldown_frames: u32,
    /// Frames since the last scale-up attempt (saturating).
    frames_since_raise: u32,
    /// While positive, scale-ups are locked out because a recent raise
    /// immediately overloaded the GPU and had to be reverted (anti-flap).
    raise_lockout_frames: u32,
}

impl PerfGovernor {
    const SCALES: [f32; 4] = [0.5, 0.65, 0.8, 1.0];
    /// A drop this soon after a raise counts as a failed raise.
    const RAISE_PROBE_FRAMES: u32 = 600;
    /// How long a failed raise blocks further raise attempts (~1 min).
    const RAISE_LOCKOUT_FRAMES: u32 = 3600;

    pub fn new() -> Self {
        Self {
            ema_frame_ms: 16.0,
            scale: 0.8,
            cooldown_frames: 0,
            frames_since_raise: u32::MAX,
            raise_lockout_frames: 0,
        }
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn update(&mut self, frame_ms: f32) {
        self.ema_frame_ms = self.ema_frame_ms * 0.95 + frame_ms.clamp(0.0, 100.0) * 0.05;
        self.frames_since_raise = self.frames_since_raise.saturating_add(1);
        self.raise_lockout_frames = self.raise_lockout_frames.saturating_sub(1);
        if self.cooldown_frames > 0 {
            self.cooldown_frames -= 1;
            return;
        }
        let idx = Self::SCALES
            .iter()
            .position(|&s| s == self.scale)
            .unwrap_or(2);
        if self.ema_frame_ms > 33.0 && idx > 0 {
            // Sustained under ~30 fps: drop one resolution step. If this
            // happens right after a raise, the raise failed — lock raises
            // out for a while so the scale doesn't oscillate.
            if self.frames_since_raise < Self::RAISE_PROBE_FRAMES {
                self.raise_lockout_frames = Self::RAISE_LOCKOUT_FRAMES;
            }
            self.scale = Self::SCALES[idx - 1];
            self.cooldown_frames = 120;
        } else if self.ema_frame_ms < 17.5
            && idx + 1 < Self::SCALES.len()
            && self.raise_lockout_frames == 0
        {
            // Holding 60Hz vsync (rAF EMA floors at ~16.7 ms, so a lower
            // threshold would never fire): probe one resolution step up.
            self.scale = Self::SCALES[idx + 1];
            self.cooldown_frames = 240;
            self.frames_since_raise = 0;
        }
    }
}

/// The single coarse LOD used for progressive availability: every missing
/// chunk is first loaded at this LOD (voxels 2x the size, ~1/8 the cost) so
/// the whole streaming footprint becomes visible before any chunk is refined.
const COARSE_LOD: u8 = 1;
/// Budget units one fine load costs; a coarse load costs 1. Coarse generation
/// touches ~1/8 the voxels but has fixed per-chunk overhead, so 4 (not 8)
/// keeps the worst-case tick cost at the old two-fine-loads level.
const FINE_LOAD_COST: u32 = 4;

/// Backrooms level ids the engine can noclip between.
const LEVEL_BACKROOMS: u32 = 0;
/// Level 1, the Habitable Zone (physical doors lead here from Level 0).
const LEVEL_HABITABLE: u32 = 1;
const LEVEL_GRASSLAND: u32 = 34;
/// Horizontal reach within which a supply pickup fires.
const PICKUP_REACH: f32 = 0.8;
/// Carried supplies beyond immediate need, per kind.
const CARRY_CAP: u32 = 3;
/// A carried supply is consumed when its vital falls under this level.
const AUTO_CONSUME_AT: f32 = 0.5;
/// Seconds between level-door transits (re-arming, not gameplay pacing).
const DOOR_COOLDOWN_S: f32 = 2.0;
/// Thermal inertia: seconds for the felt ambient to close ~63% of the gap
/// to the instantaneous lit-scene temperature.
const THERMAL_TAU_S: f32 = 8.0;
/// Where death and Level-1 arrival place the eye, per level.
fn level_spawn(level: u32, config_spawn: [f32; 3]) -> [f32; 3] {
    if level == LEVEL_HABITABLE {
        [3.0, 1.7, 3.0]
    } else {
        config_spawn
    }
}
/// Keep pushing into a wall this long before a noclip roll happens...
const NOCLIP_PUSH_SECONDS: f32 = 1.2;
/// ...then one roll per second of continued pushing, at this probability.
const NOCLIP_CHANCE: f32 = 0.2;
/// Flare lifetime in seconds (fade begins in the final tenth).
const FLARE_LIFETIME_S: f32 = 90.0;
/// Global active-flare cap; dropping past it silently retires the oldest.
const FLARE_CAP: usize = 12;
/// Flares farther than this from the camera contribute no dynamic light.
const FLARE_CULL_DISTANCE: f32 = 40.0;
/// Local warm glow radius of one flare.
const FLARE_LIGHT_RADIUS: f32 = 9.0;

/// Per-frame intensity gain of one fixture's authored flicker mode.
/// Deterministic in (id, time): every machine shows the same buzz.
/// Mode 1 is a tired ballast — a fast shallow shimmer with slow beats.
/// Mode 2 is a dying tube — mostly lit, with hard sputtering dropouts.
fn fixture_flicker_gain(id: u64, mode: u8, time: f32) -> f32 {
    let phase = (id % 977) as f32 * 0.61;
    match mode {
        1 => {
            let t = time * 47.0 + phase;
            (0.86 + 0.10 * t.sin() * (t * 0.093).cos() + 0.04 * (time * 3.1 + phase).sin())
                .clamp(0.7, 1.0)
        }
        2 => {
            // Coarse deterministic gate: ~9 decisions per second per tube.
            let step = (time * 9.0 + phase) as u64;
            let mut h = id ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            h ^= h >> 31;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            if (h >> 40) % 100 < 24 {
                0.06
            } else {
                0.92
            }
        }
        _ => 1.0,
    }
}

/// Inside a blackout, drift cells closer than this may still be revealed by
/// a flare or the flashlight cone, so the rear shift never touches them.
/// Blackout ambient is near-black past a few units; 5 u is already beyond
/// anything the player can resolve behind their own shoulder, and the
/// tighter radius lets the dark rearrange almost at the player's heels.
const BLACKOUT_HIDE_DISTANCE: f32 = 5.0;
/// Cadence of the in-blackout rear shift. Aggressive on purpose: the way
/// in is gone the moment it leaves the corner of the eye, so a blackout is
/// almost impossible to back out of. Each firing's rebuild work is still
/// metered by the ordinary install budget.
const BLACKOUT_SHIFT_PERIOD_S: f32 = 0.75;
/// Flare flame height above the floor slab.
const FLARE_HEIGHT: f32 = 0.3;

/// One dropped flare: pure runtime world-space state. Never serialized into
/// chunks, never part of the reality snapshot, and preserved untouched by
/// ordinary chunk streaming.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flare {
    pub position: [f32; 3],
    pub age_seconds: f32,
    /// Monotonic drop order; doubles as the deterministic flicker phase.
    pub sequence: u64,
}

impl Flare {
    pub fn remaining_seconds(&self) -> f32 {
        (FLARE_LIFETIME_S - self.age_seconds).max(0.0)
    }
}

pub struct Engine {
    config: EngineConfig,
    player: Player,
    policy: StreamingPolicy,
    visual_policy: StreamingPolicy,
    store: ChunkStore,
    pool: AtlasPool,
    world: CollisionWorld,
    draws: Vec<ChunkDraw>,
    atlas_nodes: usize,
    governor: PerfGovernor,
    renderer: Box<dyn RendererPort>,
    source: Box<dyn ChunkSourcePort>,
    /// Outstanding background loads (async sources only). The complete
    /// request is retained so a late completion can neither clear nor replace
    /// a newer request for the same spatial chunk.
    pending: HashMap<ChunkKey, ChunkRequest>,
    /// Finished background loads waiting for install budget. Workers can
    /// return several chunks in one tick; installing them all at once (mesh
    /// upload + collision rebuild per chunk) is exactly the frame hitch the
    /// worker pool exists to avoid, so installs are metered per tick and the
    /// overflow waits here. Every entry is re-validated at install time.
    completed_backlog: Vec<CompletedChunk>,
    /// Monotonic worker-order identity. Kept independent from chunk/LOD so a
    /// re-request after eviction is distinguishable from its predecessor.
    next_request_id: u32,
    /// Immutable encounter state used by every request in the active reality.
    /// Traversal/event handling will replace this in a later slice; epoch zero
    /// is sufficient to establish the transport and validation contract now.
    reality: RealitySnapshot,
    /// Resident red-room chunks that may safely rebuild after the player has
    /// crossed the occluded inner threshold. Other remaps stay lazy until
    /// ordinary eviction, so directly visible pillar/blackout geometry never
    /// pops.
    forced_reloads: HashSet<ChunkKey>,
    /// Peripheral Shift bookkeeping: fabric drift cells the player is (or
    /// recently was) near. A cell that leaves the far radius has become
    /// genuinely unobserved and its drift epoch advances, so the fabric the
    /// player walks back into has lawfully rearranged.
    drift_near: HashSet<(i64, i64)>,
    /// Seconds until the next in-blackout rear shift may fire.
    blackout_shift_cooldown: f32,
    /// Debug telemetry for the rear shift: (player inside a blackout,
    /// eligible cells at last firing, total cells shifted this session).
    blackout_shift_debug: (bool, usize, usize),
    /// Last non-hazard eye position used by pit-lattice recovery.
    last_safe_position: [f32; 3],
    /// Active Backrooms level; chunks are requested for this level.
    level: u32,
    /// How long the player has been pushing into a wall without moving.
    push_seconds: f32,
    /// Seconds until the next noclip roll is allowed.
    noclip_cooldown: f32,
    /// xorshift state for the noclip dice.
    rng: u64,
    /// Active dropped flares, oldest first.
    flares: Vec<Flare>,
    /// Monotonic flare drop counter.
    flare_sequence: u64,
    /// Session pedometer: resolved horizontal movement (world units).
    distance_m: f64,
    /// Engine-time seconds, used only for flare flicker phase.
    time_seconds: f64,
    /// Survival vitals; advanced once per tick under the felt ambient.
    vitals: PlayerVitals,
    /// Carried supplies awaiting auto-consumption.
    almond_bottles: u32,
    rations: u32,
    /// Smoothed ambient temperature at the player, °C.
    ambient_c: f32,
    /// Seconds until a level door may fire again.
    door_cooldown: f32,
    /// Times the wanderer has succumbed this session.
    deaths: u32,
}

impl Engine {
    pub fn new(
        config: EngineConfig,
        renderer: Box<dyn RendererPort>,
        source: Box<dyn ChunkSourcePort>,
    ) -> Self {
        let radius = config.chunk_radius.clamp(0, 2);
        let visual_radius = if renderer.uses_surface_meshes() {
            4
        } else {
            radius
        };
        let mut player = Player::new(config.spawn);
        player.yaw = config.spawn_yaw;
        Self {
            player,
            policy: StreamingPolicy {
                chunk_size: config.chunk_size,
                radius,
            },
            visual_policy: StreamingPolicy {
                chunk_size: config.chunk_size,
                radius: visual_radius,
            },
            store: ChunkStore::new(),
            pool: AtlasPool::new(),
            world: CollisionWorld::new(),
            draws: Vec::new(),
            atlas_nodes: 0,
            governor: PerfGovernor::new(),
            renderer,
            source,
            pending: HashMap::new(),
            completed_backlog: Vec::new(),
            next_request_id: 1,
            reality: RealitySnapshot::default(),
            forced_reloads: HashSet::new(),
            drift_near: HashSet::new(),
            blackout_shift_cooldown: 0.0,
            blackout_shift_debug: (false, 0, 0),
            last_safe_position: config.spawn,
            level: config.initial_level,
            push_seconds: 0.0,
            noclip_cooldown: 0.0,
            rng: (config.seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            flares: Vec::new(),
            flare_sequence: 0,
            distance_m: 0.0,
            time_seconds: 0.0,
            vitals: PlayerVitals::default(),
            almond_bottles: 0,
            rations: 0,
            ambient_c: thermal::dark_baseline_c(config.initial_level),
            door_cooldown: 0.0,
            deaths: 0,
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

    /// Phase through reality by pushing into walls. From Level 0 the fall is
    /// into the grassland; from anywhere else — including Level 1, exactly
    /// as the wiki warns — a noclip drops the wanderer back into Level 0.
    fn noclip(&mut self) {
        let (target, arrival) = if self.level == LEVEL_BACKROOMS {
            // Into the grassland you phase in place.
            (LEVEL_GRASSLAND, None)
        } else {
            // The way back drops you at the spawn clearing so you can't
            // rematerialize inside a wall.
            (LEVEL_BACKROOMS, Some(self.config.spawn))
        };
        self.switch_level(target, arrival);
    }

    /// Switches the active level: drop every resident chunk so the streamer
    /// rebuilds the world from the new level's generator, then optionally
    /// relocate the player to an arrival point in the new coordinate space.
    fn switch_level(&mut self, target: u32, arrival: Option<[f32; 3]>) {
        self.level = target;
        self.store.retain_keys(&[]);
        // In-flight loads are for the old level; their completions will be
        // rejected by the level check, and clearing pending lets the new
        // level re-request the same keys immediately.
        self.pending.clear();
        self.completed_backlog.clear();
        self.forced_reloads.clear();
        // The whole world is regenerated: drop the pool so the next stream
        // pass relayouts and re-uploads from scratch.
        self.pool = AtlasPool::new();
        self.renderer.clear_surfaces();
        self.world.rebuild([].iter());
        self.draws.clear();
        self.push_seconds = 0.0;
        // Flares are world objects of the level they were lit in.
        self.flares.clear();
        // Leaving the level unobserves everything at once: every tracked
        // drift cell rearranges, so phasing out and back never returns the
        // wanderer to the hallways they left.
        let watched: Vec<(i64, i64)> = self.drift_near.drain().collect();
        for (cx, cz) in watched {
            self.reality = self.reality.with_fabric_drift_advanced(cx, cz);
        }
        if let Some(arrival) = arrival {
            self.player.relocate(arrival);
        }
        self.last_safe_position = self.player.position;
        // The felt air snaps toward the new level's baseline over the
        // ordinary inertia; no need to reset it here.
        self.door_cooldown = DOOR_COOLDOWN_S;
    }

    /// The active Backrooms level id.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Per-level atmosphere shared by every renderer: the grassland is an
    /// open daylight plane, Level 1 carries its canonical low fog, and
    /// everything else keeps the interior palette.
    pub fn environment(&self) -> Environment {
        match self.level {
            LEVEL_GRASSLAND => Environment::daylight(),
            LEVEL_HABITABLE => Environment::habitable(),
            _ => Environment::interior(),
        }
    }

    /// Survival vitals (read-only view for presenters/tests).
    pub fn vitals(&self) -> PlayerVitals {
        self.vitals
    }

    /// Test-only vitals override, for exercising attrition paths without
    /// simulating twenty real minutes.
    #[cfg(test)]
    pub(crate) fn set_vitals_for_test(&mut self, vitals: PlayerVitals) {
        self.vitals = vitals;
    }

    /// Immutable encounter-state snapshot currently used to key generation.
    /// A later traversal layer will own safe snapshot transitions; exposing
    /// the current value here does not itself mutate or remap the world.
    pub fn reality_snapshot(&self) -> &RealitySnapshot {
        &self.reality
    }

    /// Advances the simulation one frame and issues the draw.
    pub fn tick(&mut self, dt_seconds: f32, input: &InputFrame) {
        // Clamp dt so a background-tab hitch can't teleport the player.
        let dt = dt_seconds.clamp(0.0, 0.1);
        self.governor.update(dt_seconds * 1000.0);
        self.time_seconds += dt as f64;

        let old_pos = self.player.position;
        if input.locked {
            self.player.apply_look(input.look_dx, input.look_dy);
            self.player.step(dt, &input.intent, &self.world);

            // Pedometer: measured here — immediately after the one
            // collision-resolved integration step — so relocations that
            // happen later in the tick (pit recovery, noclip, red-loop
            // handling) and out-of-band teleports can never inflate it.
            // Look input contributes nothing: only position deltas count.
            let dx = self.player.position[0] - old_pos[0];
            let dz = self.player.position[2] - old_pos[2];
            let step = (dx * dx + dz * dz).sqrt();
            // Anything faster than sprint speed in one frame is a
            // discontinuity, not walking.
            if step < 1.0 {
                self.distance_m += step as f64;
            }
        }
        self.update_flares(dt, input);
        self.update_anomalies(old_pos);
        self.update_provisions(dt);
        self.update_peripheral_shift(dt);
        self.update_noclip(dt, input, old_pos);

        self.stream_chunks();

        // Front-to-back chunk order: the shader marches chunks nearest-first
        // and its per-pixel insertion sort degenerates to a single linear
        // pass when the uniform table is already sorted by camera distance.
        let cam = self.player.position;
        self.draws.sort_by(|a, b| {
            let d2 = |c: &ChunkDraw| {
                let h = c.world_size * 0.5;
                let dx = c.origin[0] + h - cam[0];
                let dy = c.origin[1] + h - cam[1];
                let dz = c.origin[2] + h - cam[2];
                dx * dx + dy * dy + dz * dz
            };
            d2(a).total_cmp(&d2(b))
        });

        let mut scene_lights = select_scene_lights(
            self.store
                .iter_ordered()
                .flat_map(|chunk| chunk.payload.surface.lights.iter()),
            self.player.position,
        );
        // Fluorescent flicker is applied here, once, CPU-side — exactly like
        // flare flicker — so every renderer sees the same fluctuating world
        // and no shader needs a clock.
        let flicker_time = self.time_seconds as f32;
        for light in &mut scene_lights {
            light.intensity *= fixture_flicker_gain(light.id, light.flicker_mode, flicker_time);
        }

        // Survival: the lit scene *is* the heat source. The felt ambient
        // chases the instantaneous value with thermal inertia, then the
        // vitals advance under it (heat accelerates thirst).
        let instantaneous_c = thermal::ambient_celsius(
            self.level,
            self.player.position,
            &scene_lights,
            self.world.boxes(),
        );
        let blend = (dt / THERMAL_TAU_S).clamp(0.0, 1.0);
        self.ambient_c += (instantaneous_c - self.ambient_c) * blend;
        self.vitals = self.vitals.advanced(dt, self.ambient_c);
        self.auto_consume_supplies();
        if self.vitals.is_dead() {
            self.succumb();
        }

        let frame = FrameParams {
            camera_pos: self.player.position,
            yaw: self.player.yaw,
            pitch: self.player.pitch,
            flashlight: input.flashlight,
            dynamic_lights: self.flare_lights(),
            dynamic_light_count: self.flare_light_count(),
            scene_lights,
            environment: self.environment(),
        };
        self.renderer.draw(&frame, &self.draws);
    }

    /// Ages, caps, and drops flares. Placement prefers a spot slightly ahead
    /// of the player's feet but never inside solid geometry; the fallback is
    /// the player's own (guaranteed valid) position.
    fn update_flares(&mut self, dt: f32, input: &InputFrame) {
        for flare in &mut self.flares {
            flare.age_seconds += dt;
        }
        self.flares
            .retain(|flare| flare.age_seconds < FLARE_LIFETIME_S);

        if !(input.drop_flare && input.locked) {
            return;
        }
        let fwd = self.player.forward();
        let eye = self.player.position;
        let ahead = [eye[0] + fwd[0] * 0.9, eye[1], eye[2] + fwd[2] * 0.9];
        let spot = if self.world.collides(ahead) {
            eye
        } else {
            ahead
        };
        if self.flares.len() >= FLARE_CAP {
            // Oldest first by construction; no count UI, no modal — the
            // oldest flare simply retires.
            self.flares.remove(0);
        }
        self.flare_sequence += 1;
        self.flares.push(Flare {
            position: [spot[0], FLARE_HEIGHT, spot[2]],
            age_seconds: 0.0,
            sequence: self.flare_sequence,
        });
    }

    /// A regenerated chunk may lawfully differ (wake remaps beyond the
    /// eviction ring). A flare standing where new geometry now stands would
    /// desynchronize render and collision, so the deterministic policy is:
    /// remaps never move a flare through a wall — a buried flare is
    /// extinguished the moment the world around it becomes solid.
    fn extinguish_buried_flares(&mut self) {
        if self.flares.is_empty() {
            return;
        }
        let world = &self.world;
        self.flares.retain(|flare| {
            let probe = [
                flare.position[0],
                flare.position[1] + 0.2,
                flare.position[2],
            ];
            !world.boxes().iter().any(|b| Self::point_in_aabb(probe, b))
        });
    }

    fn point_in_aabb(p: [f32; 3], b: &crate::application::collision::Aabb) -> bool {
        p[0] >= b.min[0]
            && p[0] <= b.max[0]
            && p[1] >= b.min[1]
            && p[1] <= b.max[1]
            && p[2] >= b.min[2]
            && p[2] <= b.max[2]
    }

    /// The nearest active flares as per-frame dynamic lights, distance
    /// culled, with CPU-side flicker and end-of-life fade baked into the
    /// intensity so renderers stay clockless.
    fn flare_lights(&self) -> [DynamicLight; MAX_DYNAMIC_LIGHTS] {
        let mut out = [DynamicLight::default(); MAX_DYNAMIC_LIGHTS];
        for (slot, flare) in self.nearest_flares().into_iter().enumerate() {
            let fade = (flare.remaining_seconds() / (FLARE_LIFETIME_S * 0.1)).clamp(0.0, 1.0);
            let phase = self.time_seconds as f32 * 13.0 + flare.sequence as f32 * 1.7;
            let flicker = 0.85 + 0.15 * phase.sin() * (phase * 0.37).cos();
            out[slot] = DynamicLight {
                position: flare.position,
                color: [1.0, 0.45, 0.18],
                radius: FLARE_LIGHT_RADIUS,
                intensity: 1.6 * fade * flicker,
            };
        }
        out
    }

    fn flare_light_count(&self) -> u8 {
        self.nearest_flares().len() as u8
    }

    fn nearest_flares(&self) -> Vec<&Flare> {
        let cam = self.player.position;
        let mut near: Vec<&Flare> = self
            .flares
            .iter()
            .filter(|f| {
                let dx = f.position[0] - cam[0];
                let dz = f.position[2] - cam[2];
                dx * dx + dz * dz < FLARE_CULL_DISTANCE * FLARE_CULL_DISTANCE
            })
            .collect();
        near.sort_by(|a, b| {
            let d2 = |f: &Flare| {
                let dx = f.position[0] - cam[0];
                let dz = f.position[2] - cam[2];
                dx * dx + dz * dz
            };
            d2(a).total_cmp(&d2(b))
        });
        near.truncate(MAX_DYNAMIC_LIGHTS);
        near
    }

    /// Active flares (world-space runtime state), oldest first.
    pub fn flares(&self) -> &[Flare] {
        &self.flares
    }

    /// Session pedometer reading in meters.
    pub fn distance_m(&self) -> f32 {
        self.distance_m as f32
    }

    /// Squared distance from the player to the nearest point of a chunk's
    /// 2D footprint; 0 inside the chunk. Drives the fine/coarse LOD choice.
    fn chunk_dist2(&self, origin_x: f32, origin_z: f32) -> f32 {
        let cs = self.config.chunk_size;
        let (px, pz) = (self.player.position[0], self.player.position[2]);
        let dx = (origin_x - px).max(px - (origin_x + cs)).max(0.0);
        let dz = (origin_z - pz).max(pz - (origin_z + cs)).max(0.0);
        dx * dx + dz * dz
    }

    fn update_anomalies(&mut self, old_pos: [f32; 3]) {
        if self.level != LEVEL_BACKROOMS {
            self.last_safe_position = self.player.position;
            return;
        }

        let hazards: Vec<_> = self.store.all_pit_hazards().copied().collect();
        if let Some(hazard) = hazards
            .iter()
            .find(|h| h.contains(self.player.position[0], self.player.position[2]))
        {
            let authored = [
                hazard.recovery.x,
                self.player.position[1],
                hazard.recovery.z,
            ];
            let recovery = if self.world.collides(authored) {
                self.last_safe_position
            } else {
                authored
            };
            self.player.relocate(recovery);
            self.last_safe_position = recovery;
            return;
        }

        let mut gates: Vec<_> = self.store.all_traversal_gates().copied().collect();
        gates.sort_by_key(|g| g.id);
        gates.dedup_by_key(|g| g.id);
        for gate in gates {
            let Some(direction) = gate.crossing(
                [old_pos[0], old_pos[2]],
                [self.player.position[0], self.player.position[2]],
            ) else {
                continue;
            };
            if gate.kind == TraversalGateKind::RedThreshold && direction != gate.forward {
                continue;
            }
            let next_reality = self.reality.with_advanced_gate(&gate, direction);
            if next_reality == self.reality {
                continue;
            }
            self.reality = next_reality;

            if gate.anomaly_kind == AnomalyKind::RedRoom {
                // A committed Red Room selects another infinite Level 0
                // address. Every resident chunk belongs to the old address,
                // so replacing only the vestibule would splice two realities
                // at an arbitrary streaming boundary.
                let targets: Vec<_> = self
                    .store
                    .iter_ordered()
                    .map(|chunk| chunk_key(chunk.origin.0, chunk.origin.1))
                    .collect();
                for key in targets {
                    // A pre-threshold refinement for this key is no longer
                    // authoritative; its late completion cannot clear the new
                    // exact request identity.
                    self.pending.remove(&key);
                    self.forced_reloads.insert(key);
                }
            }
        }
        self.last_safe_position = self.player.position;
    }

    /// Supply pickups and level doors, both exported by generation beside
    /// the geometry. Pickups fold into the reality snapshot (a consumed
    /// bottle never regenerates); doors switch the streamed level.
    fn update_provisions(&mut self, dt: f32) {
        self.door_cooldown = (self.door_cooldown - dt).max(0.0);
        let (px, pz) = (self.player.position[0], self.player.position[2]);

        // -- pickups ---------------------------------------------------------
        // Halo overlap can surface the same item from two chunks; dedupe by
        // id so one bottle never quenches twice.
        let mut grabbed: Vec<_> = self
            .store
            .all_supply_items()
            .filter(|item| {
                item.within_reach(px, pz, PICKUP_REACH)
                    && !self.reality.supply_consumed(item.id)
            })
            .copied()
            .collect();
        grabbed.sort_by_key(|item| item.id);
        grabbed.dedup_by_key(|item| item.id);
        for item in grabbed {
            self.reality = self.reality.with_supply_consumed(item.id);
            match item.kind {
                SupplyKind::AlmondWater => {
                    if self.vitals.hydration < 0.98 || self.almond_bottles >= CARRY_CAP {
                        self.vitals = self.vitals.drank_almond_water();
                    } else {
                        self.almond_bottles += 1;
                    }
                }
                SupplyKind::Ration => {
                    if self.vitals.satiety < 0.98 || self.rations >= CARRY_CAP {
                        self.vitals = self.vitals.ate_ration();
                    } else {
                        self.rations += 1;
                    }
                }
            }
            // Rebuild the chunk holding the marker under the advanced
            // reality so the bottle visibly leaves the world.
            let cs = self.config.chunk_size;
            let key = chunk_key(
                (item.position.x / cs).floor() * cs,
                (item.position.z / cs).floor() * cs,
            );
            self.pending.remove(&key);
            self.forced_reloads.insert(key);
        }

        // -- level doors ------------------------------------------------------
        if self.door_cooldown > 0.0 {
            return;
        }
        let transit = self
            .store
            .all_level_exits()
            .find(|exit| exit.contains(px, pz))
            .copied();
        if let Some(exit) = transit {
            let arrival = [
                exit.arrival.x,
                self.player.position[1],
                exit.arrival.z,
            ];
            self.switch_level(exit.target_level, Some(arrival));
        }
    }

    /// Consumes carried supplies once their vital runs low.
    fn auto_consume_supplies(&mut self) {
        if self.vitals.hydration < AUTO_CONSUME_AT && self.almond_bottles > 0 {
            self.almond_bottles -= 1;
            self.vitals = self.vitals.drank_almond_water();
        }
        if self.vitals.satiety < AUTO_CONSUME_AT && self.rations > 0 {
            self.rations -= 1;
            self.vitals = self.vitals.ate_ration();
        }
    }

    /// Death by attrition: the wanderer wakes at the level's arrival point
    /// with reset vitals and empty pockets. The world — drift, consumed
    /// supplies, sealed rooms — remembers everything.
    fn succumb(&mut self) {
        self.deaths += 1;
        self.vitals = PlayerVitals::default();
        self.almond_bottles = 0;
        self.rations = 0;
        let spawn = level_spawn(self.level, self.config.spawn);
        self.player.relocate(spawn);
        self.last_safe_position = spawn;
    }

    /// Squared distance from a point to a drift cell's world rectangle
    /// (0 inside the cell).
    fn drift_cell_dist2(cell: (i64, i64), px: f32, pz: f32) -> f32 {
        let x0 = cell.0 as f32 * FABRIC_DRIFT_CELL;
        let z0 = cell.1 as f32 * FABRIC_DRIFT_CELL;
        let dx = (x0 - px).max(px - (x0 + FABRIC_DRIFT_CELL)).max(0.0);
        let dz = (z0 - pz).max(pz - (z0 + FABRIC_DRIFT_CELL)).max(0.0);
        dx * dx + dz * dz
    }

    /// Every drift cell any resident chunk could touch counts as observed.
    /// The far radius adds hysteresis of a couple of cells, so a cell only
    /// drifts once the player has genuinely abandoned it.
    fn drift_near_radius(&self) -> f32 {
        (self.visual_policy.radius as f32 + 1.0) * self.config.chunk_size
    }

    fn drift_far_radius(&self) -> f32 {
        self.drift_near_radius() + 2.0 * FABRIC_DRIFT_CELL
    }

    /// The Peripheral Shift: "whenever not directly observed, the layout can
    /// warp, stretch, or rearrange itself."
    ///
    /// Tier 1 — abandoned territory. Cells the player walks near are marked;
    /// when one falls beyond the far radius every chunk of it has long been
    /// evicted, its drift epoch advances, and whatever streams back in later
    /// is a lawfully different warren (corridors, assemblies, and anomalies
    /// never move — navigation survives; hallway memory does not).
    ///
    /// Tier 2 — inside a blackout the shift stalks the player in real time:
    /// on a slow cadence, cells that lie entirely behind the player's facing,
    /// beyond what any light could reveal, and fully inside the blackout's
    /// bounds advance immediately and their resident chunks rebuild. The
    /// darkness hides the swap; turning around is never a way back.
    fn update_peripheral_shift(&mut self, dt: f32) {
        if self.level != LEVEL_BACKROOMS {
            return;
        }
        let (px, pz) = (self.player.position[0], self.player.position[2]);

        // -- tier 1: mark near cells, drift abandoned ones -------------------
        let near = self.drift_near_radius();
        let min_cx = fabric_drift_cell_of(px - near);
        let max_cx = fabric_drift_cell_of(px + near);
        let min_cz = fabric_drift_cell_of(pz - near);
        let max_cz = fabric_drift_cell_of(pz + near);
        for cz in min_cz..=max_cz {
            for cx in min_cx..=max_cx {
                if Self::drift_cell_dist2((cx, cz), px, pz) <= near * near {
                    self.drift_near.insert((cx, cz));
                }
            }
        }
        let far2 = self.drift_far_radius() * self.drift_far_radius();
        let abandoned: Vec<(i64, i64)> = self
            .drift_near
            .iter()
            .copied()
            .filter(|&cell| Self::drift_cell_dist2(cell, px, pz) > far2)
            .collect();
        for cell in abandoned {
            self.drift_near.remove(&cell);
            self.reality = self.reality.with_fabric_drift_advanced(cell.0, cell.1);
        }

        // -- tier 2: the blackout rearranges behind the player ---------------
        self.blackout_shift_cooldown = (self.blackout_shift_cooldown - dt).max(0.0);
        let blackout_bounds = self
            .store
            .all_traversal_gates()
            .find(|gate| {
                gate.anomaly_kind == AnomalyKind::BlackoutExpanse
                    && gate.affected_bounds.contains(px, pz)
            })
            .map(|gate| gate.affected_bounds);
        let Some(bounds) = blackout_bounds else {
            self.blackout_shift_debug.0 = false;
            return;
        };
        self.blackout_shift_debug.0 = true;
        if self.blackout_shift_cooldown > 0.0 {
            return;
        }
        let forward = self.player.forward();
        // The rear shift reaches past the streaming footprint on purpose:
        // cells it advances beyond residency simply rebuild on approach,
        // exactly like tier-1 drift.
        let reach = self.drift_near_radius().max(3.0 * FABRIC_DRIFT_CELL);
        let mut shifted = Vec::new();
        for cz in fabric_drift_cell_of(pz - reach)..=fabric_drift_cell_of(pz + reach) {
            for cx in fabric_drift_cell_of(px - reach)..=fabric_drift_cell_of(px + reach) {
                let cx_w = (cx as f32 + 0.5) * FABRIC_DRIFT_CELL;
                let cz_w = (cz as f32 + 0.5) * FABRIC_DRIFT_CELL;
                // Center-based checks on purpose: a 40 u cell near the
                // blackout's edge or the player's flank still shifts. The
                // sliver a check this loose can expose sits in blackout
                // darkness — and a half-glimpsed wall that wasn't there is
                // the intended experience, not a bug.
                let inside = bounds.contains(cx_w, cz_w);
                let behind =
                    (cx_w - px) * forward[0] + (cz_w - pz) * forward[2] < -4.0;
                let hidden =
                    Self::drift_cell_dist2((cx, cz), px, pz) > BLACKOUT_HIDE_DISTANCE.powi(2);
                if inside && behind && hidden {
                    shifted.push((cx, cz));
                }
            }
        }
        self.blackout_shift_debug = (true, shifted.len(), self.blackout_shift_debug.2);
        if shifted.is_empty() {
            return;
        }
        self.blackout_shift_debug.2 += shifted.len();
        self.blackout_shift_cooldown = BLACKOUT_SHIFT_PERIOD_S;
        for &(cx, cz) in &shifted {
            self.reality = self.reality.with_fabric_drift_advanced(cx, cz);
        }
        // Rebuild the resident chunks of the shifted cells in place. The
        // install path's player-overlap check keeps the swap safe, and the
        // request carries the advanced reality by construction.
        let cell_bounds: Vec<WorldBounds> = shifted
            .iter()
            .map(|&(cx, cz)| {
                WorldBounds::new(
                    cx as f32 * FABRIC_DRIFT_CELL,
                    cz as f32 * FABRIC_DRIFT_CELL,
                    (cx + 1) as f32 * FABRIC_DRIFT_CELL,
                    (cz + 1) as f32 * FABRIC_DRIFT_CELL,
                )
            })
            .collect();
        let cs = self.config.chunk_size;
        let targets: Vec<ChunkKey> = self
            .store
            .iter_ordered()
            .filter(|chunk| {
                let rect = WorldBounds::new(
                    chunk.origin.0,
                    chunk.origin.1,
                    chunk.origin.0 + cs,
                    chunk.origin.1 + cs,
                );
                cell_bounds.iter().any(|cell| cell.intersects(rect))
            })
            .map(|chunk| chunk_key(chunk.origin.0, chunk.origin.1))
            .collect();
        for key in targets {
            self.pending.remove(&key);
            self.forced_reloads.insert(key);
        }
    }

    fn make_request_for(
        &mut self,
        origin_x: f32,
        origin_z: f32,
        lod: u8,
        reality: RealitySnapshot,
    ) -> ChunkRequest {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        if self.next_request_id == 0 {
            self.next_request_id = 1;
        }
        ChunkRequest {
            request_id,
            origin_x,
            origin_z,
            level: self.level,
            lod,
            reality,
        }
    }

    fn make_request(&mut self, origin_x: f32, origin_z: f32, lod: u8) -> ChunkRequest {
        self.make_request_for(origin_x, origin_z, lod, self.reality.clone())
    }

    /// Installs finished background loads, discarding stale work: a result
    /// for the wrong level (noclip happened), for a chunk that left the
    /// streaming footprint, or for a LOD no better than what is already
    /// resident. Refinement stays monotonic exactly like the sync path.
    ///
    /// Installs are metered by the same per-tick cost budget as synchronous
    /// loads (coarse = 1, fine = `FINE_LOAD_COST`): when a burst of worker
    /// results lands in one frame, the overflow waits in
    /// `completed_backlog` instead of stacking mesh uploads and collision
    /// rebuilds into a single frame.
    fn install_completed(
        &mut self,
        keep: &[ChunkKey],
        loaded: &mut Vec<ChunkKey>,
        changed: &mut bool,
    ) {
        // A transport failure is not a completion, but it must retire the
        // exact pending order or that chunk can never be scheduled again.
        // Old worker callbacks cannot cancel a newer retry because request
        // identity includes the engine-assigned monotonic id and reality.
        for failed in self.source.poll_failed_requests() {
            let key = chunk_key(failed.origin_x, failed.origin_z);
            let is_current = self
                .pending
                .get(&key)
                .is_some_and(|pending| pending.is_same_request(&failed));
            if is_current {
                self.pending.remove(&key);
            }
        }

        let fresh = self.source.poll_completed();
        self.completed_backlog.extend(fresh);
        let mut budget = (self.config.max_loads_per_tick as u32 * FINE_LOAD_COST).max(1);
        let mut deferred: Vec<CompletedChunk> = Vec::new();
        for done in self.completed_backlog.drain(..).collect::<Vec<_>>() {
            let install_cost = if done.request.lod == 0 {
                FINE_LOAD_COST
            } else {
                1
            };
            if budget < install_cost {
                deferred.push(done);
                continue;
            }
            let req = &done.request;
            let key = chunk_key(req.origin_x, req.origin_z);
            // A completion is authoritative only while the exact order it
            // echoes is still pending. In particular, an old result for the
            // same chunk must not remove a newer pending request.
            let is_current = self
                .pending
                .get(&key)
                .is_some_and(|pending| pending.is_same_request(req));
            if !is_current {
                continue;
            }
            self.pending.remove(&key);
            if req.level != self.level || !keep.contains(&key) {
                continue;
            }
            let is_forced = self.forced_reloads.contains(&key);
            let improves = match self.store.get(key) {
                None => true,
                // LOD is monotonic only inside one exact reality. A chunk
                // from a replaced reality is replaced even if its old LOD was
                // finer; comparing their LODs would preserve stale geometry.
                Some(resident) if resident.is_in_reality(&req.reality) => req.lod < resident.lod,
                Some(_) => is_forced,
            };
            if !improves {
                continue;
            }
            if is_forced {
                let player = player_aabb(self.player.position);
                if done
                    .payload
                    .collision
                    .iter()
                    .any(|solid| solid.intersects(&player))
                {
                    // Safety beats the effect: keep the old resident reality
                    // but do not abandon the transition. Defer this chunk and
                    // restore its pending status so we try again next tick.
                    self.pending.insert(key, req.clone());
                    deferred.push(done);
                    continue;
                }
            }
            budget -= install_cost;
            self.store.insert(
                key,
                LoadedChunk::new(
                    (req.origin_x, req.origin_z),
                    req.lod,
                    req.reality.clone(),
                    done.payload,
                ),
            );
            if !loaded.contains(&key) {
                loaded.push(key);
            }
            *changed = true;
            self.forced_reloads.remove(&key);
        }
        self.completed_backlog = deferred;
        // Drop pending entries that left the footprint so their slots free
        // up; a late completion for them is discarded by the keep check.
        self.pending
            .retain(|key, request| keep.contains(key) && request.level == self.level);
    }

    /// Tops up the background request queue: coarse availability for every
    /// missing chunk (nearest first), then at most one fine refinement in
    /// flight at a time, mirroring the sync path's priorities.
    fn issue_requests(&mut self, desired: &[(f32, f32)], desired_visual: &[(f32, f32)]) {
        // Keep roughly a worker pool's worth of requests in flight; more
        // would just build a stale backlog behind a moving player.
        let max_pending = (self.config.max_loads_per_tick * 2).max(4);

        let forced: Vec<_> = self
            .forced_reloads
            .iter()
            .filter_map(|&key| {
                self.store
                    .get(key)
                    .map(|chunk| (key, chunk.origin.0, chunk.origin.1, chunk.lod))
            })
            .collect();
        for (key, ox, oz, lod) in forced {
            if self.pending.len() >= max_pending {
                break;
            }
            if !self.pending.contains_key(&key) {
                let request = self.make_request(ox, oz, lod);
                self.pending.insert(key, request.clone());
                self.source.request(request);
            }
        }

        for &(ox, oz) in desired_visual {
            if self.pending.len() >= max_pending {
                break;
            }
            let key = chunk_key(ox, oz);
            if !self.store.contains(key) && !self.pending.contains_key(&key) {
                let request = self.make_request(ox, oz, COARSE_LOD);
                self.pending.insert(key, request.clone());
                self.source.request(request);
            }
        }

        let fine_in_flight = self.pending.values().any(|request| request.lod == 0);
        if fine_in_flight || self.pending.len() >= max_pending {
            return;
        }
        let fine_d2 = self.config.fine_distance * self.config.fine_distance;
        let target = desired.iter().copied().find(|&(ox, oz)| {
            let key = chunk_key(ox, oz);
            self.chunk_dist2(ox, oz) <= fine_d2
                && !self.pending.contains_key(&key)
                && self.store.get(key).is_some_and(|c| c.lod > 0)
        });
        if let Some((ox, oz)) = target {
            let key = chunk_key(ox, oz);
            let resident_reality = self.store.get(key).unwrap().reality.clone();
            let request = self.make_request_for(ox, oz, 0, resident_reality);
            self.pending.insert(key, request.clone());
            self.source.request(request);
        }
    }

    /// Progressive, error-driven chunk streaming.
    ///
    /// Availability first: every missing chunk (nearest first) loads at the
    /// cheap coarse LOD, so the whole footprint renders before any chunk is
    /// refined. Refinement second: with leftover budget, the nearest
    /// resident coarse chunk within `fine_distance` reloads at full
    /// resolution in place. Chunks beyond `fine_distance` stay coarse — at
    /// that range a coarse voxel projects to about a fine voxel's pixels —
    /// and refinement is monotonic while resident, so there is no flapping.
    ///
    /// Both phases share one per-tick cost budget (coarse = 1 unit, fine =
    /// `FINE_LOAD_COST`), keeping the worst-case tick at the old cost of
    /// `max_loads_per_tick` fine loads. Newly loaded and refined chunks
    /// upload only their own atlas slot rows; a full re-upload only happens
    /// on pool relayouts (first load, bigger chunks, level switch) or on
    /// back ends without partial-update support.
    fn stream_chunks(&mut self) {
        let surface_renderer = self.renderer.uses_surface_meshes();
        let desired = self
            .policy
            .desired_origins(self.player.position[0], self.player.position[2]);
        let desired_visual = self
            .visual_policy
            .desired_origins(self.player.position[0], self.player.position[2]);
        let keep: Vec<_> = desired_visual
            .iter()
            .map(|&(x, z)| chunk_key(x, z))
            .collect();

        // Free the atlas slots of chunks about to be evicted.
        let evicted: Vec<ChunkKey> = self
            .store
            .iter_ordered()
            .map(|c| chunk_key(c.origin.0, c.origin.1))
            .filter(|k| !keep.contains(k))
            .collect();
        let mut changed = self.store.retain_keys(&keep);
        self.forced_reloads.retain(|key| keep.contains(key));
        if surface_renderer {
            if !evicted.is_empty() {
                self.renderer.remove_surfaces(&evicted);
            }
        } else {
            for key in evicted {
                self.pool.release(key);
            }
        }

        let mut loaded: Vec<ChunkKey> = Vec::new();
        if self.source.is_async() {
            // Background pipeline: install validated completions, then top
            // up the request queue. The frame never blocks on generation.
            self.install_completed(&keep, &mut loaded, &mut changed);
            self.issue_requests(&desired, &desired_visual);
        } else {
            let mut budget = self.config.max_loads_per_tick as u32 * FINE_LOAD_COST;

            // Red-room closure is a semantic transition, not ordinary LOD
            // work. Rebuild the small affected set first and install each
            // complete payload atomically across its geometry/collision/light
            // views. The threshold bend keeps this set out of direct sight.
            let forced: Vec<_> = self
                .forced_reloads
                .iter()
                .filter_map(|&key| {
                    self.store
                        .get(key)
                        .map(|c| (key, c.origin.0, c.origin.1, c.lod))
                })
                .collect();
            for (key, ox, oz, lod) in forced {
                let payload = self
                    .source
                    .load_with_reality(ox, oz, self.level, lod, &self.reality);
                let player = player_aabb(self.player.position);
                if payload
                    .collision
                    .iter()
                    .any(|solid| solid.intersects(&player))
                {
                    self.forced_reloads.remove(&key);
                    continue;
                }
                self.store.insert(
                    key,
                    LoadedChunk::new((ox, oz), lod, self.reality.clone(), payload),
                );
                self.forced_reloads.remove(&key);
                loaded.push(key);
                changed = true;
            }

            // Phase 1 — availability: missing chunks come in coarse, nearest
            // first. Only a leftover budget flows into refinement, so a moving
            // player always fills holes before sharpening anything.
            for &(ox, oz) in &desired_visual {
                if budget == 0 {
                    break;
                }
                let key = chunk_key(ox, oz);
                if !self.store.contains(key) {
                    let payload = self.source.load_with_reality(
                        ox,
                        oz,
                        self.level,
                        COARSE_LOD,
                        &self.reality,
                    );
                    self.store.insert(
                        key,
                        LoadedChunk::new((ox, oz), COARSE_LOD, self.reality.clone(), payload),
                    );
                    loaded.push(key);
                    budget -= 1;
                    changed = true;
                }
            }

            // Phase 2 — refinement: nearest coarse chunk inside the fine ring.
            let fine_d2 = self.config.fine_distance * self.config.fine_distance;
            while budget >= FINE_LOAD_COST {
                let target = desired.iter().copied().find(|&(ox, oz)| {
                    self.chunk_dist2(ox, oz) <= fine_d2
                        && self.store.get(chunk_key(ox, oz)).is_some_and(|c| c.lod > 0)
                });
                let Some((ox, oz)) = target else { break };
                let key = chunk_key(ox, oz);
                let resident_reality = self.store.get(key).unwrap().reality.clone();
                let payload =
                    self.source
                        .load_with_reality(ox, oz, self.level, 0, &resident_reality);
                self.store.insert(
                    key,
                    LoadedChunk::new((ox, oz), 0, resident_reality, payload),
                );
                if !loaded.contains(&key) {
                    loaded.push(key);
                }
                budget -= FINE_LOAD_COST;
                changed = true;
            }
        }

        if !changed {
            return;
        }

        // The default path uploads only changed/replaced meshes. It keeps
        // the SVO payload in memory for collision and exact debug queries,
        // but avoids atlas rebasing and per-pixel traversal entirely.
        if surface_renderer {
            let surface_updates: Vec<SurfaceChunk<'_>> = loaded
                .iter()
                .filter_map(|&key| {
                    self.store.get(key).map(|chunk| SurfaceChunk {
                        key,
                        origin: [chunk.origin.0, 0.0, chunk.origin.1],
                        mesh: &chunk.payload.surface,
                    })
                })
                .collect();
            if !surface_updates.is_empty() {
                self.renderer.upload_surfaces(&surface_updates);
            }

            self.draws.clear();
            for chunk in self.store.iter_ordered() {
                self.draws.push(ChunkDraw {
                    origin: [chunk.origin.0, 0.0, chunk.origin.1],
                    root_index: 0,
                    world_size: chunk.payload.world_size,
                    voxel_size: chunk.payload.voxel_size,
                    svo_depth: chunk.payload.svo_depth,
                });
            }
            self.draws.truncate(MAX_CHUNKS);
            self.atlas_nodes = self
                .store
                .iter_ordered()
                .map(|c| c.payload.nodes.len() / 4)
                .sum();
            let boxes: Vec<_> = self.store.all_collision_boxes().copied().collect();
            self.world.rebuild(boxes.iter());
            self.extinguish_buried_flares();
            return;
        }

        // Size the pool for the streaming footprint and the largest chunk.
        let num_slots = ((2 * self.policy.radius + 1).pow(2) as usize).min(MAX_CHUNKS);
        let rows = self
            .store
            .iter_ordered()
            .map(|c| payload_rows(&c.payload))
            .max()
            .unwrap_or(1)
            .max(1);
        let mut need_full = self.pool.ensure_layout(num_slots, rows);

        if !need_full {
            for &key in &loaded {
                let uploaded = match self.pool.assign(key) {
                    Some(slot) => {
                        let payload = &self.store.get(key).expect("just inserted").payload;
                        let block = self.pool.rebased_block(slot, payload);
                        let first_row = self.pool.slot_first_row(slot) as u32;
                        self.renderer.upload_atlas_rows(first_row, &block)
                    }
                    None => false,
                };
                if !uploaded {
                    need_full = true;
                    break;
                }
            }
        }

        if need_full {
            let keys: Vec<ChunkKey> = self
                .store
                .iter_ordered()
                .map(|c| chunk_key(c.origin.0, c.origin.1))
                .collect();
            for key in keys {
                let _ = self.pool.assign(key);
            }
            let texels = self
                .pool
                .full_texels(|k| self.store.get(k).map(|c| &c.payload));
            self.renderer.upload_atlas(&texels);
        }

        self.draws.clear();
        for chunk in self.store.iter_ordered() {
            let key = chunk_key(chunk.origin.0, chunk.origin.1);
            if let Some(offset) = self.pool.node_offset_of(key) {
                self.draws.push(ChunkDraw {
                    origin: [chunk.origin.0, 0.0, chunk.origin.1],
                    root_index: (offset + chunk.payload.root as usize) as i32,
                    world_size: chunk.payload.world_size,
                    voxel_size: chunk.payload.voxel_size,
                    svo_depth: chunk.payload.svo_depth,
                });
            }
        }
        self.draws.truncate(MAX_CHUNKS);
        self.atlas_nodes = self
            .store
            .iter_ordered()
            .map(|c| c.payload.nodes.len() / 4)
            .sum();

        let boxes: Vec<_> = self.store.all_collision_boxes().copied().collect();
        self.world.rebuild(boxes.iter());
        self.extinguish_buried_flares();
    }

    pub fn player(&self) -> &Player {
        &self.player
    }

    pub fn teleport_player(&mut self, position: [f32; 3], yaw: f32, pitch: f32) {
        self.player.position = position;
        self.player.yaw = yaw;
        self.player.pitch = pitch;
    }

    pub fn chunk_size(&self) -> f32 {
        self.config.chunk_size
    }

    pub fn is_chunk_resident(&self, key: (i64, i64)) -> bool {
        self.store.contains(key)
    }

    pub fn collision_world(&self) -> &CollisionWorld {
        &self.world
    }

    /// Multi-line plain-text report of the live anomaly machinery, for the
    /// debug overlay: reality-snapshot epochs, resident traversal gates and
    /// pit hazards (nearest first), and the streaming pipeline counters.
    /// Pure formatting over engine state — no browser types, natively
    /// testable.
    pub fn anomaly_debug_text(&self) -> String {
        use std::fmt::Write;
        let p = self.player.position;
        let mut out = String::with_capacity(1024);
        let _ = writeln!(
            out,
            "pos ({:.1}, {:.1}, {:.1})  level {}  yaw {:.2}",
            p[0], p[1], p[2], self.level, self.player.yaw
        );
        let _ = writeln!(
            out,
            "chunks {} resident ({} fine) | pending {} | backlog {} | forced reloads {}",
            self.store.len(),
            self.store.iter_ordered().filter(|c| c.lod == 0).count(),
            self.pending.len(),
            self.completed_backlog.len(),
            self.forced_reloads.len(),
        );
        let _ = writeln!(
            out,
            "flares {} | push {:.2}s",
            self.flares.len(),
            self.push_seconds
        );

        let stamps = self.reality.stamps();
        let _ = writeln!(out, "reality: {} stamp(s)", stamps.len());
        let _ = writeln!(
            out,
            "peripheral shift: {} drifted cell(s), {} watched",
            self.reality.fabric_drifts().len(),
            self.drift_near.len()
        );
        let (inside, eligible, total) = self.blackout_shift_debug;
        let _ = writeln!(
            out,
            "blackout shift: inside={inside} eligible={eligible} \
             shifted_total={total} cooldown={:.1}s",
            self.blackout_shift_cooldown
        );
        for stamp in stamps {
            let _ = writeln!(
                out,
                "  {:016x} epoch {}  plane {:.1} {:?} {:?}",
                stamp.instance_id,
                stamp.epoch,
                stamp.gate_plane(),
                stamp.gate_axis,
                stamp.travel_direction,
            );
        }

        let mut gates: Vec<_> = self.store.all_traversal_gates().copied().collect();
        gates.sort_by_key(|g| g.id);
        gates.dedup_by_key(|g| g.id);
        gates.sort_by(|a, b| {
            let d = |g: &vackrooms::domain::entities::anomaly::TraversalGate| match g.axis {
                vackrooms::domain::entities::anomaly::Axis2::X => (g.plane - p[0]).abs(),
                vackrooms::domain::entities::anomaly::Axis2::Z => (g.plane - p[2]).abs(),
            };
            d(a).total_cmp(&d(b))
        });
        let _ = writeln!(out, "gates resident: {}", gates.len());
        for gate in gates.iter().take(6) {
            let dist = match gate.axis {
                vackrooms::domain::entities::anomaly::Axis2::X => (gate.plane - p[0]).abs(),
                vackrooms::domain::entities::anomaly::Axis2::Z => (gate.plane - p[2]).abs(),
            };
            let epoch = self.reality.lookup(gate.instance_id).map_or(0, |s| s.epoch);
            let _ = writeln!(
                out,
                "  {:?}/{:?} of {:?} {:08x}  plane {:.1} on {:?}  d={:.1}  epoch {}",
                gate.kind,
                gate.forward,
                gate.anomaly_kind,
                (gate.instance_id & 0xFFFF_FFFF) as u32,
                gate.plane,
                gate.axis,
                dist,
                epoch,
            );
        }

        let mut pits: Vec<_> = self.store.all_pit_hazards().copied().collect();
        pits.sort_by_key(|h| h.id);
        pits.dedup_by_key(|h| h.id);
        pits.sort_by(|a, b| {
            let d = |h: &vackrooms::domain::entities::anomaly::PitHazard| {
                let dx = h.center.x - p[0];
                let dz = h.center.z - p[2];
                dx * dx + dz * dz
            };
            d(a).total_cmp(&d(b))
        });
        let _ = writeln!(out, "pit hazards resident: {}", pits.len());
        for pit in pits.iter().take(4) {
            let dx = pit.center.x - p[0];
            let dz = pit.center.z - p[2];
            let _ = writeln!(
                out,
                "  {:08x} at ({:.1}, {:.1})  half {:.1}  d={:.1}  recovery ({:.1}, {:.1})",
                (pit.id & 0xFFFF_FFFF) as u32,
                pit.center.x,
                pit.center.z,
                pit.half_side,
                (dx * dx + dz * dz).sqrt(),
                pit.recovery.x,
                pit.recovery.z,
            );
        }
        out
    }

    pub fn stats(&self) -> HudStats {
        let spawn_key = chunk_key(
            (self.config.spawn[0] / self.config.chunk_size).floor() * self.config.chunk_size,
            (self.config.spawn[2] / self.config.chunk_size).floor() * self.config.chunk_size,
        );
        HudStats {
            resident_chunks: self.store.len(),
            fine_chunks: self.store.iter_ordered().filter(|c| c.lod == 0).count(),
            atlas_nodes: self.atlas_nodes,
            collision_boxes: self.world.len(),
            resolution_scale: self.governor.scale(),
            ready: self.store.contains(spawn_key),
            distance_m: self.distance_m as f32,
            hydration: self.vitals.hydration,
            satiety: self.vitals.satiety,
            condition: self.vitals.condition,
            almond_bottles: self.almond_bottles,
            rations: self.rations,
            ambient_c: self.ambient_c,
            deaths: self.deaths,
            level: self.level,
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
    use vackrooms::domain::entities::anomaly::LevelExit;
    use vackrooms::domain::entities::supplies::SupplyItem;
    use vackrooms::entities::models::Position;

    /// Synchronous source whose every chunk carries one almond water beside
    /// the spawn and one door to Level 1 a few steps away.
    struct ProvisionedChunkSource;

    impl ChunkSourcePort for ProvisionedChunkSource {
        fn load(&self, origin_x: f32, origin_z: f32, level: u32, _lod: u8) -> ChunkPayload {
            let in_chunk = |x: f32, z: f32| {
                x >= origin_x && x < origin_x + 10.0 && z >= origin_z && z < origin_z + 10.0
            };
            let mut supply_items = vec![];
            let mut level_exits = vec![];
            if level == 0 && in_chunk(5.0, 5.0) {
                supply_items.push(SupplyItem {
                    id: 41,
                    kind: SupplyKind::AlmondWater,
                    position: Position::new(5.2, 5.2),
                    rest_y: 0.0,
                });
                level_exits.push(LevelExit {
                    id: 90,
                    target_level: 1,
                    center: Position::new(8.0, 5.0),
                    half_extent: 0.7,
                    arrival: Position::new(3.0, 3.0),
                });
            }
            ChunkPayload {
                root: 0,
                nodes: [1u32, 0, 0, 0].repeat(1024),
                world_size: 12.8,
                voxel_size: 0.2,
                svo_depth: 6,
                surface: crate::application::ports::SurfaceMeshPayload::empty(0),
                collision: vec![],
                traversal_gates: vec![],
                pit_hazards: vec![],
                supply_items,
                level_exits,
            }
        }
    }

    fn provisioned_engine() -> Engine {
        Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(ProvisionedChunkSource),
        )
    }

    #[test]
    fn pickup_folds_into_reality_and_quenches_or_stocks() {
        let mut engine = provisioned_engine();
        let input = InputFrame::default();
        // Tick 1 streams the spawn chunk; tick 2 grabs the bottle at reach.
        engine.tick(1.0 / 60.0, &input);
        engine.tick(1.0 / 60.0, &input);
        assert!(engine.reality_snapshot().supply_consumed(41));
        // Fresh vitals are near-full, so the bottle is carried, not drunk.
        assert_eq!(engine.stats().almond_bottles, 1);

        // The carried bottle is drunk the moment thirst runs low.
        engine.set_vitals_for_test(PlayerVitals {
            hydration: 0.2,
            satiety: 0.9,
            condition: 0.9,
        });
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(engine.stats().almond_bottles, 0);
        assert!(engine.vitals().hydration > 0.5, "auto-drink restored thirst");
    }

    #[test]
    fn stepping_through_a_door_switches_to_level_one_and_back_is_possible() {
        let mut engine = provisioned_engine();
        let input = InputFrame::default();
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(engine.level(), 0);

        // Walk the player into the door trigger between ticks (the tick
        // itself only integrates real input; the test moves the eye).
        engine.player.relocate([8.0, 1.7, 5.0]);
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(engine.level(), 1, "door transit must fire");
        // Arrival is the authored Level 1 plaza point.
        assert_eq!(engine.player.position[0], 3.0);
        assert_eq!(engine.player.position[2], 3.0);
        // The world was flushed and re-streamed from the new level's
        // generator within the same tick (synchronous source).
        assert!(engine.stats().resident_chunks > 0);
    }

    #[test]
    fn attrition_death_respawns_with_reset_vitals_and_empty_pockets() {
        let mut engine = provisioned_engine();
        let input = InputFrame::default();
        engine.tick(1.0 / 60.0, &input);
        engine.set_vitals_for_test(PlayerVitals {
            hydration: 0.0,
            satiety: 0.0,
            condition: 0.001,
        });
        engine.player.relocate([9.0, 1.7, 9.0]);
        engine.tick(1.0, &input);
        let stats = engine.stats();
        assert_eq!(stats.deaths, 1);
        assert!(stats.condition > 0.9, "vitals reset on respawn");
        assert_eq!(engine.player.position, EngineConfig::default().spawn);
    }

    #[test]
    fn lit_scenes_run_hotter_than_darkness_and_heat_costs_water() {
        // Domain-level sanity for the wiring: the engine's felt ambient
        // starts at the level baseline and vitals depend on it.
        let engine = provisioned_engine();
        assert_eq!(
            engine.stats().ambient_c,
            crate::application::thermal::dark_baseline_c(0)
        );
    }

    #[derive(Default)]
    struct RecordingRenderer {
        uploads: Rc<RefCell<Vec<usize>>>,
        row_uploads: Rc<RefCell<Vec<u32>>>,
        draws: Rc<RefCell<Vec<usize>>>,
        /// When true the renderer accepts partial row updates like the GPU
        /// driver; when false it forces the full-upload fallback.
        supports_rows: bool,
    }

    impl RendererPort for RecordingRenderer {
        fn upload_atlas(&mut self, texels: &[u32]) {
            self.uploads.borrow_mut().push(texels.len());
        }
        fn upload_atlas_rows(&mut self, first_row: u32, _texels: &[u32]) -> bool {
            if self.supports_rows {
                self.row_uploads.borrow_mut().push(first_row);
            }
            self.supports_rows
        }
        fn draw(&mut self, _frame: &FrameParams, chunks: &[ChunkDraw]) {
            self.draws.borrow_mut().push(chunks.len());
        }
    }

    #[derive(Default)]
    struct SurfaceRecordingRenderer {
        atlas_uploads: Rc<RefCell<usize>>,
        surface_uploads: Rc<RefCell<Vec<usize>>>,
        removed: Rc<RefCell<Vec<usize>>>,
    }

    impl RendererPort for SurfaceRecordingRenderer {
        fn uses_surface_meshes(&self) -> bool {
            true
        }
        fn upload_surfaces(&mut self, chunks: &[SurfaceChunk<'_>]) {
            self.surface_uploads.borrow_mut().push(chunks.len());
        }
        fn remove_surfaces(&mut self, keys: &[crate::application::ports::SurfaceChunkKey]) {
            self.removed.borrow_mut().push(keys.len());
        }
        fn upload_atlas(&mut self, _texels: &[u32]) {
            *self.atlas_uploads.borrow_mut() += 1;
        }
        fn draw(&mut self, _frame: &FrameParams, _chunks: &[ChunkDraw]) {}
    }

    struct FlatChunkSource;

    impl ChunkSourcePort for FlatChunkSource {
        fn load(&self, origin_x: f32, _origin_z: f32, _level: u32, _lod: u8) -> ChunkPayload {
            ChunkPayload {
                root: 0,
                nodes: [1u32, 0, 0, 0].repeat(1024), // one padded row of air leaves
                world_size: 12.8,
                voxel_size: 0.2,
                svo_depth: 6,
                surface: crate::application::ports::SurfaceMeshPayload::empty(0),
                collision: vec![Aabb::new([origin_x, 0.0, 0.0], [origin_x + 0.2, 3.0, 0.2])],
                traversal_gates: vec![],
                pit_hazards: vec![],
                supply_items: vec![],
                level_exits: vec![],
            }
        }
    }

    /// Chunk source that traps the player: one huge collision box around the
    /// spawn, and it records which level each load was for.
    struct TrappingChunkSource {
        levels: Rc<RefCell<Vec<u32>>>,
    }

    impl ChunkSourcePort for TrappingChunkSource {
        fn load(&self, _x: f32, _z: f32, level: u32, _lod: u8) -> ChunkPayload {
            self.levels.borrow_mut().push(level);
            ChunkPayload {
                root: 0,
                nodes: [1u32, 0, 0, 0].repeat(1024),
                world_size: 12.8,
                voxel_size: 0.2,
                svo_depth: 6,
                surface: crate::application::ports::SurfaceMeshPayload::empty(0),
                collision: vec![Aabb::new([-100.0, 0.0, -100.0], [100.0, 3.0, 100.0])],
                traversal_gates: vec![],
                pit_hazards: vec![],
                supply_items: vec![],
                level_exits: vec![],
            }
        }
    }

    /// Background source double: records requests, completes only what the
    /// test explicitly finishes, and panics on any blocking load.
    #[derive(Default)]
    struct AsyncFakeSource {
        requests: Rc<RefCell<Vec<ChunkRequest>>>,
        ready: Rc<RefCell<Vec<crate::application::ports::CompletedChunk>>>,
        failed: Rc<RefCell<Vec<ChunkRequest>>>,
    }

    impl ChunkSourcePort for AsyncFakeSource {
        fn load(&self, _x: f32, _z: f32, _level: u32, _lod: u8) -> ChunkPayload {
            unreachable!("async sources must never be block-loaded by the engine");
        }
        fn is_async(&self) -> bool {
            true
        }
        fn request(&mut self, request: ChunkRequest) {
            self.requests.borrow_mut().push(request);
        }
        fn poll_completed(&mut self) -> Vec<crate::application::ports::CompletedChunk> {
            self.ready.borrow_mut().drain(..).collect()
        }
        fn poll_failed_requests(&mut self) -> Vec<ChunkRequest> {
            self.failed.borrow_mut().drain(..).collect()
        }
    }

    fn completed(request: ChunkRequest) -> crate::application::ports::CompletedChunk {
        let payload = FlatChunkSource.load(
            request.origin_x,
            request.origin_z,
            request.level,
            request.lod,
        );
        crate::application::ports::CompletedChunk { request, payload }
    }

    fn test_request(
        request_id: u32,
        origin_x: f32,
        origin_z: f32,
        level: u32,
        lod: u8,
    ) -> ChunkRequest {
        ChunkRequest {
            request_id,
            origin_x,
            origin_z,
            level,
            lod,
            reality: RealitySnapshot::default(),
        }
    }

    #[test]
    fn async_source_streams_without_blocking_and_refines_monotonically() {
        let source = AsyncFakeSource::default();
        let requests = source.requests.clone();
        let ready = source.ready.clone();
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(source),
        );
        let input = InputFrame::default();

        // First tick issues requests but installs nothing — the frame
        // never waits for generation.
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(engine.stats().resident_chunks, 0);
        assert!(!engine.stats().ready);
        assert!(!requests.borrow().is_empty());

        // Fulfil whatever is asked until the footprint is fine everywhere.
        for _ in 0..40 {
            let fulfil: Vec<_> = requests.borrow_mut().drain(..).collect();
            for req in fulfil {
                ready.borrow_mut().push(completed(req));
            }
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.stats().resident_chunks, 9);
        assert_eq!(engine.stats().fine_chunks, 9);
        assert!(engine.stats().ready);

        // Steady state: no further requests.
        let before = requests.borrow().len();
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(requests.borrow().len(), before);
    }

    #[test]
    fn worker_result_bursts_install_across_ticks_not_in_one_frame() {
        let source = AsyncFakeSource::default();
        let ready = source.ready.clone();
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(source),
        );
        let input = InputFrame::default();
        engine.tick(1.0 / 60.0, &input);

        // A worst-case burst: fine results for all nine footprint chunks
        // land in the same frame.
        let desired = engine
            .policy
            .desired_origins(engine.player.position[0], engine.player.position[2]);
        let mut id = 90_000;
        for (ox, oz) in desired {
            let req = test_request(id, ox, oz, 0, 0);
            id += 1;
            engine.pending.insert(chunk_key(ox, oz), req.clone());
            ready.borrow_mut().push(completed(req));
        }

        // Budget: max_loads_per_tick (2) fine-equivalents per tick, so the
        // burst spreads over ceil(9 / 2) ticks instead of hitching one frame.
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(engine.stats().fine_chunks, 2);
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(engine.stats().fine_chunks, 4);
        for _ in 0..3 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.stats().fine_chunks, 9, "backlog must fully drain");
    }

    #[test]
    fn anomaly_debug_text_reports_reality_and_gates() {
        use vackrooms::domain::entities::anomaly::{
            AnomalyKind, Axis2, AxisDirection, TraversalGate, TraversalGateKind, WorldBounds,
        };
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(FlatChunkSource),
        );
        let gate = TraversalGate {
            id: 7,
            instance_id: 0xABCD_EF01,
            anomaly_kind: AnomalyKind::RedRoom,
            kind: TraversalGateKind::RedThreshold,
            axis: Axis2::X,
            plane: 5.0,
            span_min: 0.0,
            span_max: 10.0,
            forward: AxisDirection::Positive,
            affected_bounds: WorldBounds::new(0.0, 0.0, 10.0, 10.0),
        };
        let mut payload = FlatChunkSource.load(0.0, 0.0, 0, 0);
        payload.traversal_gates.push(gate);
        engine.store.insert(
            chunk_key(0.0, 0.0),
            LoadedChunk::new((0.0, 0.0), 0, RealitySnapshot::empty(), payload),
        );
        engine.player.position = [6.0, 1.7, 5.0];
        engine.update_anomalies([4.0, 1.7, 5.0]);

        let text = engine.anomaly_debug_text();
        assert!(text.contains("reality: 1 stamp(s)"), "{text}");
        assert!(text.contains("epoch 1"), "{text}");
        assert!(text.contains("gates resident: 1"), "{text}");
        assert!(text.contains("RedRoom"), "{text}");
    }

    #[test]
    fn stale_completions_are_discarded() {
        let source = AsyncFakeSource::default();
        let requests = source.requests.clone();
        let ready = source.ready.clone();
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(source),
        );
        let input = InputFrame::default();
        engine.tick(1.0 / 60.0, &input);
        requests.borrow_mut().clear();

        // Wrong level: the engine is on level 0.
        ready
            .borrow_mut()
            .push(completed(test_request(10_001, 0.0, 0.0, 34, 1)));
        // Outside the streaming footprint entirely.
        ready
            .borrow_mut()
            .push(completed(test_request(10_002, 500.0, 500.0, 0, 1)));
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(
            engine.stats().resident_chunks,
            0,
            "stale completions must not be installed"
        );
    }

    #[test]
    fn old_completion_cannot_clear_a_newer_request_for_the_same_chunk() {
        let source = AsyncFakeSource::default();
        let requests = source.requests.clone();
        let ready = source.ready.clone();
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(source),
        );
        let input = InputFrame::default();
        engine.tick(1.0 / 60.0, &input);

        let current = requests.borrow()[0].clone();
        let key = chunk_key(current.origin_x, current.origin_z);
        let mut stale = current.clone();
        stale.request_id = stale.request_id.wrapping_sub(1);
        ready.borrow_mut().push(completed(stale));
        engine.tick(1.0 / 60.0, &input);

        assert!(
            engine
                .pending
                .get(&key)
                .is_some_and(|pending| pending.is_same_request(&current)),
            "a stale completion must leave the newer request pending"
        );
        assert!(engine.store.get(key).is_none());

        ready.borrow_mut().push(completed(current.clone()));
        engine.tick(1.0 / 60.0, &input);
        let installed = engine.store.get(key).expect("current completion installs");
        assert!(installed.is_in_reality(&current.reality));
    }

    #[test]
    fn failed_async_request_is_retried_but_stale_failure_cannot_cancel_retry() {
        let source = AsyncFakeSource::default();
        let requests = source.requests.clone();
        let failed = source.failed.clone();
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(source),
        );
        let input = InputFrame::default();
        engine.tick(1.0 / 60.0, &input);

        let first = requests.borrow()[0].clone();
        let key = chunk_key(first.origin_x, first.origin_z);
        requests.borrow_mut().clear();
        failed.borrow_mut().push(first.clone());
        engine.tick(1.0 / 60.0, &input);

        let retry = engine
            .pending
            .get(&key)
            .expect("a transport failure is retried on the same tick")
            .clone();
        assert_ne!(retry.request_id, first.request_id);
        assert!(
            requests
                .borrow()
                .iter()
                .any(|request| request.is_same_request(&retry))
        );

        requests.borrow_mut().clear();
        failed.borrow_mut().push(first);
        engine.tick(1.0 / 60.0, &input);
        assert!(
            engine
                .pending
                .get(&key)
                .is_some_and(|pending| pending.is_same_request(&retry)),
            "a late failure for the old order must preserve its current retry"
        );
        assert!(requests.borrow().is_empty(), "no duplicate may be queued");
    }

    #[test]
    fn late_coarse_result_never_downgrades_a_fine_chunk() {
        let source = AsyncFakeSource::default();
        let requests = source.requests.clone();
        let ready = source.ready.clone();
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(source),
        );
        let input = InputFrame::default();
        for _ in 0..40 {
            let fulfil: Vec<_> = requests.borrow_mut().drain(..).collect();
            for req in fulfil {
                ready.borrow_mut().push(completed(req));
            }
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.stats().fine_chunks, 9);

        // A leftover coarse result for the spawn chunk arrives late.
        let late = test_request(10_003, 0.0, 0.0, 0, 1);
        engine.pending.insert(chunk_key(0.0, 0.0), late.clone());
        ready.borrow_mut().push(completed(late));
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(
            engine.stats().fine_chunks,
            9,
            "refinement must stay monotonic under out-of-order results"
        );
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
    fn surface_renderer_streaming_skips_svo_atlas_uploads() {
        let renderer = SurfaceRecordingRenderer::default();
        let atlas_uploads = renderer.atlas_uploads.clone();
        let surface_uploads = renderer.surface_uploads.clone();
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(renderer),
            Box::new(FlatChunkSource),
        );
        engine.tick(1.0 / 60.0, &InputFrame::default());

        assert_eq!(
            *atlas_uploads.borrow(),
            0,
            "surface path must not build atlas uploads"
        );
        assert!(
            surface_uploads.borrow().iter().sum::<usize>() > 0,
            "newly resident chunks need incremental mesh uploads"
        );
    }

    #[test]
    fn streams_coarse_first_then_refines_nearest_first() {
        let (mut engine, uploads, _draws) = engine_with_recorder();
        let input = InputFrame::default();

        // 9 chunks are desired (radius 1). The per-tick budget is
        // max_loads_per_tick * FINE_LOAD_COST = 8 units; a coarse load costs
        // 1, a fine load 4. Tick 1: eight coarse loads — the world is
        // visible (and `ready`) after one tick instead of five.
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(engine.stats().resident_chunks, 8);
        assert_eq!(engine.stats().fine_chunks, 0);
        assert!(engine.stats().ready, "coarse spawn chunk flips ready");

        // Tick 2: the last coarse load + one refinement. Ticks 3-6: two
        // refinements each; every chunk is within fine_distance (15) here.
        for expected_fine in [1usize, 3, 5, 7, 9] {
            engine.tick(1.0 / 60.0, &input);
            assert_eq!(engine.stats().resident_chunks, 9);
            assert_eq!(engine.stats().fine_chunks, expected_fine);
        }
        assert_eq!(uploads.borrow().len(), 6, "one upload per changed tick");

        // Subsequent ticks should not load anything or re-upload.
        for _ in 0..5 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.stats().resident_chunks, 9);
        assert_eq!(engine.stats().fine_chunks, 9);
        assert_eq!(uploads.borrow().len(), 6);
    }

    #[test]
    fn chunks_beyond_fine_distance_stay_coarse() {
        let renderer = RecordingRenderer::default();
        let mut engine = Engine::new(
            EngineConfig {
                // Spawn is at (5,5) mid-chunk: every neighbour chunk's
                // nearest point is >= 5 units away, so only the player's own
                // chunk sits inside the fine ring.
                fine_distance: 3.0,
                ..EngineConfig::default()
            },
            Box::new(renderer),
            Box::new(FlatChunkSource),
        );
        let input = InputFrame::default();
        for _ in 0..30 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.stats().resident_chunks, 9);
        assert_eq!(
            engine.stats().fine_chunks,
            1,
            "distant chunks must keep their cheap coarse LOD"
        );
    }

    #[test]
    fn partial_row_uploads_replace_full_uploads_after_first_layout() {
        let renderer = RecordingRenderer {
            supports_rows: true,
            ..Default::default()
        };
        let uploads = renderer.uploads.clone();
        let row_uploads = renderer.row_uploads.clone();
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(renderer),
            Box::new(FlatChunkSource),
        );
        let input = InputFrame::default();
        for _ in 0..10 {
            engine.tick(1.0 / 60.0, &input);
        }
        assert_eq!(engine.stats().resident_chunks, 9);
        assert_eq!(engine.stats().fine_chunks, 9);
        // The first tick sizes the pool (full upload of its 8 coarse
        // chunks); every later load — the 9th coarse chunk plus all 9 in-
        // place refinements — goes through the partial row path.
        assert_eq!(uploads.borrow().len(), 1, "exactly one full upload");
        assert_eq!(row_uploads.borrow().len(), 10, "remaining loads partial");
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
            6,
            "no uploads once resident set is stable and fully refined"
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

    /// Tier 1 of the Peripheral Shift: walking far from a neighborhood
    /// advances its drift epoch, so the fabric that streams back in later is
    /// keyed to a different reality — while the territory around the player
    /// stays exactly as observed.
    #[test]
    fn abandoned_territory_gains_a_fabric_drift_epoch() {
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(FlatChunkSource),
        );
        let input = InputFrame::default();
        engine.tick(1.0 / 60.0, &input);
        assert!(
            engine.reality.fabric_drifts().is_empty(),
            "nothing may drift while the player stands in it"
        );
        let origin_cell = (
            vackrooms::domain::entities::anomaly::fabric_drift_cell_of(5.0),
            vackrooms::domain::entities::anomaly::fabric_drift_cell_of(5.0),
        );
        assert!(engine.drift_near.contains(&origin_cell));

        // Abandon the spawn neighborhood entirely.
        engine.player.relocate([500.0, 1.7, 500.0]);
        engine.tick(1.0 / 60.0, &input);
        assert!(
            engine
                .reality
                .fabric_drift_epoch(5.0, 5.0)
                >= 1,
            "abandoned territory must rearrange"
        );
        assert_eq!(
            engine.reality.fabric_drift_epoch(500.0, 500.0),
            0,
            "the territory around the player never drifts"
        );
        // Returning and leaving again drifts it again.
        engine.player.relocate([5.0, 1.7, 5.0]);
        engine.tick(1.0 / 60.0, &input);
        let first = engine.reality.fabric_drift_epoch(5.0, 5.0);
        engine.player.relocate([500.0, 1.7, 500.0]);
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(engine.reality.fabric_drift_epoch(5.0, 5.0), first + 1);
    }

    /// Tier 2: inside a blackout, cells wholly behind the player and beyond
    /// any light's reach shift in real time and their chunks force-rebuild.
    /// The space ahead and everything outside the blackout hold still.
    #[test]
    fn blackout_shifts_the_space_behind_the_player_while_inside() {
        use vackrooms::domain::entities::anomaly::{
            AnomalyKind, Axis2, AxisDirection, TraversalGate, TraversalGateKind, WorldBounds,
        };
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(FlatChunkSource),
        );
        let gate = TraversalGate {
            id: 11,
            instance_id: 401,
            anomaly_kind: AnomalyKind::BlackoutExpanse,
            kind: TraversalGateKind::Remap,
            axis: Axis2::X,
            plane: 0.0,
            span_min: -200.0,
            span_max: 200.0,
            forward: AxisDirection::Positive,
            affected_bounds: WorldBounds::new(-200.0, -200.0, 200.0, 200.0),
        };
        let mut payload = FlatChunkSource.load(0.0, 0.0, 0, 0);
        payload.traversal_gates.push(gate);
        engine.store.insert(
            chunk_key(0.0, 0.0),
            LoadedChunk::new((0.0, 0.0), 0, RealitySnapshot::empty(), payload),
        );
        // A resident chunk in the rear cell (player at z=5 facing -Z: the
        // cell z in [40, 80) is wholly behind and beyond the hide distance).
        engine.store.insert(
            chunk_key(0.0, 40.0),
            LoadedChunk::new(
                (0.0, 40.0),
                0,
                RealitySnapshot::empty(),
                FlatChunkSource.load(0.0, 40.0, 0, 0),
            ),
        );
        engine.player.position = [5.0, 1.7, 5.0];
        engine.player.yaw = 0.0; // forward = -Z
        engine.update_peripheral_shift(1.0 / 60.0);

        assert!(
            engine.reality.fabric_drift_epoch(5.0, 60.0) >= 1,
            "the cell behind the player must shift"
        );
        assert_eq!(
            engine.reality.fabric_drift_epoch(5.0, -60.0),
            0,
            "the cell ahead of the player must hold still"
        );
        assert!(
            engine.forced_reloads.contains(&chunk_key(0.0, 40.0)),
            "resident chunks of the shifted cell must rebuild"
        );
        // The cadence gate: an immediate second pass changes nothing more.
        let stamps = engine.reality.fabric_drifts().len();
        engine.update_peripheral_shift(1.0 / 60.0);
        assert_eq!(engine.reality.fabric_drifts().len(), stamps);
    }

    #[test]
    fn red_threshold_advances_reality_and_replaces_the_resident_world() {
        use vackrooms::domain::entities::anomaly::{
            AnomalyKind, Axis2, AxisDirection, TraversalGate, TraversalGateKind, WorldBounds,
        };
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(FlatChunkSource),
        );
        let gate = TraversalGate {
            id: 7,
            instance_id: 99,
            anomaly_kind: AnomalyKind::RedRoom,
            kind: TraversalGateKind::RedThreshold,
            axis: Axis2::X,
            plane: 5.0,
            span_min: 0.0,
            span_max: 10.0,
            forward: AxisDirection::Positive,
            affected_bounds: WorldBounds::new(0.0, 0.0, 10.0, 10.0),
        };
        let mut payload = FlatChunkSource.load(0.0, 0.0, 0, 0);
        payload.traversal_gates.push(gate);
        engine.store.insert(
            chunk_key(0.0, 0.0),
            LoadedChunk::new((0.0, 0.0), 0, RealitySnapshot::empty(), payload),
        );
        engine.store.insert(
            chunk_key(20.0, 0.0),
            LoadedChunk::new(
                (20.0, 0.0),
                0,
                RealitySnapshot::empty(),
                FlatChunkSource.load(20.0, 0.0, 0, 0),
            ),
        );
        engine.player.position = [6.0, 1.7, 5.0];
        engine.update_anomalies([4.0, 1.7, 5.0]);
        assert_eq!(engine.reality.lookup(99).unwrap().epoch, 1);
        assert!(engine.forced_reloads.contains(&chunk_key(0.0, 0.0)));
        assert!(
            engine.forced_reloads.contains(&chunk_key(20.0, 0.0)),
            "a distant resident must not remain in the base Level 0 address"
        );

        // Crossing the same directional threshold outward is not closure.
        let before = engine.reality.clone();
        engine.player.position = [4.0, 1.7, 5.0];
        engine.update_anomalies([6.0, 1.7, 5.0]);
        assert_eq!(engine.reality, before);
    }

    #[test]
    fn entering_a_pit_uses_authored_relocation() {
        use vackrooms::domain::entities::anomaly::PitHazard;
        use vackrooms::entities::models::Position;
        let mut engine = Engine::new(
            EngineConfig::default(),
            Box::new(RecordingRenderer::default()),
            Box::new(FlatChunkSource),
        );
        let mut payload = FlatChunkSource.load(0.0, 0.0, 0, 0);
        payload.pit_hazards.push(PitHazard {
            id: 1,
            instance_id: 2,
            center: Position::new(5.0, 5.0),
            half_side: 0.6,
            depth: 2.4,
            recovery: Position::new(7.0, 7.0),
        });
        engine.store.insert(
            chunk_key(0.0, 0.0),
            LoadedChunk::new((0.0, 0.0), 0, RealitySnapshot::empty(), payload),
        );
        engine.last_safe_position = [4.0, 1.7, 5.0];
        engine.player.position = [5.0, 1.7, 5.0];
        engine.update_anomalies([4.9, 1.7, 5.0]);
        assert_eq!(engine.player.position, [7.0, 1.7, 7.0]);
        engine.update_anomalies([7.0, 1.7, 7.0]);
        assert_eq!(engine.player.position, [7.0, 1.7, 7.0]);
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

    #[test]
    fn governor_raises_resolution_when_holding_60hz_vsync() {
        let mut governor = PerfGovernor::new();
        // rAF on a 60Hz display floors at ~16.7ms even with GPU headroom.
        for _ in 0..600 {
            governor.update(16.7);
        }
        assert_eq!(governor.scale(), 1.0, "must reach native res under vsync");
    }

    #[test]
    fn governor_locks_out_raises_after_a_failed_probe() {
        let mut governor = PerfGovernor::new();
        for _ in 0..300 {
            governor.update(16.7); // raises to 1.0 almost immediately
        }
        assert_eq!(governor.scale(), 1.0);
        for _ in 0..600 {
            governor.update(40.0); // raise fails: overloaded at 1.0
        }
        assert!(governor.scale() < 1.0);
        let settled = governor.scale();
        for _ in 0..600 {
            governor.update(16.7); // healthy again, but inside the lockout
        }
        assert_eq!(governor.scale(), settled, "raise must stay locked out");
    }
}
