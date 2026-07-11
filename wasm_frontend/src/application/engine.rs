//! The frame-loop orchestrator: the application's central use case.
//!
//! `Engine::tick` is called once per animation frame by the browser driver
//! with plain data in (`InputFrame`, dt) and drives everything else through
//! ports: chunk loading via [`ChunkSourcePort`], drawing via [`RendererPort`].
//! No browser types appear anywhere in this file, so the whole game loop is
//! natively unit-testable.

use std::collections::HashMap;

use crate::application::atlas::{AtlasPool, MAX_CHUNKS, payload_rows};
use crate::application::collision::CollisionWorld;
use crate::application::player::{MoveIntent, Player};
use crate::application::ports::{
    ChunkDraw, ChunkRequest, ChunkSourcePort, FrameParams, RendererPort, SurfaceChunk,
};
use crate::application::streaming::{
    ChunkKey, ChunkStore, LoadedChunk, StreamingPolicy, chunk_key,
};

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
    /// Chunks resident at full resolution; the rest are coarse LOD.
    pub fine_chunks: usize,
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
const LEVEL_GRASSLAND: u32 = 34;
/// Keep pushing into a wall this long before a noclip roll happens...
const NOCLIP_PUSH_SECONDS: f32 = 1.2;
/// ...then one roll per second of continued pushing, at this probability.
const NOCLIP_CHANCE: f32 = 0.2;

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
    /// Outstanding background loads (async sources only): key → requested
    /// LOD. Prevents duplicate requests while a chunk is in flight.
    pending: HashMap<ChunkKey, u8>,
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
        // In-flight loads are for the old level; their completions will be
        // rejected by the level check, and clearing pending lets the new
        // level re-request the same keys immediately.
        self.pending.clear();
        // The whole world is regenerated: drop the pool so the next stream
        // pass relayouts and re-uploads from scratch.
        self.pool = AtlasPool::new();
        self.renderer.clear_surfaces();
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

        let frame = FrameParams {
            camera_pos: self.player.position,
            yaw: self.player.yaw,
            pitch: self.player.pitch,
            flashlight: input.flashlight,
        };
        self.renderer.draw(&frame, &self.draws);
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

    /// Installs finished background loads, discarding stale work: a result
    /// for the wrong level (noclip happened), for a chunk that left the
    /// streaming footprint, or for a LOD no better than what is already
    /// resident. Refinement stays monotonic exactly like the sync path.
    fn install_completed(
        &mut self,
        keep: &[ChunkKey],
        loaded: &mut Vec<ChunkKey>,
        changed: &mut bool,
    ) {
        for done in self.source.poll_completed() {
            let req = done.request;
            let key = chunk_key(req.origin_x, req.origin_z);
            self.pending.remove(&key);
            if req.level != self.level || !keep.contains(&key) {
                continue;
            }
            let improves = match self.store.get(key) {
                None => true,
                Some(resident) => req.lod < resident.lod,
            };
            if !improves {
                continue;
            }
            self.store.insert(
                key,
                LoadedChunk {
                    origin: (req.origin_x, req.origin_z),
                    lod: req.lod,
                    payload: done.payload,
                },
            );
            if !loaded.contains(&key) {
                loaded.push(key);
            }
            *changed = true;
        }
        // Drop pending entries that left the footprint so their slots free
        // up; a late completion for them is discarded by the keep check.
        self.pending.retain(|key, _| keep.contains(key));
    }

    /// Tops up the background request queue: coarse availability for every
    /// missing chunk (nearest first), then at most one fine refinement in
    /// flight at a time, mirroring the sync path's priorities.
    fn issue_requests(&mut self, desired: &[(f32, f32)], desired_visual: &[(f32, f32)]) {
        // Keep roughly a worker pool's worth of requests in flight; more
        // would just build a stale backlog behind a moving player.
        let max_pending = (self.config.max_loads_per_tick * 2).max(4);

        for &(ox, oz) in desired_visual {
            if self.pending.len() >= max_pending {
                break;
            }
            let key = chunk_key(ox, oz);
            if !self.store.contains(key) && !self.pending.contains_key(&key) {
                self.pending.insert(key, COARSE_LOD);
                self.source.request(ChunkRequest {
                    origin_x: ox,
                    origin_z: oz,
                    level: self.level,
                    lod: COARSE_LOD,
                });
            }
        }

        let fine_in_flight = self.pending.values().any(|&lod| lod == 0);
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
            self.pending.insert(chunk_key(ox, oz), 0);
            self.source.request(ChunkRequest {
                origin_x: ox,
                origin_z: oz,
                level: self.level,
                lod: 0,
            });
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
        let keep: Vec<_> = desired_visual.iter().map(|&(x, z)| chunk_key(x, z)).collect();

        // Free the atlas slots of chunks about to be evicted.
        let evicted: Vec<ChunkKey> = self
            .store
            .iter_ordered()
            .map(|c| chunk_key(c.origin.0, c.origin.1))
            .filter(|k| !keep.contains(k))
            .collect();
        let mut changed = self.store.retain_keys(&keep);
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

            // Phase 1 — availability: missing chunks come in coarse, nearest
            // first. Only a leftover budget flows into refinement, so a moving
            // player always fills holes before sharpening anything.
            for &(ox, oz) in &desired_visual {
                if budget == 0 {
                    break;
                }
                let key = chunk_key(ox, oz);
                if !self.store.contains(key) {
                    let payload = self.source.load(ox, oz, self.level, COARSE_LOD);
                    self.store.insert(
                        key,
                        LoadedChunk {
                            origin: (ox, oz),
                            lod: COARSE_LOD,
                            payload,
                        },
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
                let payload = self.source.load(ox, oz, self.level, 0);
                self.store.insert(
                    key,
                    LoadedChunk {
                        origin: (ox, oz),
                        lod: 0,
                        payload,
                    },
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
                surface: crate::application::ports::SurfaceMeshPayload::empty(0),
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
        fn load(&self, _x: f32, _z: f32, level: u32, _lod: u8) -> ChunkPayload {
            self.levels.borrow_mut().push(level);
            ChunkPayload {
                root: 0,
                nodes: [1u32, 0, 0, 0].repeat(1024),
                world_size: 12.8,
                surface: crate::application::ports::SurfaceMeshPayload::empty(0),
                collision: vec![Aabb::new([-100.0, 0.0, -100.0], [100.0, 3.0, 100.0])],
            }
        }
    }

    /// Background source double: records requests, completes only what the
    /// test explicitly finishes, and panics on any blocking load.
    #[derive(Default)]
    struct AsyncFakeSource {
        requests: Rc<RefCell<Vec<ChunkRequest>>>,
        ready: Rc<RefCell<Vec<crate::application::ports::CompletedChunk>>>,
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
    }

    fn completed(request: ChunkRequest) -> crate::application::ports::CompletedChunk {
        crate::application::ports::CompletedChunk {
            request,
            payload: FlatChunkSource.load(request.origin_x, request.origin_z, request.level, request.lod),
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
        ready.borrow_mut().push(completed(ChunkRequest {
            origin_x: 0.0,
            origin_z: 0.0,
            level: 34,
            lod: 1,
        }));
        // Outside the streaming footprint entirely.
        ready.borrow_mut().push(completed(ChunkRequest {
            origin_x: 500.0,
            origin_z: 500.0,
            level: 0,
            lod: 1,
        }));
        engine.tick(1.0 / 60.0, &input);
        assert_eq!(
            engine.stats().resident_chunks,
            0,
            "stale completions must not be installed"
        );
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
        ready.borrow_mut().push(completed(ChunkRequest {
            origin_x: 0.0,
            origin_z: 0.0,
            level: 0,
            lod: 1,
        }));
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
