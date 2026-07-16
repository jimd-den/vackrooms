//! Chunk streaming policy and the in-memory chunk store.
//!
//! The world is an infinite plane of square chunks. Each frame the policy
//! computes the set of chunk origins that should be resident around the
//! player; the engine loads missing ones (budgeted, to avoid frame hitches —
//! wasm has no threads, so time-slicing replaces the background meshing
//! thread a native engine would use) and evicts the rest.

use std::collections::HashMap;

use crate::application::collision::Aabb;
use crate::application::ports::ChunkPayload;
use vackrooms::domain::entities::anomaly::RealitySnapshot;
use vackrooms::domain::entities::anomaly::{LevelExit, PitHazard, TraversalGate};
use vackrooms::domain::entities::supplies::SupplyItem;

/// Integer key for a chunk, quantized from its world origin.
/// Origins are always integer multiples of `chunk_size`, so rounding to
/// millimetres gives a stable, hashable key.
pub type ChunkKey = (i64, i64);

pub fn chunk_key(origin_x: f32, origin_z: f32) -> ChunkKey {
    (
        (origin_x * 1000.0).round() as i64,
        (origin_z * 1000.0).round() as i64,
    )
}

/// Decides which chunks should be resident for a given player position.
#[derive(Debug, Clone, Copy)]
pub struct StreamingPolicy {
    /// World-space side length of a chunk.
    pub chunk_size: f32,
    /// Chebyshev radius in chunks: 1 -> 3x3 resident set, 2 -> 5x5.
    pub radius: i32,
}

impl StreamingPolicy {
    /// Returns the desired chunk origins, sorted nearest-first so the load
    /// budget is always spent on the most urgent chunk.
    pub fn desired_origins(&self, player_x: f32, player_z: f32) -> Vec<(f32, f32)> {
        let px = if player_x.is_finite() { player_x } else { 0.0 };
        let pz = if player_z.is_finite() { player_z } else { 0.0 };
        let base_x = (px / self.chunk_size).floor() * self.chunk_size;
        let base_z = (pz / self.chunk_size).floor() * self.chunk_size;

        let mut origins = Vec::with_capacity(((2 * self.radius + 1).pow(2)) as usize);
        for dz in -self.radius..=self.radius {
            for dx in -self.radius..=self.radius {
                origins.push((
                    base_x + dx as f32 * self.chunk_size,
                    base_z + dz as f32 * self.chunk_size,
                ));
            }
        }
        origins.sort_by(|a, b| {
            let da = (a.0 + self.chunk_size * 0.5 - px).powi(2)
                + (a.1 + self.chunk_size * 0.5 - pz).powi(2);
            let db = (b.0 + self.chunk_size * 0.5 - px).powi(2)
                + (b.1 + self.chunk_size * 0.5 - pz).powi(2);
            da.total_cmp(&db)
        });
        origins
    }
}

/// A resident chunk: payload plus its world origin and current level of
/// detail (0 = full resolution; higher = coarser, see `ChunkSourcePort`).
#[derive(Debug, Clone)]
pub struct LoadedChunk {
    pub origin: (f32, f32),
    pub lod: u8,
    /// Exact immutable encounter state from which every payload view
    /// (collision, SVO, mesh and lighting) was derived.
    pub reality: RealitySnapshot,
    /// Cached identity for diagnostics and fast rejection paths. Exact
    /// equality still uses `reality`, so a hash collision cannot mix worlds.
    pub reality_fingerprint: u64,
    pub payload: ChunkPayload,
}

impl LoadedChunk {
    pub fn new(
        origin: (f32, f32),
        lod: u8,
        reality: RealitySnapshot,
        payload: ChunkPayload,
    ) -> Self {
        let reality_fingerprint = reality.fingerprint();
        Self {
            origin,
            lod,
            reality,
            reality_fingerprint,
            payload,
        }
    }

    pub fn is_in_reality(&self, reality: &RealitySnapshot) -> bool {
        self.reality_fingerprint == reality.fingerprint() && &self.reality == reality
    }
}

/// Keyed store of resident chunks. Iteration order is the insertion order of
/// keys (tracked separately) so atlas offsets stay deterministic.
#[derive(Debug, Default)]
pub struct ChunkStore {
    chunks: HashMap<ChunkKey, LoadedChunk>,
    order: Vec<ChunkKey>,
}

impl ChunkStore {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn contains(&self, key: ChunkKey) -> bool {
        self.chunks.contains_key(&key)
    }

    pub fn get(&self, key: ChunkKey) -> Option<&LoadedChunk> {
        self.chunks.get(&key)
    }

    pub fn insert(&mut self, key: ChunkKey, chunk: LoadedChunk) {
        if self.chunks.insert(key, chunk).is_none() {
            self.order.push(key);
        }
    }

    /// Removes every chunk whose key is not in `keep`, returning whether
    /// anything changed.
    pub fn retain_keys(&mut self, keep: &[ChunkKey]) -> bool {
        let before = self.order.len();
        self.order.retain(|k| keep.contains(k));
        self.chunks.retain(|k, _| keep.contains(k));
        before != self.order.len()
    }

    /// Chunks in deterministic (insertion) order.
    pub fn iter_ordered(&self) -> impl Iterator<Item = &LoadedChunk> {
        self.order.iter().filter_map(|k| self.chunks.get(k))
    }

    /// All collision boxes of all resident chunks.
    pub fn all_collision_boxes(&self) -> impl Iterator<Item = &Aabb> {
        self.iter_ordered().flat_map(|c| c.payload.collision.iter())
    }

    pub fn all_traversal_gates(&self) -> impl Iterator<Item = &TraversalGate> {
        self.iter_ordered()
            .flat_map(|c| c.payload.traversal_gates.iter())
    }

    pub fn all_supply_items(&self) -> impl Iterator<Item = &SupplyItem> {
        self.iter_ordered()
            .flat_map(|c| c.payload.supply_items.iter())
    }

    pub fn all_level_exits(&self) -> impl Iterator<Item = &LevelExit> {
        self.iter_ordered()
            .flat_map(|c| c.payload.level_exits.iter())
    }

    pub fn all_pit_hazards(&self) -> impl Iterator<Item = &PitHazard> {
        self.iter_ordered()
            .flat_map(|c| c.payload.pit_hazards.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desired_origins_covers_square_around_player() {
        let policy = StreamingPolicy {
            chunk_size: 10.0,
            radius: 1,
        };
        let origins = policy.desired_origins(5.0, 5.0);
        assert_eq!(origins.len(), 9);
        // Nearest-first: the chunk containing the player comes first.
        assert_eq!(origins[0], (0.0, 0.0));
        assert!(origins.contains(&(-10.0, -10.0)));
        assert!(origins.contains(&(10.0, 10.0)));
    }

    #[test]
    fn desired_origins_handles_negative_coordinates() {
        let policy = StreamingPolicy {
            chunk_size: 10.0,
            radius: 1,
        };
        let origins = policy.desired_origins(-0.1, -0.1);
        assert_eq!(origins[0], (-10.0, -10.0));
    }

    #[test]
    fn store_eviction_reports_change() {
        let mut store = ChunkStore::new();
        let payload = ChunkPayload {
            root: 0,
            nodes: vec![],
            world_size: 10.0,
            voxel_size: 0.2,
            svo_depth: 6,
            surface: crate::application::ports::SurfaceMeshPayload::empty(0),
            collision: vec![],
            traversal_gates: vec![],
            pit_hazards: vec![],
            supply_items: vec![],
            level_exits: vec![],
        };
        store.insert(
            chunk_key(0.0, 0.0),
            LoadedChunk::new((0.0, 0.0), 0, RealitySnapshot::default(), payload.clone()),
        );
        store.insert(
            chunk_key(10.0, 0.0),
            LoadedChunk::new((10.0, 0.0), 0, RealitySnapshot::default(), payload),
        );
        assert_eq!(store.len(), 2);
        assert!(store.retain_keys(&[chunk_key(0.0, 0.0)]));
        assert_eq!(store.len(), 1);
        assert!(!store.retain_keys(&[chunk_key(0.0, 0.0)]));
    }
}
