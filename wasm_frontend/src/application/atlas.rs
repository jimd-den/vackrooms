//! Pooled SVO node atlas: stable, fixed-size chunk slots inside one shared
//! RGBA32UI texture.
//!
//! This is the raymarcher's analog of "vertex pooling" in mesh-based voxel
//! engines. The naive approach (concatenate every resident chunk's node
//! array and re-upload the whole texture on any change) makes every chunk
//! load cost O(world) in CPU copies and texture bandwidth. The pool instead
//! gives each chunk a fixed row-aligned slot: loading a chunk rebases and
//! uploads *only that slot's rows*, and evicting a chunk is free (its slot is
//! simply reused later; stale texels are never referenced by the draw table).
//!
//! Every chunk's node array is row-padded to whole 1024-texel rows by the
//! core `OctreeGpuSerializer`, so slots stay row-aligned by construction.

use crate::application::ports::ChunkPayload;
use crate::application::streaming::ChunkKey;

/// The shader's fixed per-frame chunk table size (uniform array length).
pub const MAX_CHUNKS: usize = 25;

/// Texels (= SVO nodes) per atlas row. Must match the shader's `decodeNode`
/// constant and the core serializer's row padding.
pub const ROW_TEXELS: usize = 1024;

/// One air-leaf texel (type flag 1, voxel type 0): what free space decodes to.
const AIR_LEAF: [u32; 4] = [1, 0, 0, 0];

#[derive(Debug, Default)]
pub struct AtlasPool {
    /// Rows per slot. Grows (forcing a relayout) when a chunk won't fit.
    slot_rows: usize,
    /// Slot occupancy; index is the slot number.
    slots: Vec<Option<ChunkKey>>,
}

impl AtlasPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Nodes (texels) per slot.
    pub fn slot_nodes(&self) -> usize {
        self.slot_rows * ROW_TEXELS
    }

    /// Total rows in the pooled texture.
    pub fn total_rows(&self) -> usize {
        self.slot_rows * self.slots.len()
    }

    /// Ensures the pool holds at least `num_slots` slots of at least `rows`
    /// rows each. Returns `true` when the layout changed — every slot
    /// assignment is then dropped and the caller must reassign all resident
    /// chunks and re-upload the whole pool.
    pub fn ensure_layout(&mut self, num_slots: usize, rows: usize) -> bool {
        let changed = rows > self.slot_rows || num_slots > self.slots.len();
        if changed {
            self.slot_rows = self.slot_rows.max(rows);
            let len = self.slots.len().max(num_slots);
            self.slots.clear();
            self.slots.resize(len, None);
        }
        changed
    }

    /// Assigns (or finds) the slot for `key`. Returns `None` when the pool
    /// is full — the caller sized it for the streaming radius, so that is a
    /// logic error handled by falling back to a full relayout.
    pub fn assign(&mut self, key: ChunkKey) -> Option<usize> {
        if let Some(i) = self.slots.iter().position(|s| *s == Some(key)) {
            return Some(i);
        }
        let free = self.slots.iter().position(|s| s.is_none())?;
        self.slots[free] = Some(key);
        Some(free)
    }

    pub fn release(&mut self, key: ChunkKey) {
        for slot in &mut self.slots {
            if *slot == Some(key) {
                *slot = None;
            }
        }
    }

    /// First texture row of a slot.
    pub fn slot_first_row(&self, slot: usize) -> usize {
        slot * self.slot_rows
    }

    /// Node offset of `key`'s slot within the pooled atlas.
    pub fn node_offset_of(&self, key: ChunkKey) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| *s == Some(key))
            .map(|i| i * self.slot_nodes())
    }

    /// Whether `payload` fits in the current slot size.
    pub fn fits(&self, payload: &ChunkPayload) -> bool {
        payload.nodes.len() / 4 <= self.slot_nodes()
    }

    /// The slot-sized texel block for one chunk: child pointers rebased to
    /// the slot's node offset, tail padded with air leaves.
    pub fn rebased_block(&self, slot: usize, payload: &ChunkPayload) -> Vec<u32> {
        let node_offset = (slot * self.slot_nodes()) as u32;
        let mut block = Vec::with_capacity(self.slot_nodes() * 4);
        block.extend_from_slice(&payload.nodes);
        // Internal nodes (type flag 0) store child indices local to the
        // chunk; shift them into pool space.
        for i in 0..(payload.nodes.len() / 4) {
            if block[i * 4] == 0 {
                block[i * 4 + 1] += node_offset;
            }
        }
        while block.len() < self.slot_nodes() * 4 {
            block.extend_from_slice(&AIR_LEAF);
        }
        block
    }

    /// The whole pool as one texel stream (initial upload, relayouts, and
    /// back ends without partial-update support). `lookup` resolves a slot's
    /// chunk key to its payload.
    pub fn full_texels<'a>(
        &self,
        lookup: impl Fn(ChunkKey) -> Option<&'a ChunkPayload>,
    ) -> Vec<u32> {
        let mut texels = Vec::with_capacity(self.total_rows() * ROW_TEXELS * 4);
        for (slot, occupant) in self.slots.iter().enumerate() {
            match occupant.and_then(&lookup) {
                Some(payload) => texels.extend_from_slice(&self.rebased_block(slot, payload)),
                None => {
                    for _ in 0..self.slot_nodes() {
                        texels.extend_from_slice(&AIR_LEAF);
                    }
                }
            }
        }
        texels
    }
}

/// Rows needed to hold a payload's node array.
pub fn payload_rows(payload: &ChunkPayload) -> usize {
    (payload.nodes.len() / 4).div_ceil(ROW_TEXELS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::streaming::chunk_key;

    fn payload(root: u32, node_count: usize) -> ChunkPayload {
        ChunkPayload {
            root,
            nodes: vec![0; node_count * 4],
            world_size: 12.8,
            voxel_size: 0.2,
            svo_depth: 6,
            surface: crate::application::ports::SurfaceMeshPayload::empty(0),
            collision: vec![],
            traversal_gates: vec![],
            pit_hazards: vec![],
            supply_items: vec![],
            level_exits: vec![],
        }
    }

    #[test]
    fn slots_are_stable_across_unrelated_evictions() {
        let mut pool = AtlasPool::new();
        pool.ensure_layout(4, 2);
        let (a, b, c) = (
            chunk_key(0.0, 0.0),
            chunk_key(10.0, 0.0),
            chunk_key(20.0, 0.0),
        );
        let slot_a = pool.assign(a).unwrap();
        let slot_b = pool.assign(b).unwrap();
        pool.release(a);
        let slot_c = pool.assign(c).unwrap();
        // b never moved; c reused a's freed slot.
        assert_eq!(pool.node_offset_of(b), Some(slot_b * pool.slot_nodes()));
        assert_eq!(slot_c, slot_a);
    }

    #[test]
    fn rebased_block_shifts_internal_child_pointers_only() {
        let mut pool = AtlasPool::new();
        pool.ensure_layout(2, 1);

        let mut p = payload(0, 2);
        // node 0: internal, child_base 8; node 1: leaf, voxel_type 42.
        p.nodes[0] = 0;
        p.nodes[1] = 8;
        p.nodes[4] = 1;
        p.nodes[5] = 42;

        let block = pool.rebased_block(1, &p);
        assert_eq!(block.len(), pool.slot_nodes() * 4);
        assert_eq!(block[1], 8 + pool.slot_nodes() as u32);
        assert_eq!(block[4], 1, "leaf type flag untouched");
        assert_eq!(block[5], 42, "leaf payload untouched");
        // Padding decodes as air leaves.
        assert_eq!(&block[8..12], &AIR_LEAF);
    }

    #[test]
    fn growing_the_layout_invalidates_assignments() {
        let mut pool = AtlasPool::new();
        assert!(pool.ensure_layout(2, 1));
        let key = chunk_key(0.0, 0.0);
        pool.assign(key).unwrap();
        assert!(!pool.ensure_layout(2, 1), "same layout is a no-op");
        assert!(pool.ensure_layout(2, 3), "bigger slots relayout");
        assert_eq!(pool.node_offset_of(key), None);
    }

    #[test]
    fn full_texels_covers_every_slot() {
        let mut pool = AtlasPool::new();
        pool.ensure_layout(3, 1);
        let key = chunk_key(0.0, 0.0);
        let slot = pool.assign(key).unwrap();
        let p = payload(0, 4);
        let texels = pool.full_texels(|k| (k == key).then_some(&p));
        assert_eq!(texels.len(), 3 * pool.slot_nodes() * 4);
        // Unassigned slots are air.
        let other_slot = (slot + 1) % 3;
        let base = other_slot * pool.slot_nodes() * 4;
        assert_eq!(&texels[base..base + 4], &AIR_LEAF);
    }
}
