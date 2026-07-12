//! The pluggable level-generator port: each Backrooms "level" (Level 0
//! offices, Level 34-B grassland, ...) is one implementation.
//!
//! Contract for implementors:
//! * **Seamless tiling** — derive ALL geometry from *world-space* coordinates
//!   (chunk origin + local voxel index), never from chunk-local positions, so
//!   independently generated chunks line up at their borders.
//! * **Walkability** — the ground plane must stay connected: never fully
//!   enclose a region the player could be in. Doorways/gaps must be at least
//!   ~1 world unit wide.
//! * **Determinism** — same (chunk_pos, seed, config) must always produce the
//!   same grid; only the injected [`NoiseProvider`] may be used for variety.
//! * Lighting is NOT the generator's job: place `VOXEL_LIGHT` sources and the
//!   orchestrating use case runs the BFS lighting pass afterwards.

use crate::domain::entities::anomaly::RealitySnapshot;
use crate::domain::entities::voxel_grid::VoxelGrid;
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::NoiseProvider;

/// Well-known level ids (the `GeneratorConfig::level` / noclip targets).
pub const LEVEL_BACKROOMS: u32 = 0;
pub const LEVEL_LEGACY_OFFICES: u32 = 1;
pub const LEVEL_GRASSLAND: u32 = 34;

pub trait LevelGenerator {
    /// Fills one chunk. `chunk_pos` is the chunk origin in world units.
    fn generate(
        &self,
        chunk_pos: Position,
        seed: u32,
        config: GeneratorConfig,
        noise: &dyn NoiseProvider,
    ) -> VoxelGrid;

    /// Fills one chunk under a deterministic encounter-state snapshot.
    ///
    /// Most levels are entirely immutable and inherit this implementation.
    /// Level 0 overrides it so anomaly infill/topology can be a pure function
    /// of the same snapshot that keyed the chunk request. Keeping the legacy
    /// `generate` entry point makes stateless tools and tests explicit users
    /// of epoch zero.
    fn generate_with_reality(
        &self,
        chunk_pos: Position,
        seed: u32,
        config: GeneratorConfig,
        noise: &dyn NoiseProvider,
        _reality: &RealitySnapshot,
    ) -> VoxelGrid {
        self.generate(chunk_pos, seed, config, noise)
    }
}
