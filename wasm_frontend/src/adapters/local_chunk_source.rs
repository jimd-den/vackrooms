//! In-wasm chunk source: implements the application's [`ChunkSourcePort`] by
//! driving the core engine use cases directly. There is no server round-trip;
//! the entire pipeline — procedural generation, BFS lighting, SVO build, GPU
//! serialization, collision extraction — runs inside the wasm module.
//!
//! Pipeline per chunk:
//!   GenerateChunkArchitectureUseCase  (VoxelGrid, lighting included)
//!     -> BuildOctreeUseCase           (SparseVoxelOctree)
//!     -> OctreeGpuSerializer          (row-padded RGBA32UI texel stream)
//!     -> collision walk               (solid leaves -> world-space AABBs)

use vackrooms::adapters::octree_gpu_serializer::OctreeGpuSerializer;
use vackrooms::domain::entities::sparse_voxel_octree::{SparseVoxelOctree, SvoNode};
use vackrooms::domain::use_cases::build_octree::BuildOctreeUseCase;
use vackrooms::entities::models::Position;
use vackrooms::use_cases::generate_chunk::{GenerateChunkArchitectureUseCase, GeneratorConfig};
use vackrooms::use_cases::ports::{NULL_TELEMETRY, NoiseProvider, TelemetryPort};

use crate::application::collision::Aabb;
use crate::application::ports::{ChunkPayload, ChunkSourcePort};

/// Voxel types that block the player. FLOOR/CEILING/LIGHT/GRASS/WATER are
/// visual-only: including them would make the player collide with the floor
/// they stand on.
const SOLID_TYPES: [u8; 3] = [
    vackrooms::domain::entities::voxel_grid::VOXEL_WALL,
    vackrooms::domain::entities::voxel_grid::VOXEL_RED_WALL,
    vackrooms::domain::entities::voxel_grid::VOXEL_TREE,
];

pub struct LocalChunkSource<N: NoiseProvider> {
    noise: N,
    telemetry: &'static dyn TelemetryPort,
    seed: u32,
    config: GeneratorConfig,
}

impl<N: NoiseProvider> LocalChunkSource<N> {
    pub fn new(noise: N, seed: u32, config: GeneratorConfig) -> Self {
        Self {
            noise,
            telemetry: &NULL_TELEMETRY,
            seed,
            config,
        }
    }

    pub fn with_telemetry(
        noise: N,
        seed: u32,
        config: GeneratorConfig,
        telemetry: &'static dyn TelemetryPort,
    ) -> Self {
        Self {
            noise,
            telemetry,
            seed,
            config,
        }
    }
}

impl<N: NoiseProvider> ChunkSourcePort for LocalChunkSource<N> {
    fn load(&self, origin_x: f32, origin_z: f32, level: u32, lod: u8) -> ChunkPayload {
        let config = self.config.with_level(level).at_lod(lod);
        let generator =
            GenerateChunkArchitectureUseCase::with_telemetry(&self.noise, self.telemetry);
        let grid = generator.execute(Position::new(origin_x, origin_z), self.seed, config);

        let svo =
            BuildOctreeUseCase::new().execute(&grid, config.svo_depth(), config.svo_world_size());

        let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
        let collision = extract_collision_boxes(&svo, origin_x, origin_z, config.voxel_scale);

        ChunkPayload {
            root: svo.root as u32,
            nodes: gpu.texel_data,
            world_size: config.svo_world_size(),
            collision,
        }
    }
}

/// Walks the SVO and emits one world-space AABB per solid leaf region.
/// Uniform subtrees collapsed by the SVO become single large boxes for free,
/// which keeps the collision set small.
pub fn extract_collision_boxes(
    svo: &SparseVoxelOctree,
    origin_x: f32,
    origin_z: f32,
    voxel_scale: f32,
) -> Vec<Aabb> {
    let mut boxes = Vec::new();
    let size = 1u32 << svo.depth;
    walk(
        svo,
        svo.root,
        0,
        0,
        0,
        size,
        origin_x,
        origin_z,
        voxel_scale,
        &mut boxes,
    );
    boxes
}

#[allow(clippy::too_many_arguments)]
fn walk(
    svo: &SparseVoxelOctree,
    node_idx: usize,
    x: u32,
    y: u32,
    z: u32,
    size: u32,
    origin_x: f32,
    origin_z: f32,
    voxel_scale: f32,
    out: &mut Vec<Aabb>,
) {
    match svo.nodes[node_idx] {
        SvoNode::Leaf { voxel_type, .. } => {
            if SOLID_TYPES.contains(&voxel_type) {
                let min = [
                    origin_x + x as f32 * voxel_scale,
                    y as f32 * voxel_scale,
                    origin_z + z as f32 * voxel_scale,
                ];
                let extent = size as f32 * voxel_scale;
                out.push(Aabb::new(
                    min,
                    [min[0] + extent, min[1] + extent, min[2] + extent],
                ));
            }
        }
        SvoNode::Internal {
            child_base_index,
            child_mask,
        } => {
            let half = size / 2;
            for child in 0..8u32 {
                if child_mask & (1 << child) != 0 {
                    walk(
                        svo,
                        child_base_index as usize + child as usize,
                        x + (child & 1) * half,
                        y + ((child >> 1) & 1) * half,
                        z + ((child >> 2) & 1) * half,
                        half,
                        origin_x,
                        origin_z,
                        voxel_scale,
                        out,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vackrooms::domain::entities::voxel_grid::{VOXEL_FLOOR, VOXEL_WALL, VoxelGrid};
    use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;

    #[test]
    fn wall_leaf_becomes_world_space_box_and_floor_does_not() {
        let mut grid = VoxelGrid::new(4, 4, 4);
        grid.set(1, 0, 2, VOXEL_WALL);
        grid.set(0, 0, 0, VOXEL_FLOOR);
        let svo = BuildOctreeUseCase::new().execute(&grid, 2, 2.0);

        let boxes = extract_collision_boxes(&svo, 10.0, 20.0, 0.5);
        assert_eq!(boxes.len(), 1);
        let b = &boxes[0];
        assert_eq!(b.min, [10.5, 0.0, 21.0]);
        assert_eq!(b.max, [11.0, 0.5, 21.5]);
    }

    #[test]
    fn generated_chunk_produces_row_padded_nodes_and_some_collision() {
        let source =
            LocalChunkSource::new(SimpleNoiseProvider::new(), 42, GeneratorConfig::low_spec());
        let payload = source.load(10.0, 10.0, 0, 0);
        // Row padding: node stream is a whole number of 1024-texel rows.
        assert_eq!(payload.nodes.len() % (1024 * 4), 0);
        assert!(payload.world_size > 0.0);
        assert!(
            !payload.collision.is_empty(),
            "a maze chunk must have walls"
        );
    }

    #[test]
    fn coarse_lod_is_smaller_but_covers_the_same_world_cube() {
        let source =
            LocalChunkSource::new(SimpleNoiseProvider::new(), 42, GeneratorConfig::low_spec());
        let fine = source.load(10.0, 10.0, 0, 0);
        let coarse = source.load(10.0, 10.0, 0, 1);
        assert_eq!(
            coarse.world_size, fine.world_size,
            "LODs must be interchangeable to the renderer"
        );
        assert!(
            coarse.nodes.len() < fine.nodes.len(),
            "coarse SVO must be smaller: {} vs {}",
            coarse.nodes.len(),
            fine.nodes.len()
        );
        assert!(
            !coarse.collision.is_empty(),
            "coarse chunks still collide (they cover the pre-refinement window)"
        );
    }
}
