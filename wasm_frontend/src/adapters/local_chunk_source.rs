//! In-wasm chunk source: implements the application's [`ChunkSourcePort`] by
//! driving the core engine use cases directly. There is no server round-trip;
//! the entire pipeline — procedural generation, voxel diffuse bake, SVO build, GPU
//! serialization, collision extraction — runs inside the wasm module.
//!
//! Pipeline per chunk:
//!   GenerateChunkArchitectureUseCase  (VoxelGrid, lighting included)
//!     -> BuildOctreeUseCase           (SparseVoxelOctree)
//!     -> OctreeGpuSerializer          (row-padded RGBA32UI texel stream)
//!     -> collision walk               (solid leaves -> world-space AABBs)

use vackrooms::adapters::octree_gpu_serializer::OctreeGpuSerializer;
use vackrooms::domain::entities::anomaly::RealitySnapshot;
use vackrooms::domain::entities::sparse_voxel_octree::{SparseVoxelOctree, SvoNode};
use vackrooms::domain::entities::voxel_grid::{
    FACE_OCCLUDED_NEGATIVE_X, FACE_OCCLUDED_NEGATIVE_Y, FACE_OCCLUDED_NEGATIVE_Z,
    FACE_OCCLUDED_POSITIVE_X, FACE_OCCLUDED_POSITIVE_Y, FACE_OCCLUDED_POSITIVE_Z, VOXEL_AIR,
    VoxelGrid,
};
use vackrooms::domain::use_cases::build_octree::BuildOctreeUseCase;
use vackrooms::entities::models::Position;
use vackrooms::use_cases::generate_chunk::{GenerateChunkArchitectureUseCase, GeneratorConfig};
use vackrooms::use_cases::ports::{NULL_TELEMETRY, NoiseProvider, TelemetryPort};

use crate::adapters::surface_mesh::build_surface_mesh;
use crate::application::collision::Aabb;
use crate::application::ports::{ChunkPayload, ChunkSourcePort};

/// Voxel types that block the player. FLOOR/CEILING/LIGHT/GRASS/WATER and
/// the carpet/fluid classes are visual-only: including them would make the
/// player collide with the floor they stand on. Shared with the core so a
/// new wall class can never render solid but collide hollow.
const SOLID_TYPES: [u8; 8] = vackrooms::domain::entities::voxel_grid::SOLID_MATERIALS;

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

    fn generate_payload(
        &self,
        origin_x: f32,
        origin_z: f32,
        level: u32,
        lod: u8,
        reality: &RealitySnapshot,
    ) -> ChunkPayload {
        let config = self.config.with_level(level).at_lod(lod);
        let generator =
            GenerateChunkArchitectureUseCase::with_telemetry(&self.noise, self.telemetry);

        // Generate one extra voxel around the X/Z perimeter, then crop the
        // authoritative interior for SVO/collision. The greedy mesher reads
        // the halo while deciding boundary faces, so it never invents a face
        // merely because the adjacent streamed chunk has not uploaded yet.
        let halo_config = GeneratorConfig {
            chunk_size: config.chunk_size + config.voxel_scale * 2.0,
            ..config
        };
        let halo_world_origin = [
            origin_x - config.voxel_scale,
            0.0,
            origin_z - config.voxel_scale,
        ];
        let halo_grid = generator.execute_with_reality(
            Position::new(halo_world_origin[0], halo_world_origin[2]),
            self.seed,
            halo_config,
            reality,
        );
        let grid = crop_lateral_halo(&halo_grid, 1);
        let surface = build_surface_mesh(&halo_grid, config.voxel_scale, lod, 1, halo_world_origin);

        let svo_depth = config.svo_depth();
        let svo = BuildOctreeUseCase::new().execute(&grid, svo_depth, config.svo_world_size());

        let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
        let collision = extract_collision_boxes(&svo, origin_x, origin_z, config.voxel_scale);

        ChunkPayload {
            root: svo.root as u32,
            nodes: gpu.texel_data,
            world_size: config.svo_world_size(),
            voxel_size: config.voxel_scale,
            svo_depth: svo_depth as u8,
            surface,
            collision,
            traversal_gates: halo_grid.traversal_gates.clone(),
            pit_hazards: halo_grid.pit_hazards.clone(),
            supply_items: halo_grid.supply_items.clone(),
            level_exits: halo_grid.level_exits.clone(),
        }
    }
}

impl<N: NoiseProvider> ChunkSourcePort for LocalChunkSource<N> {
    fn load(&self, origin_x: f32, origin_z: f32, level: u32, lod: u8) -> ChunkPayload {
        self.generate_payload(origin_x, origin_z, level, lod, &RealitySnapshot::default())
    }

    fn load_with_reality(
        &self,
        origin_x: f32,
        origin_z: f32,
        level: u32,
        lod: u8,
        reality: &RealitySnapshot,
    ) -> ChunkPayload {
        self.generate_payload(origin_x, origin_z, level, lod, reality)
    }
}

/// Copies the actual chunk interior out of a generated X/Z halo while
/// retaining voxel type and colored baked lighting, then deriving exact
/// six-direction neighbor occupancy while the halo is still available.
fn crop_lateral_halo(halo: &VoxelGrid, padding: usize) -> VoxelGrid {
    assert!(halo.width() > padding * 2 && halo.depth() > padding * 2);
    let mut inner = VoxelGrid::new(
        halo.width() - padding * 2,
        halo.height(),
        halo.depth() - padding * 2,
    );
    for z in 0..inner.depth() {
        for y in 0..inner.height() {
            for x in 0..inner.width() {
                let hx = x + padding;
                let hz = z + padding;
                inner.set(x, y, z, halo.get(hx, y, hz));
                inner.set_light_rgb(x, y, z, halo.get_light_rgb(hx, y, hz));
                inner.set_face_occlusion(x, y, z, neighbor_occlusion_mask(halo, hx, y, hz));
            }
        }
    }
    inner
}

/// Exact face closure for one generated voxel, including the lateral halo.
///
/// The returned bits travel unchanged through BuildOctree and the GPU atlas.
/// Computing them before the crop is what makes a streamed chunk boundary
/// authoritative even while its adjacent chunk is unloaded or represented by
/// an overlapping power-of-two padded SVO root.
fn neighbor_occlusion_mask(grid: &VoxelGrid, x: usize, y: usize, z: usize) -> u8 {
    if grid.get(x, y, z) == VOXEL_AIR {
        return 0;
    }

    let occupied = |x: isize, y: isize, z: isize| {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < grid.width()
            && (y as usize) < grid.height()
            && (z as usize) < grid.depth()
            && grid.get(x as usize, y as usize, z as usize) != VOXEL_AIR
    };
    let x = x as isize;
    let y = y as isize;
    let z = z as isize;
    let mut mask = 0;
    if occupied(x + 1, y, z) {
        mask |= FACE_OCCLUDED_POSITIVE_X;
    }
    if occupied(x - 1, y, z) {
        mask |= FACE_OCCLUDED_NEGATIVE_X;
    }
    if occupied(x, y + 1, z) {
        mask |= FACE_OCCLUDED_POSITIVE_Y;
    }
    if occupied(x, y - 1, z) {
        mask |= FACE_OCCLUDED_NEGATIVE_Y;
    }
    if occupied(x, y, z + 1) {
        mask |= FACE_OCCLUDED_POSITIVE_Z;
    }
    if occupied(x, y, z - 1) {
        mask |= FACE_OCCLUDED_NEGATIVE_Z;
    }
    mask
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
    use vackrooms::domain::entities::voxel_grid::{
        FACE_OCCLUDED_NEGATIVE_X, FACE_OCCLUDED_NEGATIVE_Y, FACE_OCCLUDED_POSITIVE_X,
        FACE_OCCLUDED_POSITIVE_Y, VOXEL_FLOOR, VOXEL_WALL, VoxelGrid,
    };
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
    fn generated_ceiling_panels_reach_the_runtime_light_contract() {
        let source =
            LocalChunkSource::new(SimpleNoiseProvider::new(), 42, GeneratorConfig::low_spec());
        // Seed 42's main-spine chunk contains the fixtures visible from the
        // deterministic browser spawn used by the GPU regression tests.
        let payload = source.load(0.0, 30.0, 0, 0);

        assert!(
            !payload.surface.lights.is_empty(),
            "voxelized emissive panels must not disappear before frame lighting"
        );
        for light in &payload.surface.lights {
            assert!(light.enabled);
            assert!(light.position[1] > payload.voxel_size);
            assert!(light.half_size.into_iter().all(|extent| extent > 0.0));
            assert!(
                light.intensity > 1.0,
                "panel radiance was normalized to darkness"
            );
            assert_ne!(light.kind, crate::application::ports::LightKind::Point);
        }
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

    #[test]
    fn halo_crop_matches_direct_chunk_generation() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let generator = GenerateChunkArchitectureUseCase::new(&noise);
        let direct = generator.execute(Position::new(10.0, -10.0), 42, config);
        let halo = generator.execute(
            Position::new(10.0 - config.voxel_scale, -10.0 - config.voxel_scale),
            42,
            GeneratorConfig {
                chunk_size: config.chunk_size + config.voxel_scale * 2.0,
                ..config
            },
        );
        let cropped = crop_lateral_halo(&halo, 1);
        assert_eq!(direct.width(), cropped.width());
        assert_eq!(direct.height(), cropped.height());
        assert_eq!(direct.depth(), cropped.depth());
        for z in 0..direct.depth() {
            for y in 0..direct.height() {
                for x in 0..direct.width() {
                    assert_eq!(direct.get(x, y, z), cropped.get(x, y, z));
                    assert_eq!(
                        direct.get_light_rgb(x, y, z),
                        cropped.get_light_rgb(x, y, z)
                    );
                    assert_eq!(
                        cropped.get_face_occlusion(x, y, z),
                        neighbor_occlusion_mask(&halo, x + 1, y, z + 1),
                    );
                }
            }
        }
    }

    #[test]
    fn halo_closes_chunk_boundary_faces_without_hiding_the_floor_top() {
        let mut halo = VoxelGrid::new(5, 3, 5);
        // Logical voxel (0, 1, 1), plus a real neighbor just beyond the
        // negative-X chunk boundary and a supporting voxel below it.
        halo.set(1, 1, 2, VOXEL_FLOOR);
        halo.set(0, 1, 2, VOXEL_WALL);
        halo.set(1, 0, 2, VOXEL_WALL);

        // Mirror the boundary condition at the other side. These halo cells
        // represent adjacent deterministic chunks even though their padded
        // SVO root cubes would overlap the current root in world space.
        halo.set(3, 1, 2, VOXEL_FLOOR);
        halo.set(4, 1, 2, VOXEL_WALL);

        let cropped = crop_lateral_halo(&halo, 1);
        let left = cropped.get_face_occlusion(0, 1, 1);
        assert_ne!(left & FACE_OCCLUDED_NEGATIVE_X, 0);
        assert_ne!(left & FACE_OCCLUDED_NEGATIVE_Y, 0);
        assert_eq!(left & FACE_OCCLUDED_POSITIVE_Y, 0);

        let right = cropped.get_face_occlusion(2, 1, 1);
        assert_ne!(right & FACE_OCCLUDED_POSITIVE_X, 0);
        assert_eq!(right & FACE_OCCLUDED_POSITIVE_Y, 0);
    }
}
