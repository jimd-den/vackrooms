use crate::domain::entities::voxel_grid::VoxelGrid;
use crate::adapters::voxel_mapper::VoxelMapper;
use crate::adapters::json_presenter::JsonPresenter;
use crate::domain::use_cases::build_octree::BuildOctreeUseCase;
use crate::adapters::octree_gpu_serializer::OctreeGpuSerializer;

/// WebRendererAdapter acts as a Presenter in Clean Architecture.
/// It translates the core domain object (VoxelGrid) into formats
/// suitable for the Web Presentation layer.
pub struct WebRendererAdapter;

impl WebRendererAdapter {
    /// Translates VoxelGrid into a JSON string representing greedy-meshed quads.
    pub fn to_json(grid: &VoxelGrid, _chunk_size: f32) -> String {
        let mapper = VoxelMapper::new(0.1);
        let quads = mapper.map_voxel_grid(grid);
        JsonPresenter::render_voxels(&quads)
    }

    /// Converts VoxelGrid into a Sparse Voxel Octree (SVO) and serializes
    /// the flattened node texture buffer and structural metadata to JSON.
    pub fn to_octree_json(grid: &VoxelGrid, depth: u32, world_size: f32) -> String {
        let builder = BuildOctreeUseCase::new();
        let svo = builder.execute(grid, depth, world_size);
        let gpu_data = OctreeGpuSerializer::serialize_to_gpu_data(&svo);

        let mut json = String::with_capacity(gpu_data.texel_data.len() * 8 + 128);
        json.push_str("{\"root\":");
        json.push_str(&svo.root.to_string());
        json.push_str(",\"depth\":");
        json.push_str(&svo.depth.to_string());
        json.push_str(",\"world_size\":");
        json.push_str(&svo.world_size.to_string());
        json.push_str(",\"nodes\":[");

        for (i, &val) in gpu_data.texel_data.iter().enumerate() {
            json.push_str(&val.to_string());
            if i < gpu_data.texel_data.len() - 1 {
                json.push(',');
            }
        }
        json.push_str("]}");
        json
    }

    /// High-performance Binary Serializer for SVO Raymarching:
    /// Compiles SVO nodes directly to little-endian bytes to avoid JSON parsing cost.
    ///
    /// BINARY SCHEMATIC (Byte offset -> payload):
    /// - `0..4`: `root` node index (u32, little-endian)
    /// - `4..8`: `depth` of tree (u32, little-endian)
    /// - `8..12`: `world_size` scale (f32, little-endian)
    /// - `12..16`: `texture_width` (u32, little-endian)
    /// - `16..20`: `texture_height` (u32, little-endian)
    /// - `20..24`: `voxel_scale` (f32, little-endian)
    /// - `24..28`: `chunk_size` (f32, little-endian)
    /// - `28..`: flat `u32` texel array data (4 * width * height * 4 bytes)
    pub fn to_octree_binary(
        grid: &VoxelGrid,
        depth: u32,
        world_size: f32,
        voxel_scale: f32,
        chunk_size: f32,
    ) -> Vec<u8> {
        let builder = BuildOctreeUseCase::new();
        let svo = builder.execute(grid, depth, world_size);
        let gpu_data = OctreeGpuSerializer::serialize_to_gpu_data(&svo);

        let header_size = 28; // 7 values * 4 bytes
        let mut binary = Vec::with_capacity(header_size + gpu_data.texel_data.len() * 4);

        binary.extend_from_slice(&(svo.root as u32).to_le_bytes());
        binary.extend_from_slice(&depth.to_le_bytes());
        binary.extend_from_slice(&world_size.to_le_bytes());
        binary.extend_from_slice(&gpu_data.texture_width.to_le_bytes());
        binary.extend_from_slice(&gpu_data.texture_height.to_le_bytes());
        binary.extend_from_slice(&voxel_scale.to_le_bytes());
        binary.extend_from_slice(&chunk_size.to_le_bytes());

        for &val in &gpu_data.texel_data {
            binary.extend_from_slice(&val.to_le_bytes());
        }

        binary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::voxel_grid::VOXEL_WALL;

    #[test]
    fn test_to_json_format() {
        let mut grid = VoxelGrid::new(2, 2, 2);
        grid.set(0, 0, 0, VOXEL_WALL);
        let json = WebRendererAdapter::to_json(&grid, 2.0);
        assert!(json.starts_with("["));
    }

    #[test]
    fn test_to_octree_binary_format() {
        let mut grid = VoxelGrid::new(2, 2, 2);
        grid.set(0, 0, 0, VOXEL_WALL);
        let binary = WebRendererAdapter::to_octree_binary(&grid, 8, 25.6, 0.1, 20.0);
        
        // Header is 28 bytes
        assert!(binary.len() >= 28);
        // Total bytes must be aligned to 4-byte boundaries
        assert_eq!(binary.len() % 4, 0);
    }
}
