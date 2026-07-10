//! Converts the core greedy-quad output into the browser renderer's compact
//! indexed mesh payload. Geometry stays chunk-local and fixed-point so WebGL
//! uploads are small and camera precision does not degrade far from origin.

use vackrooms::adapters::voxel_mapper::{FaceDirection, MergedQuad, VoxelMapper, VoxelType};
use vackrooms::domain::entities::voxel_grid::{
    VOXEL_CEILING, VOXEL_FLOOR, VOXEL_GRASS, VOXEL_LIGHT, VOXEL_RED_LIGHT, VOXEL_RED_WALL,
    VOXEL_TREE, VOXEL_WALL, VOXEL_WATER, VoxelGrid,
};

use crate::application::collision::Aabb;
use crate::application::ports::{POSITION_FIXED_SCALE, PackedVertex, SurfaceMeshPayload};

/// Builds a seam-safe surface payload from a grid that includes a one-voxel
/// X/Z halo. The mapper consults that halo before emitting boundary faces.
pub fn build_surface_mesh(
    halo_grid: &VoxelGrid,
    voxel_scale: f32,
    lod: u8,
    lateral_padding: usize,
) -> SurfaceMeshPayload {
    let mapper = VoxelMapper::new(voxel_scale);
    let quads = mapper.map_voxel_grid_with_padding(halo_grid, lateral_padding);
    let width = halo_grid.width() - lateral_padding * 2;
    let depth = halo_grid.depth() - lateral_padding * 2;
    let bounds = Aabb::new(
        [0.0, 0.0, 0.0],
        [
            width as f32 * voxel_scale,
            halo_grid.height() as f32 * voxel_scale,
            depth as f32 * voxel_scale,
        ],
    );
    let mut lights = Vec::new();
    for rl in &halo_grid.runtime_lights {
        lights.push(crate::application::ports::LightSource {
            id: ((rl.world_pos[0].to_bits() as u64) << 32) | rl.world_pos[2].to_bits() as u64,
            position: rl.world_pos,
            half_size: rl.half_size,
            color: rl.rgb,
            radius: rl.range,
            intensity: rl.intensity,
            flicker_mode: 0,
            enabled: rl.enabled,
        });
    }

    let mut mesh = SurfaceMeshPayload {
        vertices: Vec::with_capacity(quads.len() * 4),
        indices: Vec::with_capacity(quads.len() * 6),
        bounds,
        lod,
        light_volume: Vec::with_capacity(width * halo_grid.height() * depth * 3),
        light_volume_size: [width as u32, halo_grid.height() as u32, depth as u32],
        lights,
        faces: crate::adapters::face_instances::build_face_instances(&quads, voxel_scale),
        voxel_scale,
    };
    for z in 0..depth {
        for y in 0..halo_grid.height() {
            for x in 0..width {
                let [r, g, b] = halo_grid.get_light_rgb(x + lateral_padding, y, z + lateral_padding);
                // The grid light is 0-15, WebGL expects 0-255 for gl.UNSIGNED_BYTE RGB textures.
                mesh.light_volume.push(r * 17);
                mesh.light_volume.push(g * 17);
                mesh.light_volume.push(b * 17);
            }
        }
    }
    for quad in &quads {
        append_quad(&mut mesh, quad, voxel_scale);
    }
    mesh
}

fn append_quad(mesh: &mut SurfaceMeshPayload, quad: &MergedQuad, voxel_scale: f32) {
    let x0 = quad.x;
    let x1 = quad.x + quad.w;
    let y0 = quad.y;
    let y1 = quad.y + quad.h;
    let z0 = quad.z;
    let z1_horizontal = quad.z + quad.h;
    let z1_x_face = quad.z + quad.w;

    // `w` and `h` span different local axes per face. Move the positive
    // faces to the far voxel plane so the indexed mesh is a closed shell.
    let points = match quad.dir {
        FaceDirection::Up => {
            let y = quad.y + voxel_scale;
            [
                [x0, y, z0],
                [x0, y, z1_horizontal],
                [x1, y, z1_horizontal],
                [x1, y, z0],
            ]
        }
        FaceDirection::Down => [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z1_horizontal],
            [x0, y0, z1_horizontal],
        ],
        FaceDirection::North => [
            [x0, y0, quad.z],
            [x0, y1, quad.z],
            [x1, y1, quad.z],
            [x1, y0, quad.z],
        ],
        FaceDirection::South => {
            let z = quad.z + voxel_scale;
            [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
        }
        FaceDirection::East => {
            let x = quad.x + voxel_scale;
            [
                [x, y0, z0],
                [x, y1, z0],
                [x, y1, z1_x_face],
                [x, y0, z1_x_face],
            ]
        }
        FaceDirection::West => [
            [quad.x, y0, z0],
            [quad.x, y0, z1_x_face],
            [quad.x, y1, z1_x_face],
            [quad.x, y1, z0],
        ],
    };

    let base = mesh.vertices.len() as u32;
    let normal_axis = normal_axis(quad.dir);
    let material = material_id(quad.v_type);
    for position in points {
        mesh.vertices.push(PackedVertex {
            position: position.map(pack_position),
            normal_axis,
            material,
            static_indirect: quad.light,
            ao: quad.ao,
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn pack_position(value: f32) -> u16 {
    (value * POSITION_FIXED_SCALE)
        .round()
        .clamp(0.0, u16::MAX as f32) as u16
}

fn normal_axis(dir: FaceDirection) -> u8 {
    match dir {
        FaceDirection::Up => 0,
        FaceDirection::Down => 1,
        FaceDirection::North => 2,
        FaceDirection::South => 3,
        FaceDirection::East => 4,
        FaceDirection::West => 5,
    }
}

pub(crate) fn material_id(v_type: VoxelType) -> u8 {
    match v_type {
        VoxelType::Wall => VOXEL_WALL,
        VoxelType::Floor => VOXEL_FLOOR,
        VoxelType::Ceiling => VOXEL_CEILING,
        VoxelType::Light => VOXEL_LIGHT,
        VoxelType::RedWall => VOXEL_RED_WALL,
        VoxelType::Grass => VOXEL_GRASS,
        VoxelType::Water => VOXEL_WATER,
        VoxelType::Tree => VOXEL_TREE,
        VoxelType::RedLight => VOXEL_RED_LIGHT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_voxel_becomes_indexed_closed_surface() {
        let mut grid = VoxelGrid::new(3, 2, 3);
        grid.set(1, 0, 1, VOXEL_WALL);
        let mesh = build_surface_mesh(&grid, 1.0, 0, 1);
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
    }
}
