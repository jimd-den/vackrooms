//! Converts the core greedy-quad output into the browser renderer's compact
//! indexed mesh payload. Geometry stays chunk-local and fixed-point so GPU
//! uploads are small and camera precision does not degrade far from origin.

use vackrooms::adapters::material_palette::DEFAULT_MATERIAL_PALETTE;
use vackrooms::adapters::voxel_mapper::{FaceDirection, MergedQuad, VoxelMapper};
use vackrooms::domain::entities::voxel_grid::VoxelGrid;

use vackrooms::domain::entities::probe_grid::{
    DEFAULT_PROBE_DEPTH, DEFAULT_PROBE_HEIGHT, DEFAULT_PROBE_WIDTH, ProbeGrid,
};

use crate::application::collision::Aabb;
use crate::application::ports::{
    POSITION_FIXED_SCALE, PackedVertex, RenderArtifactNeeds, SurfaceMeshPayload,
};

/// Builds a seam-safe surface payload from a grid that includes a one-voxel
/// X/Z halo. The mapper consults that halo before emitting boundary faces.
pub fn build_surface_mesh(
    halo_grid: &VoxelGrid,
    voxel_scale: f32,
    lod: u8,
    lateral_padding: usize,
) -> SurfaceMeshPayload {
    build_surface_artifacts(
        halo_grid,
        voxel_scale,
        lod,
        lateral_padding,
        RenderArtifactNeeds::SURFACE,
    )
}

/// Builds only the raster representation selected by the renderer. Greedy
/// quads are shared by both outputs, but indexed vertices and face instances
/// are materialized independently so neither strategy pays for the other.
pub fn build_surface_artifacts(
    halo_grid: &VoxelGrid,
    voxel_scale: f32,
    lod: u8,
    lateral_padding: usize,
    artifacts: RenderArtifactNeeds,
) -> SurfaceMeshPayload {
    let mapper = VoxelMapper::new(voxel_scale, &DEFAULT_MATERIAL_PALETTE);
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
    let probe_grid = ProbeGrid::from_voxel_grid(
        halo_grid,
        DEFAULT_PROBE_WIDTH,
        DEFAULT_PROBE_HEIGHT,
        DEFAULT_PROBE_DEPTH,
    );
    let (pw, ph, pd) = probe_grid.dimensions();
    let mut mesh = SurfaceMeshPayload {
        vertices: Vec::with_capacity(if artifacts.indexed_surface_mesh() {
            quads.len() * 4
        } else {
            0
        }),
        indices: Vec::with_capacity(if artifacts.indexed_surface_mesh() {
            quads.len() * 6
        } else {
            0
        }),
        bounds,
        lod,
        faces: if artifacts.face_splats() {
            crate::adapters::face_instances::build_face_instances(&quads, voxel_scale)
        } else {
            crate::application::ports::FaceInstanceSet::empty()
        },
        voxel_scale,
        light_volume_bytes: probe_grid.as_bytes().to_vec(),
        light_volume_dims: [pw as u32, ph as u32, pd as u32],
    };
    if artifacts.indexed_surface_mesh() {
        for quad in &quads {
            append_quad(&mut mesh, quad, voxel_scale);
        }
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
    let material = quad.material;
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

#[cfg(test)]
mod tests {
    use super::*;
    use vackrooms::domain::entities::voxel_grid::VOXEL_WALL;

    #[test]
    fn isolated_voxel_becomes_indexed_closed_surface() {
        let mut grid = VoxelGrid::new(3, 2, 3);
        grid.set(1, 0, 1, VOXEL_WALL);
        let mesh = build_surface_mesh(&grid, 1.0, 0, 1);
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
        assert!(mesh.faces.instances.is_empty());
    }

    #[test]
    fn splat_artifacts_do_not_materialize_the_indexed_mesh() {
        let mut grid = VoxelGrid::new(4, 2, 4);
        grid.set(1, 0, 1, VOXEL_WALL);
        let artifacts = build_surface_artifacts(&grid, 1.0, 0, 1, RenderArtifactNeeds::SPLAT);

        assert!(artifacts.vertices.is_empty());
        assert!(artifacts.indices.is_empty());
        assert!(!artifacts.faces.instances.is_empty());
    }
}
