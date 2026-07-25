//! Converts greedy-merged quads into the splat renderer's compact face
//! instances: one error-selected, axis-aligned surface rectangle each.
//!
//! The instances derive from the *same* [`MergedQuad`] list as the indexed
//! mesh, so the splat renderer's visible boundary and the mesh-based shadow
//! caster can never disagree. Two transformations happen on the way:
//!
//! * **Extent capping** — a merged quad longer than [`MAX_FACE_EXTENT_UNITS`]
//!   on either axis is split into pieces, because the splat path lights each
//!   face with one flat sample; one sample across a 10 u wall would smear.
//! * **Cell bucketing** — instances are sorted by [`FACE_CELL_UNITS`] cell so
//!   the driver can draw or cull contiguous ranges with zero per-frame
//!   rebuild work.
//!
//! Like the rest of the adapters layer this is platform-free and natively
//! unit-tested.

use vackrooms::adapters::voxel_mapper::{FaceDirection, MergedQuad};

use crate::application::ports::{
    FACE_INSTANCE_FLAG_EMISSIVE, FaceCellRange, FaceInstanceSet, POSITION_FIXED_SCALE,
    PackedFaceInstance,
};

/// Longest face edge in world units before a quad is split. Keeps one flat
/// per-face light/shadow sample honest on large merged planes.
pub const MAX_FACE_EXTENT_UNITS: f32 = 2.5;
/// World size of one culling cell (matches the extent cap so a face never
/// spans more than ~two cells).
pub const FACE_CELL_UNITS: f32 = 2.5;

/// Builds the face-instance page for one chunk from its greedy quads.
/// `voxel_scale` is the world size of one voxel cell at this LOD; quad
/// geometry is exact multiples of it.
pub fn build_face_instances(quads: &[MergedQuad], voxel_scale: f32) -> FaceInstanceSet {
    let max_cells = ((MAX_FACE_EXTENT_UNITS / voxel_scale).floor() as u32).max(1);
    let mut instances = Vec::with_capacity(quads.len());
    for quad in quads {
        emit_capped(quad, voxel_scale, max_cells, &mut instances);
    }

    // Sort by culling cell so each cell's instances form one contiguous run.
    let cell_of = |inst: &PackedFaceInstance| -> [u8; 3] {
        let mut cell = [0u8; 3];
        for axis in 0..3 {
            let world = inst.position[axis] as f32 / POSITION_FIXED_SCALE;
            cell[axis] = (world / FACE_CELL_UNITS).floor().clamp(0.0, 255.0) as u8;
        }
        cell
    };
    instances.sort_by_key(|inst| {
        let c = cell_of(inst);
        (c[0], c[1], c[2])
    });

    let mut cells: Vec<FaceCellRange> = Vec::new();
    for (index, inst) in instances.iter().enumerate() {
        let cell = cell_of(inst);
        match cells.last_mut() {
            Some(range) if range.cell == cell => range.count += 1,
            _ => cells.push(FaceCellRange {
                cell,
                offset: index as u32,
                count: 1,
            }),
        }
    }

    FaceInstanceSet {
        instances,
        cells,
        cell_size: FACE_CELL_UNITS,
    }
}

/// Face frame of one quad: the fixed plane coordinate plus the two in-plane
/// axes. U/V axis assignment matches the splat vertex shader (U = X for
/// Y/Z-normal faces, Y for X-normal faces; V = the remaining axis).
struct FaceFrame {
    /// World axis index the face is perpendicular to.
    plane_axis: usize,
    /// Face plane coordinate along `plane_axis`.
    plane: f32,
    u_axis: usize,
    v_axis: usize,
    min_u: f32,
    min_v: f32,
    cells_u: u32,
    cells_v: u32,
}

fn face_frame(quad: &MergedQuad, s: f32) -> FaceFrame {
    let cells = |len: f32| ((len / s).round() as u32).max(1);
    match quad.dir {
        // Horizontal faces: `w` spans X, `h` spans Z (see surface_mesh).
        FaceDirection::Up | FaceDirection::Down => FaceFrame {
            plane_axis: 1,
            plane: if quad.dir == FaceDirection::Up {
                quad.y + s
            } else {
                quad.y
            },
            u_axis: 0,
            v_axis: 2,
            min_u: quad.x,
            min_v: quad.z,
            cells_u: cells(quad.w),
            cells_v: cells(quad.h),
        },
        // Z-normal faces: `w` spans X, `h` spans Y.
        FaceDirection::North | FaceDirection::South => FaceFrame {
            plane_axis: 2,
            plane: if quad.dir == FaceDirection::South {
                quad.z + s
            } else {
                quad.z
            },
            u_axis: 0,
            v_axis: 1,
            min_u: quad.x,
            min_v: quad.y,
            cells_u: cells(quad.w),
            cells_v: cells(quad.h),
        },
        // X-normal faces: `h` spans Y, `w` spans Z.
        FaceDirection::East | FaceDirection::West => FaceFrame {
            plane_axis: 0,
            plane: if quad.dir == FaceDirection::East {
                quad.x + s
            } else {
                quad.x
            },
            u_axis: 1,
            v_axis: 2,
            min_u: quad.y,
            min_v: quad.z,
            cells_u: cells(quad.h),
            cells_v: cells(quad.w),
        },
    }
}

fn emit_capped(quad: &MergedQuad, s: f32, max_cells: u32, out: &mut Vec<PackedFaceInstance>) {
    let frame = face_frame(quad, s);
    let material = quad.material;
    let flags = if vackrooms::domain::entities::voxel_grid::EMISSIVE_MATERIALS.contains(&material) {
        FACE_INSTANCE_FLAG_EMISSIVE
    } else {
        0
    };
    let normal_axis = normal_axis(quad.dir);

    let mut u_off = 0;
    while u_off < frame.cells_u {
        let piece_u = max_cells.min(frame.cells_u - u_off);
        let mut v_off = 0;
        while v_off < frame.cells_v {
            let piece_v = max_cells.min(frame.cells_v - v_off);
            let center_u = frame.min_u + (u_off as f32 + piece_u as f32 * 0.5) * s;
            let center_v = frame.min_v + (v_off as f32 + piece_v as f32 * 0.5) * s;
            let mut position = [0.0f32; 3];
            position[frame.plane_axis] = frame.plane;
            position[frame.u_axis] = center_u;
            position[frame.v_axis] = center_v;
            out.push(PackedFaceInstance {
                position: position.map(pack_position),
                extent_u: piece_u.min(u8::MAX as u32) as u8,
                extent_v: piece_v.min(u8::MAX as u32) as u8,
                normal_axis,
                material,
                baked_light: quad.light,
                ao: quad.ao,
                flags,
                reserved: [0; 3],
            });
            v_off += piece_v;
        }
        u_off += piece_u;
    }
}

fn pack_position(value: f32) -> u16 {
    (value * POSITION_FIXED_SCALE)
        .round()
        .clamp(0.0, u16::MAX as f32) as u16
}

/// Same encoding as `surface_mesh::normal_axis` / the shaders' `normalForAxis`.
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
    use vackrooms::adapters::material_palette::DEFAULT_MATERIAL_PALETTE;
    use vackrooms::adapters::voxel_mapper::VoxelMapper;
    use vackrooms::domain::entities::voxel_grid::{VOXEL_WALL, VoxelGrid};

    fn world(pos: u16) -> f32 {
        pos as f32 / POSITION_FIXED_SCALE
    }

    #[test]
    fn isolated_voxel_yields_six_unit_faces_on_their_planes() {
        let mut grid = VoxelGrid::new(3, 2, 3);
        grid.set(1, 0, 1, VOXEL_WALL);
        let quads = VoxelMapper::new(1.0, &DEFAULT_MATERIAL_PALETTE)
            .map_voxel_grid_with_padding(&grid, 1);
        let set = build_face_instances(&quads, 1.0);

        assert_eq!(set.instances.len(), 6);
        for inst in &set.instances {
            assert_eq!((inst.extent_u, inst.extent_v), (1, 1));
        }
        // The up face sits on the voxel's top plane at its center.
        let up = set
            .instances
            .iter()
            .find(|i| i.normal_axis == 0)
            .expect("up face");
        assert_eq!(world(up.position[1]), 1.0);
        assert_eq!(world(up.position[0]), 0.5);
        assert_eq!(world(up.position[2]), 0.5);
        // The down face sits on y = 0.
        let down = set
            .instances
            .iter()
            .find(|i| i.normal_axis == 1)
            .expect("down face");
        assert_eq!(world(down.position[1]), 0.0);
    }

    #[test]
    fn extent_cap_splits_large_planes_and_conserves_area() {
        // A 30x30 slab with a 1-voxel halo: the mapper merges the 28x28
        // interior top/bottom into single quads, which must split.
        let mut grid = VoxelGrid::new(30, 1, 30);
        for z in 0..30 {
            for x in 0..30 {
                grid.set(x, 0, z, VOXEL_WALL);
            }
        }
        let quads = VoxelMapper::new(1.0, &DEFAULT_MATERIAL_PALETTE)
            .map_voxel_grid_with_padding(&grid, 1);
        let set = build_face_instances(&quads, 1.0);

        let max_cells = (MAX_FACE_EXTENT_UNITS.floor() as u32).max(1);
        let mut area_up = 0u32;
        for inst in &set.instances {
            assert!(inst.extent_u as u32 <= max_cells);
            assert!(inst.extent_v as u32 <= max_cells);
            if inst.normal_axis == 0 {
                area_up += inst.extent_u as u32 * inst.extent_v as u32;
            }
        }
        assert_eq!(area_up, 28 * 28, "capping must conserve covered area");
    }

    #[test]
    fn cell_ranges_partition_instances_exactly() {
        let mut grid = VoxelGrid::new(30, 4, 30);
        for z in 0..30 {
            for x in 0..30 {
                grid.set(x, 0, z, VOXEL_WALL);
                if (x / 7 + z / 5) % 3 == 0 {
                    grid.set(x, 1, z, VOXEL_WALL);
                }
            }
        }
        let quads = VoxelMapper::new(0.5, &DEFAULT_MATERIAL_PALETTE)
            .map_voxel_grid_with_padding(&grid, 1);
        let set = build_face_instances(&quads, 0.5);

        let mut covered = 0u32;
        for range in &set.cells {
            assert_eq!(range.offset, covered, "ranges must be contiguous");
            assert!(range.count > 0);
            for i in range.offset..range.offset + range.count {
                let inst = &set.instances[i as usize];
                for axis in 0..3 {
                    let cell = (world(inst.position[axis]) / set.cell_size).floor() as u8;
                    assert_eq!(cell, range.cell[axis]);
                }
            }
            covered += range.count;
        }
        assert_eq!(covered as usize, set.instances.len());
    }

    #[test]
    fn instances_describe_same_boundary_as_quads() {
        // Every quad corner stays covered: total instance area equals total
        // quad area (in cells) for every face direction.
        let mut grid = VoxelGrid::new(8, 3, 8);
        for z in 2..6 {
            for x in 2..6 {
                grid.set(x, 0, z, VOXEL_WALL);
                grid.set(x, 1, z, VOXEL_WALL);
            }
        }
        let s = 0.4;
        let quads = VoxelMapper::new(s, &DEFAULT_MATERIAL_PALETTE)
            .map_voxel_grid_with_padding(&grid, 1);
        let set = build_face_instances(&quads, s);

        let quad_area: f32 = quads.iter().map(|q| (q.w / s) * (q.h / s)).sum();
        let inst_area: u32 = set
            .instances
            .iter()
            .map(|i| i.extent_u as u32 * i.extent_v as u32)
            .sum();
        assert_eq!(inst_area as f32, quad_area);
    }
}
