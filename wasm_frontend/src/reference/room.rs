//! Deterministic room fixtures used by renderer regression tests.

use crate::application::ports::Environment;
use vackrooms::domain::entities::voxel_grid::{
    VOXEL_CEILING, VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_WALL, VoxelGrid,
};

const ROOM_EDGE_VOXELS: usize = 8;
const ROOM_CEILING_Y: usize = 6;
const PANEL_X: usize = 4;
const PANEL_Z: usize = 2;
const REFERENCE_VOXEL_SIZE: f32 = 0.5;

/// Test camera position and orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraSpec {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_degrees: f32,
}

/// A platform-free dense scene description. Rendering metadata is deliberately
/// absent: [`crate::reference::scene::build_render_scene`] owns SVO
/// construction and atlas serialization.
pub struct RoomScene {
    voxels: VoxelGrid,
    voxel_size: f32,
    origin: [f32; 3],
}

impl RoomScene {
    /// One closed 4x4x4-world-unit room with a single 0.5-unit fluorescent
    /// ceiling panel. The panel replaces one ceiling voxel and its underside
    /// borders air; the entire floor remains ordinary non-emissive carpet.
    pub fn single_ceiling_fixture() -> Self {
        let mut voxels = VoxelGrid::new(ROOM_EDGE_VOXELS, ROOM_EDGE_VOXELS, ROOM_EDGE_VOXELS);
        fill_floor(&mut voxels);
        fill_boundary_walls(&mut voxels);
        fill_ceiling(&mut voxels);
        voxels.set(PANEL_X, ROOM_CEILING_Y, PANEL_Z, VOXEL_LIGHT);

        Self {
            voxels,
            voxel_size: REFERENCE_VOXEL_SIZE,
            origin: [0.0; 3],
        }
    }

    pub fn voxels(&self) -> &VoxelGrid {
        &self.voxels
    }

    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    pub(crate) fn into_parts(self) -> (VoxelGrid, f32, [f32; 3]) {
        (self.voxels, self.voxel_size, self.origin)
    }
}

fn fill_floor(voxels: &mut VoxelGrid) {
    for z in 0..ROOM_EDGE_VOXELS {
        for x in 0..ROOM_EDGE_VOXELS {
            voxels.set(x, 0, z, VOXEL_FLOOR);
        }
    }
}

fn fill_boundary_walls(voxels: &mut VoxelGrid) {
    for y in 1..=ROOM_CEILING_Y {
        for edge in 0..ROOM_EDGE_VOXELS {
            voxels.set(0, y, edge, VOXEL_WALL);
            voxels.set(ROOM_EDGE_VOXELS - 1, y, edge, VOXEL_WALL);
            voxels.set(edge, y, 0, VOXEL_WALL);
            voxels.set(edge, y, ROOM_EDGE_VOXELS - 1, VOXEL_WALL);
        }
    }
}

fn fill_ceiling(voxels: &mut VoxelGrid) {
    for z in 1..ROOM_EDGE_VOXELS - 1 {
        for x in 1..ROOM_EDGE_VOXELS - 1 {
            voxels.set(x, ROOM_CEILING_Y, z, VOXEL_CEILING);
        }
    }
}

pub trait RoomFixture {
    fn id(&self) -> &'static str;
    fn build(&self) -> RoomScene;
    fn camera(&self) -> CameraSpec;
    fn environment(&self) -> Environment;
}
