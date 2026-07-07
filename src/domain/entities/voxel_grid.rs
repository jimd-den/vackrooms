/// Represents a dense 3D grid of voxels.
/// This is the foundation for our "smaller voxels" requirement.
pub struct VoxelGrid {
    width: usize,
    height: usize,
    depth: usize,
    data: Vec<u8>,
    light_data: Vec<u8>,
}

pub const VOXEL_AIR: u8 = 0;
pub const VOXEL_WALL: u8 = 1;
pub const VOXEL_FLOOR: u8 = 2;
pub const VOXEL_CEILING: u8 = 3;
pub const VOXEL_LIGHT: u8 = 4;
pub const VOXEL_RED_WALL: u8 = 5;

impl VoxelGrid {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let size = width * height * depth;
        Self {
            width,
            height,
            depth,
            data: vec![VOXEL_AIR; size],
            light_data: vec![0; size],
        }
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn depth(&self) -> usize { self.depth }

    fn index(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x < self.width && y < self.height && z < self.depth {
            Some(y * (self.width * self.depth) + z * self.width + x)
        } else {
            None
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel_type: u8) {
        if let Some(idx) = self.index(x, y, z) {
            self.data[idx] = voxel_type;
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        if let Some(idx) = self.index(x, y, z) {
            self.data[idx]
        } else {
            VOXEL_AIR
        }
    }

    pub fn set_light(&mut self, x: usize, y: usize, z: usize, level: u8) {
        if let Some(idx) = self.index(x, y, z) {
            self.light_data[idx] = level;
        }
    }

    pub fn get_light(&self, x: usize, y: usize, z: usize) -> u8 {
        if let Some(idx) = self.index(x, y, z) {
            self.light_data[idx]
        } else {
            0
        }
    }
}
