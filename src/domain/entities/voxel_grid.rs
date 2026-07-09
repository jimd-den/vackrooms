/// Represents a dense 3D grid of voxels.
/// This is the foundation for our "smaller voxels" requirement.
pub struct VoxelGrid {
    width: usize,
    height: usize,
    depth: usize,
    data: Vec<u8>,
    /// Scalar light level 0-15 (the max of the RGB channels), kept for
    /// presenters and bakes that only need brightness.
    light_data: Vec<u8>,
    /// Colored flood-fill light, 0-15 per channel.
    light_rgb: Vec<[u8; 3]>,
    face_occlusion: Vec<u8>,
}

pub const VOXEL_AIR: u8 = 0;
pub const VOXEL_WALL: u8 = 1;
pub const VOXEL_FLOOR: u8 = 2;
pub const VOXEL_CEILING: u8 = 3;
pub const VOXEL_LIGHT: u8 = 4;
pub const VOXEL_RED_WALL: u8 = 5;
/// Walkable ground cover (grassland level). Visual-only, like FLOOR.
pub const VOXEL_GRASS: u8 = 6;
/// Walkable shallow water (grassland lakes). Visual-only, like FLOOR.
pub const VOXEL_WATER: u8 = 7;
/// Tree trunk (grassland). Solid: blocks the player like WALL.
pub const VOXEL_TREE: u8 = 8;
pub const VOXEL_RED_LIGHT: u8 = 9;

impl VoxelGrid {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let size = width * height * depth;
        Self {
            width,
            height,
            depth,
            data: vec![VOXEL_AIR; size],
            light_data: vec![0; size],
            light_rgb: vec![[0; 3]; size],
            face_occlusion: vec![0; size],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn depth(&self) -> usize {
        self.depth
    }

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

    /// Sets a neutral (white) light level: all three channels get `level`.
    pub fn set_light(&mut self, x: usize, y: usize, z: usize, level: u8) {
        self.set_light_rgb(x, y, z, [level; 3]);
    }

    pub fn get_light(&self, x: usize, y: usize, z: usize) -> u8 {
        if let Some(idx) = self.index(x, y, z) {
            self.light_data[idx]
        } else {
            0
        }
    }

    /// Sets colored light (0-15 per channel); the scalar level becomes the
    /// max of the channels.
    pub fn set_light_rgb(&mut self, x: usize, y: usize, z: usize, rgb: [u8; 3]) {
        if let Some(idx) = self.index(x, y, z) {
            self.light_rgb[idx] = rgb;
            self.light_data[idx] = rgb[0].max(rgb[1]).max(rgb[2]);
        }
    }

    pub fn get_light_rgb(&self, x: usize, y: usize, z: usize) -> [u8; 3] {
        if let Some(idx) = self.index(x, y, z) {
            self.light_rgb[idx]
        } else {
            [0; 3]
        }
    }

    pub fn set_face_occlusion(&mut self, x: usize, y: usize, z: usize, mask: u8) {
        if let Some(idx) = self.index(x, y, z) {
            self.face_occlusion[idx] = mask;
        }
    }

    pub fn get_face_occlusion(&self, x: usize, y: usize, z: usize) -> u8 {
        if let Some(idx) = self.index(x, y, z) {
            self.face_occlusion[idx]
        } else {
            0
        }
    }
}
