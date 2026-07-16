use crate::domain::entities::voxel_grid::{
    FACE_OCCLUDED_NEGATIVE_X, FACE_OCCLUDED_NEGATIVE_Y, FACE_OCCLUDED_NEGATIVE_Z,
    FACE_OCCLUDED_POSITIVE_X, FACE_OCCLUDED_POSITIVE_Y, FACE_OCCLUDED_POSITIVE_Z, MATERIAL_COLORS,
    VOXEL_AIR, VOXEL_ALMOND_WATER, VOXEL_CEILING, VOXEL_CONCRETE_FLOOR, VOXEL_CONCRETE_WALL,
    VOXEL_CRATE, VOXEL_DAMAGED_WALL, VOXEL_DEEP_CARPET, VOXEL_DRY_CARPET, VOXEL_FLOOR, VOXEL_FLUID,
    VOXEL_GLIMMER, VOXEL_GRASS, VOXEL_LIGHT, VOXEL_METAL_DOOR, VOXEL_PALE_WALL, VOXEL_PIPE,
    VOXEL_RED_LIGHT, VOXEL_RED_WALL, VOXEL_STICKY_CARPET, VOXEL_TILE_FLOOR, VOXEL_TREE, VOXEL_WALL,
    VOXEL_WATER, VoxelGrid,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelType {
    Wall,
    Floor,
    Ceiling,
    Light,
    RedWall,
    Grass,
    Water,
    Tree,
    RedLight,
    PaleWall,
    DamagedWall,
    DryCarpet,
    DeepCarpet,
    StickyCarpet,
    Fluid,
    Glimmer,
    ConcreteWall,
    TileFloor,
    ConcreteFloor,
    Crate,
    Pipe,
    MetalDoor,
    AlmondWater,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceDirection {
    Up,
    Down,
    North,
    South,
    East,
    West,
}

impl FaceDirection {
    fn occlusion_bit(self) -> u8 {
        match self {
            FaceDirection::East => FACE_OCCLUDED_POSITIVE_X,
            FaceDirection::West => FACE_OCCLUDED_NEGATIVE_X,
            FaceDirection::Up => FACE_OCCLUDED_POSITIVE_Y,
            FaceDirection::Down => FACE_OCCLUDED_NEGATIVE_Y,
            FaceDirection::South => FACE_OCCLUDED_POSITIVE_Z,
            FaceDirection::North => FACE_OCCLUDED_NEGATIVE_Z,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MergedQuad {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
    pub h: f32,
    pub dir: FaceDirection,
    pub v_type: VoxelType,
    /// Base material color. Lighting stays separate so renderers can shade
    /// consistently instead of baking a different color per presentation.
    pub color: u32,
    /// Scalar baked voxel light, 0--15.
    pub light: u8,
    /// Directional face occlusion bit for this exposed face (0 or 1).
    pub ao: u8,
}

/// Occupancy lookup used by the surface extractor. Coordinates are relative
/// to the meshed chunk; implementations may resolve a one-voxel halo from
/// adjacent chunks or a deterministically generated border. Missing lateral
/// neighbors must be conservative (solid), never treated as air.
pub trait VoxelNeighborhood {
    fn voxel(&self, x: i32, y: i32, z: i32) -> u8;
}

struct GridNeighborhood<'a> {
    grid: &'a VoxelGrid,
    origin: [usize; 3],
    conservative_lateral_border: bool,
}

impl VoxelNeighborhood for GridNeighborhood<'_> {
    fn voxel(&self, x: i32, y: i32, z: i32) -> u8 {
        let gx = self.origin[0] as i32 + x;
        let gy = self.origin[1] as i32 + y;
        let gz = self.origin[2] as i32 + z;
        if gy < 0 || gy >= self.grid.height() as i32 {
            return VOXEL_AIR;
        }
        if gx < 0 || gx >= self.grid.width() as i32 || gz < 0 || gz >= self.grid.depth() as i32 {
            return if self.conservative_lateral_border {
                VOXEL_WALL
            } else {
                VOXEL_AIR
            };
        }
        self.grid.get(gx as usize, gy as usize, gz as usize)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FaceKey {
    v_type: VoxelType,
    color: u32,
    light: u8,
    ao: u8,
}

pub struct VoxelMapper {
    pub voxel_scale: f32,
}

impl VoxelMapper {
    pub fn new(voxel_scale: f32) -> Self {
        Self { voxel_scale }
    }

    /// Maps the entire grid into merged exposed quads. This compatibility
    /// helper treats out-of-grid X/Z as air; streamed rendering should use
    /// [`Self::map_voxel_grid_with_padding`] instead.
    pub fn map_voxel_grid(&self, grid: &VoxelGrid) -> Vec<MergedQuad> {
        let neighborhood = GridNeighborhood {
            grid,
            origin: [0, 0, 0],
            conservative_lateral_border: false,
        };
        self.map_voxel_region(
            grid,
            [0, 0, 0],
            [grid.width(), grid.height(), grid.depth()],
            &neighborhood,
        )
    }

    /// Greedily meshes the interior of a grid carrying a one-voxel X/Z halo.
    /// Faces across a chunk edge are emitted only when the halo says the
    /// adjacent voxel is air, eliminating duplicate boundary faces and the
    /// pop that comes from assuming an unavailable neighbor is air.
    pub fn map_voxel_grid_with_padding(
        &self,
        grid: &VoxelGrid,
        lateral_padding: usize,
    ) -> Vec<MergedQuad> {
        assert!(lateral_padding > 0, "surface meshing requires a voxel halo");
        assert!(grid.width() > lateral_padding * 2 && grid.depth() > lateral_padding * 2);
        let neighborhood = GridNeighborhood {
            grid,
            origin: [lateral_padding, 0, lateral_padding],
            conservative_lateral_border: true,
        };
        self.map_voxel_region(
            grid,
            [lateral_padding, 0, lateral_padding],
            [
                grid.width() - lateral_padding * 2,
                grid.height(),
                grid.depth() - lateral_padding * 2,
            ],
            &neighborhood,
        )
    }

    /// Surface extraction against a supplied occupancy neighborhood. The
    /// dense `grid` supplies material/light/AO attributes for the interior;
    /// `neighborhood` decides whether each candidate face is exposed.
    pub fn map_voxel_region(
        &self,
        grid: &VoxelGrid,
        interior_origin: [usize; 3],
        dimensions: [usize; 3],
        neighborhood: &dyn VoxelNeighborhood,
    ) -> Vec<MergedQuad> {
        let mut quads = Vec::new();
        let scale = self.voxel_scale;
        let [w_dim, h_dim, d_dim] = dimensions;

        let attrs = |x: usize, y: usize, z: usize, v: u8, dir: FaceDirection| {
            self.face_key(
                grid,
                interior_origin[0] + x,
                interior_origin[1] + y,
                interior_origin[2] + z,
                v,
                dir,
            )
        };
        let mut append = |x: f32, y: f32, z: f32, w: f32, h: f32, dir, key: FaceKey| {
            quads.push(MergedQuad {
                x,
                y,
                z,
                w,
                h,
                dir,
                v_type: key.v_type,
                color: key.color,
                light: key.light,
                ao: key.ao,
            });
        };

        // +Y / -Y faces: 2D slices in X/Z.
        for y in 0..h_dim {
            for (dir, dy) in [(FaceDirection::Up, 1), (FaceDirection::Down, -1)] {
                let get_face = |x: usize, z: usize| -> Option<FaceKey> {
                    let v = neighborhood.voxel(x as i32, y as i32, z as i32);
                    (v != VOXEL_AIR
                        && neighborhood.voxel(x as i32, y as i32 + dy, z as i32) == VOXEL_AIR)
                        .then(|| attrs(x, y, z, v, dir))
                };
                for (u, v, qw, qh, key) in self.greedy_mesh_2d(w_dim, d_dim, &get_face) {
                    append(
                        u as f32 * scale,
                        y as f32 * scale,
                        v as f32 * scale,
                        qw as f32 * scale,
                        qh as f32 * scale,
                        dir,
                        key,
                    );
                }
            }
        }

        // -Z / +Z faces: 2D slices in X/Y.
        for z in 0..d_dim {
            for (dir, dz) in [(FaceDirection::North, -1), (FaceDirection::South, 1)] {
                let get_face = |x: usize, y: usize| -> Option<FaceKey> {
                    let v = neighborhood.voxel(x as i32, y as i32, z as i32);
                    (v != VOXEL_AIR
                        && neighborhood.voxel(x as i32, y as i32, z as i32 + dz) == VOXEL_AIR)
                        .then(|| attrs(x, y, z, v, dir))
                };
                for (u, v, qw, qh, key) in self.greedy_mesh_2d(w_dim, h_dim, &get_face) {
                    append(
                        u as f32 * scale,
                        v as f32 * scale,
                        z as f32 * scale,
                        qw as f32 * scale,
                        qh as f32 * scale,
                        dir,
                        key,
                    );
                }
            }
        }

        // +X / -X faces: 2D slices in Z/Y.
        for x in 0..w_dim {
            for (dir, dx) in [(FaceDirection::East, 1), (FaceDirection::West, -1)] {
                let get_face = |z: usize, y: usize| -> Option<FaceKey> {
                    let v = neighborhood.voxel(x as i32, y as i32, z as i32);
                    (v != VOXEL_AIR
                        && neighborhood.voxel(x as i32 + dx, y as i32, z as i32) == VOXEL_AIR)
                        .then(|| attrs(x, y, z, v, dir))
                };
                for (u, v, qw, qh, key) in self.greedy_mesh_2d(d_dim, h_dim, &get_face) {
                    append(
                        x as f32 * scale,
                        v as f32 * scale,
                        u as f32 * scale,
                        qw as f32 * scale,
                        qh as f32 * scale,
                        dir,
                        key,
                    );
                }
            }
        }

        quads
    }

    fn face_key(
        &self,
        grid: &VoxelGrid,
        x: usize,
        y: usize,
        z: usize,
        voxel: u8,
        dir: FaceDirection,
    ) -> FaceKey {
        let ao = u8::from(grid.get_face_occlusion(x, y, z) & dir.occlusion_bit() != 0);
        FaceKey {
            v_type: Self::voxel_type(voxel),
            color: Self::material_color(voxel),
            light: grid.get_light(x, y, z),
            ao,
        }
    }

    fn voxel_type(voxel: u8) -> VoxelType {
        match voxel {
            VOXEL_WALL => VoxelType::Wall,
            VOXEL_FLOOR => VoxelType::Floor,
            VOXEL_CEILING => VoxelType::Ceiling,
            VOXEL_LIGHT => VoxelType::Light,
            VOXEL_RED_LIGHT => VoxelType::RedLight,
            VOXEL_RED_WALL => VoxelType::RedWall,
            VOXEL_GRASS => VoxelType::Grass,
            VOXEL_WATER => VoxelType::Water,
            VOXEL_TREE => VoxelType::Tree,
            VOXEL_PALE_WALL => VoxelType::PaleWall,
            VOXEL_DAMAGED_WALL => VoxelType::DamagedWall,
            VOXEL_DRY_CARPET => VoxelType::DryCarpet,
            VOXEL_DEEP_CARPET => VoxelType::DeepCarpet,
            VOXEL_STICKY_CARPET => VoxelType::StickyCarpet,
            VOXEL_FLUID => VoxelType::Fluid,
            VOXEL_GLIMMER => VoxelType::Glimmer,
            VOXEL_CONCRETE_WALL => VoxelType::ConcreteWall,
            VOXEL_TILE_FLOOR => VoxelType::TileFloor,
            VOXEL_CONCRETE_FLOOR => VoxelType::ConcreteFloor,
            VOXEL_CRATE => VoxelType::Crate,
            VOXEL_PIPE => VoxelType::Pipe,
            VOXEL_METAL_DOOR => VoxelType::MetalDoor,
            VOXEL_ALMOND_WATER => VoxelType::AlmondWater,
            _ => VoxelType::Wall,
        }
    }

    fn material_color(voxel: u8) -> u32 {
        // Single authority: the palette table beside the material ids.
        MATERIAL_COLORS
            .get(voxel as usize)
            .copied()
            .unwrap_or(0x000000)
    }

    /// Helper that performs 2D greedy meshing on a slice.
    fn greedy_mesh_2d(
        &self,
        w_slice: usize,
        h_slice: usize,
        get_face: &dyn Fn(usize, usize) -> Option<FaceKey>,
    ) -> Vec<(usize, usize, usize, usize, FaceKey)> {
        let mut visited = vec![vec![false; h_slice]; w_slice];
        let mut quads = Vec::new();

        for v in 0..h_slice {
            for u in 0..w_slice {
                if visited[u][v] {
                    continue;
                }

                if let Some(key) = get_face(u, v) {
                    // Find max width along u
                    let mut quad_w = 1;
                    while u + quad_w < w_slice {
                        if visited[u + quad_w][v] {
                            break;
                        }
                        if let Some(next) = get_face(u + quad_w, v) {
                            if next == key {
                                quad_w += 1;
                                continue;
                            }
                        }
                        break;
                    }

                    // Find max height along v
                    let mut quad_h = 1;
                    'expand_h: while v + quad_h < h_slice {
                        for du in 0..quad_w {
                            if visited[u + du][v + quad_h] {
                                break 'expand_h;
                            }
                            if let Some(next) = get_face(u + du, v + quad_h) {
                                if next != key {
                                    break 'expand_h;
                                }
                            } else {
                                break 'expand_h;
                            }
                        }
                        quad_h += 1;
                    }

                    // Mark matched area as visited
                    for dv in 0..quad_h {
                        for du in 0..quad_w {
                            visited[u + du][v + dv] = true;
                        }
                    }

                    quads.push((u, v, quad_w, quad_h, key));
                }
            }
        }
        quads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_mapping() {
        let mut grid = VoxelGrid::new(2, 2, 2);
        grid.set(0, 0, 0, VOXEL_WALL);

        let mapper = VoxelMapper::new(1.0);
        let quads = mapper.map_voxel_grid(&grid);

        // Single isolated wall block should have 6 faces merged as 6 quads of size 1x1
        assert_eq!(quads.len(), 6);
        for q in quads {
            assert_eq!(q.w, 1.0);
            assert_eq!(q.h, 1.0);
        }
    }

    #[test]
    fn test_greedy_merging() {
        let mut grid = VoxelGrid::new(4, 1, 1);
        grid.set(0, 0, 0, VOXEL_WALL);
        grid.set(1, 0, 0, VOXEL_WALL);
        grid.set(2, 0, 0, VOXEL_WALL);
        grid.set(3, 0, 0, VOXEL_WALL);

        let mapper = VoxelMapper::new(1.0);
        let quads = mapper.map_voxel_grid(&grid);

        // The Up faces of these 4 aligned blocks should merge into a single 4x1 quad!
        let up_quads: Vec<&MergedQuad> = quads
            .iter()
            .filter(|q| q.dir == FaceDirection::Up)
            .collect();
        assert_eq!(up_quads.len(), 1);
        assert_eq!(up_quads[0].w, 4.0);
        assert_eq!(up_quads[0].h, 1.0);
    }

    #[test]
    fn padded_mesher_does_not_emit_face_against_solid_halo() {
        // X/Z padding is one cell. The rightmost interior voxel touches a
        // solid halo voxel, so its +X face must not be generated.
        let mut grid = VoxelGrid::new(4, 1, 3);
        grid.set(2, 0, 1, VOXEL_WALL); // interior x = 1
        grid.set(3, 0, 1, VOXEL_WALL); // +X halo

        let mapper = VoxelMapper::new(1.0);
        let quads = mapper.map_voxel_grid_with_padding(&grid, 1);
        assert!(
            !quads.iter().any(|q| q.dir == FaceDirection::East),
            "solid neighbor halo must suppress the boundary face"
        );
    }

    #[test]
    fn greedy_merge_respects_baked_light_and_ao() {
        let mut grid = VoxelGrid::new(2, 1, 1);
        grid.set(0, 0, 0, VOXEL_WALL);
        grid.set(1, 0, 0, VOXEL_WALL);
        grid.set_light(0, 0, 0, 5);
        grid.set_light(1, 0, 0, 6);

        let mapper = VoxelMapper::new(1.0);
        let up: Vec<_> = mapper
            .map_voxel_grid(&grid)
            .into_iter()
            .filter(|q| q.dir == FaceDirection::Up)
            .collect();
        assert_eq!(up.len(), 2, "different baked light must split a quad");
    }
}
