use crate::domain::entities::voxel_grid::{
    VOXEL_AIR, VOXEL_CEILING, VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_RED_WALL, VOXEL_WALL, VOXEL_RED_LIGHT, VoxelGrid,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoxelType {
    Wall,
    Floor,
    Ceiling,
    Light,
    RedWall,
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

#[derive(Debug, Clone, PartialEq)]
pub struct MergedQuad {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
    pub h: f32,
    pub dir: FaceDirection,
    pub v_type: VoxelType,
    pub color: u32,
}

pub struct VoxelMapper {
    pub voxel_scale: f32,
}

impl VoxelMapper {
    pub fn new(voxel_scale: f32) -> Self {
        Self { voxel_scale }
    }

    /// Maps the 3D VoxelGrid into a list of merged quads using Greedy Meshing.
    pub fn map_voxel_grid(&self, grid: &VoxelGrid) -> Vec<MergedQuad> {
        let mut quads = Vec::new();
        let scale = self.voxel_scale;

        let apply_light = |base_color: u32, light_level: u8| -> u32 {
            let normalized = light_level as f32 / 15.0;
            let mult = 0.25 + (normalized * normalized) * 0.75;
            let r = ((base_color >> 16) & 0xFF) as f32 * mult;
            let g = ((base_color >> 8) & 0xFF) as f32 * mult;
            let b = (base_color & 0xFF) as f32 * mult;
            ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
        };

        let get_voxel_type = |v_id: u8| -> VoxelType {
            match v_id {
                VOXEL_WALL => VoxelType::Wall,
                VOXEL_FLOOR => VoxelType::Floor,
                VOXEL_CEILING => VoxelType::Ceiling,
                VOXEL_LIGHT | VOXEL_RED_LIGHT => VoxelType::Light,
                VOXEL_RED_WALL => VoxelType::RedWall,
                _ => VoxelType::Wall,
            }
        };

        let get_voxel_color = |x: usize, y: usize, z: usize| -> u32 {
            let v_id = grid.get(x, y, z);
            let ll = grid.get_light(x, y, z);
            let base_color = match v_id {
                VOXEL_WALL => 0xddcc66,
                VOXEL_FLOOR => 0x998811,
                VOXEL_CEILING => 0xcccccc,
                VOXEL_LIGHT => 0xffffff,
                VOXEL_RED_LIGHT => 0xff4444,
                VOXEL_RED_WALL => 0x880000,
                _ => 0x000000,
            };

            if v_id == VOXEL_LIGHT || v_id == VOXEL_RED_LIGHT {
                base_color
            } else {
                apply_light(base_color, ll)
            }
        };

        let w_dim = grid.width();
        let h_dim = grid.height();
        let d_dim = grid.depth();

        // 1. UP FACES (+Y)
        for y in 0..h_dim {
            let get_face = |x: usize, z: usize| -> Option<(VoxelType, u32)> {
                let v = grid.get(x, y, z);
                if v != VOXEL_AIR && (y == h_dim - 1 || grid.get(x, y + 1, z) == VOXEL_AIR) {
                    Some((get_voxel_type(v), get_voxel_color(x, y, z)))
                } else {
                    None
                }
            };
            let merged = self.greedy_mesh_2d(w_dim, d_dim, &get_face);
            for (u, v, qw, qh, vt, col) in merged {
                quads.push(MergedQuad {
                    x: u as f32 * scale,
                    y: y as f32 * scale,
                    z: v as f32 * scale,
                    w: qw as f32 * scale,
                    h: qh as f32 * scale,
                    dir: FaceDirection::Up,
                    v_type: vt,
                    color: col,
                });
            }
        }

        // 2. DOWN FACES (-Y)
        for y in 0..h_dim {
            let get_face = |x: usize, z: usize| -> Option<(VoxelType, u32)> {
                let v = grid.get(x, y, z);
                if v != VOXEL_AIR && (y == 0 || grid.get(x, y - 1, z) == VOXEL_AIR) {
                    Some((get_voxel_type(v), get_voxel_color(x, y, z)))
                } else {
                    None
                }
            };
            let merged = self.greedy_mesh_2d(w_dim, d_dim, &get_face);
            for (u, v, qw, qh, vt, col) in merged {
                quads.push(MergedQuad {
                    x: u as f32 * scale,
                    y: y as f32 * scale,
                    z: v as f32 * scale,
                    w: qw as f32 * scale,
                    h: qh as f32 * scale,
                    dir: FaceDirection::Down,
                    v_type: vt,
                    color: col,
                });
            }
        }

        // 3. NORTH FACES (-Z)
        for z in 0..d_dim {
            let get_face = |x: usize, y: usize| -> Option<(VoxelType, u32)> {
                let v = grid.get(x, y, z);
                if v != VOXEL_AIR && (z == 0 || grid.get(x, y, z - 1) == VOXEL_AIR) {
                    Some((get_voxel_type(v), get_voxel_color(x, y, z)))
                } else {
                    None
                }
            };
            let merged = self.greedy_mesh_2d(w_dim, h_dim, &get_face);
            for (u, v, qw, qh, vt, col) in merged {
                quads.push(MergedQuad {
                    x: u as f32 * scale,
                    y: v as f32 * scale,
                    z: z as f32 * scale,
                    w: qw as f32 * scale,
                    h: qh as f32 * scale,
                    dir: FaceDirection::North,
                    v_type: vt,
                    color: col,
                });
            }
        }

        // 4. SOUTH FACES (+Z)
        for z in 0..d_dim {
            let get_face = |x: usize, y: usize| -> Option<(VoxelType, u32)> {
                let v = grid.get(x, y, z);
                if v != VOXEL_AIR && (z == d_dim - 1 || grid.get(x, y, z + 1) == VOXEL_AIR) {
                    Some((get_voxel_type(v), get_voxel_color(x, y, z)))
                } else {
                    None
                }
            };
            let merged = self.greedy_mesh_2d(w_dim, h_dim, &get_face);
            for (u, v, qw, qh, vt, col) in merged {
                quads.push(MergedQuad {
                    x: u as f32 * scale,
                    y: v as f32 * scale,
                    z: z as f32 * scale,
                    w: qw as f32 * scale,
                    h: qh as f32 * scale,
                    dir: FaceDirection::South,
                    v_type: vt,
                    color: col,
                });
            }
        }

        // 5. EAST FACES (+X)
        for x in 0..w_dim {
            let get_face = |z: usize, y: usize| -> Option<(VoxelType, u32)> {
                let v = grid.get(x, y, z);
                if v != VOXEL_AIR && (x == w_dim - 1 || grid.get(x + 1, y, z) == VOXEL_AIR) {
                    Some((get_voxel_type(v), get_voxel_color(x, y, z)))
                } else {
                    None
                }
            };
            let merged = self.greedy_mesh_2d(d_dim, h_dim, &get_face);
            for (u, v, qw, qh, vt, col) in merged {
                quads.push(MergedQuad {
                    x: x as f32 * scale,
                    y: v as f32 * scale,
                    z: u as f32 * scale,
                    w: qw as f32 * scale,
                    h: qh as f32 * scale,
                    dir: FaceDirection::East,
                    v_type: vt,
                    color: col,
                });
            }
        }

        // 6. WEST FACES (-X)
        for x in 0..w_dim {
            let get_face = |z: usize, y: usize| -> Option<(VoxelType, u32)> {
                let v = grid.get(x, y, z);
                if v != VOXEL_AIR && (x == 0 || grid.get(x - 1, y, z) == VOXEL_AIR) {
                    Some((get_voxel_type(v), get_voxel_color(x, y, z)))
                } else {
                    None
                }
            };
            let merged = self.greedy_mesh_2d(d_dim, h_dim, &get_face);
            for (u, v, qw, qh, vt, col) in merged {
                quads.push(MergedQuad {
                    x: x as f32 * scale,
                    y: v as f32 * scale,
                    z: u as f32 * scale,
                    w: qw as f32 * scale,
                    h: qh as f32 * scale,
                    dir: FaceDirection::West,
                    v_type: vt,
                    color: col,
                });
            }
        }

        quads
    }

    /// Helper that performs 2D greedy meshing on a slice.
    fn greedy_mesh_2d(
        &self,
        w_slice: usize,
        h_slice: usize,
        get_face: &dyn Fn(usize, usize) -> Option<(VoxelType, u32)>,
    ) -> Vec<(usize, usize, usize, usize, VoxelType, u32)> {
        let mut visited = vec![vec![false; h_slice]; w_slice];
        let mut quads = Vec::new();

        for v in 0..h_slice {
            for u in 0..w_slice {
                if visited[u][v] {
                    continue;
                }

                if let Some((v_type, color)) = get_face(u, v) {
                    // Find max width along u
                    let mut quad_w = 1;
                    while u + quad_w < w_slice {
                        if visited[u + quad_w][v] {
                            break;
                        }
                        if let Some((next_type, next_color)) = get_face(u + quad_w, v) {
                            if next_type == v_type && next_color == color {
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
                            if let Some((next_type, next_color)) = get_face(u + du, v + quad_h) {
                                if next_type != v_type || next_color != color {
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

                    quads.push((u, v, quad_w, quad_h, v_type, color));
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
}
