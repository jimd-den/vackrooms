use crate::domain::entities::voxel_grid::{VOXEL_AIR, VOXEL_LIGHT, VoxelGrid};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A simple 3D DDA (Digital Differential Analyzer) Raycaster
fn cast_ray(
    grid: &VoxelGrid,
    x: f32,
    y: f32,
    z: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    max_steps: usize,
) -> Option<(usize, usize, usize, u8)> {
    let step_x = if dx > 0.0 { 1 } else { -1 };
    let step_y = if dy > 0.0 { 1 } else { -1 };
    let step_z = if dz > 0.0 { 1 } else { -1 };

    let mut t_max_x = if dx != 0.0 {
        (x.floor() + if dx > 0.0 { 1.0 } else { 0.0 } - x) / dx
    } else {
        f32::INFINITY
    };
    let mut t_max_y = if dy != 0.0 {
        (y.floor() + if dy > 0.0 { 1.0 } else { 0.0 } - y) / dy
    } else {
        f32::INFINITY
    };
    let mut t_max_z = if dz != 0.0 {
        (z.floor() + if dz > 0.0 { 1.0 } else { 0.0 } - z) / dz
    } else {
        f32::INFINITY
    };

    let t_delta_x = if dx != 0.0 {
        (1.0 / dx).abs()
    } else {
        f32::INFINITY
    };
    let t_delta_y = if dy != 0.0 {
        (1.0 / dy).abs()
    } else {
        f32::INFINITY
    };
    let t_delta_z = if dz != 0.0 {
        (1.0 / dz).abs()
    } else {
        f32::INFINITY
    };

    let mut ix = x as isize;
    let mut iy = y as isize;
    let mut iz = z as isize;

    for _ in 0..max_steps {
        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                ix += step_x;
                t_max_x += t_delta_x;
            } else {
                iz += step_z;
                t_max_z += t_delta_z;
            }
        } else {
            if t_max_y < t_max_z {
                iy += step_y;
                t_max_y += t_delta_y;
            } else {
                iz += step_z;
                t_max_z += t_delta_z;
            }
        }

        if ix < 0
            || iy < 0
            || iz < 0
            || ix >= grid.width() as isize
            || iy >= grid.height() as isize
            || iz >= grid.depth() as isize
        {
            break;
        }

        let vx = ix as usize;
        let vy = iy as usize;
        let vz = iz as usize;

        let v = grid.get(vx, vy, vz);
        if v != VOXEL_AIR {
            return Some((vx, vy, vz, v));
        }
    }
    None
}

/// Bakes lighting into the VoxelGrid using Monte Carlo Path Tracing
pub fn bake_path_traced_lighting(grid: &mut VoxelGrid) {
    let w = grid.width();
    let h = grid.height();
    let d = grid.depth();

    let rays_per_voxel = 64;
    let max_bounces = 1; // 0 for direct only, 1 for one bounce
    let max_steps = 30; // Max ray distance

    let mut rng = StdRng::seed_from_u64(42);

    // We only need to bake lighting into AIR blocks that are adjacent to solid blocks,
    // but for simplicity we can bake all AIR blocks, or directly bake solid blocks.
    // Actually, baking solid blocks is better. We evaluate light slightly outside the solid block face.
    // Since our map_voxel_grid just uses `grid.get_light(x, y, z)` for the solid block itself,
    // we can just store the light level in the solid block!

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let v = grid.get(x, y, z);
                if v == VOXEL_AIR || v == VOXEL_LIGHT {
                    continue;
                }

                // It's a solid block. Let's do path tracing from its center slightly offset by normals.
                // To keep it simple, we shoot rays from the center of the block. If it immediately hits itself,
                // we ignore it. Actually, start rays from the adjacent AIR voxels.

                let mut total_light = 0.0;

                // Find an adjacent air voxel to act as the emission point
                let mut px = x as f32 + 0.5;
                let mut py = y as f32 + 0.5;
                let mut pz = z as f32 + 0.5;

                // Shift point slightly towards the air
                if x > 0 && grid.get(x - 1, y, z) == VOXEL_AIR {
                    px -= 0.55;
                } else if x < w - 1 && grid.get(x + 1, y, z) == VOXEL_AIR {
                    px += 0.55;
                } else if y > 0 && grid.get(x, y - 1, z) == VOXEL_AIR {
                    py -= 0.55;
                } else if y < h - 1 && grid.get(x, y + 1, z) == VOXEL_AIR {
                    py += 0.55;
                } else if z > 0 && grid.get(x, y, z - 1) == VOXEL_AIR {
                    pz -= 0.55;
                } else if z < d - 1 && grid.get(x, y, z + 1) == VOXEL_AIR {
                    pz += 0.55;
                } else {
                    continue; // Completely surrounded by solids, no light needed
                }

                for _ in 0..rays_per_voxel {
                    // Random spherical direction
                    let theta = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
                    let phi = (rng.random::<f32>() * 2.0 - 1.0).acos();
                    let dx = phi.sin() * theta.cos();
                    let dy = phi.cos();
                    let dz = phi.sin() * theta.sin();

                    if let Some((_hx, _hy, _hz, hit_type)) =
                        cast_ray(grid, px, py, pz, dx, dy, dz, max_steps)
                    {
                        if hit_type == VOXEL_LIGHT {
                            total_light += 1.0;
                        } else if max_bounces > 0 {
                            // Secondary bounce (simplified)
                            // We don't do a full recursive call to avoid stack/performance issues,
                            // just a small probability addition to simulate ambient bounce.
                            // If we hit a wall, it bounces a bit of light.
                            total_light += 0.05;
                        }
                    }
                }

                // Normalize and map to 0-15 scale for compatibility with voxel_mapper
                let mut light_level = ((total_light / rays_per_voxel as f32) * 15.0 * 2.0) as u8;
                if light_level > 15 {
                    light_level = 15;
                }

                grid.set_light(x, y, z, light_level);
            }
        }
    }
}

/// Bakes directional face occlusion using DDA raycasting.
/// Casts short rays (1-4 voxels) outward from the 6 cardinal faces of each solid block.
/// If a ray hits another solid block within the occlusion distance, that face gets shaded (bit set to 1).
pub fn bake_face_occlusion(grid: &mut VoxelGrid) {
    let w = grid.width();
    let h = grid.height();
    let d = grid.depth();

    let occlusion_dist = 4; // Max steps to cast

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let v = grid.get(x, y, z);
                if v == VOXEL_AIR || v == VOXEL_LIGHT {
                    continue; // We only need occlusion for solid faces
                }

                let mut mask = 0u8;
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;
                let pz = z as f32 + 0.5;

                // +X face
                if x < w - 1 && grid.get(x + 1, y, z) == VOXEL_AIR {
                    if let Some(_) =
                        cast_ray(grid, px + 0.51, py, pz, 1.0, 0.0, 0.0, occlusion_dist)
                    {
                        mask |= 1;
                    }
                }
                // -X face
                if x > 0 && grid.get(x - 1, y, z) == VOXEL_AIR {
                    if let Some(_) =
                        cast_ray(grid, px - 0.51, py, pz, -1.0, 0.0, 0.0, occlusion_dist)
                    {
                        mask |= 2;
                    }
                }
                // +Y face
                if y < h - 1 && grid.get(x, y + 1, z) == VOXEL_AIR {
                    if let Some(_) =
                        cast_ray(grid, px, py + 0.51, pz, 0.0, 1.0, 0.0, occlusion_dist)
                    {
                        mask |= 4;
                    }
                }
                // -Y face
                if y > 0 && grid.get(x, y - 1, z) == VOXEL_AIR {
                    if let Some(_) =
                        cast_ray(grid, px, py - 0.51, pz, 0.0, -1.0, 0.0, occlusion_dist)
                    {
                        mask |= 8;
                    }
                }
                // +Z face
                if z < d - 1 && grid.get(x, y, z + 1) == VOXEL_AIR {
                    if let Some(_) =
                        cast_ray(grid, px, py, pz + 0.51, 0.0, 0.0, 1.0, occlusion_dist)
                    {
                        mask |= 16;
                    }
                }
                // -Z face
                if z > 0 && grid.get(x, y, z - 1) == VOXEL_AIR {
                    if let Some(_) =
                        cast_ray(grid, px, py, pz - 0.51, 0.0, 0.0, -1.0, occlusion_dist)
                    {
                        mask |= 32;
                    }
                }

                grid.set_face_occlusion(x, y, z, mask);
            }
        }
    }
}
