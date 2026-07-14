//! The legacy "office blueprint" generator (`?level=1`), preserved intact.
//!
//! This is the pre-planning era of the engine: a chunk-local Growing Tree
//! maze with BSP-ish rooms, microbiome zones, and flyweight room stamps. It
//! is kept for the `/legacy` client and as a reference for what the planned
//! Level 0 replaced. New work belongs in the planning pipeline
//! (`world_topology` -> `region_plan` -> `backrooms_level`); nothing in this
//! module is consulted by Level 0 or the grassland level.

use crate::domain::entities::grid::Grid;
use crate::domain::entities::voxel_grid::{
    VOXEL_AIR, VOXEL_CEILING, VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_RED_WALL, VOXEL_WALL, VoxelGrid,
};
use crate::domain::use_cases::generate_maze::{GrowingTreeGenerator, MazeGenerator};
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::{NoiseProvider, TelemetryPort};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Represents the internal bounding box of a Room.
#[derive(Debug, Clone)]
struct Room {
    name: &'static str,
    x0: usize,
    x1: usize,
    z0: usize,
    z1: usize,
}

/// Flyweight Room Stamps
#[derive(Debug, Clone)]
struct RoomStamp {
    width: usize,
    depth: usize,
    data: Vec<u8>,
}

impl RoomStamp {
    fn cubicles() -> Self {
        let width = 20;
        let depth = 20;
        let mut data = vec![0; width * depth];
        for z in 2..8 {
            data[z * width + 5] = 1;
        }
        for x in 2..8 {
            data[5 * width + x] = 1;
        }
        for z in 12..18 {
            data[z * width + 15] = 1;
        }
        for x in 12..18 {
            data[15 * width + x] = 1;
        }
        Self { width, depth, data }
    }

    fn showroom() -> Self {
        let width = 16;
        let depth = 16;
        let mut data = vec![0; width * depth];
        for z in 2..6 {
            for x in 2..6 {
                data[z * width + x] = 2;
            }
            for x in 10..14 {
                data[z * width + x] = 2;
            }
        }
        for z in 10..14 {
            for x in 2..6 {
                data[z * width + x] = 2;
            }
            for x in 10..14 {
                data[z * width + x] = 2;
            }
        }
        Self { width, depth, data }
    }

    fn waiting() -> Self {
        let width = 20;
        let depth = 20;
        let mut data = vec![0; width * depth];
        for x in 4..16 {
            data[6 * width + x] = 3;
            data[14 * width + x] = 3;
        }
        Self { width, depth, data }
    }

    fn storage() -> Self {
        let width = 12;
        let depth = 16;
        let mut data = vec![0; width * depth];
        for z in 2..14 {
            data[z * width + 2] = 1;
            data[z * width + 9] = 1;
        }
        Self { width, depth, data }
    }

    fn lounge() -> Self {
        let width = 18;
        let depth = 18;
        let mut data = vec![0; width * depth];
        for z in 6..12 {
            for x in 6..12 {
                if z == 6 || z == 11 || x == 6 || x == 11 {
                    data[z * width + x] = 3;
                }
            }
        }
        Self { width, depth, data }
    }

    fn server_room() -> Self {
        let width = 14;
        let depth = 20;
        let mut data = vec![0; width * depth];
        for x in [3, 7, 10].iter() {
            for z in 2..18 {
                if z % 4 != 0 {
                    data[z * width + x] = 2;
                }
            }
        }
        Self { width, depth, data }
    }

    fn cafeteria() -> Self {
        let width = 24;
        let depth = 24;
        let mut data = vec![0; width * depth];
        for z in [4, 10, 16, 20].iter() {
            for x in [4, 10, 16, 20].iter() {
                data[z * width + x] = 1;
                data[*z * width + x + 1] = 1;
                data[(z + 1) * width + x] = 1;
                data[(z + 1) * width + x + 1] = 1;
            }
        }
        Self { width, depth, data }
    }

    fn maintenance() -> Self {
        let width = 10;
        let depth = 10;
        let mut data = vec![0; width * depth];
        data[2 * width + 2] = 2;
        data[2 * width + 7] = 2;
        data[7 * width + 2] = 2;
        data[7 * width + 7] = 2;
        data[4 * width + 4] = 3;
        data[4 * width + 5] = 3;
        data[5 * width + 4] = 3;
        data[5 * width + 5] = 3;
        Self { width, depth, data }
    }

    fn stairway() -> Self {
        let width = 16;
        let depth = 16;
        let mut data = vec![0; width * depth];
        for z in 0..16 {
            for x in 0..16 {
                if x > 9 {
                    data[z * width + x] = 4; // raised
                } else if x == 9 {
                    data[z * width + x] = 5; // step 2
                } else if x == 8 {
                    data[z * width + x] = 6; // step 1
                }
            }
        }
        Self { width, depth, data }
    }
}

/// Generates one legacy blueprint chunk. Pure in `(chunk_pos, seed, config)`
/// given a deterministic noise provider; telemetry is observation only.
pub(crate) fn generate_legacy_blueprint(
    noise_provider: &dyn NoiseProvider,
    telemetry: &dyn TelemetryPort,
    chunk_pos: Position,
    seed: u32,
    config: GeneratorConfig,
) -> VoxelGrid {
        let start_micros = telemetry.now_micros();

        let width = (config.chunk_size / config.voxel_scale).round() as usize;
        let depth = (config.chunk_size / config.voxel_scale).round() as usize;

        // Cell voxel borders derive from world space (cells are 5.0 units) so
        // every LOD of a chunk puts its walls on the same world planes.
        // Truncating a fixed voxels-per-cell instead compresses the maze at
        // scales where 5.0/voxel_scale is fractional.
        let cell_border = |c: usize| (c as f32 * 5.0 / config.voxel_scale).round() as usize;
        let cell_w = ((config.chunk_size / 5.0).round() as usize).max(1);
        let cell_d = cell_w;

        let mut abstract_grid = Grid::new(cell_w, cell_d);

        // --- ZONE ASSIGNMENT PASS ---
        let mut max_chunk_height_units = 3.0_f32;
        for cz in 0..cell_d {
            for cx in 0..cell_w {
                let wx = chunk_pos.x * config.chunk_size + (cx as f32) * 5.0;
                let wz = chunk_pos.z * config.chunk_size + (cz as f32) * 5.0;

                // Low frequency noise for zone clustering
                let n = noise_provider.evaluate_2d(
                    seed ^ 0x2b8f_a43c,
                    crate::entities::models::Position::new(wx * 0.05, wz * 0.05),
                ); // n is in [-1, 1]

                use crate::domain::entities::cell::MicrobiomeZone;
                let mut zone = MicrobiomeZone::Standard;

                if n < -0.7 {
                    zone = MicrobiomeZone::Blackout;
                } else if n < -0.4 {
                    zone = MicrobiomeZone::Holes;
                } else if n > 0.85 - (config.tuning.atria as f32 * 0.1) {
                    zone = MicrobiomeZone::Atrium;
                } else if n > 0.6 {
                    zone = MicrobiomeZone::PillarField;
                } else if n > 0.4 {
                    zone = MicrobiomeZone::Arch;
                } else if n > 0.2 && n < 0.25 {
                    zone = MicrobiomeZone::RedRoom;
                }

                if let Some(cell) = abstract_grid.get_mut(cx, cz) {
                    cell.zone = zone;
                }

                let cell_h = match zone {
                    MicrobiomeZone::Atrium => 12.0,  // 4.0 * 3.0x
                    MicrobiomeZone::Blackout => 3.2, // 4.0 * 0.8x
                    _ => 4.0,
                };
                if cell_h > max_chunk_height_units {
                    max_chunk_height_units = cell_h;
                }
            }
        }

        let height = (max_chunk_height_units / config.voxel_scale).ceil() as usize + 2; // +2 for floor and ceiling bounds

        telemetry.log(&format!(
            "[INFO] Entering GenerateChunkArchitectureUseCase::execute. Args: chunk_pos={:?}, seed={}, chunk_size={} (Voxel dimensions: {}x{}x{})",
            chunk_pos, seed, config.chunk_size, width, height, depth
        ));

        let mut grid = VoxelGrid::new(width, height, depth);

        // STARTING HUB OVERRIDE: Chunk (0,0) is a massive open plaza that feeds seamlessly into the maze
        if chunk_pos.x.abs() < 1.0 && chunk_pos.z.abs() < 1.0 {
            for z in 0..depth {
                for x in 0..width {
                    grid.set(x, 0, z, VOXEL_FLOOR);
                    grid.set(x, height - 1, z, VOXEL_CEILING);

                    // Scatter ceiling panels without replacing walkable floor material.
                    if x > 0 && z > 0 && x % 20 == 0 && z % 20 == 0 {
                        grid.set(x, height - 1, z, VOXEL_LIGHT);
                    }
                }
            }
            let lighting =
                crate::use_cases::bake_voxel_lighting::VoxelLightingSettings::with_default_range(
                    config.voxel_scale,
                )
                .expect("GeneratorConfig voxel_scale must be finite and greater than zero");
            crate::use_cases::bake_voxel_lighting::bake_voxel_lighting(&mut grid, lighting);
            return grid;
        }

        // ==========================================
        // DYNAMIC GENERATION (Growing Tree & BSP)
        // ==========================================
        let mut rng = StdRng::seed_from_u64(
            (seed as u64) ^ (chunk_pos.x.to_bits() as u64) ^ (chunk_pos.z.to_bits() as u64),
        );
        // Per-voxel cosmetic noise (floor holes, dotted pillars) draws from
        // its own stream: the number of those draws depends on voxel
        // resolution, and letting them share the structural RNG would
        // desynchronize wall/door placement between LODs of the same chunk.
        let mut detail_rng = StdRng::seed_from_u64(
            (seed as u64)
                ^ (chunk_pos.x.to_bits() as u64)
                ^ (chunk_pos.z.to_bits() as u64)
                ^ 0xD57A_11ED,
        );

        let maze_gen = GrowingTreeGenerator {
            junction_density: config.tuning.junction_density,
        };
        maze_gen.generate(&mut abstract_grid, &mut rng);

        // Organic room segmentation
        let mut rooms = Vec::new();
        let mut room_grid = vec![None; cell_w * cell_d];

        // Tunable parameters
        let num_room_attempts = 15;
        for _ in 0..num_room_attempts {
            let cx = rng.random_range(0..cell_w);
            let cz = rng.random_range(0..cell_d);
            let rw = rng.random_range(1..=3);
            let rd = rng.random_range(1..=3);

            if cx + rw <= cell_w && cz + rd <= cell_d {
                let mut overlap = false;
                for i in 0..rw {
                    for j in 0..rd {
                        if room_grid[(cz + j) * cell_w + cx + i].is_some() {
                            overlap = true;
                        }
                    }
                }
                if !overlap {
                    let room_id = rooms.len();
                    for i in 0..rw {
                        for j in 0..rd {
                            room_grid[(cz + j) * cell_w + cx + i] = Some(room_id);
                            if let Some(cell) = abstract_grid.get_mut(cx + i, cz + j) {
                                if j > 0 {
                                    cell.walls[0] = false;
                                }
                                if i < rw - 1 {
                                    cell.walls[1] = false;
                                }
                                if j < rd - 1 {
                                    cell.walls[2] = false;
                                }
                                if i > 0 {
                                    cell.walls[3] = false;
                                }
                            }
                        }
                    }
                    rooms.push(Room {
                        name: "Organic",
                        x0: cell_border(cx),
                        x1: cell_border(cx + rw) - 1,
                        z0: cell_border(cz),
                        z1: cell_border(cz + rd) - 1,
                    });
                }
            }
        }

        // Hallway reservation (Part 1)
        let mut non_room_indices = Vec::new();
        for i in 0..(cell_w * cell_d) {
            if room_grid[i].is_none() {
                non_room_indices.push(i);
            }
        }

        let target_corridors = (non_room_indices.len() as f32 * 0.20) as usize;
        let mut corridor_count = 0;

        use rand::seq::SliceRandom;
        non_room_indices.shuffle(&mut rng);

        for &idx in &non_room_indices {
            if corridor_count >= target_corridors {
                break;
            }
            let cx = idx % cell_w;
            let cz = idx / cell_w;
            if abstract_grid.get(cx, cz).unwrap().is_corridor {
                continue;
            }

            abstract_grid.get_mut(cx, cz).unwrap().is_corridor = true;
            corridor_count += 1;

            // Extend North
            let mut current_cz = cz;
            while current_cz > 0 && !abstract_grid.get(cx, current_cz).unwrap().walls[0] {
                current_cz -= 1;
                if room_grid[current_cz * cell_w + cx].is_some()
                    || abstract_grid.get(cx, current_cz).unwrap().is_corridor
                {
                    break;
                }
                abstract_grid.get_mut(cx, current_cz).unwrap().is_corridor = true;
                corridor_count += 1;
            }
            // Extend South
            let mut current_cz = cz;
            while current_cz < cell_d - 1 && !abstract_grid.get(cx, current_cz).unwrap().walls[2] {
                current_cz += 1;
                if room_grid[current_cz * cell_w + cx].is_some()
                    || abstract_grid.get(cx, current_cz).unwrap().is_corridor
                {
                    break;
                }
                abstract_grid.get_mut(cx, current_cz).unwrap().is_corridor = true;
                corridor_count += 1;
            }
            // Extend West
            let mut current_cx = cx;
            while current_cx > 0 && !abstract_grid.get(current_cx, cz).unwrap().walls[3] {
                current_cx -= 1;
                if room_grid[cz * cell_w + current_cx].is_some()
                    || abstract_grid.get(current_cx, cz).unwrap().is_corridor
                {
                    break;
                }
                abstract_grid.get_mut(current_cx, cz).unwrap().is_corridor = true;
                corridor_count += 1;
            }
            // Extend East
            let mut current_cx = cx;
            while current_cx < cell_w - 1 && !abstract_grid.get(current_cx, cz).unwrap().walls[1] {
                current_cx += 1;
                if room_grid[cz * cell_w + current_cx].is_some()
                    || abstract_grid.get(current_cx, cz).unwrap().is_corridor
                {
                    break;
                }
                abstract_grid.get_mut(current_cx, cz).unwrap().is_corridor = true;
                corridor_count += 1;
            }
        }

        // Draw abstract grid to VoxelGrid
        for cz in 0..cell_d {
            for cx in 0..cell_w {
                let cell = abstract_grid.get(cx, cz).unwrap();
                let v_x0 = cell_border(cx);
                let v_z0 = cell_border(cz);
                let v_x1 = (cell_border(cx + 1) - 1).min(width - 1);
                let v_z1 = (cell_border(cz + 1) - 1).min(depth - 1);

                use crate::domain::entities::cell::MicrobiomeZone;
                let cell_h_units = match cell.zone {
                    MicrobiomeZone::Atrium => 12.0,
                    MicrobiomeZone::Blackout => 3.2,
                    _ => 4.0,
                };
                let wall_max_y = (cell_h_units / config.voxel_scale).ceil() as usize;
                let door_max_y = (2.5 / config.voxel_scale).ceil() as usize; // Doorways are 2.5 units tall

                let wall_type = VOXEL_WALL;

                // Floors and Ceilings
                for vx in v_x0..=v_x1 {
                    for vz in v_z0..=v_z1 {
                        grid.set(vx, 0, vz, VOXEL_FLOOR);

                        if cell.zone == MicrobiomeZone::Holes {
                            if vx > v_x0 + 5 && vx < v_x1 - 5 && vz > v_z0 + 5 && vz < v_z1 - 5 {
                                if detail_rng.random_bool(0.05) {
                                    grid.set(vx, 0, vz, VOXEL_AIR);
                                }
                            }
                        }

                        if cell.zone != MicrobiomeZone::Atrium {
                            // Normal ceilings with hanging light
                            grid.set(vx, wall_max_y, vz, VOXEL_CEILING); // Always close the ceiling
                            if cell.zone != MicrobiomeZone::Blackout
                                && (vx % 10 == 0 && vz % 10 == 0)
                            {
                                if cell.zone == MicrobiomeZone::RedRoom {
                                    grid.set(
                                        vx,
                                        wall_max_y - 1,
                                        vz,
                                        crate::domain::entities::voxel_grid::VOXEL_RED_LIGHT,
                                    );
                                } else {
                                    grid.set(vx, wall_max_y - 1, vz, VOXEL_LIGHT);
                                }
                            }
                            for y in (wall_max_y + 1)..height {
                                grid.set(vx, y, vz, VOXEL_CEILING);
                            }
                        } else {
                            // Atrium ceiling and bright hanging light
                            grid.set(vx, wall_max_y, vz, VOXEL_CEILING);
                            if vx % 10 == 0 && vz % 10 == 0 {
                                grid.set(vx, wall_max_y - 1, vz, VOXEL_LIGHT);
                            }
                        }
                    }
                }

                if cell.zone == MicrobiomeZone::PillarField {
                    for y in 1..=wall_max_y {
                        grid.set(v_x0, y, v_z0, wall_type);
                        grid.set(v_x1, y, v_z0, wall_type);
                        grid.set(v_x0, y, v_z1, wall_type);
                        grid.set(v_x1, y, v_z1, wall_type);
                        if detail_rng.random_bool(0.3) {
                            grid.set((v_x0 + v_x1 + 1) / 2, y, (v_z0 + v_z1 + 1) / 2, wall_type);
                        }
                    }
                    continue;
                }

                // South Wall (z1)
                if cz < cell_d - 1 {
                    let same_room = room_grid[cz * cell_w + cx]
                        == room_grid[(cz + 1) * cell_w + cx]
                        && room_grid[cz * cell_w + cx].is_some();

                    if !same_room {
                        let width_choice = rng.random_range(0..100);
                        let hole_units = if width_choice < 15 {
                            1.2
                        } else if width_choice < 85 {
                            2.0
                        } else {
                            3.5
                        };
                        let hole_w = (hole_units / config.voxel_scale) as usize;
                        // Drawn in centimetres of the 5.0-unit cell span, not
                        // voxels, so every LOD picks the same door position.
                        let hole_cm = (hole_units * 100.0) as u32;
                        let max_offset_cm = 500u32.saturating_sub(hole_cm + 50);
                        let offset_cm = if max_offset_cm > 50 {
                            rng.random_range(50..max_offset_cm)
                        } else {
                            50
                        };
                        let offset = (offset_cm as f32 / 100.0 / config.voxel_scale) as usize;
                        let hole_x0 = v_x0 + offset;
                        let hole_x1 = hole_x0 + hole_w;

                        for vx in v_x0..=v_x1 {
                            let in_hole = vx >= hole_x0 && vx < hole_x1;
                            if cell.walls[2] || !in_hole {
                                for y in 1..=wall_max_y {
                                    grid.set(vx, y, v_z1, wall_type);
                                }
                            } else if cell.zone == MicrobiomeZone::Arch {
                                let dx =
                                    (vx as isize - (hole_x0 + hole_w / 2) as isize).abs() as usize;
                                let arch_top = door_max_y;
                                let block_y = arch_top.saturating_sub(dx);
                                for y in block_y..=wall_max_y {
                                    grid.set(vx, y, v_z1, wall_type);
                                }
                            } else {
                                for y in door_max_y..=wall_max_y {
                                    grid.set(vx, y, v_z1, wall_type);
                                }
                            }
                        }
                    }
                } else if cz == cell_d - 1 {
                    // Chunk boundary (force holes for connectivity)
                    let hole_w = (2.0 / config.voxel_scale) as usize;
                    let hole_x0 = (v_x0 + v_x1 + 1) / 2 - hole_w / 2;
                    for vx in v_x0..=v_x1 {
                        let is_hole = vx >= hole_x0 && vx < hole_x0 + hole_w;
                        if !is_hole {
                            for y in 1..=wall_max_y {
                                grid.set(vx, y, v_z1, wall_type);
                            }
                        } else if cell.zone == MicrobiomeZone::Arch {
                            let dx = (vx as isize - (hole_x0 + hole_w / 2) as isize).abs() as usize;
                            let arch_top = door_max_y;
                            let block_y = arch_top.saturating_sub(dx);
                            for y in block_y..=wall_max_y {
                                grid.set(vx, y, v_z1, wall_type);
                            }
                        } else {
                            for y in door_max_y..=wall_max_y {
                                grid.set(vx, y, v_z1, wall_type);
                            }
                        }
                    }
                }

                // East Wall (x1)
                if cx < cell_w - 1 {
                    let same_room = room_grid[cz * cell_w + cx] == room_grid[cz * cell_w + cx + 1]
                        && room_grid[cz * cell_w + cx].is_some();

                    if !same_room {
                        let width_choice = rng.random_range(0..100);
                        let hole_units = if width_choice < 15 {
                            1.2
                        } else if width_choice < 85 {
                            2.0
                        } else {
                            3.5
                        };
                        let hole_w = (hole_units / config.voxel_scale) as usize;
                        // Centimetre draw, same reasoning as the south wall.
                        let hole_cm = (hole_units * 100.0) as u32;
                        let max_offset_cm = 500u32.saturating_sub(hole_cm + 50);
                        let offset_cm = if max_offset_cm > 50 {
                            rng.random_range(50..max_offset_cm)
                        } else {
                            50
                        };
                        let offset = (offset_cm as f32 / 100.0 / config.voxel_scale) as usize;
                        let hole_z0 = v_z0 + offset;
                        let hole_z1 = hole_z0 + hole_w;

                        for vz in v_z0..=v_z1 {
                            let in_hole = vz >= hole_z0 && vz < hole_z1;
                            if cell.walls[1] || !in_hole {
                                for y in 1..=wall_max_y {
                                    grid.set(v_x1, y, vz, wall_type);
                                }
                            } else if cell.zone == MicrobiomeZone::Arch {
                                let dz =
                                    (vz as isize - (hole_z0 + hole_w / 2) as isize).abs() as usize;
                                let arch_top = door_max_y;
                                let block_y = arch_top.saturating_sub(dz);
                                for y in block_y..=wall_max_y {
                                    grid.set(v_x1, y, vz, wall_type);
                                }
                            } else {
                                for y in door_max_y..=wall_max_y {
                                    grid.set(v_x1, y, vz, wall_type);
                                }
                            }
                        }
                    }
                } else if cx == cell_w - 1 {
                    let hole_w = (2.0 / config.voxel_scale) as usize;
                    let hole_z0 = (v_z0 + v_z1 + 1) / 2 - hole_w / 2;
                    for vz in v_z0..=v_z1 {
                        let is_hole = vz >= hole_z0 && vz < hole_z0 + hole_w;
                        if !is_hole {
                            for y in 1..=wall_max_y {
                                grid.set(v_x1, y, vz, wall_type);
                            }
                        } else if cell.zone == MicrobiomeZone::Arch {
                            let dz = (vz as isize - (hole_z0 + hole_w / 2) as isize).abs() as usize;
                            let arch_top = door_max_y;
                            let block_y = arch_top.saturating_sub(dz);
                            for y in block_y..=wall_max_y {
                                grid.set(v_x1, y, vz, wall_type);
                            }
                        } else {
                            for y in door_max_y..=wall_max_y {
                                grid.set(v_x1, y, vz, wall_type);
                            }
                        }
                    }
                }

                // Draw narrower corridors for 2+ consecutive runs
                let mut is_ns_hall = false;
                let mut is_ew_hall = false;
                if cell.is_corridor {
                    let mut ns_run = 1;
                    if !cell.walls[0]
                        && cz > 0
                        && abstract_grid.get(cx, cz - 1).unwrap().is_corridor
                    {
                        ns_run += 1;
                    }
                    if !cell.walls[2]
                        && cz < cell_d - 1
                        && abstract_grid.get(cx, cz + 1).unwrap().is_corridor
                    {
                        ns_run += 1;
                    }

                    let mut ew_run = 1;
                    if !cell.walls[1]
                        && cx < cell_w - 1
                        && abstract_grid.get(cx + 1, cz).unwrap().is_corridor
                    {
                        ew_run += 1;
                    }
                    if !cell.walls[3]
                        && cx > 0
                        && abstract_grid.get(cx - 1, cz).unwrap().is_corridor
                    {
                        ew_run += 1;
                    }

                    if ns_run >= 2 {
                        is_ns_hall = true;
                    }
                    if ew_run >= 2 {
                        is_ew_hall = true;
                    }
                }

                if is_ns_hall && !is_ew_hall {
                    let inset_x = ((v_x1 - v_x0 + 1).saturating_sub(26)) / 2;
                    for vx in v_x0..=(v_x0 + inset_x) {
                        for vz in v_z0..=v_z1 {
                            for y in 1..=wall_max_y {
                                grid.set(vx, y, vz, wall_type);
                            }
                        }
                    }
                    for vx in (v_x1 - inset_x)..=v_x1 {
                        for vz in v_z0..=v_z1 {
                            for y in 1..=wall_max_y {
                                grid.set(vx, y, vz, wall_type);
                            }
                        }
                    }
                } else if is_ew_hall && !is_ns_hall {
                    let inset_z = ((v_z1 - v_z0 + 1).saturating_sub(26)) / 2;
                    for vz in v_z0..=(v_z0 + inset_z) {
                        for vx in v_x0..=v_x1 {
                            for y in 1..=wall_max_y {
                                grid.set(vx, y, vz, wall_type);
                            }
                        }
                    }
                    for vz in (v_z1 - inset_z)..=v_z1 {
                        for vx in v_x0..=v_x1 {
                            for y in 1..=wall_max_y {
                                grid.set(vx, y, vz, wall_type);
                            }
                        }
                    }
                } else if is_ns_hall && is_ew_hall {
                    let inset_x = ((v_x1 - v_x0 + 1).saturating_sub(26)) / 2;
                    let inset_z = ((v_z1 - v_z0 + 1).saturating_sub(26)) / 2;
                    for y in 1..=wall_max_y {
                        for vx in v_x0..=(v_x0 + inset_x) {
                            for vz in v_z0..=(v_z0 + inset_z) {
                                grid.set(vx, y, vz, wall_type);
                            }
                        }
                        for vx in (v_x1 - inset_x)..=v_x1 {
                            for vz in v_z0..=(v_z0 + inset_z) {
                                grid.set(vx, y, vz, wall_type);
                            }
                        }
                        for vx in v_x0..=(v_x0 + inset_x) {
                            for vz in (v_z1 - inset_z)..=v_z1 {
                                grid.set(vx, y, vz, wall_type);
                            }
                        }
                        for vx in (v_x1 - inset_x)..=v_x1 {
                            for vz in (v_z1 - inset_z)..=v_z1 {
                                grid.set(vx, y, vz, wall_type);
                            }
                        }
                    }
                }
            }
        }

        // Chunk Boundaries (North and West)
        for cz in 0..cell_d {
            let cell = abstract_grid.get(0, cz).unwrap();
            use crate::domain::entities::cell::MicrobiomeZone;
            let cell_h_units = match cell.zone {
                MicrobiomeZone::Atrium => 12.0,
                MicrobiomeZone::Blackout => 3.2,
                _ => 4.0,
            };
            let wall_max_y = (cell_h_units / config.voxel_scale).ceil() as usize;
            let door_max_y = (2.5 / config.voxel_scale).ceil() as usize;
            let wall_type = if cell.zone == MicrobiomeZone::RedRoom {
                VOXEL_RED_WALL
            } else {
                VOXEL_WALL
            };

            if cell.zone == MicrobiomeZone::PillarField {
                continue;
            }

            let v_z0 = cell_border(cz);
            let v_z1 = (cell_border(cz + 1) - 1).min(depth - 1);
            let hole_w = (2.0 / config.voxel_scale) as usize;
            let hole_z0 = (v_z0 + v_z1 + 1) / 2 - hole_w / 2;
            for vz in v_z0..=v_z1 {
                let is_hole = vz >= hole_z0 && vz < hole_z0 + hole_w;
                if !is_hole {
                    for y in 1..=wall_max_y {
                        grid.set(0, y, vz, wall_type);
                    }
                } else if cell.zone == MicrobiomeZone::Arch {
                    let dz = (vz as isize - (hole_z0 + hole_w / 2) as isize).abs() as usize;
                    let arch_top = door_max_y;
                    let block_y = arch_top.saturating_sub(dz);
                    for y in block_y..=wall_max_y {
                        grid.set(0, y, vz, wall_type);
                    }
                } else {
                    for y in door_max_y..=wall_max_y {
                        grid.set(0, y, vz, wall_type);
                    }
                }
            }
        }
        for cx in 0..cell_w {
            let cell = abstract_grid.get(cx, 0).unwrap();
            use crate::domain::entities::cell::MicrobiomeZone;
            let cell_h_units = match cell.zone {
                MicrobiomeZone::Atrium => 12.0,
                MicrobiomeZone::Blackout => 3.2,
                _ => 4.0,
            };
            let wall_max_y = (cell_h_units / config.voxel_scale).ceil() as usize;
            let door_max_y = (2.5 / config.voxel_scale).ceil() as usize;
            let wall_type = if cell.zone == MicrobiomeZone::RedRoom {
                VOXEL_RED_WALL
            } else {
                VOXEL_WALL
            };

            if cell.zone == MicrobiomeZone::PillarField {
                continue;
            }

            let v_x0 = cell_border(cx);
            let v_x1 = (cell_border(cx + 1) - 1).min(width - 1);
            let hole_w = (2.0 / config.voxel_scale) as usize;
            let hole_x0 = (v_x0 + v_x1 + 1) / 2 - hole_w / 2;
            for vx in v_x0..=v_x1 {
                let is_hole = vx >= hole_x0 && vx < hole_x0 + hole_w;
                if !is_hole {
                    for y in 1..=wall_max_y {
                        grid.set(vx, y, 0, wall_type);
                    }
                } else if cell.zone == MicrobiomeZone::Arch {
                    let dx = (vx as isize - (hole_x0 + hole_w / 2) as isize).abs() as usize;
                    let arch_top = door_max_y;
                    let block_y = arch_top.saturating_sub(dx);
                    for y in block_y..=wall_max_y {
                        grid.set(vx, y, 0, wall_type);
                    }
                } else {
                    for y in door_max_y..=wall_max_y {
                        grid.set(vx, y, 0, wall_type);
                    }
                }
            }
        }

        // Apply Room Stamps to generated rooms
        let stamps = vec![
            RoomStamp::cubicles(),
            RoomStamp::showroom(),
            RoomStamp::waiting(),
            RoomStamp::storage(),
            RoomStamp::lounge(),
            RoomStamp::server_room(),
            RoomStamp::cafeteria(),
            RoomStamp::maintenance(),
        ];

        for room in &rooms {
            let mut possible_stamps = Vec::new();
            for stamp in &stamps {
                let max_stamps_w = (room.x1 - room.x0) / stamp.width.max(1);
                let max_stamps_d = (room.z1 - room.z0) / stamp.depth.max(1);
                if max_stamps_w > 0 && max_stamps_d > 0 {
                    possible_stamps.push(stamp);
                }
            }
            // Both rolls happen whether or not a stamp fits: which stamps fit
            // depends on voxel resolution, and skipping draws would
            // desynchronize the structural RNG between LODs of the chunk.
            let stamp_roll = rng.random_range(0..usize::MAX);
            let stairs_roll = rng.random_bool(0.15 * (config.tuning.stairs_density as f64));
            if !possible_stamps.is_empty() {
                let mut stamp = possible_stamps[stamp_roll % possible_stamps.len()];

                // Low-frequency stairs chance override
                let stairway = RoomStamp::stairway();
                if stairs_roll {
                    stamp = &stairway;
                }

                let spacing = (6.0 * (config.voxel_scale / 0.1)) as usize;
                let mut start_x = room.x0 + spacing;
                while start_x + stamp.width <= room.x1.saturating_sub(spacing) {
                    let mut start_z = room.z0 + spacing;
                    while start_z + stamp.depth <= room.z1.saturating_sub(spacing) {
                        if validate_stamp_placement(&grid, room, stamp, start_x, start_z) {
                            place_stamp(&mut grid, stamp, start_x, start_z);
                        }
                        start_z += stamp.depth + spacing;
                    }
                    start_x += stamp.width + spacing;
                }
            }

            // Basic room lighting
            let cx = (room.x0 + room.x1) / 2;
            let is_strip = rng.random_bool(0.3); // 30% chance for strip lights
            let mut z = room.z0 + 8;

            while z <= room.z1.saturating_sub(8) {
                if is_strip {
                    for offset in 0..4 {
                        if z + offset <= room.z1 {
                            grid.set(cx, height - 1, z + offset, VOXEL_LIGHT);
                        }
                    }
                } else {
                    grid.set(cx, height - 1, z, VOXEL_LIGHT);
                }
                z += 12;
            }
        }

        // Basic corridor lighting
        for cz in (8..depth).step_by(16) {
            for cx in (8..width).step_by(16) {
                if grid.get(cx, height - 1, cz) == VOXEL_CEILING && grid.get(cx, 1, cz) == VOXEL_AIR
                {
                    grid.set(cx, height - 1, cz, VOXEL_LIGHT);
                }
            }
        }

        // ==========================================
        // PHYSICAL VOXEL LIGHTING BAKE
        // ==========================================
        let lighting =
            crate::use_cases::bake_voxel_lighting::VoxelLightingSettings::with_default_range(
                config.voxel_scale,
            )
            .expect("GeneratorConfig voxel_scale must be finite and greater than zero");
        crate::use_cases::bake_voxel_lighting::bake_voxel_lighting(&mut grid, lighting);

        // Bake directional face occlusion
        crate::domain::use_cases::path_tracer::bake_face_occlusion(&mut grid);

        let elapsed_micros = telemetry.now_micros().saturating_sub(start_micros);
        telemetry.log(&format!(
            "[TELEMETRY] execute completed. Duration={}us, OutputVoxelGridSize={}x{}x{}",
            elapsed_micros,
            grid.width(),
            grid.height(),
            grid.depth()
        ));

        grid
}

fn validate_stamp_placement(
    grid: &VoxelGrid,
    room: &Room,
    stamp: &RoomStamp,
    start_x: usize,
    start_z: usize,
) -> bool {
    if start_x < room.x0
        || start_x + stamp.width > room.x1
        || start_z < room.z0
        || start_z + stamp.depth > room.z1
    {
        return false;
    }

    for lz in 0..stamp.depth {
        for lx in 0..stamp.width {
            let vx = start_x + lx;
            let vz = start_z + lz;
            if stamp.data[lz * stamp.width + lx] > 0 {
                if grid.get(vx, 1, vz) != VOXEL_AIR && grid.get(vx, 1, vz) != VOXEL_FLOOR {
                    return false;
                }
            }
        }
    }

    true
}

fn place_stamp(grid: &mut VoxelGrid, stamp: &RoomStamp, start_x: usize, start_z: usize) {
    let max_y = grid.height() - 2;
    for sz in 0..stamp.depth {
        for sx in 0..stamp.width {
            let val = stamp.data[sz * stamp.width + sx];
            let vx = start_x + sx;
            let vz = start_z + sz;
            if val == 1 {
                for y in 1..=3 {
                    grid.set(vx, y, vz, VOXEL_WALL);
                }
            } else if val == 2 {
                for y in 1..=max_y {
                    grid.set(vx, y, vz, VOXEL_WALL);
                }
            } else if val == 3 {
                for y in 1..=1 {
                    grid.set(vx, y, vz, VOXEL_WALL);
                }
            } else if val == 4 {
                // Raised floor
                grid.set(vx, 1, vz, VOXEL_FLOOR);
                grid.set(vx, 2, vz, VOXEL_FLOOR);
            } else if val == 5 {
                // Step 1
                grid.set(vx, 1, vz, VOXEL_FLOOR);
            } else if val == 6 { // Step 2
                // Just floor, already set
            }
        }
    }
}

