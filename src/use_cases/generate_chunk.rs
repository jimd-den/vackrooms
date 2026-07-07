use crate::entities::models::Position;
use crate::use_cases::ports::{NoiseProvider, TelemetryPort, NULL_TELEMETRY};
use crate::domain::entities::voxel_grid::{
    VoxelGrid, VOXEL_AIR, VOXEL_WALL, VOXEL_FLOOR, VOXEL_CEILING, VOXEL_LIGHT, VOXEL_RED_WALL,
};

/// Dynamic config configuration profile to scale SVO dimensions and voxel grid.
#[derive(Debug, Clone, Copy)]
pub struct GeneratorConfig {
    pub chunk_size: f32,
    pub voxel_scale: f32,
}

impl GeneratorConfig {
    pub fn high_spec() -> Self {
        Self { chunk_size: 20.0, voxel_scale: 0.1 }
    }
    
    pub fn low_spec() -> Self {
        Self { chunk_size: 10.0, voxel_scale: 0.2 }
    }

    pub fn svo_depth(&self) -> u32 {
        if self.voxel_scale > 0.15 { 6 } else { 8 }
    }
    
    pub fn svo_world_size(&self) -> f32 {
        let size = 1 << self.svo_depth();
        size as f32 * self.voxel_scale
    }
}

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
        for z in 2..8 { data[z * width + 5] = 1; }
        for x in 2..8 { data[5 * width + x] = 1; }
        for z in 12..18 { data[z * width + 15] = 1; }
        for x in 12..18 { data[15 * width + x] = 1; }
        Self { width, depth, data }
    }

    fn showroom() -> Self {
        let width = 16;
        let depth = 16;
        let mut data = vec![0; width * depth];
        for z in 2..6 {
            for x in 2..6 { data[z * width + x] = 2; }
            for x in 10..14 { data[z * width + x] = 2; }
        }
        for z in 10..14 {
            for x in 2..6 { data[z * width + x] = 2; }
            for x in 10..14 { data[z * width + x] = 2; }
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
}

pub struct GenerateChunkArchitectureUseCase<'a> {
    noise_provider: &'a dyn NoiseProvider,
    telemetry: &'a dyn TelemetryPort,
}

impl<'a> GenerateChunkArchitectureUseCase<'a> {
    /// Constructs the use case with silent telemetry (tests, wasm default).
    pub fn new(noise_provider: &'a dyn NoiseProvider) -> Self {
        Self { noise_provider, telemetry: &NULL_TELEMETRY }
    }

    /// Constructs the use case with an injected telemetry sink (native server,
    /// browser console, ...). Keeps the Dependency Rule intact: the use case
    /// only knows the TelemetryPort trait.
    pub fn with_telemetry(noise_provider: &'a dyn NoiseProvider, telemetry: &'a dyn TelemetryPort) -> Self {
        Self { noise_provider, telemetry }
    }

    pub fn execute(&self, chunk_pos: Position, seed: u32, config: GeneratorConfig) -> VoxelGrid {
        let start_micros = self.telemetry.now_micros();

        let width = (config.chunk_size / config.voxel_scale) as usize;
        let height = (3.0 / config.voxel_scale) as usize;
        let depth = (config.chunk_size / config.voxel_scale) as usize;

        self.telemetry.log(&format!(
            "[INFO] Entering GenerateChunkArchitectureUseCase::execute. Args: chunk_pos={:?}, seed={}, chunk_size={} (Voxel dimensions: {}x{}x{})",
            chunk_pos, seed, config.chunk_size, width, height, depth
        ));

        let mut grid = VoxelGrid::new(width, height, depth);

        // STARTING HUB OVERRIDE: Chunk (0,0) is a massive open plaza
        if chunk_pos.x.abs() < 1.0 && chunk_pos.z.abs() < 1.0 {
            for z in 0..depth {
                for x in 0..width {
                    grid.set(x, 0, z, VOXEL_FLOOR);
                    grid.set(x, height - 1, z, VOXEL_CEILING);
                    
                    // Add some scattered lights to the ceiling and floor so it's bright
                    if x > 0 && z > 0 && x % 20 == 0 && z % 20 == 0 {
                        grid.set(x, height - 1, z, VOXEL_LIGHT);
                        grid.set(x, 0, z, VOXEL_LIGHT); // Also light up the floor
                    }
                    
                    // Only put walls exactly at the absolute outer edge so you don't fall off the world if adjacent chunks aren't loaded yet
                    if x == 0 || x == width - 1 || z == 0 || z == depth - 1 {
                        // Leave holes for the standard corridors so it connects to other chunks!
                        let w_f = width as f32;
                        let d_f = depth as f32;
                        let main_wall_x1 = (0.465 * w_f) as usize;
                        let main_wall_x2 = (0.53 * w_f) as usize;
                        let z_split1 = (0.275 * d_f) as usize;
                        let z_split2 = (0.325 * d_f) as usize;
                        let z_split3 = (0.675 * d_f) as usize;
                        let z_split4 = (0.725 * d_f) as usize;
                        
                        let is_z_corridor = z >= z_split1 && z < z_split2 || z >= z_split3 && z < z_split4;
                        let is_x_corridor = x >= main_wall_x1 && x < main_wall_x2;
                        
                        let is_hole = ( (x == 0 || x == width - 1) && is_z_corridor ) || 
                                      ( (z == 0 || z == depth - 1) && is_x_corridor );
                                      
                        if !is_hole {
                            for y in 1..height-1 {
                                grid.set(x, y, z, VOXEL_WALL);
                            }
                        }
                    }
                }
            }
            
            crate::domain::use_cases::calculate_lighting::calculate_voxel_lighting(&mut grid);
            return grid;
        }

        // Scale-invariant coordinate splits using fractions
        let w_f = width as f32;
        let d_f = depth as f32;

        let main_wall_x1 = (0.465 * w_f) as usize;
        let main_wall_x2 = (0.53 * w_f) as usize;
        
        let z_split1 = (0.275 * d_f) as usize;
        let z_split2 = (0.325 * d_f) as usize;
        let z_split3 = (0.675 * d_f) as usize;
        let z_split4 = (0.725 * d_f) as usize;

        // ==========================================
        // PASS 1: BLUEPRINT (Deterministic layout)
        // ==========================================
        let wall_max_y = height - 2;
        for z in 0..depth {
            for x in 0..width {
                grid.set(x, 0, z, VOXEL_FLOOR);
                grid.set(x, height - 1, z, VOXEL_CEILING);

                let is_main_wall = (x == main_wall_x1 || x == main_wall_x2) && 
                                   (z < z_split1 || (z >= z_split2 && z < z_split3) || z >= z_split4);
                let is_branch1_wall = (x < main_wall_x1) && (z == z_split1 || z == z_split2 - 1);
                let is_branch2_wall = (x >= main_wall_x2) && (z == z_split3 || z == z_split4 - 1);
                
                let is_boundary_wall = 
                    (z == 0 && (x < main_wall_x1 + 1 || x >= main_wall_x2)) ||
                    (z == depth - 1 && (x < main_wall_x1 + 1 || x >= main_wall_x2)) ||
                    (x == 0 && (z < z_split1 || (z >= z_split2 && z < z_split3) || z >= z_split4)) ||
                    (x == width - 1 && (z < z_split1 || (z >= z_split2 && z < z_split3) || z >= z_split4));

                if is_main_wall || is_branch1_wall || is_branch2_wall || is_boundary_wall {
                    for y in 1..=wall_max_y {
                        grid.set(x, y, z, VOXEL_WALL);
                    }
                } else {
                    for y in 1..=wall_max_y {
                        grid.set(x, y, z, VOXEL_AIR);
                    }
                }
            }
        }

        // Carve doorways
        let d_z1 = (0.14 * d_f) as usize;
        let d_z2 = (0.50 * d_f) as usize;
        let d_z3 = (0.85 * d_f) as usize;
        let d_z4 = (0.35 * d_f) as usize;
        let d_z5 = (0.85 * d_f) as usize;

        self.carve_doorway(&mut grid, main_wall_x1, d_z1, 1, wall_max_y);
        self.carve_doorway(&mut grid, main_wall_x1, d_z2, 1, wall_max_y);
        self.carve_doorway(&mut grid, main_wall_x1, d_z3, 1, wall_max_y);
        self.carve_doorway(&mut grid, main_wall_x2, d_z4, 1, wall_max_y);
        self.carve_doorway(&mut grid, main_wall_x2, d_z5, 1, wall_max_y);

        // Room interior stamp placement
        let rooms = vec![
            Room { name: "Room 1", x0: 1, x1: main_wall_x1 - 1, z0: 1, z1: z_split1 - 1 },
            Room { name: "Room 2", x0: 1, x1: main_wall_x1 - 1, z0: z_split2, z1: z_split3 - 1 },
            Room { name: "Room 3", x0: 1, x1: main_wall_x1 - 1, z0: z_split4, z1: depth - 2 },
            Room { name: "Room 4", x0: main_wall_x2 + 1, x1: width - 2, z0: 1, z1: z_split3 - 1 },
            Room { name: "Room 5", x0: main_wall_x2 + 1, x1: width - 2, z0: z_split4, z1: depth - 2 },
        ];

        let stamps = vec![
            RoomStamp::cubicles(),
            RoomStamp::showroom(),
            RoomStamp::waiting(),
        ];

        for room in &rooms {
            let stamp = match room.name {
                "Room 2" => &stamps[0], // Cubicles
                "Room 4" => &stamps[1], // Showroom
                _ => &stamps[2],        // Waiting
            };

            let spacing = (6.0 * (config.voxel_scale / 0.1)) as usize;
            let mut start_x = room.x0 + spacing;
            while start_x + stamp.width <= room.x1.saturating_sub(spacing) {
                let mut start_z = room.z0 + spacing;
                while start_z + stamp.depth <= room.z1.saturating_sub(spacing) {
                    if self.validate_stamp_placement(&grid, room, stamp, start_x, start_z) {
                        self.place_stamp(&mut grid, stamp, start_x, start_z);
                    }
                    start_z += stamp.depth + spacing;
                }
                start_x += stamp.width + spacing;
            }
        }

        // Lights
        for room in &rooms {
            let cx = (room.x0 + room.x1) / 2;
            let mut z = room.z0 + 8;
            while z <= room.z1 - 8 {
                grid.set(cx, height - 1, z, VOXEL_LIGHT);
                z += 12;
            }

            if room.x1 - room.x0 > 50 {
                let cx1 = room.x0 + (room.x1 - room.x0) / 4;
                let cx2 = room.x0 + 3 * (room.x1 - room.x0) / 4;
                let mut z = room.z0 + 8;
                while z <= room.z1 - 8 {
                    grid.set(cx1, height - 1, z, VOXEL_LIGHT);
                    grid.set(cx2, height - 1, z, VOXEL_LIGHT);
                    z += 12;
                }
            }
        }

        // Corridor light runs
        let mut cz = 8;
        let corridor_x = (0.50 * w_f) as usize;
        while cz < depth - 8 {
            grid.set(corridor_x, height - 1, cz, VOXEL_LIGHT);
            cz += 16;
        }

        // ==========================================
        // PASS 2: UNCANNY EROSION (Noise-driven)
        // ==========================================
        let mut mutated_grid = VoxelGrid::new(width, height, depth);
        for z in 0..depth {
            for x in 0..width {
                for y in 0..height {
                    mutated_grid.set(x, y, z, grid.get(x, y, z));
                }

                for y in 1..height {
                    let current = grid.get(x, y, z);
                    let noise_val = self.noise_provider.evaluate_2d(
                        seed,
                        Position::new(
                            x as f32 + (y as f32 * 31.415) + (chunk_pos.x * 23.5),
                            z as f32 + (chunk_pos.z * 23.5)
                        )
                    );

                    if current == VOXEL_WALL && y > 0 && y < height - 1 {
                        if noise_val > 0.94 {
                            mutated_grid.set(x, y, z, VOXEL_RED_WALL);
                        }
                    }

                    if current == VOXEL_LIGHT {
                        if noise_val > 0.80 {
                            mutated_grid.set(x, y, z, VOXEL_CEILING);
                        }
                    }

                    if current == VOXEL_CEILING && y == height - 1 {
                        if noise_val > 0.90 {
                            mutated_grid.set(x, y - 1, z, VOXEL_CEILING);
                            mutated_grid.set(x, y, z, VOXEL_WALL);
                        }
                    }
                }

                if z > 0 && z < depth - 1 && x > 0 && x < width - 1 {
                    if grid.get(x, 1, z) == VOXEL_WALL {
                        let neighbors = [
                            grid.get(x + 1, 1, z) == VOXEL_WALL,
                            grid.get(x - 1, 1, z) == VOXEL_WALL,
                            grid.get(x, 1, z + 1) == VOXEL_WALL,
                            grid.get(x, 1, z - 1) == VOXEL_WALL,
                        ];
                        let walls_count = neighbors.iter().filter(|&&v| v).count();
                        if walls_count == 1 {
                            let mut h = 1;
                            while h < height - 1 && grid.get(x, h + 1, z) == VOXEL_WALL {
                                h += 1;
                            }
                            
                            if h <= 16 {
                                let noise_val = self.noise_provider.evaluate_2d(
                                    seed.wrapping_add(x as u32),
                                    Position::new(
                                        x as f32 + (chunk_pos.x * 23.5),
                                        z as f32 + (chunk_pos.z * 23.5)
                                    )
                                );
                                if noise_val > 0.75 {
                                    let offset = if neighbors[0] { (-1, 0) }
                                                 else if neighbors[1] { (1, 0) }
                                                 else if neighbors[2] { (0, -1) }
                                                 else { (0, 1) };
                                    let tx = (x as isize + offset.0) as usize;
                                    let tz = (z as isize + offset.1) as usize;
                                    if grid.get(tx, 1, tz) == VOXEL_AIR {
                                        for y in 1..=h {
                                            mutated_grid.set(tx, y, tz, VOXEL_WALL);
                                        }
                                    }
                                } else if noise_val < -0.75 {
                                    for y in 1..=h {
                                        mutated_grid.set(x, y, z, VOXEL_AIR);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // E. Doorway closure check
        let doorways = vec![
            (main_wall_x1, d_z1),
            (main_wall_x1, d_z2),
            (main_wall_x1, d_z3),
            (main_wall_x2, d_z4),
            (main_wall_x2, d_z5),
        ];
        for &(cx, cz) in &doorways {
            let noise_val = self.noise_provider.evaluate_2d(
                seed.wrapping_add(99),
                Position::new(cx as f32 + (chunk_pos.x * 23.5), cz as f32 + (chunk_pos.z * 23.5))
            );
            if noise_val > 0.88 {
                let mut test_grid = VoxelGrid::new(width, height, depth);
                for tz in 0..depth {
                    for ty in 0..height {
                        for tx in 0..width {
                            test_grid.set(tx, ty, tz, mutated_grid.get(tx, ty, tz));
                        }
                    }
                }
                
                for ty in 1..=wall_max_y {
                    test_grid.set(cx, ty, cz, VOXEL_WALL);
                    test_grid.set(cx, ty, cz + 1, VOXEL_WALL);
                    test_grid.set(cx, ty, cz + 2, VOXEL_WALL);
                    test_grid.set(cx, ty, cz + 3, VOXEL_WALL);
                    test_grid.set(cx, ty, cz + 4, VOXEL_WALL);
                    test_grid.set(cx, ty, cz + 5, VOXEL_WALL);
                }

                if self.validate_global_walkability(&test_grid, corridor_x, main_wall_x1, main_wall_x2, d_z1, d_z2, d_z3, d_z4, d_z5) {
                    for ty in 1..=wall_max_y {
                        mutated_grid.set(cx, ty, cz, VOXEL_WALL);
                        mutated_grid.set(cx, ty, cz + 1, VOXEL_WALL);
                        mutated_grid.set(cx, ty, cz + 2, VOXEL_WALL);
                        mutated_grid.set(cx, ty, cz + 3, VOXEL_WALL);
                        mutated_grid.set(cx, ty, cz + 4, VOXEL_WALL);
                        mutated_grid.set(cx, ty, cz + 5, VOXEL_WALL);
                    }
                }
            }
        }

        // ==========================================
        // LIGHTING PROPAGATION (BFS Flood fill only)
        // ==========================================
        crate::domain::use_cases::calculate_lighting::calculate_voxel_lighting(&mut mutated_grid);

        let elapsed_micros = self.telemetry.now_micros().saturating_sub(start_micros);
        self.telemetry.log(&format!(
            "[TELEMETRY] execute completed. Duration={}us, OutputVoxelGridSize={}x{}x{}",
            elapsed_micros, mutated_grid.width(), mutated_grid.height(), mutated_grid.depth()
        ));

        mutated_grid
    }

    fn carve_doorway(&self, grid: &mut VoxelGrid, x: usize, z: usize, y0: usize, y1: usize) {
        for y in y0..=y1 {
            grid.set(x, y, z, VOXEL_AIR);
        }
        for dz in 1..6 {
            for y in y0..=y1 {
                grid.set(x, y, z + dz, VOXEL_AIR);
            }
        }
    }

    fn validate_stamp_placement(
        &self,
        grid: &VoxelGrid,
        room: &Room,
        stamp: &RoomStamp,
        start_x: usize,
        start_z: usize,
    ) -> bool {
        if start_x < room.x0 || start_x + stamp.width > room.x1 ||
           start_z < room.z0 || start_z + stamp.depth > room.z1 {
            return false;
        }

        for lz in 0..stamp.depth {
            for lx in 0..stamp.width {
                let vx = start_x + lx;
                let vz = start_z + lz;
                if stamp.data[lz * stamp.width + lx] > 0 {
                    if grid.get(vx, 1, vz) != VOXEL_AIR {
                        return false;
                    }
                }
            }
        }

        let mut temp_grid = VoxelGrid::new(grid.width(), grid.height(), grid.depth());
        for z in 0..grid.depth() {
            for y in 0..grid.height() {
                for x in 0..grid.width() {
                    temp_grid.set(x, y, z, grid.get(x, y, z));
                }
            }
        }

        for lz in 0..stamp.depth {
            for lx in 0..stamp.width {
                let vx = start_x + lx;
                let vz = start_z + lz;
                let t = stamp.data[lz * stamp.width + lx];
                if t == 1 {
                    for y in 1..=16 { temp_grid.set(vx, y, vz, VOXEL_WALL); }
                } else if t == 2 {
                    for y in 1..=28 { temp_grid.set(vx, y, vz, VOXEL_WALL); }
                } else if t == 3 {
                    for y in 1..=10 { temp_grid.set(vx, y, vz, VOXEL_WALL); }
                }
            }
        }

        let mut d_x = 0;
        let mut d_z = 0;
        let mut found = false;

        for z in room.z0..room.z1 {
            if temp_grid.get(room.x0 - 1, 1, z) == VOXEL_AIR {
                d_x = room.x0 - 1;
                d_z = z;
                found = true;
                break;
            }
            if temp_grid.get(room.x1, 1, z) == VOXEL_AIR {
                d_x = room.x1;
                d_z = z;
                found = true;
                break;
            }
        }

        if !found {
            return false;
        }

        let mut total_empty = 0;
        for z in room.z0..room.z1 {
            for x in room.x0..room.x1 {
                if temp_grid.get(x, 1, z) == VOXEL_AIR {
                    total_empty += 1;
                }
            }
        }

        let mut visited = vec![vec![false; grid.depth()]; grid.width()];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((d_x, d_z));
        visited[d_x][d_z] = true;

        let mut reached = 0;

        while let Some((cx, cz)) = queue.pop_front() {
            let dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];
            for &(dx, dz) in &dirs {
                let nx = cx as isize + dx;
                let nz = cz as isize + dz;
                if nx >= (room.x0 - 1) as isize && nx <= room.x1 as isize &&
                   nz >= (room.z0 - 1) as isize && nz <= room.z1 as isize {
                    let nx = nx as usize;
                    let nz = nz as usize;
                    if !visited[nx][nz] && temp_grid.get(nx, 1, nz) == VOXEL_AIR {
                        visited[nx][nz] = true;
                        queue.push_back((nx, nz));
                        if nx >= room.x0 && nx < room.x1 && nz >= room.z0 && nz < room.z1 {
                            reached += 1;
                        }
                    }
                }
            }
        }

        if total_empty == 0 { return true; }
        let ratio = (reached as f32) / (total_empty as f32);
        ratio >= 0.8
    }

    fn place_stamp(&self, grid: &mut VoxelGrid, stamp: &RoomStamp, start_x: usize, start_z: usize) {
        for lz in 0..stamp.depth {
            for lx in 0..stamp.width {
                let vx = start_x + lx;
                let vz = start_z + lz;
                let t = stamp.data[lz * stamp.width + lx];
                if t == 1 {
                    for y in 1..=16 { grid.set(vx, y, vz, VOXEL_WALL); }
                } else if t == 2 {
                    for y in 1..=28 { grid.set(vx, y, vz, VOXEL_WALL); }
                } else if t == 3 {
                    for y in 1..=10 { grid.set(vx, y, vz, VOXEL_WALL); }
                }
            }
        }
    }

    fn validate_global_walkability(
        &self,
        grid: &VoxelGrid,
        corridor_x: usize,
        main_wall_x1: usize,
        main_wall_x2: usize,
        d_z1: usize,
        d_z2: usize,
        d_z3: usize,
        d_z4: usize,
        d_z5: usize,
    ) -> bool {
        let w = grid.width();
        let d = grid.depth();
        let mut visited = vec![vec![false; d]; w];
        let mut queue = std::collections::VecDeque::new();
        
        let start_x = corridor_x;
        let start_z = 0;
        queue.push_back((start_x, start_z));
        visited[start_x][start_z] = true;
        
        let mut reached_targets = 0;
        let targets = vec![
            (corridor_x, d - 1),
            (0, (0.3 * d as f32) as usize),
            (w - 1, (0.7 * d as f32) as usize),
            (main_wall_x1, d_z1),
            (main_wall_x1, d_z2),
            (main_wall_x1, d_z3),
            (main_wall_x2, d_z4),
            (main_wall_x2, d_z5),
        ];

        while let Some((cx, cz)) = queue.pop_front() {
            for &(tx, tz) in &targets {
                if cx == tx && cz == tz {
                    reached_targets += 1;
                }
            }

            if reached_targets == targets.len() {
                return true;
            }

            let dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];
            for &(dx, dz) in &dirs {
                let nx = cx as isize + dx;
                let nz = cz as isize + dz;
                if nx >= 0 && nx < w as isize && nz >= 0 && nz < d as isize {
                    let nx = nx as usize;
                    let nz = nz as usize;
                    if !visited[nx][nz] && grid.get(nx, 1, nz) == VOXEL_AIR {
                        visited[nx][nz] = true;
                        queue.push_back((nx, nz));
                    }
                }
            }
        }
        
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockNoiseProvider {
        value: f32,
    }
    impl NoiseProvider for MockNoiseProvider {
        fn evaluate_2d(&self, _seed: u32, _pos: Position) -> f32 {
            self.value
        }
    }

    #[test]
    fn test_voxel_native_blueprint_walkable() {
        let noise = MockNoiseProvider { value: 0.0 };
        let generator = GenerateChunkArchitectureUseCase::new(&noise);
        let config = GeneratorConfig::high_spec();
        let grid = generator.execute(Position::new(0.0, 0.0), 42, config);

        assert_eq!(grid.width(), 200);
        assert_eq!(grid.height(), 30);
        assert_eq!(grid.depth(), 200);
    }
}
