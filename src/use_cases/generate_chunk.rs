use crate::entities::models::Position;
use crate::use_cases::ports::{NoiseProvider, TelemetryPort, NULL_TELEMETRY};
use crate::domain::entities::voxel_grid::{
    VoxelGrid, VOXEL_AIR, VOXEL_WALL, VOXEL_FLOOR, VOXEL_CEILING, VOXEL_LIGHT, VOXEL_RED_WALL,
};
use rand::{Rng, RngExt, SeedableRng};
use rand::rngs::StdRng;
use crate::domain::use_cases::generate_maze::{GrowingTreeGenerator, MazeGenerator};
use crate::domain::entities::grid::Grid;

/// User-tunable knobs for the level generators. All values are multipliers
/// around the defaults (1.0); 0 disables the feature, ~2 saturates it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelTuning {
    /// Structural column / pillar density.
    pub pillars: f32,
    /// Office wall segment density.
    pub walls: f32,
    /// How much of the world vaults into tall atria.
    pub atria: f32,
    /// Ceiling light panel density.
    pub lights: f32,
    /// Dead-end and junction frequency (Growing Tree loop breaking).
    pub junction_density: f32,
    /// Frequency of stairs / vertical traversal generation.
    pub stairs_density: f32,
}

impl Default for LevelTuning {
    fn default() -> Self {
        Self { 
            pillars: 1.0, 
            walls: 1.0, 
            atria: 1.0, 
            lights: 1.0,
            junction_density: 1.0,
            stairs_density: 1.0,
        }
    }
}

/// Dynamic config configuration profile to scale SVO dimensions and voxel grid.
#[derive(Debug, Clone, Copy)]
pub struct GeneratorConfig {
    pub chunk_size: f32,
    pub voxel_scale: f32,
    /// Which level generator fills the chunks (see `level_generator`):
    /// 0 = Backrooms (default), 1 = legacy office blueprint, 34 = grassland.
    pub level: u32,
    /// User-facing generation knobs (URL query params in the browser).
    pub tuning: LevelTuning,
}

impl GeneratorConfig {
    pub fn high_spec() -> Self {
        Self { chunk_size: 20.0, voxel_scale: 0.1, level: 0, tuning: LevelTuning::default() }
    }

    pub fn low_spec() -> Self {
        Self { chunk_size: 10.0, voxel_scale: 0.2, level: 0, tuning: LevelTuning::default() }
    }

    pub fn with_level(mut self, level: u32) -> Self {
        self.level = level;
        self
    }

    pub fn with_tuning(mut self, tuning: LevelTuning) -> Self {
        self.tuning = tuning;
        self
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
        // Pluggable levels: everything except the legacy office blueprint
        // (level 1, kept inline below) goes through the LevelGenerator port.
        if config.level == 34 { // Grassland
            use crate::use_cases::grassland_level::GrasslandLevel;
            use crate::use_cases::level_generator::{LevelGenerator, LEVEL_GRASSLAND};

            let start_micros = self.telemetry.now_micros();
            let generator: &dyn LevelGenerator = &GrasslandLevel;
            let mut grid = generator.generate(chunk_pos, seed, config, self.noise_provider);
            crate::domain::use_cases::calculate_lighting::calculate_voxel_lighting(&mut grid);

            let elapsed_micros = self.telemetry.now_micros().saturating_sub(start_micros);
            self.telemetry.log(&format!(
                "[TELEMETRY] level {} chunk generated. Duration={}us, Grid={}x{}x{}",
                config.level, elapsed_micros, grid.width(), grid.height(), grid.depth()
            ));
            return grid;
        }

        let start_micros = self.telemetry.now_micros();

        let width = (config.chunk_size / config.voxel_scale) as usize;
        let height = (3.0 / config.voxel_scale) as usize;
        let depth = (config.chunk_size / config.voxel_scale) as usize;

        self.telemetry.log(&format!(
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
                    
                    // Add some scattered lights to the ceiling and floor so it's bright
                    if x > 0 && z > 0 && x % 20 == 0 && z % 20 == 0 {
                        grid.set(x, height - 1, z, VOXEL_LIGHT);
                        grid.set(x, 0, z, VOXEL_LIGHT);
                    }
                }
            }
            crate::domain::use_cases::calculate_lighting::calculate_voxel_lighting(&mut grid);
            return grid;
        }

        // ==========================================
        // DYNAMIC GENERATION (Growing Tree & BSP)
        // ==========================================
        let mut rng = StdRng::seed_from_u64((seed as u64) ^ (chunk_pos.x.to_bits() as u64) ^ (chunk_pos.z.to_bits() as u64));
        
        let cell_vw = (5.0 / config.voxel_scale) as usize;
        let cell_vd = (5.0 / config.voxel_scale) as usize;
        let cell_w = width / cell_vw;
        let cell_d = depth / cell_vd;
        
        let mut abstract_grid = Grid::new(cell_w, cell_d);
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
                                if j > 0 { cell.walls[0] = false; }
                                if i < rw - 1 { cell.walls[1] = false; }
                                if j < rd - 1 { cell.walls[2] = false; }
                                if i > 0 { cell.walls[3] = false; }
                            }
                        }
                    }
                    rooms.push(Room {
                        name: "Organic",
                        x0: cx * cell_vw,
                        x1: (cx + rw) * cell_vw - 1,
                        z0: cz * cell_vd,
                        z1: (cz + rd) * cell_vd - 1,
                    });
                }
            }
        }

        // Draw abstract grid to VoxelGrid
        let wall_max_y = height - 2;
        for cz in 0..cell_d {
            for cx in 0..cell_w {
                let cell = abstract_grid.get(cx, cz).unwrap();
                let v_x0 = cx * cell_vw;
                let v_z0 = cz * cell_vd;
                let v_x1 = v_x0 + cell_vw - 1;
                let v_z1 = v_z0 + cell_vd - 1;

                // Floors and Ceilings
                for vx in v_x0..=v_x1 {
                    for vz in v_z0..=v_z1 {
                        grid.set(vx, 0, vz, VOXEL_FLOOR);
                        grid.set(vx, height - 1, vz, VOXEL_CEILING);
                    }
                }
                
                // South Wall (z1)
                if cz < cell_d - 1 {
                    let same_room = room_grid[cz * cell_w + cx] == room_grid[(cz + 1) * cell_w + cx] 
                        && room_grid[cz * cell_w + cx].is_some();
                    
                    if !same_room {
                        let width_choice = rng.random_range(0..100);
                        let hole_units = if width_choice < 15 { 1.2 } else if width_choice < 85 { 2.0 } else { 3.5 };
                        let hole_w = (hole_units / config.voxel_scale) as usize;
                        let min_offset = (0.5 / config.voxel_scale) as usize;
                        let max_offset = cell_vw.saturating_sub(hole_w + min_offset);
                        let offset = if max_offset > min_offset { rng.random_range(min_offset..max_offset) } else { min_offset };
                        let hole_x0 = v_x0 + offset;
                        let hole_x1 = hole_x0 + hole_w;

                        for vx in v_x0..=v_x1 {
                            if cell.walls[2] || vx < hole_x0 || vx >= hole_x1 {
                                for y in 1..=wall_max_y { grid.set(vx, y, v_z1, VOXEL_WALL); }
                            }
                        }
                    }
                } else if cz == cell_d - 1 {
                    // Chunk boundary (force holes for connectivity)
                    let hole_w = (2.0 / config.voxel_scale) as usize;
                    let hole_x0 = v_x0 + cell_vw / 2 - hole_w / 2;
                    for vx in v_x0..=v_x1 {
                        let is_hole = vx >= hole_x0 && vx < hole_x0 + hole_w;
                        if !is_hole {
                            for y in 1..=wall_max_y { grid.set(vx, y, v_z1, VOXEL_WALL); }
                        }
                    }
                }

                // East Wall (x1)
                if cx < cell_w - 1 {
                    let same_room = room_grid[cz * cell_w + cx] == room_grid[cz * cell_w + cx + 1] 
                        && room_grid[cz * cell_w + cx].is_some();
                    
                    if !same_room {
                        let width_choice = rng.random_range(0..100);
                        let hole_units = if width_choice < 15 { 1.2 } else if width_choice < 85 { 2.0 } else { 3.5 };
                        let hole_w = (hole_units / config.voxel_scale) as usize;
                        let min_offset = (0.5 / config.voxel_scale) as usize;
                        let max_offset = cell_vd.saturating_sub(hole_w + min_offset);
                        let offset = if max_offset > min_offset { rng.random_range(min_offset..max_offset) } else { min_offset };
                        let hole_z0 = v_z0 + offset;
                        let hole_z1 = hole_z0 + hole_w;

                        for vz in v_z0..=v_z1 {
                            if cell.walls[1] || vz < hole_z0 || vz >= hole_z1 {
                                for y in 1..=wall_max_y { grid.set(v_x1, y, vz, VOXEL_WALL); }
                            }
                        }
                    }
                } else if cx == cell_w - 1 {
                    let hole_w = (2.0 / config.voxel_scale) as usize;
                    let hole_z0 = v_z0 + cell_vd / 2 - hole_w / 2;
                    for vz in v_z0..=v_z1 {
                        let is_hole = vz >= hole_z0 && vz < hole_z0 + hole_w;
                        if !is_hole {
                            for y in 1..=wall_max_y { grid.set(v_x1, y, vz, VOXEL_WALL); }
                        }
                    }
                }
            }
        }
        
        // Chunk Boundaries (North and West)
        for cz in 0..cell_d {
            let cell = abstract_grid.get(0, cz).unwrap();
            let v_z0 = cz * cell_vd;
            let v_z1 = v_z0 + cell_vd - 1;
            let hole_w = (2.0 / config.voxel_scale) as usize;
            let hole_z0 = v_z0 + cell_vd / 2 - hole_w / 2;
            for vz in v_z0..=v_z1 {
                let is_hole = vz >= hole_z0 && vz < hole_z0 + hole_w;
                if !is_hole {
                    for y in 1..=wall_max_y { grid.set(0, y, vz, VOXEL_WALL); }
                }
            }
        }
        for cx in 0..cell_w {
            let cell = abstract_grid.get(cx, 0).unwrap();
            let v_x0 = cx * cell_vw;
            let v_x1 = v_x0 + cell_vw - 1;
            let hole_w = (2.0 / config.voxel_scale) as usize;
            let hole_x0 = v_x0 + cell_vw / 2 - hole_w / 2;
            for vx in v_x0..=v_x1 {
                let is_hole = vx >= hole_x0 && vx < hole_x0 + hole_w;
                if !is_hole {
                    for y in 1..=wall_max_y { grid.set(vx, y, 0, VOXEL_WALL); }
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
            RoomStamp::stairway(),
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
            if !possible_stamps.is_empty() {
                let mut stamp = possible_stamps[rng.random_range(0..possible_stamps.len())];
                
                // Low-frequency stairs chance override
                if rng.random_bool(0.15 * (config.tuning.stairs_density as f64)) {
                    stamp = &stamps[8]; // stairway
                }
                
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
            
            // Basic room lighting
            let cx = (room.x0 + room.x1) / 2;
            let mut z = room.z0 + 8;
            while z <= room.z1.saturating_sub(8) {
                grid.set(cx, height - 1, z, VOXEL_LIGHT);
                z += 12;
            }
        }
        
        // Basic corridor lighting
        for cz in (8..depth).step_by(16) {
            for cx in (8..width).step_by(16) {
                if grid.get(cx, height - 1, cz) == VOXEL_CEILING && grid.get(cx, 1, cz) == VOXEL_AIR {
                    grid.set(cx, height - 1, cz, VOXEL_LIGHT);
                }
            }
        }


        // ==========================================
        // LIGHTING PROPAGATION (BFS Flood fill only)
        // ==========================================
        crate::domain::use_cases::calculate_lighting::calculate_voxel_lighting(&mut grid);

        let elapsed_micros = self.telemetry.now_micros().saturating_sub(start_micros);
        self.telemetry.log(&format!(
            "[TELEMETRY] execute completed. Duration={}us, OutputVoxelGridSize={}x{}x{}",
            elapsed_micros, grid.width(), grid.height(), grid.depth()
        ));

        grid
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
                    if grid.get(vx, 1, vz) != VOXEL_AIR && grid.get(vx, 1, vz) != VOXEL_FLOOR {
                        return false;
                    }
                }
            }
        }
        
        true
    }

    fn place_stamp(&self, grid: &mut VoxelGrid, stamp: &RoomStamp, start_x: usize, start_z: usize) {
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
                } else if val == 4 { // Raised floor
                    grid.set(vx, 1, vz, VOXEL_FLOOR);
                    grid.set(vx, 2, vz, VOXEL_FLOOR);
                } else if val == 5 { // Step 1
                    grid.set(vx, 1, vz, VOXEL_FLOOR);
                } else if val == 6 { // Step 2
                    // Just floor, already set
                }
            }
        }
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
        let config = GeneratorConfig::high_spec().with_level(1);
        let grid = generator.execute(Position::new(0.0, 0.0), 42, config);

        assert_eq!(grid.width(), 200);
        assert_eq!(grid.height(), 30);
        assert_eq!(grid.depth(), 200);
    }

    #[test]
    fn test_chunk_seeding_varies_output() {
        let noise = MockNoiseProvider { value: 0.0 };
        let generator = GenerateChunkArchitectureUseCase::new(&noise);
        let config = GeneratorConfig::high_spec().with_level(0);
        
        let grid1 = generator.execute(Position::new(10.0, 10.0), 42, config);
        let grid2 = generator.execute(Position::new(10.0, 10.0), 43, config);
        
        let mut diff_count = 0;
        for z in 0..grid1.depth() {
            for x in 0..grid1.width() {
                if grid1.get(x, 1, z) != grid2.get(x, 1, z) {
                    diff_count += 1;
                }
            }
        }
        
        assert!(diff_count > 100, "Expected significant voxel differences between seeds");
    }

    #[test]
    fn test_junction_density_increases_connections() {
        use crate::domain::use_cases::generate_maze::{GrowingTreeGenerator, MazeGenerator};
        use crate::domain::entities::grid::Grid;
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        
        let mut grid_low = Grid::new(20, 20);
        let mut rng1 = StdRng::seed_from_u64(42);
        let gen_low = GrowingTreeGenerator { junction_density: 0.5 };
        gen_low.generate(&mut grid_low, &mut rng1);
        
        let mut grid_high = Grid::new(20, 20);
        let mut rng2 = StdRng::seed_from_u64(42);
        let gen_high = GrowingTreeGenerator { junction_density: 2.0 };
        gen_high.generate(&mut grid_high, &mut rng2);
        
        let mut open_low = 0;
        let mut open_high = 0;
        
        for z in 0..20 {
            for x in 0..20 {
                if let Some(cell) = grid_low.get(x, z) {
                    open_low += cell.walls.iter().filter(|&&w| !w).count();
                }
                if let Some(cell) = grid_high.get(x, z) {
                    open_high += cell.walls.iter().filter(|&&w| !w).count();
                }
            }
        }
        
        assert!(open_high > open_low, "Expected higher junction density to produce more open walls ({} vs {})", open_high, open_low);
    }

    #[test]
    fn test_stairs_density_scaling() {
        let noise = MockNoiseProvider { value: 0.0 };
        let generator = GenerateChunkArchitectureUseCase::new(&noise);
        
        let mut config_zero = GeneratorConfig::high_spec().with_level(0);
        config_zero.tuning.stairs_density = 0.0;
        
        let mut config_high = GeneratorConfig::high_spec().with_level(0);
        config_high.tuning.stairs_density = 3.0;
        
        let mut stairs_zero_count = 0;
        let mut stairs_high_count = 0;
        
        for i in 0..5 {
            let grid_zero = generator.execute(Position::new(i as f32, 0.0), 42, config_zero);
            for z in 0..grid_zero.depth() {
                for x in 0..grid_zero.width() {
                    if grid_zero.get(x, 1, z) == VOXEL_FLOOR && grid_zero.get(x, 2, z) == VOXEL_FLOOR {
                        stairs_zero_count += 1;
                    }
                }
            }
            
            let grid_high = generator.execute(Position::new(i as f32, 0.0), 42, config_high);
            for z in 0..grid_high.depth() {
                for x in 0..grid_high.width() {
                    if grid_high.get(x, 1, z) == VOXEL_FLOOR && grid_high.get(x, 2, z) == VOXEL_FLOOR {
                        stairs_high_count += 1;
                    }
                }
            }
        }
        
        assert_eq!(stairs_zero_count, 0, "Expected zero stairs with density 0.0");
        assert!(stairs_high_count > 0, "Expected stairs to generate with density 3.0");
    }
}
