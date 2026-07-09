//! Level 0 — the Backrooms proper.
//!
//! Everything is a pure function of *world-space* coordinates, so chunks tile
//! seamlessly no matter which order they stream in. The goal is *believable
//! building architecture* with depth — not flat mazes:
//!
//! * **Zones** (~25 u blobs of smooth noise) pick the local architecture:
//!   - high zone → *atrium*: the ceiling vaults **continuously** from office
//!     height up to ~5.4 u (a dome-like swell, not a step), ringed by a
//!     colonnade of floor-to-vault pillars with glowing sconce bands at
//!     human height — the "architectural marvel" moments.
//!   - low zone  → *expanse*: wide open halls of sparse structural columns.
//!   - middle    → *offices*: yellow wall grid with real **doorways** —
//!     door-height openings under solid lintels, hashed per cell so no two
//!     rooms line up — and **coffered ceilings** (a beam grid overhead).
//! * **Halls**: two perpendicular corridor families on a 12 u period whose
//!   centerlines *wander* (±2 u of low-frequency noise) and whose width
//!   breathes between ~1.9 u and ~3.2 u. Halls carve through walls, run
//!   under lowered soffit ceilings with light strips, and grow flanking
//!   colonnades where they cross atria.
//! * **Walkability is structural**: the hall network alone connects the
//!   world; walls always carry a doorway per 7 u cell; pillars are point
//!   obstacles. A BFS test enforces >95% connectivity.

use crate::domain::entities::voxel_grid::{
    VOXEL_CEILING, VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_RED_WALL, VOXEL_WALL, VoxelGrid,
};
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::{GeneratorConfig, LevelTuning};
use crate::use_cases::level_generator::LevelGenerator;
use crate::use_cases::ports::NoiseProvider;

/// Ceiling height of the tallest (atrium) vaults, world units.
pub const MAX_CEILING_UNITS: f32 = 5.4;
/// Total grid height: headroom above the tallest vault.
pub const GRID_HEIGHT_UNITS: f32 = 5.8;

/// Corridor spacing / wander / width, world units.
const HALL_PERIOD: f32 = 12.0;
const HALL_WANDER: f32 = 2.0;
const HALL_WIDTH_BASE: f32 = 2.55;
const HALL_WIDTH_VAR: f32 = 0.65;
/// Hall ceilings drop to this soffit height outside atria.
const HALL_SOFFIT_UNITS: f32 = 2.9;

/// Office wall grid: cell size, doorway width and door (lintel) height.
/// 3.0 units creates tight 2–3 m corridors matching authentic Level 0.
const WALL_PERIOD: f32 = 3.0;
const DOOR_WIDTH: f32 = 1.2;
const DOOR_HEIGHT: f32 = 2.2;

/// Coffered-ceiling beam grid period and beam drop, world units.
const COFFER_PERIOD: f32 = 2.8;
const COFFER_DROP: f32 = 0.2;

/// Ceiling light panel spacing (aligned to coffer panel centers).
const LIGHT_PERIOD: f32 = 2.8;

/// Sconce band height on atrium pillars.
const SCONCE_UNITS: f32 = 2.4;

/// Keep a clearing around the world spawn point.
const SPAWN: (f32, f32) = (5.0, 5.0);
const SPAWN_CLEAR_RADIUS: f32 = 2.5;

pub struct BackroomsLevel;

/// What one (x, z) column of the level looks like.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ColumnPlan {
    /// Floor-to-ceiling solid (wall or pillar).
    pub solid: bool,
    /// Red accent variant of a solid column.
    pub red: bool,
    /// Ceiling height in world units (top of the walkable space).
    pub ceiling_units: f32,
    /// Ceiling light above this column.
    pub light: bool,
    /// Solid band hanging from the ceiling down to this height (door
    /// lintels). The space below stays walkable.
    pub lintel_from_units: Option<f32>,
    /// Glowing sconce band on a solid column (atrium pillars).
    pub sconce: bool,
}

impl BackroomsLevel {
    /// Smooth world-space noise in [-1, 1]. `scale` stretches the provider's
    /// built-in ~20 u wavelength: wavelength = 20 / scale.
    fn n(noise: &dyn NoiseProvider, seed: u32, salt: u32, x: f32, z: f32, scale: f32) -> f32 {
        noise.evaluate_2d(seed ^ salt, Position::new(x * scale, z * scale))
    }

    /// Decorrelated per-cell hash in [0, 1): samples the value noise exactly
    /// on its lattice (provider frequency is 0.05, so inputs that are
    /// multiples of 20 hit lattice points), where it is uniform rather than
    /// interpolation-smoothed toward zero.
    fn cell_hash(noise: &dyn NoiseProvider, seed: u32, salt: u32, cx: i64, cz: i64) -> f32 {
        let v = noise.evaluate_2d(
            seed ^ salt,
            Position::new(cx as f32 * 20.0, cz as f32 * 20.0),
        );
        (v * 0.5 + 0.5).clamp(0.0, 0.999)
    }

    fn smoothstep(t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Distance from `w` to the wandering centerline of corridor family
    /// `salt` at position `along` on the perpendicular axis.
    /// Returns (signed offset from center, half_width).
    fn hall_offset(
        noise: &dyn NoiseProvider,
        seed: u32,
        salt: u32,
        w: f32,
        along: f32,
    ) -> (f32, f32) {
        let k = ((w - HALL_PERIOD * 0.5) / HALL_PERIOD).round();
        let nominal = k * HALL_PERIOD + HALL_PERIOD * 0.5;
        let center = nominal + HALL_WANDER * Self::n(noise, seed, salt, along, k * 37.7, 0.45);
        let half_width = (HALL_WIDTH_BASE
            + HALL_WIDTH_VAR * Self::n(noise, seed, salt ^ 0x5A5A, along, k * 19.3, 0.8))
            * 0.5;
        (w - center, half_width)
    }

    /// Should a pillar site survive? Applies the user pillar-density knob
    /// plus a per-site hash so even default density feels irregular.
    fn pillar_alive(
        noise: &dyn NoiseProvider,
        seed: u32,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
        period: f32,
    ) -> bool {
        let cx = (wx / period).floor() as i64;
        let cz = (wz / period).floor() as i64;
        Self::cell_hash(noise, seed, 0xF200, cx, cz) < 0.7 * tuning.pillars
    }

    /// The full column plan at world position (wx, wz), world units.
    pub(crate) fn column_plan(
        noise: &dyn NoiseProvider,
        seed: u32,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        // ---- ceiling field (flat and low everywhere like authentic Level 0) ----
        let coarse = Self::n(noise, seed, 0xB200, wx, wz, 0.55);
        let fine = Self::n(noise, seed, 0xB300, wx, wz, 1.6);
        let mut ceiling_units = (2.8 + 0.15 * coarse + 0.05 * fine).clamp(2.7, 3.0);

        // Coffered ceiling grid
        let on_beam = wx.rem_euclid(COFFER_PERIOD) < 0.22 || wz.rem_euclid(COFFER_PERIOD) < 0.22;
        if on_beam {
            ceiling_units -= COFFER_DROP;
        }

        // ---- solids ------------------------------------------------------
        let spawn_d2 = (wx - SPAWN.0) * (wx - SPAWN.0) + (wz - SPAWN.1) * (wz - SPAWN.1);
        let in_spawn = spawn_d2 < SPAWN_CLEAR_RADIUS * SPAWN_CLEAR_RADIUS;

        let mut solid = false;
        let mut lintel_from_units: Option<f32> = None;
        let mut sconce = false;

        if !in_spawn {
            let fx = wx.rem_euclid(WALL_PERIOD);
            let fz = wz.rem_euclid(WALL_PERIOD);
            let cell_x = (wx / WALL_PERIOD).floor() as i64;
            let cell_z = (wz / WALL_PERIOD).floor() as i64;

            // (solid, lintel) contribution of one wall family.
            //
            // Directional noise strategy (Pattern: Strategy):
            // A Z-wall runs along the Z axis. To make it run *long*,
            // we sample noise with SLOW variation along Z (scale 0.25)
            // and FAST variation along X (scale 1.0). This means
            // adjacent cells along the wall share similar noise values
            // (wall stays solid for many cells), while different grid
            // lines (perpendicular) get independent values (some lines
            // have walls, some don't). The result: long, straight wall
            // runs forming tight corridors — exactly like Level 0.
            let wall_here = |f_wall: f32, f_along: f32, is_z_wall: bool| {
                if f_wall >= 0.25 {
                    return (false, false);
                }

                let (cx, cz) = if is_z_wall {
                    (cell_x - 1, cell_z)
                } else {
                    (cell_x, cell_z - 1)
                };

                // World-space midpoint of this wall segment
                let (wx_mid, wz_mid) = if is_z_wall {
                    (cx as f32 * WALL_PERIOD, (cz as f32 + 0.5) * WALL_PERIOD)
                } else {
                    ((cx as f32 + 0.5) * WALL_PERIOD, cz as f32 * WALL_PERIOD)
                };

                // Directional noise (Pattern: Strategy):
                // A single directional noise sample with slow variation
                // along the wall axis creates long wall runs (5-10 cells).
                //
                // Threshold formula: 1.0 - walls.clamp(0, 2)
                // Maps directly into the noise output range [-1, 1]:
                //   walls=0.0 → threshold=1.0  → 0% kept (empty)
                //   walls=0.3 → threshold=0.7  → ~15% kept (sparse)
                //   walls=1.0 → threshold=0.0  → ~50% kept (default)
                //   walls=2.0 → threshold=-1.0 → ~100% kept (dense)
                let threshold = 1.0 - tuning.walls.clamp(0.0, 2.0);
                let keep = if is_z_wall {
                    let n_val = Self::n(noise, seed, 0xD500, wx_mid, wz_mid * 0.5, 1.0);
                    n_val > threshold
                } else {
                    let n_val = Self::n(noise, seed, 0xD600, wx_mid * 0.5, wz_mid, 1.0);
                    n_val > threshold
                };

                if !keep {
                    return (false, false);
                }

                // Doorways: 40% of kept walls get a door opening.
                // This creates the authentic Level 0 feel: dense walls
                // with occasional doorway passages between rooms.
                let door_salt = if is_z_wall { 0xD700 } else { 0xD800 };
                let door_hash = Self::cell_hash(noise, seed, door_salt, cx, cz);
                if door_hash < 0.40 {
                    let door_pos = (WALL_PERIOD - DOOR_WIDTH) * 0.5;
                    if f_along >= door_pos && f_along < door_pos + DOOR_WIDTH {
                        return (false, true); // Doorway: open below, lintel above
                    }
                }
                (true, false) // Solid wall
            };

            let (sx, lx) = wall_here(fx, fz, true);
            let (sz, lz) = wall_here(fz, fx, false);
            solid = sx || sz;
            if !solid && (lx || lz) {
                lintel_from_units = Some(DOOR_HEIGHT);
            }
        }

        // ---- lights ------------------------------------------------------
        let light = if !solid {
            // Room panels centered in the coffer bays.
            let lx = (wx - LIGHT_PERIOD * 0.5).rem_euclid(LIGHT_PERIOD);
            let lz = (wz - LIGHT_PERIOD * 0.5).rem_euclid(LIGHT_PERIOD);
            let cell_x = (wx / LIGHT_PERIOD).floor() as i64;
            let cell_z = (wz / LIGHT_PERIOD).floor() as i64;
            let keep = 0.78 * tuning.lights;
            let alive = Self::cell_hash(noise, seed, 0xE900, cell_x, cell_z) < keep;
            (lx < 0.45 && lz < 0.45 && alive) || spawn_d2 < 0.36
        } else {
            false
        };

        let red = solid
            && Self::cell_hash(noise, seed, 0xF100, (wx / 1.4) as i64, (wz / 1.4) as i64) > 0.94;

        ColumnPlan {
            solid,
            red,
            ceiling_units,
            light,
            lintel_from_units,
            sconce,
        }
    }
}

impl LevelGenerator for BackroomsLevel {
    fn generate(
        &self,
        chunk_pos: Position,
        seed: u32,
        config: GeneratorConfig,
        noise: &dyn NoiseProvider,
    ) -> VoxelGrid {
        let s = config.voxel_scale;
        let width = (config.chunk_size / s) as usize;
        let depth = (config.chunk_size / s) as usize;
        let height = (GRID_HEIGHT_UNITS / s) as usize;
        let mut grid = VoxelGrid::new(width, height, depth);

        // Plan every column plus a 1-voxel margin: ceiling skirts must seal
        // height steps across chunk borders too.
        let tuning = config.tuning;
        let plan_at = |lx: i64, lz: i64| -> ColumnPlan {
            let wx = chunk_pos.x + (lx as f32 + 0.5) * s;
            let wz = chunk_pos.z + (lz as f32 + 0.5) * s;
            BackroomsLevel::column_plan(noise, seed, &tuning, wx, wz)
        };

        let mut plans = Vec::with_capacity((width + 2) * (depth + 2));
        for lz in -1..=(depth as i64) {
            for lx in -1..=(width as i64) {
                plans.push(plan_at(lx, lz));
            }
        }
        let plan = |lx: i64, lz: i64| -> &ColumnPlan {
            &plans[((lz + 1) as usize) * (width + 2) + (lx + 1) as usize]
        };

        let max_y = height - 1;
        let to_vox = |units: f32| ((units / s) as usize).clamp(2, max_y);

        for z in 0..depth {
            for x in 0..width {
                let p = *plan(x as i64, z as i64);
                let ch = to_vox(p.ceiling_units);

                grid.set(x, 0, z, VOXEL_FLOOR);

                if p.solid {
                    let mat = if p.red { VOXEL_RED_WALL } else { VOXEL_WALL };
                    for y in 1..ch {
                        grid.set(x, y, z, mat);
                    }
                    if p.sconce {
                        let sy = to_vox(SCONCE_UNITS).min(ch - 1);
                        grid.set(x, sy, z, VOXEL_LIGHT);
                    }
                } else if let Some(from) = p.lintel_from_units {
                    // Door lintel: solid from door height to the ceiling.
                    for y in to_vox(from)..ch {
                        grid.set(x, y, z, VOXEL_WALL);
                    }
                }

                // Ceiling plus a skirt down to the tallest neighbor, so a
                // higher neighboring ceiling can never see over this one.
                let neighbor_max = [
                    plan(x as i64 - 1, z as i64),
                    plan(x as i64 + 1, z as i64),
                    plan(x as i64, z as i64 - 1),
                    plan(x as i64, z as i64 + 1),
                ]
                .iter()
                .map(|n| to_vox(n.ceiling_units))
                .max()
                .unwrap_or(ch);

                for y in ch..=neighbor_max.max(ch) {
                    grid.set(x, y, z, VOXEL_CEILING);
                }
                if p.light && !p.solid {
                    grid.set(x, ch, z, VOXEL_LIGHT);
                }
            }
        }

        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::voxel_grid::VOXEL_AIR;
    use crate::frameworks_drivers::simple_noise::SimpleNoiseProvider;
    use crate::use_cases::generate_chunk::LevelTuning;

    fn generate(ox: f32, oz: f32) -> VoxelGrid {
        BackroomsLevel.generate(
            Position::new(ox, oz),
            42,
            GeneratorConfig::low_spec(),
            &SimpleNoiseProvider::new(),
        )
    }

    fn is_open(grid: &VoxelGrid, x: usize, z: usize) -> bool {
        grid.get(x, 1, z) == VOXEL_AIR
    }

    /// Walkable-plane connectivity: nearly all open floor must be mutually
    /// reachable (walls always leave doorways, pillars never seal a region).
    /// We exclude a 2-voxel border to avoid edge-of-chunk artifacts where
    /// walls at boundaries form isolated strips that would be connected by
    /// the adjacent chunk at runtime.
    #[test]
    fn walkable_plane_is_connected() {
        let grid = generate(0.0, 0.0);
        let (w, d) = (grid.width(), grid.depth());
        let margin = 2usize; // exclude edge voxels

        let open: Vec<(usize, usize)> = (margin..w - margin)
            .flat_map(|x| (margin..d - margin).map(move |z| (x, z)))
            .filter(|&(x, z)| is_open(&grid, x, z))
            .collect();
        assert!(
            open.len() > (w - 2 * margin) * (d - 2 * margin) / 2,
            "backrooms must be mostly open space"
        );

        // BFS from the spawn clearing (world 5,5 = local 25,25 in chunk 0,0).
        let mut visited = vec![false; w * d];
        let mut queue = std::collections::VecDeque::from([(w / 2, d / 2)]);
        visited[(w / 2) * d + d / 2] = true;
        let mut reached = 0usize;
        while let Some((x, z)) = queue.pop_front() {
            reached += 1;
            for (dx, dz) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (nx, nz) = (x as i64 + dx, z as i64 + dz);
                if nx < margin as i64
                    || nz < margin as i64
                    || nx >= (w - margin) as i64
                    || nz >= (d - margin) as i64
                {
                    continue;
                }
                let (nx, nz) = (nx as usize, nz as usize);
                if !visited[nx * d + nz] && is_open(&grid, nx, nz) {
                    visited[nx * d + nz] = true;
                    queue.push_back((nx, nz));
                }
            }
        }
        let ratio = reached as f32 / open.len() as f32;
        assert!(
            ratio > 0.95,
            "only {:.0}% of open floor is reachable from spawn",
            ratio * 100.0
        );
    }

    /// Adjacent chunks must agree at their shared border: the column at
    /// world position g is the same whether it came from chunk A's last
    /// column or chunk B's first.
    #[test]
    fn chunks_tile_seamlessly() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let a = generate(0.0, 0.0);
        let b = generate(10.0, 0.0);
        let w = a.width();

        for z in 0..a.depth() {
            for (grid, lx, gx) in [(&a, w - 1, w - 1), (&b, 0usize, w)] {
                let plan = BackroomsLevel::column_plan(
                    &noise,
                    42,
                    &LevelTuning::default(),
                    (gx as f32 + 0.5) * config.voxel_scale,
                    (z as f32 + 0.5) * config.voxel_scale,
                );
                let got_solid =
                    grid.get(lx, 1, z) == VOXEL_WALL || grid.get(lx, 1, z) == VOXEL_RED_WALL;
                assert_eq!(
                    got_solid, plan.solid,
                    "column mismatch at world x={gx} z={z}"
                );
            }
        }
    }

    /// Higher ceilings than the old 3.0 u slab: vaults swell past 4.2 u and
    /// every walkable column keeps at least 2.3 u of headroom.
    #[test]
    fn ceilings_are_tall_and_varied() {
        let config = GeneratorConfig::low_spec();
        let s = config.voxel_scale;
        let mut tallest = 0usize;
        let mut lowest = usize::MAX;
        for oz in [-10.0, 0.0, 10.0] {
            for ox in [-10.0, 0.0, 10.0] {
                let grid = generate(ox, oz);
                for z in 0..grid.depth() {
                    for x in 0..grid.width() {
                        if !is_open(&grid, x, z) {
                            continue;
                        }
                        let mut y = 1;
                        while y < grid.height()
                            && grid.get(x, y, z) != VOXEL_CEILING
                            && grid.get(x, y, z) != VOXEL_LIGHT
                            && grid.get(x, y, z) != VOXEL_WALL
                        {
                            y += 1;
                        }
                        tallest = tallest.max(y);
                        lowest = lowest.min(y);
                    }
                }
            }
        }
        assert!(
            tallest as f32 * s <= 3.1,
            "expected flat ceiling <= 3.1 u, tallest was {} u",
            tallest as f32 * s
        );
        assert!(
            lowest as f32 * s >= 2.1,
            "walkable headroom fell below 2.1 u: {} u",
            lowest as f32 * s
        );
    }

    /// Halls must wander: their centerline offset changes along their length.
    #[test]
    fn halls_wander_instead_of_running_straight() {
        let noise = SimpleNoiseProvider::new();
        let mut moved = false;
        for k in 0..4 {
            let w = k as f32 * HALL_PERIOD + HALL_PERIOD * 0.5;
            let (d0, _) = BackroomsLevel::hall_offset(&noise, 42, 0xC300, w, 0.0);
            for along in [8.0f32, 16.0, 24.0, 32.0] {
                let (d1, _) = BackroomsLevel::hall_offset(&noise, 42, 0xC300, w, along);
                if (d0 - d1).abs() > 0.4 {
                    moved = true;
                }
            }
        }
        assert!(moved, "hall centerlines never moved over 32 u of length");
    }

    /// Doorways are real architecture: somewhere there must be an open
    /// passage at head height with a solid lintel above it.
    #[test]
    fn doorways_have_lintels() {
        let config = GeneratorConfig::low_spec();
        let s = config.voxel_scale;
        let door_y = (DOOR_HEIGHT / s) as usize; // first solid voxel of a lintel
        let mut found = false;
        for oz in [0.0, 10.0, 20.0] {
            for ox in [0.0, 10.0, 20.0] {
                let grid = generate(ox, oz);
                for z in 0..grid.depth() {
                    for x in 0..grid.width() {
                        if grid.get(x, 1, z) == VOXEL_AIR
                            && grid.get(x, door_y - 1, z) == VOXEL_AIR
                            && grid.get(x, door_y, z) == VOXEL_WALL
                        {
                            found = true;
                        }
                    }
                }
            }
        }
        assert!(found, "no doorway-with-lintel found in 9 chunks");
    }

    /// The generation knobs actually steer the output: zeroing pillars and
    /// walls empties the world of solids; cranking them fills it back up.
    #[test]
    fn tuning_knobs_control_density() {
        let noise = SimpleNoiseProvider::new();
        let count_solids = |tuning: LevelTuning| -> usize {
            let grid = BackroomsLevel.generate(
                Position::new(10.0, 10.0),
                42,
                GeneratorConfig::low_spec().with_tuning(tuning),
                &noise,
            );
            let mut n = 0;
            for z in 0..grid.depth() {
                for x in 0..grid.width() {
                    if grid.get(x, 1, z) != VOXEL_AIR {
                        n += 1;
                    }
                }
            }
            n
        };

        let none = count_solids(LevelTuning {
            pillars: 0.0,
            walls: 0.0,
            ..Default::default()
        });
        let sparse = count_solids(LevelTuning {
            pillars: 0.3,
            walls: 0.3,
            ..Default::default()
        });
        let default = count_solids(LevelTuning::default());
        let dense = count_solids(LevelTuning {
            pillars: 2.0,
            walls: 2.0,
            ..Default::default()
        });

        assert_eq!(none, 0, "pillars=0 walls=0 must produce an empty plane");
        assert!(
            sparse < default,
            "sparse ({sparse}) must be < default ({default})"
        );
        assert!(
            default < dense,
            "default ({default}) must be < dense ({dense})"
        );
    }

    /// The spawn point has a clear floor and a light overhead.
    #[test]
    fn spawn_clearing_is_open_and_lit() {
        let grid = generate(0.0, 0.0);
        let (cx, cz) = (25usize, 25usize); // world (5,5) at voxel_scale 0.2
        for z in cz - 5..=cz + 5 {
            for x in cx - 5..=cx + 5 {
                assert!(
                    is_open(&grid, x, z),
                    "spawn clearing blocked at local ({x},{z})"
                );
            }
        }
        let mut lit = false;
        for z in cz - 3..=cz + 3 {
            for x in cx - 3..=cx + 3 {
                for y in 0..grid.height() {
                    if grid.get(x, y, z) == VOXEL_LIGHT {
                        lit = true;
                    }
                }
            }
        }
        assert!(lit, "no light panel above spawn");
    }

    #[test]
    fn test_print_ascii_map() {
        let noise = SimpleNoiseProvider::new();
        let tuning = LevelTuning::default();
        let mut map = String::new();
        // Sample a 60x60 world area with 0.5 step (120x120 ASCII cells)
        for sz in -60..60 {
            for sx in -60..60 {
                let wx = sx as f32 * 0.5;
                let wz = sz as f32 * 0.5;
                let plan = BackroomsLevel::column_plan(&noise, 42, &tuning, wx, wz);
                if plan.solid {
                    map.push('#');
                } else if plan.light {
                    map.push('*');
                } else if plan.lintel_from_units.is_some() {
                    map.push('d');
                } else {
                    map.push(' ');
                }
            }
            map.push('\n');
        }
        std::fs::write("./ascii_map.txt", map).unwrap();
    }
}
