//! Level 0 — the Backrooms proper.
//!
//! Everything is a pure function of *world-space* coordinates, so chunks tile
//! seamlessly no matter which order they stream in. The goal is *believable
//! building architecture* with depth — not flat mazes:
//!
//! * **Ceiling regions** use broad smooth fields instead of a low default
//!   slab: regular dropped ceilings are 3.2--3.6 u, open expanses 3.8--4.4 u,
//!   vaults 4.5--5.4 u, and compression zones are rare deliberate contrast.
//! * **Halls**: one dominant cross-region route is wide enough to read as
//!   circulation through a vast building, with at most two interior branches.
//!   Its edge walls persist in long runs and open into large rooms or fabric,
//!   not a repeated doorway grid.
//! * **Fabric**: the default fill *is* the labyrinth — an irregular warren
//!   of rooms on a hidden 7.2 u lattice, chained through hashed doorways
//!   (binary-tree rule guarantees global connectivity chunk-locally) and
//!   merged by porosity-driven wall dropout so no grid is ever readable.
//!   Open expanses are the exception that punctuates it, never the default.

use crate::domain::entities::anomaly::{
    AnomalyInstance, AnomalyKind, RealitySnapshot, WorldBounds,
};
use crate::domain::entities::architecture::{
    AssemblyInstance, CeilingLanguage, CirculationSpine, RegionPlan, SpaceProgram,
    StructuralSystem, StructuralSystemInstance,
};
use crate::domain::entities::voxel_grid::{
    VOXEL_CEILING, VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_WALL, VoxelGrid,
};
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::{GeneratorConfig, LevelTuning};
use crate::use_cases::level_generator::LevelGenerator;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::region_plan::{
    PLAN_WALL_T, REGION_SIZE, generate_region_plan, region_index, spawn_point,
};

/// Ceiling height of the tallest (atrium) vaults, world units.
pub const MAX_CEILING_UNITS: f32 = 5.4;
/// Total grid height: headroom above the tallest vault.
pub const GRID_HEIGHT_UNITS: f32 = 5.8;

const DOOR_WIDTH: f32 = 1.2;
const DOOR_HEIGHT: f32 = 2.2;

/// The default fabric room lattice. Rooms are chained through hashed
/// doorways and merged by wall dropout, so the cell size never reads as a
/// grid from inside — it is the scale of the labyrinth, not its shape.
const FABRIC_CELL: f32 = 7.2;

/// Column grid spacing inside expanses.
const EXPANSE_COLUMN_PERIOD: f32 = 7.2;

/// Ceiling light panel spacing. The coffer beam grid shares this period,
/// but beams are now a renderer shading pattern (see the splat fragment
/// shader's 2.8 u grid), never stepped ceiling geometry.
const LIGHT_PERIOD: f32 = 2.8;

/// Sconce band height on atrium pillars.
const SCONCE_UNITS: f32 = 2.4;

/// Runtime direct lights sit below their visible ceiling panel. Tall atria
/// need a longer pendant drop so the light reaches occupied space and casts
/// useful column shadows instead of flattening against the vault.
fn runtime_light_height(ceiling_units: f32, is_atrium: bool) -> f32 {
    let pendant_drop = if is_atrium { 1.6 } else { 1.1 };
    (ceiling_units - pendant_drop).max(2.4)
}

/// Within this radius of the spawn point the main corridor keeps both of its
/// edge walls fully intact: the first thing the player reads is an
/// unambiguous walled corridor, not a dissolved edge into open fabric. The
/// world only starts opening up once that grammar has been established.
const SPAWN_READABLE_RADIUS: f32 = 26.0;

pub struct BackroomsLevel;

/// What one (x, z) column of the level looks like.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ColumnPlan {
    /// Whether the walkable floor slab exists under this column.
    pub floor: bool,
    /// Floor-to-ceiling solid (wall or pillar).
    pub solid: bool,
    /// Ceiling height in world units (top of the walkable space).
    pub ceiling_units: f32,
    /// Ceiling light above this column.
    pub light: bool,
    /// The light above this column burns red (red-room corruption). Only
    /// ever true inside an assembly whose whole room is the anomaly.
    pub red_light: bool,
    /// Solid band hanging from the ceiling down to this height (door
    /// lintels). The space below stays walkable.
    pub lintel_from_units: Option<f32>,
    /// Glowing sconce band on a solid column (atrium pillars).
    pub sconce: bool,
}

/// The broad ceiling hierarchy that gives Level 0 scale without turning the
/// whole map into a warehouse. It is local implementation detail rather than
/// a world semantic: assemblies and corridors may override it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FabricCeilingBand {
    Compression,
    Regular,
    Expanse,
    Vault,
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

    /// Broad, low-frequency ceiling territory. The thresholds are chosen to
    /// bias the sampled world toward regular dropped ceilings while retaining
    /// meaningful regions of open volume and rare compression.
    fn fabric_ceiling_band(
        noise: &dyn NoiseProvider,
        seed: u32,
        wx: f32,
        wz: f32,
    ) -> FabricCeilingBand {
        // Regular dropped ceiling — the labyrinth fabric — is the default
        // condition of Level 0. Expanses and vaults are deliberate
        // punctuation the maze occasionally opens into, never the baseline.
        let field = Self::n(noise, seed, 0xAA10, wx, wz, 0.18);
        if field < -0.60 {
            FabricCeilingBand::Compression
        } else if field < 0.52 {
            FabricCeilingBand::Regular
        } else if field < 0.76 {
            FabricCeilingBand::Expanse
        } else {
            FabricCeilingBand::Vault
        }
    }

    /// Ceiling heights are architectural tiers, not terrain. The common
    /// bands are perfectly flat; open volumes vary per ~8 u ceiling *zone*
    /// (one suspended-grid bay run), never per column, and every height
    /// snaps to the fine voxel lattice so voxelization cannot add
    /// sub-voxel stair-stepping on top.
    fn fabric_ceiling_height(
        noise: &dyn NoiseProvider,
        seed: u32,
        wx: f32,
        wz: f32,
        band: FabricCeilingBand,
    ) -> f32 {
        const ZONE: f32 = 8.0;
        let zone_x = (wx / ZONE).floor() * ZONE + ZONE * 0.5;
        let zone_z = (wz / ZONE).floor() * ZONE + ZONE * 0.5;
        let detail = Self::n(noise, seed, 0xB300, zone_x, zone_z, 0.72);
        let (height, lo, hi) = match band {
            FabricCeilingBand::Compression => (2.6, 2.5, 2.8),
            FabricCeilingBand::Regular => (3.4, 3.2, 3.6),
            FabricCeilingBand::Expanse => (4.1 + 0.3 * detail, 3.8, 4.4),
            FabricCeilingBand::Vault => (4.95 + 0.45 * detail, 4.5, 5.4),
        };
        // Snap to the fine voxel lattice, then clamp last: 21 * 0.2 is
        // 4.2000003 in f32 and must not escape the band's range.
        ((height / 0.2).round() * 0.2).clamp(lo, hi)
    }

    /// Is (wx, wz) inside an open expanse of the fabric?
    fn in_expanse(noise: &dyn NoiseProvider, seed: u32, wx: f32, wz: f32) -> bool {
        matches!(
            Self::fabric_ceiling_band(noise, seed, wx, wz),
            FabricCeilingBand::Expanse | FabricCeilingBand::Vault
        )
    }

    /// The *fabric* column plan at world position (wx, wz): the endless,
    /// unplanned office fill between planned corridors and assemblies.
    pub(crate) fn column_plan(
        noise: &dyn NoiseProvider,
        seed: u32,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        // ---- ceiling field -------------------------------------------------
        let ceiling_band = Self::fabric_ceiling_band(noise, seed, wx, wz);
        let ceiling_units = Self::fabric_ceiling_height(noise, seed, wx, wz, ceiling_band);

        // In open and vaulted regions, partitions dissolve and only sparse
        // structural columns remain. The cheap dropped-ceiling material is
        // therefore carried through unexpectedly large space.
        let expanse = matches!(
            ceiling_band,
            FabricCeilingBand::Expanse | FabricCeilingBand::Vault
        );

        // Coffer beams are shading detail in the renderer, not geometry: a
        // 0.2 u drop at a 0.2 u voxel scale turned every beam into a
        // full-voxel step and read as noisy terrain overhead. The splat
        // shader darkens the same COFFER_PERIOD grid instead.

        // ---- solids ------------------------------------------------------
        let mut solid = false;
        let mut lintel_from_units: Option<f32> = None;
        let sconce = false;

        if expanse {
            // Sparse structural columns hold the expanse ceiling up.
            let cx = (wx / EXPANSE_COLUMN_PERIOD).floor() as i64;
            let cz = (wz / EXPANSE_COLUMN_PERIOD).floor() as i64;
            let on_site = wx.rem_euclid(EXPANSE_COLUMN_PERIOD) < 0.45
                && wz.rem_euclid(EXPANSE_COLUMN_PERIOD) < 0.45;
            if on_site && Self::cell_hash(noise, seed, 0xF200, cx, cz) < 0.7 * tuning.pillars {
                solid = true;
            }
        } else {
            // The default fabric *is* the Backrooms labyrinth: an irregular
            // warren of yellow rooms chained through hashed doorways.
            // Connectivity is a binary-tree rule — every cell knocks a
            // doorway through its west *or* its north wall — so all rooms
            // connect without a chunk ever seeing its neighbors. Smoothly
            // drifting porosity (whole walls dropped, wider thresholds,
            // second doorways) breaks the lattice read: it plays as one
            // endless wrong building, never as "a maze section".
            let t = PLAN_WALL_T;
            let fx = wx.rem_euclid(FABRIC_CELL);
            let fz = wz.rem_euclid(FABRIC_CELL);
            let (in_w, in_n) = (fx < t, fz < t);
            if in_w && in_n {
                // Junction posts anchor every corner; where the walls
                // around them have dropped they survive as column stubs.
                solid = tuning.walls > 0.0;
            } else if in_w || in_n {
                let cx = (wx / FABRIC_CELL).floor() as i64;
                let cz = (wz / FABRIC_CELL).floor() as i64;
                // 0 = tight labyrinth, 1 = broken-open suites; drifts over
                // ~180 u so density changes read as neighborhoods, not zones.
                let porosity =
                    (Self::n(noise, seed, 0x9010, wx, wz, 0.11) * 0.5 + 0.5).clamp(0.0, 1.0);
                let opens_west = Self::cell_hash(noise, seed, 0x9200, cx, cz) < 0.5;
                let (wall_salt, door_salt, opens_here) = if in_w {
                    (0x9300u32, 0x9500u32, opens_west)
                } else {
                    (0x9400u32, 0x9600u32, !opens_west)
                };
                // Whole-wall dropout merges rooms into larger wrong shapes.
                // Porosity varies along a run, so drops end ragged rather
                // than on clean cell boundaries. The walls knob scales
                // survival: 0 empties the fabric, 2 approaches a full grid.
                let survive = (0.92 - 0.42 * porosity) * tuning.walls.clamp(0.0, 1.5);
                if Self::cell_hash(noise, seed, wall_salt, cx, cz) < survive {
                    let along = if in_w { fz } else { fx };
                    // The binary-tree wall always gets its doorway; porous
                    // neighborhoods often cut a second one.
                    let extra = Self::cell_hash(noise, seed, door_salt ^ 0x1F, cx, cz)
                        < 0.18 + 0.42 * porosity;
                    if opens_here || extra {
                        let dh = Self::cell_hash(noise, seed, door_salt, cx, cz);
                        let width = DOOR_WIDTH + 0.2 + 2.2 * porosity;
                        let pos = t + (FABRIC_CELL - 2.0 * t - width) * dh;
                        if along >= pos && along < pos + width {
                            // Narrow thresholds sometimes keep a lintel:
                            // framed evidence of doors that once existed.
                            if width < 2.0
                                && Self::cell_hash(noise, seed, door_salt ^ 0x2E, cx, cz) < 0.35
                            {
                                lintel_from_units = Some(DOOR_HEIGHT);
                            }
                        } else {
                            solid = true;
                        }
                    } else {
                        solid = true;
                    }
                }
            }
        }

        // ---- lights ------------------------------------------------------
        let light = if !solid {
            // Open regions keep a longer, sparser fluorescent rhythm; the
            // regular dropped ceiling retains the denser office grid.
            let panel_period = if expanse { 4.8 } else { LIGHT_PERIOD };
            let lx = (wx - panel_period * 0.5).rem_euclid(panel_period);
            let lz = (wz - panel_period * 0.5).rem_euclid(panel_period);
            let cell_x = (wx / panel_period).floor() as i64;
            let cell_z = (wz / panel_period).floor() as i64;
            let keep = if expanse { 0.66 } else { 0.78 } * tuning.lights;
            let alive = Self::cell_hash(noise, seed, 0xE900, cell_x, cell_z) < keep;
            lx < 0.45 && lz < 0.45 && alive
        } else {
            false
        };

        ColumnPlan {
            floor: true,
            solid,
            ceiling_units,
            light,
            red_light: false,
            lintel_from_units,
            sconce,
        }
    }

    /// Region plans for every region a chunk (plus a margin) overlaps.
    pub(crate) fn region_plans_for(
        chunk_pos: Position,
        chunk_size: f32,
        seed: u32,
        config: &GeneratorConfig,
        noise: &dyn NoiseProvider,
    ) -> Vec<((i64, i64), RegionPlan)> {
        let m = 1.0;
        let mut out = Vec::new();
        for rz in region_index(chunk_pos.z - m)..=region_index(chunk_pos.z + chunk_size + m) {
            for rx in region_index(chunk_pos.x - m)..=region_index(chunk_pos.x + chunk_size + m) {
                out.push((
                    (rx, rz),
                    generate_region_plan(
                        seed,
                        Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                        REGION_SIZE,
                        config,
                        noise,
                    ),
                ));
            }
        }
        out
    }

    /// Is (wx, wz) on a structural column of this system?
    fn on_column(st: &StructuralSystemInstance, wx: f32, wz: f32) -> bool {
        if st.system == StructuralSystem::CoreAndShell {
            // Core-and-shell designers hide columns in walls; none inside.
            return false;
        }
        let mut mx = (wx - st.phase.0).rem_euclid(st.bay_x);
        let mz = (wz - st.phase.1).rem_euclid(st.bay_z);
        if st.system == StructuralSystem::OffsetGrid {
            let row = ((wz - st.phase.1) / st.bay_z).floor() as i64;
            if row.rem_euclid(2) == 1 {
                mx = (wx - st.phase.0 + st.bay_x * 0.5).rem_euclid(st.bay_x);
            }
        }
        mx < st.column_side && mz < st.column_side
    }

    /// Column plan for a point inside an assembly footprint or its
    /// surrounding wall band (`inside == false`).
    fn assembly_column(
        a: &AssemblyInstance,
        inside: bool,
        renovator: Option<&StructuralSystemInstance>,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        let walls_on = tuning.walls > 0.0;
        let zone = a.ceiling_zones.first();
        let mut ceiling_units = zone.map_or(3.4, |c| c.height_units);
        match zone.map(|c| c.language) {
            // Coffered reads as a shading pattern now (see the splat
            // shader), not stepped geometry.
            Some(CeilingLanguage::ExposedSoffit) => ceiling_units -= 0.2,
            _ => {}
        }
        let mut plan = ColumnPlan {
            floor: true,
            solid: false,
            ceiling_units,
            light: false,
            red_light: false,
            lintel_from_units: None,
            sconce: false,
        };

        // Entrances pierce the wall band (and win over everything solid).
        for e in &a.entrances {
            let (da, db) = if e.through_x_wall {
                ((wx - e.center.x).abs(), (wz - e.center.z).abs())
            } else {
                ((wz - e.center.z).abs(), (wx - e.center.x).abs())
            };
            if da < e.width * 0.5 && db <= PLAN_WALL_T + 0.05 {
                plan.lintel_from_units = walls_on
                    .then_some(e.lintel_units.unwrap_or(0.0))
                    .filter(|u| *u > 0.0);
                return plan;
            }
        }

        if !inside {
            // Perimeter wall band.
            plan.solid = walls_on;
            return plan;
        }

        // Interior partitions: walls on space boundaries that are not the
        // footprint perimeter, each with a centered doorway.
        let fb = a.footprint.bounds();
        if walls_on {
            for s in &a.spaces {
                let sb = s.footprint.bounds();
                if wz >= sb.1 && wz <= sb.3 {
                    for plane in [sb.0, sb.2] {
                        if (plane - fb.0).abs() > 0.1
                            && (plane - fb.2).abs() > 0.1
                            && (wx - plane).abs() < PLAN_WALL_T * 0.5
                        {
                            let door_c = (sb.1 + sb.3) * 0.5;
                            if (wz - door_c).abs() < 0.6 {
                                plan.lintel_from_units = Some(DOOR_HEIGHT);
                            } else {
                                plan.solid = true;
                            }
                        }
                    }
                }
                if wx >= sb.0 && wx <= sb.2 {
                    for plane in [sb.1, sb.3] {
                        if (plane - fb.1).abs() > 0.1
                            && (plane - fb.3).abs() > 0.1
                            && (wz - plane).abs() < PLAN_WALL_T * 0.5
                        {
                            let door_c = (sb.0 + sb.2) * 0.5;
                            if (wx - door_c).abs() < 0.6 {
                                plan.lintel_from_units = Some(DOOR_HEIGHT);
                            } else {
                                plan.solid = true;
                            }
                        }
                    }
                }
            }
        }

        // Structure: the original grid, plus the renovator's contradictory
        // grid where a renovation overlays the assembly.
        if tuning.pillars > 0.0 && !plan.solid {
            if Self::on_column(&a.structure, wx, wz) {
                plan.solid = true;
            }
            if let Some(r) = renovator {
                if a.corruption.renovation_overlay && Self::on_column(r, wx, wz) {
                    // Renovation columns intrude on the original bay rhythm;
                    // the contradiction itself is the corruption. They stay
                    // ordinary wall material — Level 0 has no red masonry.
                    plan.solid = true;
                }
            }
        }

        // Fixtures, tied to the assembly's ceiling modules. In a red room
        // every fixture burns red: the anomaly is the room's light, applied
        // after all geometry decisions and on the same fixture spacing.
        if !plan.solid && tuning.lights > 0.0 {
            for f in &a.fixtures {
                if f.lit && (wx - f.at.x).abs() <= f.half_x && (wz - f.at.z).abs() <= f.half_z {
                    plan.light = true;
                    plan.red_light = a.corruption.red_room;
                    break;
                }
            }
        }
        plan
    }

    fn corridor_ceiling(
        spine: &CirculationSpine,
        noise: &dyn NoiseProvider,
        seed: u32,
        wx: f32,
        wz: f32,
    ) -> f32 {
        // Slow, quantized drift: one sample per 12 u corridor zone snapped
        // to the voxel lattice, so long runs hold one height and then step
        // once — instead of per-column ripple overhead.
        const DRIFT_ZONE: f32 = 12.0;
        let zone_x = (wx / DRIFT_ZONE).floor() * DRIFT_ZONE + DRIFT_ZONE * 0.5;
        let zone_z = (wz / DRIFT_ZONE).floor() * DRIFT_ZONE + DRIFT_ZONE * 0.5;
        let drift = Self::n(
            noise,
            seed,
            0xCA00_u32.wrapping_add(spine.id),
            zone_x,
            zone_z,
            0.35,
        );
        let (height, lo, hi) = match spine.spine_kind {
            SpaceProgram::MainCorridor => (3.8 + 0.4 * drift, 3.4, 4.2),
            SpaceProgram::SecondaryHall => (3.3 + 0.3 * drift, 3.0, 3.6),
            _ => (3.4, 3.4, 3.4),
        };
        // Snap first, clamp last (f32 lattice snap can overshoot the band).
        ((height / 0.2).round() * 0.2).clamp(lo, hi)
    }

    /// Whether a corridor-side wall dissolves into the adjacent room/fabric.
    /// Main-route openings take 8--10 u from a 32 u macro span: around a
    /// quarter to a third of an eligible edge is directly open, while the
    /// remaining wall runs stay long enough to avoid a doorway cadence.
    fn corridor_edge_opens(
        spine: &CirculationSpine,
        noise: &dyn NoiseProvider,
        seed: u32,
        along: f32,
        is_horizontal: bool,
        wx: f32,
        wz: f32,
    ) -> bool {
        // The opening spine sequence is sacred: near spawn the corridor keeps
        // both edge walls so the player's first minute reads as circulation
        // through a building, before any dissolution into fabric.
        let sp = spawn_point(seed);
        let spawn_d2 = (wx - sp.x) * (wx - sp.x) + (wz - sp.z) * (wz - sp.z);
        if spawn_d2 < SPAWN_READABLE_RADIUS * SPAWN_READABLE_RADIUS {
            return false;
        }

        if Self::in_expanse(noise, seed, wx, wz) {
            return true;
        }

        let span = 32.0;
        let section = (along / span).floor() as i64;
        let perpendicular = if is_horizontal { wz } else { wx };
        let side = (perpendicular / spine.width.max(1.0)).floor() as i64;
        let salt = 0xCB00_u32.wrapping_add(spine.id);
        let width_hash = Self::cell_hash(noise, seed, salt, section, side);
        let phase_hash = Self::cell_hash(noise, seed, salt ^ 0x19, section, side);
        let (width, start) = match spine.spine_kind {
            SpaceProgram::MainCorridor => (8.0 + 2.0 * width_hash, 6.0 + 10.0 * phase_hash),
            _ => (4.0 + 1.6 * width_hash, 10.0 + 8.0 * phase_hash),
        };
        let offset = along.rem_euclid(span);
        offset >= start && offset < start + width
    }

    fn anomaly_hash(instance: &AnomalyInstance, epoch: u32, salt: u64, x: i64, z: i64) -> f32 {
        let mut h = instance.id
            ^ salt
            ^ (epoch as u64).rotate_left(19)
            ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (z as u64).rotate_left(37);
        h ^= h >> 30;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 31;
        (h >> 40) as f32 / (1u64 << 24) as f32
    }

    fn remap_eligible(
        instance: &AnomalyInstance,
        reality: &RealitySnapshot,
        config: &GeneratorConfig,
        wx: f32,
        wz: f32,
    ) -> Option<u32> {
        let state = instance.state(reality)?;
        (state.epoch > 0
            && instance.boundary_distance(wx, wz) > instance.entry_band
            && state.point_is_in_wake(wx, wz, config.anomalies.remap_distance))
        .then_some(state.epoch)
    }

    /// Epoch-aware macro anomaly sampler. Corridors are handled before this
    /// function, so their immutable bearing route always punches through an
    /// anomaly footprint.
    fn anomaly_column(
        instance: &AnomalyInstance,
        noise: &dyn NoiseProvider,
        seed: u32,
        config: &GeneratorConfig,
        reality: &RealitySnapshot,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        let tuning = &config.tuning;
        let (lx, lz) = instance.local_coords(wx, wz);
        let boundary = instance.boundary_distance(wx, wz);
        let perimeter = boundary <= PLAN_WALL_T;
        let skeleton = lz.abs() <= instance.skeleton_half_width;
        let seam_x = wx.rem_euclid(config.chunk_size);
        let seam_z = wz.rem_euclid(config.chunk_size);
        let away_from_chunk_seam = seam_x > PLAN_WALL_T * 2.0
            && seam_x < config.chunk_size - PLAN_WALL_T * 2.0
            && seam_z > PLAN_WALL_T * 2.0
            && seam_z < config.chunk_size - PLAN_WALL_T * 2.0;

        match instance.kind {
            AnomalyKind::PillarExpanse => {
                let lattice = instance.pillar_lattice.expect("pillar instance lattice");
                let cx = ((lx - lattice.phase_x) / lattice.bay_x).floor() as i64;
                let cz = ((lz - lattice.phase_z) / lattice.bay_z).floor() as i64;
                let fx = (lx - lattice.phase_x).rem_euclid(lattice.bay_x);
                let fz = (lz - lattice.phase_z).rem_euclid(lattice.bay_z);
                let side = instance.pillar_size(cx, cz).unwrap_or(1.2);
                let dx = fx.min(lattice.bay_x - fx);
                let dz = fz.min(lattice.bay_z - fz);
                let mut solid = perimeter || (dx < side * 0.5 && dz < side * 0.5);

                // Epochs alter plausible infill in the committed wake only.
                // The center bearing lane, boundary/entry band and chunk seam
                // guards are immutable, so old and new resident cohorts meet
                // on identical geometry.
                if !solid
                    && !skeleton
                    && away_from_chunk_seam
                    && let Some(epoch) = Self::remap_eligible(instance, reality, config, wx, wz)
                {
                    let edge_x = fx < PLAN_WALL_T || lattice.bay_x - fx < PLAN_WALL_T;
                    let edge_z = fz < PLAN_WALL_T || lattice.bay_z - fz < PLAN_WALL_T;
                    let threshold = (0.12 * config.anomalies.remap_intensity).clamp(0.0, 0.48);
                    solid = (edge_x
                        && Self::anomaly_hash(instance, epoch, 0x11F1, cx, cz) < threshold)
                        || (edge_z
                            && Self::anomaly_hash(instance, epoch, 0x11F2, cx, cz) < threshold);
                }
                if skeleton {
                    solid = false;
                }
                let light = !solid
                    && (lx - 2.4).rem_euclid(4.8) < 0.45
                    && lz.rem_euclid(4.8) < 0.45
                    && Self::anomaly_hash(instance, 0, 0x11A7, cx, cz) < 0.62 * tuning.lights;
                ColumnPlan {
                    floor: true,
                    solid,
                    ceiling_units: 4.2,
                    light,
                    red_light: false,
                    lintel_from_units: None,
                    sconce: false,
                }
            }
            AnomalyKind::BlackoutExpanse => {
                let mut plan = Self::column_plan(noise, seed ^ instance.id as u32, tuning, wx, wz);
                let depth = instance.normalized_depth(wx, wz);
                plan.ceiling_units = if depth > 0.68 { 2.6 } else { 3.4 };
                plan.light = false;
                plan.red_light = false;
                plan.solid |= perimeter;
                if skeleton {
                    plan.solid = false;
                    // Sparse ordinary-color glimmers mark the immutable
                    // recovery skeleton; palette work is intentionally later.
                    plan.light = lx.rem_euclid(32.0) < 0.45 && tuning.lights > 0.0;
                } else if away_from_chunk_seam
                    && let Some(epoch) = Self::remap_eligible(instance, reality, config, wx, wz)
                {
                    let cell = 7.2;
                    let cx = (lx / cell).floor() as i64;
                    let cz = (lz / cell).floor() as i64;
                    let fx = lx.rem_euclid(cell);
                    let fz = lz.rem_euclid(cell);
                    let edge = fx < PLAN_WALL_T || fz < PLAN_WALL_T;
                    let threshold = (0.20 * config.anomalies.remap_intensity).clamp(0.0, 0.65);
                    if edge && Self::anomaly_hash(instance, epoch, 0xB1AC, cx, cz) < threshold {
                        plan.solid = true;
                        plan.light = false;
                    }
                }
                plan
            }
            AnomalyKind::PitLattice => {
                let lattice = instance.pit_lattice.expect("pit instance lattice");
                let fx = (lx - lattice.phase_x).rem_euclid(lattice.spacing_x);
                let fz = (lz - lattice.phase_z).rem_euclid(lattice.spacing_z);
                let dx = fx.min(lattice.spacing_x - fx);
                let dz = fz.min(lattice.spacing_z - fz);
                let pit = !skeleton
                    && boundary > lattice.side
                    && dx < lattice.side * 0.5
                    && dz < lattice.side * 0.5;
                ColumnPlan {
                    floor: !pit,
                    solid: perimeter,
                    ceiling_units: 3.8,
                    light: !perimeter
                        && lx.rem_euclid(3.2) < 0.45
                        && lz.rem_euclid(3.2) < 0.45
                        && tuning.lights > 0.0,
                    red_light: false,
                    lintel_from_units: None,
                    sconce: false,
                }
            }
            AnomalyKind::RedRoom => unreachable!("red rooms overlay assemblies"),
        }
    }

    fn close_red_room(
        instance: &AnomalyInstance,
        mut plan: ColumnPlan,
        config: &GeneratorConfig,
        reality: &RealitySnapshot,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        let Some(state) = instance.state(reality) else {
            return plan;
        };
        if state.epoch == 0 {
            return plan;
        }
        let (lx, lz) = instance.local_coords(wx, wz);
        let hx = instance.footprint.half_x;
        let hz = instance.footprint.half_z;
        let ring = 2.0f32.min(hx * 0.35).min(hz * 0.35).max(1.2);
        let boundary = instance.boundary_distance(wx, wz);
        let in_core = lx.abs() < hx - ring && lz.abs() < hz - ring;
        // Entry becomes an ordinary wall after the occluded inner threshold;
        // the open band around the central core is a literal closed loop.
        plan.solid = boundary <= PLAN_WALL_T || in_core;
        plan.lintel_from_units = None;
        if plan.solid {
            plan.light = false;
        }

        // Default safety policy: a separately authored far-side breach is
        // present when escape_bias is nonzero. It is never the remembered
        // entrance and its position changes coherently with the epoch.
        if config.anomalies.red_escape_bias > 0.0 {
            let escape_z = -hz
                + ring
                + Self::anomaly_hash(instance, state.epoch, 0xE5CA, 0, 0) * (2.0 * hz - 2.0 * ring);
            if (lx - (hx - PLAN_WALL_T * 0.5)).abs() <= PLAN_WALL_T && (lz - escape_z).abs() < 0.6 {
                plan.solid = false;
            }
        }
        plan
    }

    /// The architectural column plan: corridor beats assembly beats fabric.
    pub(crate) fn plan_column(
        plan: &RegionPlan,
        noise: &dyn NoiseProvider,
        seed: u32,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        Self::plan_column_in_reality(
            plan,
            noise,
            seed,
            &GeneratorConfig::low_spec().with_tuning(*tuning),
            &RealitySnapshot::empty(),
            wx,
            wz,
        )
    }

    pub(crate) fn plan_column_in_reality(
        plan: &RegionPlan,
        noise: &dyn NoiseProvider,
        seed: u32,
        config: &GeneratorConfig,
        reality: &RealitySnapshot,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        let tuning = &config.tuning;
        // -- circulation ----------------------------------------------------
        let mut in_corridor = false;
        let mut corridor_ceiling = 0.0f32;
        let mut corridor_light = false;
        let mut corridor_wall = false;
        let mut corridor_gap = false;
        for s in &plan.corridors {
            let (d, along, is_horizontal) = s.nearest(wx, wz);
            let half = s.width * 0.5;
            if d <= half {
                in_corridor = true;
                corridor_ceiling =
                    corridor_ceiling.max(Self::corridor_ceiling(s, noise, seed, wx, wz));
                // Light strip modules follow the corridor in world space.
                if d < 0.45 && along.rem_euclid(4.0) < 1.0 {
                    corridor_light = true;
                }
            } else if d <= half + PLAN_WALL_T {
                corridor_wall = true;
                corridor_ceiling =
                    corridor_ceiling.max(Self::corridor_ceiling(s, noise, seed, wx, wz));
                if Self::corridor_edge_opens(s, noise, seed, along, is_horizontal, wx, wz) {
                    corridor_gap = true;
                }
            }
        }
        if in_corridor {
            return ColumnPlan {
                floor: true,
                solid: false,
                ceiling_units: corridor_ceiling,
                light: corridor_light && tuning.lights > 0.0,
                red_light: false,
                lintel_from_units: None,
                sconce: false,
            };
        }

        if let Some(instance) = plan
            .anomalies
            .iter()
            .filter(|a| a.kind != AnomalyKind::RedRoom && a.contains(wx, wz))
            .max_by(|a, b| {
                a.normalized_depth(wx, wz)
                    .total_cmp(&b.normalized_depth(wx, wz))
            })
        {
            return Self::anomaly_column(instance, noise, seed, config, reality, wx, wz);
        }

        // -- assemblies -------------------------------------------------------
        {
            let renovator_structure = plan.architects.get(1).map(|g| StructuralSystemInstance {
                system: g.structural_system,
                bay_x: 3.6,
                bay_z: 4.4,
                phase: (1.6, 2.4),
                column_side: 0.4,
            });
            for a in &plan.assemblies {
                let b = a.footprint.bounds();
                let t = PLAN_WALL_T;
                if wx < b.0 - t || wx > b.2 + t || wz < b.1 - t || wz > b.3 + t {
                    continue;
                }
                let inside = a.footprint.contains(wx, wz);
                let base =
                    Self::assembly_column(a, inside, renovator_structure.as_ref(), tuning, wx, wz);
                if a.corruption.red_room
                    && let Some(red) = plan
                        .anomalies
                        .iter()
                        .find(|r| r.kind == AnomalyKind::RedRoom && r.contains(wx, wz))
                {
                    return Self::close_red_room(red, base, config, reality, wx, wz);
                }
                return base;
            }
        }

        // -- corridor edge walls through fabric -------------------------------
        if corridor_wall && tuning.walls > 0.0 && !corridor_gap {
            return ColumnPlan {
                floor: true,
                solid: true,
                ceiling_units: corridor_ceiling.max(3.2),
                light: false,
                red_light: false,
                lintel_from_units: None,
                sconce: false,
            };
        }

        // -- the endless unplanned office fabric -------------------------------
        Self::column_plan(noise, seed, tuning, wx, wz)
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
        self.generate_with_reality(chunk_pos, seed, config, noise, &RealitySnapshot::empty())
    }

    fn generate_with_reality(
        &self,
        chunk_pos: Position,
        seed: u32,
        config: GeneratorConfig,
        noise: &dyn NoiseProvider,
        reality: &RealitySnapshot,
    ) -> VoxelGrid {
        let s = config.voxel_scale;
        let width = (config.chunk_size / s).round() as usize;
        let depth = (config.chunk_size / s).round() as usize;
        let height = (GRID_HEIGHT_UNITS / s) as usize;
        let mut grid = VoxelGrid::new(width, height, depth);

        // Architecture first: plan every region this chunk overlaps.
        let plans =
            BackroomsLevel::region_plans_for(chunk_pos, config.chunk_size, seed, &config, noise);
        let plan_of = |wx: f32, wz: f32| -> &RegionPlan {
            let key = (region_index(wx), region_index(wz));
            plans
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, p)| p)
                .unwrap_or(&plans[0].1)
        };

        // Export semantic crossings/hazards from the same immutable plans as
        // voxel geometry. Neighboring region plans repeat macro instances, so
        // deduplicate by stable IDs before the payload reaches the engine.
        let chunk_bounds = WorldBounds::new(
            chunk_pos.x,
            chunk_pos.z,
            chunk_pos.x + config.chunk_size,
            chunk_pos.z + config.chunk_size,
        );
        let mut seen_instances = std::collections::HashSet::new();
        let mut seen_gates = std::collections::HashSet::new();
        let mut seen_hazards = std::collections::HashSet::new();
        for (_, plan) in &plans {
            for anomaly in &plan.anomalies {
                if !seen_instances.insert(anomaly.id)
                    || !anomaly
                        .footprint
                        .bounds()
                        .intersects(chunk_bounds.expanded(1.0))
                {
                    continue;
                }
                for gate in anomaly.traversal_gates() {
                    let crosses_chunk = match gate.axis {
                        crate::domain::entities::anomaly::Axis2::X => {
                            gate.plane >= chunk_bounds.min_x - 0.01
                                && gate.plane <= chunk_bounds.max_x + 0.01
                                && gate.span_max >= chunk_bounds.min_z
                                && gate.span_min <= chunk_bounds.max_z
                        }
                        crate::domain::entities::anomaly::Axis2::Z => {
                            gate.plane >= chunk_bounds.min_z - 0.01
                                && gate.plane <= chunk_bounds.max_z + 0.01
                                && gate.span_max >= chunk_bounds.min_x
                                && gate.span_min <= chunk_bounds.max_x
                        }
                    };
                    if crosses_chunk && seen_gates.insert(gate.id) {
                        grid.traversal_gates.push(*gate);
                    }
                }
                for hazard in anomaly.pit_hazards_for_bounds(chunk_bounds) {
                    if seen_hazards.insert(hazard.id) {
                        grid.pit_hazards.push(hazard);
                    }
                }
            }
        }

        use crate::domain::entities::architecture::{LightKind, RuntimeLight};
        let mut seen = std::collections::HashSet::new();
        for (_, plan) in &plans {
            for a in &plan.assemblies {
                if a.corruption.abandoned {
                    continue;
                }
                for f in &a.fixtures {
                    if !f.lit {
                        continue;
                    }
                    let key = (f.at.x.to_bits(), f.at.z.to_bits());
                    if !seen.insert(key) {
                        continue;
                    }

                    // Region-scale anomaly interiors own their fixture
                    // rhythm. Do not leak an overwritten assembly's runtime
                    // light into a blackout/pillar/pit payload.
                    if plan_of(f.at.x, f.at.z).anomalies.iter().any(|anomaly| {
                        anomaly.kind != AnomalyKind::RedRoom && anomaly.contains(f.at.x, f.at.z)
                    }) {
                        continue;
                    }

                    let cx = f.at.x - chunk_pos.x;
                    let cz = f.at.z - chunk_pos.z;
                    if cx >= -15.0
                        && cx <= config.chunk_size + 15.0
                        && cz >= -15.0
                        && cz <= config.chunk_size + 15.0
                    {
                        let ceiling_units = a
                            .ceiling_zones
                            .iter()
                            .find(|z| z.area.contains(f.at.x, f.at.z))
                            .map(|z| z.height_units)
                            .unwrap_or(4.0);
                        let is_atrium = matches!(
                            a.program,
                            crate::domain::entities::architecture::SpaceProgram::Atrium
                        );
                        let y = runtime_light_height(ceiling_units, is_atrium);

                        let kind = if f.half_x > f.half_z * 2.0 || f.half_z > f.half_x * 2.0 {
                            LightKind::Strip
                        } else {
                            LightKind::CeilingPanel
                        };

                        // Runtime lights are collected before voxelization, so
                        // query the architectural column plan rather than the
                        // still-empty grid when rejecting pillar intersections.
                        let is_buried = BackroomsLevel::plan_column_in_reality(
                            plan_of(f.at.x, f.at.z),
                            noise,
                            seed,
                            &config,
                            reality,
                            f.at.x,
                            f.at.z,
                        )
                        .solid;

                        if !is_buried {
                            // Red rooms are exposed purely by their light
                            // color; the fixtures keep the room's spacing.
                            let rgb = if a.corruption.red_room {
                                [1.0, 0.22, 0.16]
                            } else {
                                [1.0, 0.95, 0.8]
                            };
                            grid.runtime_lights.push(RuntimeLight {
                                world_pos: [f.at.x, y, f.at.z],
                                half_size: [f.half_x, f.half_z],
                                rgb,
                                range: if is_atrium { 24.0 } else { 16.0 },
                                intensity: if is_atrium { 4.0 } else { 1.0 },
                                enabled: true,
                                kind,
                            });
                        }
                    }
                }
            }
        }

        // Plan every column plus a 1-voxel margin: ceiling skirts must seal
        // height steps across chunk borders too.
        let plan_at = |lx: i64, lz: i64| -> ColumnPlan {
            let wx = chunk_pos.x + (lx as f32 + 0.5) * s;
            let wz = chunk_pos.z + (lz as f32 + 0.5) * s;
            BackroomsLevel::plan_column_in_reality(
                plan_of(wx, wz),
                noise,
                seed,
                &config,
                reality,
                wx,
                wz,
            )
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

                if p.floor {
                    grid.set(x, 0, z, VOXEL_FLOOR);
                }

                if p.solid {
                    for y in 1..ch {
                        grid.set(x, y, z, VOXEL_WALL);
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

                // Solid columns stay wall material all the way up: capping a
                // wall with ceiling voxels painted its visible top band grey.
                let cap_mat = if p.solid { VOXEL_WALL } else { VOXEL_CEILING };
                for y in ch..=neighbor_max.max(ch) {
                    grid.set(x, y, z, cap_mat);
                }
                if p.light && !p.solid {
                    let light_mat = if p.red_light {
                        crate::domain::entities::voxel_grid::VOXEL_RED_LIGHT
                    } else {
                        VOXEL_LIGHT
                    };
                    grid.set(x, ch, z, light_mat);
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

        // BFS from the first open interior voxel.
        let &(sx, sz) = open.first().expect("some open floor exists");
        let mut visited = vec![false; w * d];
        let mut queue = std::collections::VecDeque::from([(sx, sz)]);
        visited[sx * d + sz] = true;
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

        let plans =
            BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 20.0, 42, &config, &noise);
        for z in 0..a.depth() {
            for (grid, lx, gx) in [(&a, w - 1, w - 1), (&b, 0usize, w)] {
                let wx = (gx as f32 + 0.5) * config.voxel_scale;
                let wz = (z as f32 + 0.5) * config.voxel_scale;
                let key = (
                    crate::use_cases::region_plan::region_index(wx),
                    crate::use_cases::region_plan::region_index(wz),
                );
                let plan = plans
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, p)| p)
                    .unwrap();
                let expect =
                    BackroomsLevel::plan_column(plan, &noise, 42, &LevelTuning::default(), wx, wz);
                let got_solid = grid.get(lx, 1, z) == VOXEL_WALL;
                assert_eq!(
                    got_solid, expect.solid,
                    "column mismatch at world x={gx} z={z}"
                );
            }
        }
    }

    /// The plan is authoritative: corridor centerlines must be carved open
    /// in the voxelized chunks they cross.
    #[test]
    fn corridors_from_the_plan_are_carved_open() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let plans =
            BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 80.0, 42, &config, &noise);
        let plan = &plans.iter().find(|(k, _)| *k == (0, 0)).unwrap().1;
        let spine = &plan.corridors[0];

        let mut checked = 0;
        for seg in spine.path.windows(2) {
            let (p0, p1) = (seg[0], seg[1]);
            let steps = 8;
            for k in 1..steps {
                let t = k as f32 / steps as f32;
                let (wx, wz) = (p0.x + (p1.x - p0.x) * t, p0.z + (p1.z - p0.z) * t);
                // Stay inside region (0,0) and off chunk edges.
                if !(1.0..79.0).contains(&wx) || !(1.0..79.0).contains(&wz) {
                    continue;
                }
                let (cx, cz) = ((wx / 10.0).floor() * 10.0, (wz / 10.0).floor() * 10.0);
                let grid = generate(cx, cz);
                let (lx, lz) = (
                    ((wx - cx) / config.voxel_scale) as usize,
                    ((wz - cz) / config.voxel_scale) as usize,
                );
                assert!(
                    is_open(&grid, lx.min(grid.width() - 1), lz.min(grid.depth() - 1)),
                    "main corridor blocked at world ({wx:.1}, {wz:.1})"
                );
                checked += 1;
            }
        }
        assert!(checked > 5, "spine barely sampled ({checked} points)");
    }

    #[test]
    fn circulation_uses_the_raised_ceiling_hierarchy() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let mut secondary_checked = false;
        for rx in -3i64..=3 {
            for rz in -3i64..=3 {
                let plans = BackroomsLevel::region_plans_for(
                    Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                    1.0,
                    42,
                    &config,
                    &noise,
                );
                let plan = &plans
                    .iter()
                    .find(|(key, _)| *key == (rx, rz))
                    .expect("requested region plan")
                    .1;
                for spine in &plan.corridors {
                    let segment = spine.path.windows(2).next().expect("spine segment");
                    let sample = Position::new(
                        segment[0].x * 0.45 + segment[1].x * 0.55,
                        segment[0].z * 0.45 + segment[1].z * 0.55,
                    );
                    let ceiling =
                        BackroomsLevel::corridor_ceiling(spine, &noise, 42, sample.x, sample.z);
                    let expected = match spine.spine_kind {
                        SpaceProgram::MainCorridor => 3.4..=4.2,
                        SpaceProgram::SecondaryHall => {
                            secondary_checked = true;
                            3.0..=3.6
                        }
                        _ => unreachable!("non-circulation spine"),
                    };
                    assert!(
                        expected.contains(&ceiling),
                        "{:?} ceiling {} at ({}, {})",
                        spine.spine_kind,
                        ceiling,
                        sample.x,
                        sample.z
                    );
                }
            }
        }
        assert!(secondary_checked, "sample contained no secondary branch");
    }

    /// Assemblies voxelize as walled rooms whose planned entrance is open
    /// (with a lintel when the designer's threshold language wants one).
    #[test]
    fn assemblies_have_walls_and_open_entrances() {
        let noise = SimpleNoiseProvider::new();
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.frequency = 0.0;
        let tuning = LevelTuning::default();
        let plans =
            BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 80.0, 42, &config, &noise);
        let plan = &plans.iter().find(|(k, _)| *k == (0, 0)).unwrap().1;
        assert!(!plan.assemblies.is_empty());

        for a in &plan.assemblies {
            let e = &a.entrances[0];
            // The entrance column itself: open (possibly under a lintel).
            let door =
                BackroomsLevel::plan_column(plan, &noise, 42, &tuning, e.center.x, e.center.z);
            assert!(!door.solid, "assembly {} door is walled shut", a.id);
            // Somewhere along the same front wall, clear of the door span,
            // there must be solid wall. Probe the middle of the wall band
            // (the entrance center sits exactly on the footprint boundary,
            // where containment is ambiguous).
            let b = a.footprint.bounds();
            let band = |c: f32, lo: f32, hi: f32| {
                if (c - lo).abs() < (c - hi).abs() {
                    lo - PLAN_WALL_T * 0.5
                } else {
                    hi + PLAN_WALL_T * 0.5
                }
            };
            let (lo, hi, door_along) = if e.through_x_wall {
                (b.0, b.2, e.center.x)
            } else {
                (b.1, b.3, e.center.z)
            };
            let mut solid_found = false;
            let mut along = lo + 0.3;
            while along < hi - 0.2 {
                if (along - door_along).abs() > e.width * 0.5 + 0.4 {
                    let (wx, wz) = if e.through_x_wall {
                        (along, band(e.center.z, b.1, b.3))
                    } else {
                        (band(e.center.x, b.0, b.2), along)
                    };
                    if BackroomsLevel::plan_column(plan, &noise, 42, &tuning, wx, wz).solid {
                        solid_found = true;
                        break;
                    }
                }
                along += 0.2;
            }
            assert!(
                solid_found,
                "assembly {} has no solid front wall anywhere",
                a.id
            );
        }
    }

    /// An abandoned expansion is a dark shell: it keeps its walls but none of
    /// its fixtures are lit.
    #[test]
    fn abandoned_expansions_are_unlit() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let tuning = LevelTuning::default();
        let mut found = false;
        for rx in -3i64..3 {
            for rz in -3i64..3 {
                let plans = BackroomsLevel::region_plans_for(
                    Position::new(rx as f32 * 80.0, rz as f32 * 80.0),
                    1.0,
                    42,
                    &config,
                    &noise,
                );
                let plan = &plans[0].1;
                for a in &plan.assemblies {
                    if !a.corruption.abandoned {
                        continue;
                    }
                    found = true;
                    let (x0, z0, x1, z1) = a.footprint.bounds();
                    // No interior column may carry a lit fixture.
                    let mut probe_z = z0 + 0.6;
                    while probe_z < z1 - 0.4 {
                        let mut probe_x = x0 + 0.6;
                        while probe_x < x1 - 0.4 {
                            // A corridor clipping the footprint may still run
                            // its own lit strip through the shell — that is
                            // canon ("unreachable but still lit"). Only the
                            // room's fixtures must be dark.
                            let in_corridor = plan
                                .corridors
                                .iter()
                                .any(|s| s.distance(probe_x, probe_z) <= s.width * 0.5);
                            if !in_corridor {
                                let c = BackroomsLevel::plan_column(
                                    plan, &noise, 42, &tuning, probe_x, probe_z,
                                );
                                assert!(!c.light, "abandoned assembly {} is lit", a.id);
                            }
                            probe_x += 0.8;
                        }
                        probe_z += 0.8;
                    }
                }
            }
        }
        assert!(found, "no abandoned expansion within 36 regions");
    }

    /// The baseline is broad regular dropped ceiling, with enough expansive
    /// and vaulted territory to prevent Level 0 from reading as a low maze.
    #[test]
    fn ceilings_are_vast_and_varied() {
        let noise = SimpleNoiseProvider::new();
        let mut counts = [0usize; 4];
        let mut lowest = f32::MAX;
        let mut tallest = 0.0f32;
        for z in (-600..=600).step_by(8) {
            for x in (-600..=600).step_by(8) {
                let (wx, wz) = (x as f32 + 0.5, z as f32 + 0.5);
                let band = BackroomsLevel::fabric_ceiling_band(&noise, 42, wx, wz);
                let ceiling = BackroomsLevel::fabric_ceiling_height(&noise, 42, wx, wz, band);
                let index = match band {
                    FabricCeilingBand::Compression => 0,
                    FabricCeilingBand::Regular => 1,
                    FabricCeilingBand::Expanse => 2,
                    FabricCeilingBand::Vault => 3,
                };
                counts[index] += 1;
                lowest = lowest.min(ceiling);
                tallest = tallest.max(ceiling);
            }
        }
        let total = counts.iter().sum::<usize>() as f32;
        let ratio = |index| counts[index] as f32 / total;
        // The labyrinth fabric is the default; open volumes punctuate it.
        assert!(
            (0.60..=0.85).contains(&ratio(1)),
            "ceiling territories: compression {:.1}%, regular {:.1}%, expanse {:.1}%, vault {:.1}%",
            ratio(0) * 100.0,
            ratio(1) * 100.0,
            ratio(2) * 100.0,
            ratio(3) * 100.0
        );
        assert!(
            (0.10..=0.28).contains(&ratio(2)),
            "open expanse territory was {:.1}%",
            ratio(2) * 100.0
        );
        assert!(
            (0.03..=0.15).contains(&ratio(3)),
            "vault territory was {:.1}%",
            ratio(3) * 100.0
        );
        assert!(
            (0.01..=0.10).contains(&ratio(0)),
            "compression territory was {:.1}%",
            ratio(0) * 100.0
        );
        assert!(
            lowest <= 2.8 && tallest >= 4.5,
            "ceiling range was only {lowest:.1}--{tallest:.1} u"
        );
    }

    /// Framed doorways still exist, but only as a rare architectural anomaly.
    #[test]
    fn rare_doorways_still_have_lintels() {
        let noise = SimpleNoiseProvider::new();
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.frequency = 0.0;
        let tuning = LevelTuning::default();
        let mut found = false;
        for rx in -6i64..=6 {
            for rz in -6i64..=6 {
                let plans = BackroomsLevel::region_plans_for(
                    Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                    1.0,
                    42,
                    &config,
                    &noise,
                );
                let plan = &plans
                    .iter()
                    .find(|(key, _)| *key == (rx, rz))
                    .expect("requested region plan")
                    .1;
                for a in &plan.assemblies {
                    for e in &a.entrances {
                        if e.width <= DOOR_WIDTH + 0.01 {
                            let column = BackroomsLevel::plan_column(
                                plan, &noise, 42, &tuning, e.center.x, e.center.z,
                            );
                            assert!(!column.solid, "narrow doorway is blocked");
                            assert_eq!(column.lintel_from_units, Some(DOOR_HEIGHT));
                            found = true;
                        }
                    }
                }
            }
        }
        assert!(found, "no rare doorway-with-lintel found in 169 regions");
    }

    /// The generation knobs actually steer the output: zeroing pillars and
    /// walls empties the world of solids; cranking them fills it back up.
    #[test]
    fn tuning_knobs_control_density() {
        let noise = SimpleNoiseProvider::new();
        // Aggregate over chunks in different fabric regimes (walled rooms
        // and open expanses) so both knobs have something to steer.
        let count_solids = |tuning: LevelTuning| -> usize {
            let mut n = 0;
            for (ox, oz) in [(10.0, 10.0), (30.0, 10.0), (50.0, 30.0), (10.0, 50.0)] {
                let grid = BackroomsLevel.generate(
                    Position::new(ox, oz),
                    42,
                    GeneratorConfig::low_spec().with_tuning(tuning),
                    &noise,
                );
                for z in 0..grid.depth() {
                    for x in 0..grid.width() {
                        if grid.get(x, 1, z) != VOXEL_AIR {
                            n += 1;
                        }
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

    /// Every LOD of a chunk must voxelize the same plan: coarse walls stay
    /// within one fine voxel of fine walls (the streaming engine swaps LODs
    /// of a chunk in place, so they must be faithful proxies).
    #[test]
    fn lods_of_the_same_chunk_correspond() {
        let noise = SimpleNoiseProvider::new();
        let fine = BackroomsLevel.generate(
            Position::new(10.0, 10.0),
            42,
            GeneratorConfig::low_spec(),
            &noise,
        );
        let coarse = BackroomsLevel.generate(
            Position::new(10.0, 10.0),
            42,
            GeneratorConfig::low_spec().at_lod(1),
            &noise,
        );
        let solid = |g: &VoxelGrid, x: usize, z: usize| g.get(x, 1, z) == VOXEL_WALL;
        let (mut matches, mut total) = (0usize, 0usize);
        for z in 0..coarse.depth() {
            for x in 0..coarse.width() {
                if !solid(&coarse, x, z) {
                    continue;
                }
                total += 1;
                let mut near = false;
                for dz in -1i32..=2 {
                    for dx in -1i32..=2 {
                        let (fx, fz) = (x as i32 * 2 + dx, z as i32 * 2 + dz);
                        if fx >= 0
                            && fz >= 0
                            && (fx as usize) < fine.width()
                            && (fz as usize) < fine.depth()
                            && solid(&fine, fx as usize, fz as usize)
                        {
                            near = true;
                        }
                    }
                }
                if near {
                    matches += 1;
                }
            }
        }
        assert!(total > 0, "coarse chunk has no walls at all");
        assert!(
            matches * 10 >= total * 9,
            "coarse walls stray from fine walls: {matches}/{total}"
        );
    }

    /// The player spawns *on the main corridor*: open floor along the
    /// centerline, a lit fixture within a light-strip period, and — for the
    /// readable opening sequence — solid edge walls on both sides.
    #[test]
    fn spawn_is_a_readable_walled_corridor() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let tuning = LevelTuning::default();
        let sp = crate::use_cases::region_plan::spawn_point(42);

        let plans =
            BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 80.0, 42, &config, &noise);
        let plan = &plans.iter().find(|(k, _)| *k == (0, 0)).unwrap().1;
        let spine = &plan.corridors[0];
        assert!(
            spine.distance(sp.x, sp.z) < 0.3,
            "spawn ({}, {}) is not on the main corridor centerline",
            sp.x,
            sp.z
        );

        // The corridor around spawn is open along the centerline, lit, and
        // predominantly walled on both sides. Framed entrances and branch
        // tees may pierce the run, but the edge must *read* as a wall — no
        // dissolution into open fabric inside the readable radius.
        let half = spine.width * 0.5;
        let mut lit = false;
        let mut solid_edges = [0usize; 2];
        const STEPS: usize = 16;
        for step in 0..STEPS {
            let wx = sp.x + step as f32;
            let c = BackroomsLevel::plan_column(plan, &noise, 42, &tuning, wx, sp.z);
            assert!(!c.solid, "main corridor blocked at ({wx}, {})", sp.z);
            lit |= c.light;
            for (i, side) in [-1.0, 1.0f32].iter().enumerate() {
                let wz = sp.z + side * (half + PLAN_WALL_T * 0.5);
                if BackroomsLevel::plan_column(plan, &noise, 42, &tuning, wx, wz).solid {
                    solid_edges[i] += 1;
                }
            }
        }
        assert!(lit, "no lit fixture along the first {STEPS} u of corridor");
        for (i, solid) in solid_edges.iter().enumerate() {
            assert!(
                solid * 10 >= STEPS * 6,
                "start corridor side {i} is mostly open ({solid}/{STEPS}): \
                 the opening sequence must read as a walled corridor"
            );
        }
    }

    /// The fabric connectivity invariant: every warren cell opens through
    /// its west or its north wall (doorway, dropped wall, or an override by
    /// corridor/assembly/expanse), so the labyrinth is globally connected
    /// by induction — no chunk ever needs to see its neighbors to prove it.
    #[test]
    fn every_fabric_cell_opens_west_or_north() {
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let tuning = LevelTuning::default();
        let mut cells_checked = 0usize;
        for cell_x in -40i64..40 {
            for cell_z in -40i64..40 {
                let x0 = cell_x as f32 * FABRIC_CELL;
                let z0 = cell_z as f32 * FABRIC_CELL;
                let plans = BackroomsLevel::region_plans_for(
                    Position::new(x0, z0),
                    FABRIC_CELL,
                    42,
                    &config,
                    &noise,
                );
                let plan_for = |wx: f32, wz: f32| {
                    let key = (region_index(wx), region_index(wz));
                    &plans.iter().find(|(k, _)| *k == key).unwrap().1
                };
                // The invariant belongs to *pure* fabric. Cells clipped by a
                // corridor edge or an assembly take their connectivity from
                // those systems instead (tested separately), so skip them.
                let (cx0, cz0) = (x0 - PLAN_WALL_T, z0 - PLAN_WALL_T);
                let (cx1, cz1) = (x0 + FABRIC_CELL, z0 + FABRIC_CELL);
                let clipped = plans.iter().any(|(_, p)| {
                    p.corridors.iter().any(|s| {
                        let m = s.width * 0.5 + PLAN_WALL_T + 0.1;
                        [
                            (cx0, cz0),
                            (cx1, cz0),
                            (cx0, cz1),
                            (cx1, cz1),
                            ((cx0 + cx1) * 0.5, (cz0 + cz1) * 0.5),
                        ]
                        .iter()
                        .any(|&(px, pz)| s.distance(px, pz) <= m + FABRIC_CELL)
                    }) || p.assemblies.iter().any(|a| {
                        let b = a.footprint.bounds();
                        cx0 < b.2 + PLAN_WALL_T
                            && b.0 - PLAN_WALL_T < cx1
                            && cz0 < b.3 + PLAN_WALL_T
                            && b.1 - PLAN_WALL_T < cz1
                    })
                });
                if clipped {
                    continue;
                }
                let mut open = false;
                let mut probe = |wx: f32, wz: f32| {
                    let c =
                        BackroomsLevel::plan_column(plan_for(wx, wz), &noise, 42, &tuning, wx, wz);
                    if !c.solid {
                        open = true;
                    }
                };
                // Sample along the west and north wall bands of the cell.
                let mut a = PLAN_WALL_T + 0.1;
                while a < FABRIC_CELL - PLAN_WALL_T {
                    probe(x0 + 0.2, z0 + a);
                    probe(x0 + a, z0 + 0.2);
                    a += 0.2;
                }
                assert!(
                    open,
                    "fabric cell ({cell_x},{cell_z}) is sealed on both its \
                     west and north walls"
                );
                cells_checked += 1;
            }
        }
        assert!(cells_checked > 1000, "sample too small: {cells_checked}");
    }

    /// Red is a *lighting* anomaly owned by whole rooms: some region must
    /// contain a red room whose fixtures voxelize as red lights, and no
    /// Level 0 chunk may ever contain a red wall voxel.
    #[test]
    fn red_rooms_are_lit_red_but_never_built_red() {
        use crate::domain::entities::voxel_grid::{VOXEL_RED_LIGHT, VOXEL_RED_WALL};
        let noise = SimpleNoiseProvider::new();
        let config = GeneratorConfig::low_spec();
        let tuning = LevelTuning::default();

        let mut red_room_seen = false;
        'search: for rx in -4i64..=4 {
            for rz in -4i64..=4 {
                let plans = BackroomsLevel::region_plans_for(
                    Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                    1.0,
                    42,
                    &config,
                    &noise,
                );
                let plan = &plans.iter().find(|(k, _)| *k == (rx, rz)).unwrap().1;
                for a in &plan.assemblies {
                    if !a.corruption.red_room {
                        continue;
                    }
                    assert!(!a.corruption.abandoned, "a red room must be occupied");
                    // Its lit fixtures plan red lights.
                    let f = a.fixtures.iter().find(|f| f.lit).expect("lit fixture");
                    let c = BackroomsLevel::plan_column(plan, &noise, 42, &tuning, f.at.x, f.at.z);
                    if c.light {
                        assert!(c.red_light, "red-room fixture plans a warm light");
                        red_room_seen = true;
                        break 'search;
                    }
                }
            }
        }
        assert!(red_room_seen, "no red room found within 81 regions");

        // No chunk voxelizes red walls, and red lights appear only as lights.
        for (ox, oz) in [(0.0, 0.0), (30.0, 10.0), (-40.0, 70.0), (150.0, -90.0)] {
            let grid = generate(ox, oz);
            for z in 0..grid.depth() {
                for x in 0..grid.width() {
                    for y in 0..grid.height() {
                        assert_ne!(
                            grid.get(x, y, z),
                            VOXEL_RED_WALL,
                            "red wall voxel at ({ox}+{x}, {y}, {oz}+{z})"
                        );
                    }
                    // Red lights sit at ceiling height, never at floor level.
                    assert_ne!(grid.get(x, 1, z), VOXEL_RED_LIGHT);
                }
            }
        }
    }

    fn find_macro_anomaly(kind: AnomalyKind) -> (GeneratorConfig, AnomalyInstance) {
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.frequency = 4.0;
        config.anomalies.pillar_expanses = (kind == AnomalyKind::PillarExpanse) as u8 as f32;
        config.anomalies.blackouts = (kind == AnomalyKind::BlackoutExpanse) as u8 as f32;
        config.anomalies.pit_lattices = (kind == AnomalyKind::PitLattice) as u8 as f32;
        let noise = SimpleNoiseProvider::new();
        for rz in 3i64..24 {
            for rx in 3i64..24 {
                let plans = BackroomsLevel::region_plans_for(
                    Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                    1.0,
                    42,
                    &config,
                    &noise,
                );
                if let Some(instance) = plans
                    .iter()
                    .flat_map(|(_, p)| &p.anomalies)
                    .find(|a| a.kind == kind)
                {
                    return (config, instance.clone());
                }
            }
        }
        panic!("no {kind:?} fixture found");
    }

    #[test]
    fn pillar_epoch_changes_only_committed_wake_and_preserves_bearing_lane() {
        use crate::domain::entities::anomaly::{AnomalyStateStamp, AxisDirection};
        let (mut config, instance) = find_macro_anomaly(AnomalyKind::PillarExpanse);
        config.anomalies.remap_intensity = 4.0;
        let gate = instance.gates[instance.gates.len() / 2];
        let reality = RealitySnapshot::new(vec![AnomalyStateStamp::new(
            instance.id,
            1,
            gate.plane,
            gate.axis,
            AxisDirection::Positive,
        )]);
        let empty = RealitySnapshot::empty();
        let noise = SimpleNoiseProvider::new();
        let mut changed = 0usize;
        let mut unchanged_forward = 0usize;
        let mut lz = -instance.footprint.half_z + 1.0;
        while lz < instance.footprint.half_z - 1.0 {
            let mut lx = -instance.footprint.half_x + 1.0;
            while lx < instance.footprint.half_x - 1.0 {
                let p = instance.world_coords(lx, lz);
                let a = BackroomsLevel::anomaly_column(
                    &instance, &noise, 42, &config, &empty, p.x, p.z,
                );
                let b = BackroomsLevel::anomaly_column(
                    &instance, &noise, 42, &config, &reality, p.x, p.z,
                );
                let in_wake =
                    reality.stamps()[0].point_is_in_wake(p.x, p.z, config.anomalies.remap_distance);
                if a.solid != b.solid {
                    assert!(in_wake, "geometry changed ahead of the crossed gate");
                    assert!(lz.abs() > instance.skeleton_half_width);
                    changed += 1;
                } else if !in_wake {
                    unchanged_forward += 1;
                }
                if lz.abs() <= instance.skeleton_half_width {
                    assert!(!a.solid && !b.solid, "bearing lane was blocked");
                }
                lx += 0.4;
            }
            lz += 0.4;
        }
        assert!(changed > 0, "pillar epoch produced no changed wake infill");
        assert!(unchanged_forward > 100);
    }

    #[test]
    fn blackout_has_a_recoverable_glimmer_lane_and_compressed_dark_core() {
        let (config, instance) = find_macro_anomaly(AnomalyKind::BlackoutExpanse);
        let noise = SimpleNoiseProvider::new();
        let lane = instance.world_coords(0.0, 0.0);
        let lane_plan = BackroomsLevel::anomaly_column(
            &instance,
            &noise,
            42,
            &config,
            &RealitySnapshot::empty(),
            lane.x,
            lane.z,
        );
        assert!(!lane_plan.solid, "blackout recovery skeleton is blocked");
        let core = instance.world_coords(0.0, instance.skeleton_half_width + 3.0);
        let core_plan = BackroomsLevel::anomaly_column(
            &instance,
            &noise,
            42,
            &config,
            &RealitySnapshot::empty(),
            core.x,
            core.z,
        );
        assert!(!core_plan.light, "blackout core has an ordinary fixture");
        assert_eq!(core_plan.ceiling_units, 2.6);
    }

    #[test]
    fn pit_lattice_omits_real_floor_and_exports_relocation_hazards() {
        let (config, instance) = find_macro_anomaly(AnomalyKind::PitLattice);
        let hazards = instance.pit_hazards_for_bounds(instance.footprint.bounds());
        assert!(hazards.len() > 20, "pit lattice is not a room-scale hazard");
        let h = hazards[0];
        let plan = BackroomsLevel::anomaly_column(
            &instance,
            &SimpleNoiseProvider::new(),
            42,
            &config,
            &RealitySnapshot::empty(),
            h.center.x,
            h.center.z,
        );
        assert!(!plan.floor, "pit center still voxelizes a floor slab");
        assert!(!h.contains(h.recovery.x, h.recovery.z));

        let ox = (h.center.x / config.chunk_size).floor() * config.chunk_size;
        let oz = (h.center.z / config.chunk_size).floor() * config.chunk_size;
        let grid = BackroomsLevel.generate_with_reality(
            Position::new(ox, oz),
            42,
            config,
            &SimpleNoiseProvider::new(),
            &RealitySnapshot::empty(),
        );
        assert!(grid.pit_hazards.iter().any(|x| x.id == h.id));
    }

    #[test]
    fn red_threshold_closes_the_remembered_entrance_into_a_loop() {
        use crate::domain::entities::anomaly::AxisDirection;
        let noise = SimpleNoiseProvider::new();
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.pillar_expanses = 0.0;
        config.anomalies.blackouts = 0.0;
        config.anomalies.pit_lattices = 0.0;
        let mut fixture = None;
        for rz in -6i64..=6 {
            for rx in -6i64..=6 {
                let plans = BackroomsLevel::region_plans_for(
                    Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                    1.0,
                    42,
                    &config,
                    &noise,
                );
                let plan = &plans.iter().find(|(k, _)| *k == (rx, rz)).unwrap().1;
                if let Some(red) = plan
                    .anomalies
                    .iter()
                    .find(|a| a.kind == AnomalyKind::RedRoom)
                {
                    fixture = Some((plan.clone(), red.clone()));
                    break;
                }
            }
            if fixture.is_some() {
                break;
            }
        }
        let (plan, red) = fixture.expect("red-room fixture");
        let gate = red.gates[0];
        let reality = RealitySnapshot::new(vec![
            crate::domain::entities::anomaly::AnomalyStateStamp::new(
                red.id,
                1,
                gate.plane,
                gate.axis,
                AxisDirection::Positive,
            ),
        ]);
        let entrance = plan
            .assemblies
            .iter()
            .find(|a| a.corruption.red_room)
            .unwrap()
            .entrances[0];
        let open = BackroomsLevel::plan_column_in_reality(
            &plan,
            &noise,
            42,
            &config,
            &RealitySnapshot::empty(),
            entrance.center.x,
            entrance.center.z,
        );
        let closed = BackroomsLevel::plan_column_in_reality(
            &plan,
            &noise,
            42,
            &config,
            &reality,
            entrance.center.x,
            entrance.center.z,
        );
        assert!(!open.solid, "red room is closed before threshold entry");
        assert!(closed.solid, "remembered red-room entrance did not close");

        let ring_point = red.world_coords(red.footprint.half_x - 1.0, 0.0);
        let ring = BackroomsLevel::close_red_room(
            &red,
            ColumnPlan {
                floor: true,
                solid: false,
                ceiling_units: 3.4,
                light: false,
                red_light: false,
                lintel_from_units: None,
                sconce: false,
            },
            &config,
            &reality,
            ring_point.x,
            ring_point.z,
        );
        assert!(!ring.solid, "closed red room has no traversable loop");
    }

    #[test]
    fn test_print_ascii_map() {
        let noise = SimpleNoiseProvider::new();
        let tuning = LevelTuning::default();
        let config = GeneratorConfig::low_spec();
        // The whole of region (0,0) at 0.5 u per character.
        let plans =
            BackroomsLevel::region_plans_for(Position::new(0.0, 0.0), 80.0, 42, &config, &noise);
        let plan = &plans.iter().find(|(k, _)| *k == (0, 0)).unwrap().1;
        let mut map = String::new();
        for sz in 0..160 {
            for sx in 0..160 {
                let wx = sx as f32 * 0.5 + 0.25;
                let wz = sz as f32 * 0.5 + 0.25;
                let col = BackroomsLevel::plan_column(plan, &noise, 42, &tuning, wx, wz);
                if col.solid {
                    map.push('#');
                } else if col.light {
                    map.push('*');
                } else if col.lintel_from_units.is_some() {
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
