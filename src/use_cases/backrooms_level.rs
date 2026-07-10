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
//! * **Fabric**: partitions are sparse 18 u fragments. Framed doors occur on
//!   only a small portion of retained walls; broad openings and complete
//!   dissolves into expanses do most of the connectivity work.

use crate::domain::entities::architecture::{
    AssemblyInstance, CeilingLanguage, CirculationSpine, RegionPlan, SpaceProgram,
    StructuralSystem, StructuralSystemInstance,
};
use crate::domain::entities::voxel_grid::{
    VOXEL_CEILING, VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_RED_WALL, VOXEL_WALL, VoxelGrid,
};
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::{GeneratorConfig, LevelTuning};
use crate::use_cases::level_generator::LevelGenerator;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::region_plan::{PLAN_WALL_T, REGION_SIZE, generate_region_plan, region_index};

/// Ceiling height of the tallest (atrium) vaults, world units.
pub const MAX_CEILING_UNITS: f32 = 5.4;
/// Total grid height: headroom above the tallest vault.
pub const GRID_HEIGHT_UNITS: f32 = 5.8;

/// Fragmented partition fabric: long 18 u wall runs can survive before a
/// meaningful opening, unlike the old 6 u office-cell cadence.
const WALL_PERIOD: f32 = 18.0;
const DOOR_WIDTH: f32 = 1.2;
const DOOR_HEIGHT: f32 = 2.2;

/// Column grid spacing inside expanses.
const EXPANSE_COLUMN_PERIOD: f32 = 7.2;

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
        let field = Self::n(noise, seed, 0xAA10, wx, wz, 0.18);
        if field < -0.60 {
            FabricCeilingBand::Compression
        } else if field < 0.36 {
            FabricCeilingBand::Regular
        } else if field < 0.68 {
            FabricCeilingBand::Expanse
        } else {
            FabricCeilingBand::Vault
        }
    }

    fn fabric_ceiling_height(
        noise: &dyn NoiseProvider,
        seed: u32,
        wx: f32,
        wz: f32,
        band: FabricCeilingBand,
    ) -> f32 {
        let detail = Self::n(noise, seed, 0xB300, wx, wz, 0.72);
        match band {
            FabricCeilingBand::Compression => (2.65 + 0.15 * detail).clamp(2.5, 2.8),
            FabricCeilingBand::Regular => (3.4 + 0.2 * detail).clamp(3.2, 3.6),
            FabricCeilingBand::Expanse => (4.1 + 0.3 * detail).clamp(3.8, 4.4),
            FabricCeilingBand::Vault => (4.95 + 0.45 * detail).clamp(4.5, 5.4),
        }
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
        let mut ceiling_units = Self::fabric_ceiling_height(noise, seed, wx, wz, ceiling_band);

        // In open and vaulted regions, partitions dissolve and only sparse
        // structural columns remain. The cheap dropped-ceiling material is
        // therefore carried through unexpectedly large space.
        let expanse = matches!(
            ceiling_band,
            FabricCeilingBand::Expanse | FabricCeilingBand::Vault
        );

        // Coffered ceiling grid (expanses run an open plenum instead).
        let on_beam = wx.rem_euclid(COFFER_PERIOD) < 0.22 || wz.rem_euclid(COFFER_PERIOD) < 0.22;
        if !expanse && on_beam {
            ceiling_units -= COFFER_DROP;
        }

        // ---- solids ------------------------------------------------------
        let spawn_d2 = (wx - SPAWN.0) * (wx - SPAWN.0) + (wz - SPAWN.1) * (wz - SPAWN.1);
        let in_spawn = spawn_d2 < SPAWN_CLEAR_RADIUS * SPAWN_CLEAR_RADIUS;

        let mut solid = false;
        let mut lintel_from_units: Option<f32> = None;
        let sconce = false;

        if !in_spawn && expanse {
            // Sparse structural columns hold the expanse ceiling up.
            let cx = (wx / EXPANSE_COLUMN_PERIOD).floor() as i64;
            let cz = (wz / EXPANSE_COLUMN_PERIOD).floor() as i64;
            let on_site = wx.rem_euclid(EXPANSE_COLUMN_PERIOD) < 0.45
                && wz.rem_euclid(EXPANSE_COLUMN_PERIOD) < 0.45;
            if on_site && Self::cell_hash(noise, seed, 0xF200, cx, cz) < 0.7 * tuning.pillars {
                solid = true;
            }
        } else if !in_spawn {
            let fx = wx.rem_euclid(WALL_PERIOD);
            let fz = wz.rem_euclid(WALL_PERIOD);
            let cell_x = (wx / WALL_PERIOD).floor() as i64;
            let cell_z = (wz / WALL_PERIOD).floor() as i64;

            // (solid, lintel) contribution of one wall family. Directional
            // noise keeps a retained partition alive over several large
            // periods, so it reads as a broken wall mass instead of cell
            // boundaries feeding a maze.
            let wall_here = |f_wall: f32, f_along: f32, is_z_wall: bool| {
                if f_wall >= PLAN_WALL_T {
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

                // Threshold formula: 1.0 - walls.clamp(0, 2), with the
                // default retaining roughly half of these *18 u* candidates.
                // The visual default is therefore open fabric punctuated by
                // long wall runs rather than close-set corridors.
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

                // A retained segment may have one meaningful interruption.
                // Framed doors are rare evidence of a former office; broad,
                // full-height portals are more common but still live on the
                // 18 u cadence, never a repetitive doorway grid.
                let door_salt = if is_z_wall { 0xD700 } else { 0xD800 };
                let door_hash = Self::cell_hash(noise, seed, door_salt, cx, cz);
                if door_hash > 0.88 {
                    let door_pos = (WALL_PERIOD - DOOR_WIDTH) * 0.5;
                    if f_along >= door_pos && f_along < door_pos + DOOR_WIDTH {
                        return (false, true); // Doorway: open below, lintel above
                    }
                } else {
                    let portal_salt = if is_z_wall { 0xD710 } else { 0xD810 };
                    let portal_hash = Self::cell_hash(noise, seed, portal_salt, cx, cz);
                    if portal_hash > 0.66 {
                        let portal_width = 4.8;
                        let portal_pos = (WALL_PERIOD - portal_width) * 0.5;
                        if f_along >= portal_pos && f_along < portal_pos + portal_width {
                            return (false, false); // Full-height broad portal
                        }
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
            // Open regions keep a longer, sparser fluorescent rhythm; the
            // regular dropped ceiling retains the denser office grid.
            let panel_period = if expanse { 4.8 } else { LIGHT_PERIOD };
            let lx = (wx - panel_period * 0.5).rem_euclid(panel_period);
            let lz = (wz - panel_period * 0.5).rem_euclid(panel_period);
            let cell_x = (wx / panel_period).floor() as i64;
            let cell_z = (wz / panel_period).floor() as i64;
            let keep = if expanse { 0.66 } else { 0.78 } * tuning.lights;
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
            Some(CeilingLanguage::Coffered) => {
                if wx.rem_euclid(COFFER_PERIOD) < 0.22 || wz.rem_euclid(COFFER_PERIOD) < 0.22 {
                    ceiling_units -= COFFER_DROP;
                }
            }
            Some(CeilingLanguage::ExposedSoffit) => ceiling_units -= 0.2,
            _ => {}
        }
        let mut plan = ColumnPlan {
            solid: false,
            red: false,
            ceiling_units,
            light: false,
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
                    plan.solid = true;
                    plan.red = true; // renovation columns read as intrusions
                }
            }
        }

        // Fixtures, tied to the assembly's ceiling modules.
        if !plan.solid && tuning.lights > 0.0 {
            for f in &a.fixtures {
                if f.lit && (wx - f.at.x).abs() <= f.half_x && (wz - f.at.z).abs() <= f.half_z {
                    plan.light = true;
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
        let drift = Self::n(noise, seed, 0xCA00_u32.wrapping_add(spine.id), wx, wz, 0.35);
        match spine.spine_kind {
            SpaceProgram::MainCorridor => (3.8 + 0.4 * drift).clamp(3.4, 4.2),
            SpaceProgram::SecondaryHall => (3.3 + 0.3 * drift).clamp(3.0, 3.6),
            _ => 3.4,
        }
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

    /// The architectural column plan: corridor beats assembly beats fabric.
    pub(crate) fn plan_column(
        plan: &RegionPlan,
        noise: &dyn NoiseProvider,
        seed: u32,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        let spawn_d2 = (wx - SPAWN.0) * (wx - SPAWN.0) + (wz - SPAWN.1) * (wz - SPAWN.1);
        let in_spawn = spawn_d2 < SPAWN_CLEAR_RADIUS * SPAWN_CLEAR_RADIUS;

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
        if in_corridor && !in_spawn {
            return ColumnPlan {
                solid: false,
                red: false,
                ceiling_units: corridor_ceiling,
                light: corridor_light && tuning.lights > 0.0,
                lintel_from_units: None,
                sconce: false,
            };
        }

        // -- assemblies -------------------------------------------------------
        if !in_spawn {
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
                return Self::assembly_column(
                    a,
                    inside,
                    renovator_structure.as_ref(),
                    tuning,
                    wx,
                    wz,
                );
            }
        }

        // -- corridor edge walls through fabric -------------------------------
        if corridor_wall && !in_spawn && tuning.walls > 0.0 && !corridor_gap {
            return ColumnPlan {
                solid: true,
                red: false,
                ceiling_units: corridor_ceiling.max(3.2),
                light: false,
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
        let s = config.voxel_scale;
        let width = (config.chunk_size / s).round() as usize;
        let depth = (config.chunk_size / s).round() as usize;
        let height = (GRID_HEIGHT_UNITS / s) as usize;
        let mut grid = VoxelGrid::new(width, height, depth);

        // Architecture first: plan every region this chunk overlaps. The
        // plans are pure functions of (seed, region), so any chunk in the
        // region sees the identical plan.
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

        // Plan every column plus a 1-voxel margin: ceiling skirts must seal
        // height steps across chunk borders too.
        let tuning = config.tuning;
        let plan_at = |lx: i64, lz: i64| -> ColumnPlan {
            let wx = chunk_pos.x + (lx as f32 + 0.5) * s;
            let wz = chunk_pos.z + (lz as f32 + 0.5) * s;
            BackroomsLevel::plan_column(plan_of(wx, wz), noise, seed, &tuning, wx, wz)
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
                let got_solid =
                    grid.get(lx, 1, z) == VOXEL_WALL || grid.get(lx, 1, z) == VOXEL_RED_WALL;
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
        let config = GeneratorConfig::low_spec();
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
        assert!(
            (0.45..=0.70).contains(&ratio(1)),
            "ceiling territories: compression {:.1}%, regular {:.1}%, expanse {:.1}%, vault {:.1}%",
            ratio(0) * 100.0,
            ratio(1) * 100.0,
            ratio(2) * 100.0,
            ratio(3) * 100.0
        );
        assert!(
            (0.18..=0.42).contains(&ratio(2)),
            "open expanse territory was {:.1}%",
            ratio(2) * 100.0
        );
        assert!(
            (0.05..=0.22).contains(&ratio(3)),
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
        let config = GeneratorConfig::low_spec();
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
        let solid = |g: &VoxelGrid, x: usize, z: usize| {
            g.get(x, 1, z) == VOXEL_WALL || g.get(x, 1, z) == VOXEL_RED_WALL
        };
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
