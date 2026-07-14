//! The endless unplanned office fabric: an irregular warren of yellow rooms
//! on a hidden lattice, chained through hashed doorways (binary-tree rule
//! guarantees global connectivity chunk-locally) and merged by
//! porosity-driven wall dropout so no grid is ever readable. Open expanses
//! and vaults punctuate it; they are never the default.

use crate::domain::entities::anomaly::RealitySnapshot;
use crate::entities::models::Position;
use crate::domain::entities::voxel_grid::{VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_WALL};
use crate::use_cases::generate_chunk::LevelTuning;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::region_plan::PLAN_WALL_T;

use super::{BackroomsLevel, ColumnPlan, DOOR_HEIGHT, DOOR_WIDTH};

/// The default fabric room lattice. Rooms are chained through hashed
/// doorways and merged by wall dropout, so the cell size never reads as a
/// grid from inside — it is the scale of the labyrinth, not its shape.
pub(super) const FABRIC_CELL: f32 = 7.2;

/// Column grid spacing inside expanses.
const EXPANSE_COLUMN_PERIOD: f32 = 7.2;

/// Ceiling light panel spacing. The coffer beam grid shares this period,
/// but beams are a renderer shading pattern, never stepped ceiling geometry.
pub(super) const LIGHT_PERIOD: f32 = 2.8;

/// The broad ceiling hierarchy that gives Level 0 scale without turning the
/// whole map into a warehouse. It is local implementation detail rather than
/// a world semantic: assemblies and corridors may override it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FabricCeilingBand {
    Compression,
    Regular,
    Expanse,
    Vault,
}

impl BackroomsLevel {
    /// Smooth world-space noise in [-1, 1]. `scale` stretches the provider's
    /// built-in ~20 u wavelength: wavelength = 20 / scale.
    pub(super) fn n(noise: &dyn NoiseProvider, seed: u32, salt: u32, x: f32, z: f32, scale: f32) -> f32 {
        noise.evaluate_2d(seed ^ salt, Position::new(x * scale, z * scale))
    }

    /// Decorrelated per-cell hash in [0, 1): samples the value noise exactly
    /// on its lattice (provider frequency is 0.05, so inputs that are
    /// multiples of 20 hit lattice points), where it is uniform rather than
    /// interpolation-smoothed toward zero.
    pub(super) fn cell_hash(noise: &dyn NoiseProvider, seed: u32, salt: u32, cx: i64, cz: i64) -> f32 {
        let v = noise.evaluate_2d(
            seed ^ salt,
            Position::new(cx as f32 * 20.0, cz as f32 * 20.0),
        );
        (v * 0.5 + 0.5).clamp(0.0, 0.999)
    }

    /// Broad, low-frequency ceiling territory. The thresholds are chosen to
    /// bias the sampled world toward regular dropped ceilings while retaining
    /// meaningful regions of open volume and rare compression.
    pub(super) fn fabric_ceiling_band(
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
    pub(super) fn fabric_ceiling_height(
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
    pub(super) fn in_expanse(noise: &dyn NoiseProvider, seed: u32, wx: f32, wz: f32) -> bool {
        matches!(
            Self::fabric_ceiling_band(noise, seed, wx, wz),
            FabricCeilingBand::Expanse | FabricCeilingBand::Vault
        )
    }

    /// The fabric plan as it was first observed — Peripheral Shift epoch 0
    /// everywhere. Kept for callers that deliberately freeze the fabric.
    #[cfg(test)]
    pub(crate) fn column_plan(
        noise: &dyn NoiseProvider,
        seed: u32,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        Self::column_plan_in_reality(noise, seed, tuning, &RealitySnapshot::empty(), wx, wz)
    }

    /// The *fabric* column plan at world position (wx, wz): the endless,
    /// unplanned office fill between planned corridors and assemblies.
    ///
    /// This is where the wiki's Peripheral Shift lives: `reality` carries a
    /// drift epoch per 40 u cell, advanced by the engine whenever territory
    /// goes unobserved. The epoch re-salts only the *cosmetic and porosity*
    /// decisions — wall dropout, doorway direction/position/width framing,
    /// dead lights — while the ceiling territories, porosity climate,
    /// junction posts, and expanse structure stay fixed, so a returning
    /// wanderer recognizes the neighborhood but never the hallways. Every
    /// decision reads its epoch at the deciding lattice cell's own anchor,
    /// so a wall is rebuilt whole even when a drift-cell boundary crosses it,
    /// and the binary-tree doorway rule holds per cell at any epoch mix —
    /// the labyrinth stays globally connected through every rearrangement.
    pub(crate) fn column_plan_in_reality(
        noise: &dyn NoiseProvider,
        seed: u32,
        tuning: &LevelTuning,
        reality: &RealitySnapshot,
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
                // The whole fabric cell rearranges as one: its epoch is read
                // at the cell's own center, never at the sampled column.
                let epoch = reality.fabric_drift_epoch(
                    (cx as f32 + 0.5) * FABRIC_CELL,
                    (cz as f32 + 0.5) * FABRIC_CELL,
                );
                let drift = |salt: u32| salt ^ epoch.wrapping_mul(0x9E37_79B9);
                // 0 = tight labyrinth, 1 = broken-open suites; drifts over
                // ~180 u so density changes read as neighborhoods, not zones.
                // The porosity climate is character, not layout: it survives
                // every Peripheral Shift, so a broken-open neighborhood
                // rearranges into another broken-open neighborhood.
                let porosity =
                    (Self::n(noise, seed, 0x9010, wx, wz, 0.11) * 0.5 + 0.5).clamp(0.0, 1.0);
                let opens_west = Self::cell_hash(noise, seed, drift(0x9200), cx, cz) < 0.5;
                let (wall_salt, door_salt, opens_here) = if in_w {
                    (drift(0x9300u32), drift(0x9500u32), opens_west)
                } else {
                    (drift(0x9400u32), drift(0x9600u32), !opens_west)
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
            // Which panels burned out is cosmetic memory, so the Peripheral
            // Shift re-deals it: the light that guided you out may be dead
            // when you walk back in.
            let epoch = reality.fabric_drift_epoch(
                (cell_x as f32 + 0.5) * panel_period,
                (cell_z as f32 + 0.5) * panel_period,
            );
            let salt = 0xE900 ^ epoch.wrapping_mul(0x9E37_79B9);
            let alive = Self::cell_hash(noise, seed, salt, cell_x, cell_z) < keep;
            lx < 0.45 && lz < 0.45 && alive
        } else {
            false
        };

        ColumnPlan {
            floor: true,
            floor_units: 0.0,
            solid,
            ceiling_units,
            light,
            red_light: false,
            lintel_from_units,
            wall_material: VOXEL_WALL,
            floor_material: VOXEL_FLOOR,
            light_material: VOXEL_LIGHT,
        }
    }
}
