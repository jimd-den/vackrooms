//! Tuning knobs for the CPU splatter.
//!
//! Two kinds of control live here:
//!
//! * **Quality/perf scalars** (`CpuRenderSettings`) — continuous knobs the
//!   settings menu and the URL adjust: internal resolution, LOD thresholds,
//!   fog shape, draw distance.
//! * **Optimization switches** (`toggles`) — the shared
//!   [`RenderToggles`] switchboard; every labeled optimization in this
//!   renderer can be turned off individually to fall back to the simple
//!   reference behavior.
//!
//! The struct is plain data: the browser driver copies it in once per frame
//! (`drivers::cpu_canvas`), so the rasterizer itself never touches globals.

use crate::application::render_settings::RenderToggles;

/// Per-splat shadow tracing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuShadowMode {
    /// No shadow rays at all.
    Off,
    /// Trace toward the nearest assumed ceiling fixture, amortized over
    /// frames through a position-hashed cache (see
    /// `rasterizer::SoftwareRasterizer::shadow_factor`).
    Hero,
}

/// All scalar knobs for the software rasterizer, plus the shared
/// optimization switchboard.
#[derive(Debug, Clone, Copy)]
pub struct CpuRenderSettings {
    /// Internal resolution multiplier on top of the driver's 480x270 cap
    /// (0.25 ..= 1.0).
    pub internal_scale: f32,
    /// Projected-radius threshold (px) below which a subtree becomes one
    /// MIP splat (0.25 ..= 2.0). Smaller = more detail, more nodes visited.
    pub lod_cutoff_px: f32,
    /// A collapsed solid leaf is virtually subdivided until its splat is at
    /// most this many half-pixels wide (2.0 ..= 16.0).
    pub max_splat_half_px: f32,
    /// Recursion cap for that virtual subdivision (3 ..= 8).
    pub max_virtual_depth: u8,
    /// World-space floor for virtual subdivision, so degenerate splits
    /// can't recurse into dust.
    pub min_split_size: f32,
    /// Exponential fog density (0.003 ..= 0.03).
    pub fog_density: f32,
    /// Distance where fog starts accumulating.
    pub fog_start: f32,
    /// Chunk/node rejection distance in world units.
    pub max_draw_distance: f32,
    /// Minimum subtree occupancy for a distant MIP splat to draw at all —
    /// preserves sparse far detail without spraying near-empty splats.
    pub min_mip_occupancy: f32,
    /// Per-splat shadow rays (Off / Hero).
    pub shadows: CpuShadowMode,
    /// tan(vertical FOV / 2); 0.767 = the 75-degree default.
    pub fov_tan: f32,
    /// Per-optimization on/off switches shared with the GPU paths.
    pub toggles: RenderToggles,
}

impl Default for CpuRenderSettings {
    fn default() -> Self {
        Self {
            internal_scale: 1.0,
            lod_cutoff_px: 1.0,
            max_splat_half_px: 12.0,
            max_virtual_depth: 5,
            min_split_size: 0.02,
            fog_density: 0.015,
            fog_start: 0.0,
            max_draw_distance: 96.0,
            min_mip_occupancy: 0.25,
            shadows: CpuShadowMode::Off,
            fov_tan: 0.767, // 75 deg vertical FOV
            toggles: RenderToggles::default(),
        }
    }
}
