//! CPU microvoxel rasterizer facade and frame pipeline.
//!
//! The persistent renderer owns reusable allocations, settings, telemetry,
//! atlas data, and the amortized shadow cache. Focused child modules own the
//! two dense algorithms:
//!
//! * [`target`] — framebuffer, fine depth, conservative hierarchical Z;
//! * [`traversal`] — projection, SVO walking, LOD, and virtual subdivision.
//!
//! Color remains in [`super::shading`], secondary rays in
//! [`super::raycast`], and per-frame camera math in [`super::camera`].

mod target;
mod traversal;

use crate::application::ports::{ChunkDraw, Environment, FrameParams, RendererPort};

use super::atlas::{MipNode, build_mips};
use super::camera::{Camera, dot};
use super::raycast::trace_svo;
use super::settings::{CpuRenderSettings, CpuShadowMode};
use super::shading::{self, SplatSurface};
use target::{RenderTarget, SquareSplat};
use traversal::{chunk_distance_sq, chunk_visible};

/// Per-frame counters for the HUD; tests also use them to prove termination
/// and safety-cap behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoftwareRasterizerTelemetry {
    pub visited_nodes: usize,
    pub budget_exhausted: bool,
    pub max_virtual_depth: usize,
    pub splat_count: usize,
    pub pixel_writes: usize,
}

/// Non-optional safety caps. The node cap degrades to current-fidelity/MIP
/// splats; the write cap stops inside the exact pixel loop.
const MAX_VISITED_NODES: usize = 150_000;
const MAX_PIXEL_WRITES: usize = 2_000_000;
/// Above this write level, hero shadow rays are no longer affordable.
const SHADOW_PIXEL_BUDGET: usize = 1_500_000;

/// Persistent state for the CPU renderer. The struct is intentionally the
/// allocation-owning facade; rendering algorithms live in focused modules.
pub struct SoftwareRasterizer {
    target: RenderTarget,
    /// SVO atlas texels: four `u32`s per node.
    atlas: Vec<u32>,
    mips: Vec<MipNode>,

    // Public counters are retained for compatibility; `telemetry()` is the
    // stable snapshot API used by the browser driver and tests.
    pub visited_nodes: usize,
    pub budget_exhausted: bool,
    pub max_virtual_depth_reached: usize,
    pub splat_count: usize,
    pub pixel_writes: usize,

    pub settings: CpuRenderSettings,
    /// Position-hashed hero-shadow cache; each entry retraces every fourth
    /// frame, staggered by its hash.
    shadow_cache: Vec<f32>,
    frame_index: usize,
    /// Current level atmosphere, captured once at the start of `draw`.
    environment: Environment,
}

impl SoftwareRasterizer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            target: RenderTarget::new(width, height),
            atlas: Vec::new(),
            mips: Vec::new(),
            visited_nodes: 0,
            budget_exhausted: false,
            max_virtual_depth_reached: 0,
            splat_count: 0,
            pixel_writes: 0,
            settings: CpuRenderSettings::default(),
            shadow_cache: vec![1.0; 65_536],
            frame_index: 0,
            environment: Environment::default(),
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.target.resize(width, height);
    }

    pub fn width(&self) -> usize {
        self.target.width()
    }

    pub fn height(&self) -> usize {
        self.target.height()
    }

    /// The finished frame as top-down RGBA8 rows.
    pub fn framebuffer(&self) -> &[u8] {
        self.target.rgba()
    }

    pub fn telemetry(&self) -> SoftwareRasterizerTelemetry {
        SoftwareRasterizerTelemetry {
            visited_nodes: self.visited_nodes,
            budget_exhausted: self.budget_exhausted,
            max_virtual_depth: self.max_virtual_depth_reached,
            splat_count: self.splat_count,
            pixel_writes: self.pixel_writes,
        }
    }

    /// Test seam and frame clear: atmosphere chooses the background; target
    /// storage owns color/depth/HZ reset mechanics.
    pub(super) fn clear(&mut self) {
        let background = if self.environment.outdoor {
            [
                (self.environment.sky_color[0] * 255.0) as u8,
                (self.environment.sky_color[1] * 255.0) as u8,
                (self.environment.sky_color[2] * 255.0) as u8,
            ]
        } else {
            [0, 0, 0]
        };
        self.target.clear(background);
    }

    /// Frame-policy wrapper around the pixel-only target operation. It keeps
    /// the exact in-loop write cap and renderer telemetry out of `target`.
    fn splat(&mut self, cx: f32, cy: f32, half: f32, z: f32, rgb: [u8; 3]) {
        if self.pixel_writes >= MAX_PIXEL_WRITES {
            self.budget_exhausted = true;
            return;
        }
        self.splat_count += 1;

        let result = self.target.write_splat(
            SquareSplat::new(cx, cy, half, z, rgb),
            MAX_PIXEL_WRITES - self.pixel_writes,
            self.settings.toggles.hierarchical_z,
        );
        self.pixel_writes += result.pixel_writes;
        self.budget_exhausted |= result.budget_exhausted;
    }

    /// Shades one exact box and writes its square splat. Retained as a narrow
    /// test seam; lighting itself remains in `shading`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn shade_and_splat(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        center: [f32; 3],
        world_size: f32,
        px: f32,
        py: f32,
        half_px: f32,
        dist: f32,
        base_color: [f32; 3],
        light_level: f32,
        is_emissive: bool,
        crowded_siblings: u32,
        shadow_factor: f32,
    ) {
        let surface = SplatSurface {
            center,
            world_size,
            dist,
            base_color,
            light_level,
            is_emissive,
            crowded_siblings,
            shadow_factor,
        };
        let rgb = shading::shade(
            cam,
            &self.environment,
            &self.settings,
            &self.atlas,
            chunks,
            &surface,
        );
        self.splat(px, py, half_px, dist, rgb);
    }

    /// Hero-light visibility, amortized through a position-hashed cache.
    fn shadow_factor(
        &mut self,
        chunks: &[ChunkDraw],
        center: [f32; 3],
        is_emissive: bool,
        dist: f32,
    ) -> f32 {
        let affordable = dist <= 24.0 && self.pixel_writes < SHADOW_PIXEL_BUDGET;
        if self.settings.shadows != CpuShadowMode::Hero || is_emissive || !affordable {
            return 1.0;
        }

        let ix = center[0].floor() as i32;
        let iy = center[1].floor() as i32;
        let iz = center[2].floor() as i32;
        let hash_key = (ix.wrapping_mul(73_856_093)
            ^ iy.wrapping_mul(19_349_663)
            ^ iz.wrapping_mul(83_492_791)) as usize;
        let cache_index = hash_key % self.shadow_cache.len();

        let should_trace = (hash_key + self.frame_index).is_multiple_of(4);
        if !should_trace && self.shadow_cache[cache_index] != 0.0 {
            return self.shadow_cache[cache_index];
        }

        let light_position = [
            (center[0] / 3.0).floor() * 3.0 + 1.5,
            3.6,
            (center[2] / 3.0).floor() * 3.0 + 1.5,
        ];
        let to_light = [
            light_position[0] - center[0],
            light_position[1] - center[1],
            light_position[2] - center[2],
        ];
        let distance_to_light = dot(to_light, to_light).sqrt();
        let factor = if distance_to_light > 0.001 {
            let inverse_distance = 1.0 / distance_to_light;
            let direction = to_light.map(|component| component * inverse_distance);
            let origin = [
                center[0] + direction[0] * 0.05,
                center[1] + direction[1] * 0.05,
                center[2] + direction[2] * 0.05,
            ];
            if trace_svo(
                &self.atlas,
                chunks,
                origin,
                direction,
                distance_to_light - 0.05,
                self.settings.toggles.front_to_back,
            )
            .is_some()
            {
                0.25
            } else {
                1.0
            }
        } else {
            1.0
        };
        self.shadow_cache[cache_index] = factor;
        factor
    }

    fn reset_telemetry(&mut self) {
        self.visited_nodes = 0;
        self.budget_exhausted = false;
        self.max_virtual_depth_reached = 0;
        self.splat_count = 0;
        self.pixel_writes = 0;
    }

    fn draw_flare_cores(&mut self, frame: &FrameParams, cam: &Camera) {
        for light in frame.active_dynamic_lights() {
            let relative = [
                light.position[0] - cam.pos[0],
                light.position[1] - cam.pos[1],
                light.position[2] - cam.pos[2],
            ];
            let camera_depth = dot(relative, cam.forward);
            if camera_depth <= 0.1 || camera_depth > self.settings.max_draw_distance {
                continue;
            }

            let pixel_x = cam.half_w + dot(relative, cam.right) / camera_depth * cam.focal_px;
            let pixel_y = cam.half_h - dot(relative, cam.up) / camera_depth * cam.focal_px;
            let half = (0.06 / camera_depth * cam.focal_px).clamp(1.0, 6.0);
            let intensity = light.intensity.clamp(0.0, 1.6) / 1.6;
            self.splat(
                pixel_x,
                pixel_y,
                half,
                camera_depth - 0.05,
                [
                    (255.0 * intensity) as u8,
                    (150.0 * intensity) as u8,
                    (60.0 * intensity) as u8,
                ],
            );
        }
    }
}

impl RendererPort for SoftwareRasterizer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        self.atlas = texels.to_vec();
        // MIP attributes are rebuilt on upload, never per frame.
        self.mips = build_mips(&self.atlas);
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        self.environment = frame.environment;
        self.clear();
        self.reset_telemetry();
        if self.atlas.is_empty() {
            return;
        }

        let camera = Camera::new(frame, self.width(), self.height(), &self.settings);
        let mut order: Vec<&ChunkDraw> = chunks.iter().collect();

        // OPTIMIZATION (front-to-back): nearest chunks populate depth first.
        if self.settings.toggles.front_to_back {
            order.sort_by(|a, b| {
                chunk_distance_sq(a, frame.camera_pos)
                    .total_cmp(&chunk_distance_sq(b, frame.camera_pos))
            });
        }

        for chunk in order {
            // OPTIMIZATION (distance cull): reject chunks outside the draw
            // distance or conservative screen bounds.
            if self.settings.toggles.distance_cull
                && !chunk_visible(
                    chunk,
                    &camera,
                    self.width(),
                    self.height(),
                    self.settings.max_draw_distance,
                )
            {
                continue;
            }
            self.render_chunk(&camera, chunks, chunk);
        }

        // Flare cores use the ordinary fine depth buffer, so walls occlude
        // them without a separate visibility query.
        self.draw_flare_cores(frame, &camera);
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}
