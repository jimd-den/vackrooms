//! Persistent state and public facade for CPU voxel splatting.
//!
//! This file deliberately contains no frame algorithm. It owns allocations
//! and the stable [`RendererPort`] API; purpose-named child modules explain
//! what happens to a frame:
//!
//! * [`render_cpu_frame`] orchestrates one complete frame;
//! * [`traverse_voxel_scene`] projects and walks the SVO;
//! * [`resolve_hero_light_visibility`] amortizes secondary visibility rays;
//! * [`select_lights_for_chunk`] builds exact finite-support light clusters;
//! * [`draw_visible_flare_cores`] draws depth-tested dynamic-light cores;
//! * [`write_depth_tested_splats`] owns framebuffer and hierarchical Z data.
//!
//! Color remains in [`super::shading`], secondary rays in
//! [`super::raycast`], and per-frame camera math in [`super::camera`].

mod draw_visible_flare_cores;
mod frame_work_budget;
mod project_aabb_footprint;
mod render_cpu_frame;
mod resolve_hero_light_visibility;
mod select_lights_for_chunk;
mod traverse_voxel_scene;
mod write_depth_tested_splats;

use crate::application::ports::{ChunkDraw, Environment, FrameParams, LightSource, RendererPort};
use crate::application::rendering::encode_display_color;

use super::atlas::{MipNode, build_mips};
use super::camera::Camera;
use super::settings::CpuRenderSettings;
use super::shading::{self, FrameLighting, SplatSurface};
use super::surface_geometry::{ProjectedSurfaceDepth, SurfaceExposure};
use super::update_atlas_rows::update_prepared_atlas_rows;
use frame_work_budget::FrameWorkBudget;
use resolve_hero_light_visibility::HeroLightVisibilityCache;
use write_depth_tested_splats::{DepthTestedSplatTarget, SquareSplat};

/// Per-frame counters for the HUD; tests also use them to prove termination
/// and safety-cap behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoftwareRasterizerTelemetry {
    pub visited_nodes: usize,
    pub node_soft_limit: usize,
    pub node_visit_limit: usize,
    pub node_budget_limited: bool,
    pub budget_exhausted: bool,
    pub max_virtual_depth: usize,
    pub splat_count: usize,
    pub pixel_writes: usize,
    pub pixel_write_limit: usize,
    pub pixel_budget_limited: bool,
}

/// Persistent state for the CPU renderer. The struct is intentionally the
/// allocation-owning facade; rendering algorithms live in focused modules.
pub struct SoftwareRasterizer {
    target: DepthTestedSplatTarget,
    /// SVO atlas texels: four `u32`s per node.
    atlas: Vec<u32>,
    mips: Vec<MipNode>,
    /// Reused visitation bits for one streamed pool-slot MIP rebuild.
    mip_update_scratch: Vec<bool>,
    /// Reused finite-support fixture cluster; populated once per drawn chunk.
    chunk_light_scratch: Vec<LightSource>,

    // Public counters are retained for compatibility; `telemetry()` is the
    // stable snapshot API used by the browser driver and tests.
    pub visited_nodes: usize,
    pub budget_exhausted: bool,
    pub max_virtual_depth_reached: usize,
    pub splat_count: usize,
    pub pixel_writes: usize,
    frame_work_budget: FrameWorkBudget,

    pub settings: CpuRenderSettings,
    hero_light_visibility: HeroLightVisibilityCache,
    /// Current level atmosphere, captured once at the start of `draw`.
    environment: Environment,
}

impl SoftwareRasterizer {
    pub fn new(width: usize, height: usize) -> Self {
        let settings = CpuRenderSettings::default();
        let frame_work_budget = FrameWorkBudget::for_target(&settings, width, height);
        Self {
            target: DepthTestedSplatTarget::new(width, height),
            atlas: Vec::new(),
            mips: Vec::new(),
            mip_update_scratch: Vec::new(),
            chunk_light_scratch: Vec::new(),
            visited_nodes: 0,
            budget_exhausted: false,
            max_virtual_depth_reached: 0,
            splat_count: 0,
            pixel_writes: 0,
            frame_work_budget,
            settings,
            hero_light_visibility: HeroLightVisibilityCache::new(),
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
        let node_budget_limited = self.budget_exhausted
            && self
                .frame_work_budget
                .outside_focus_reserve(self.visited_nodes);
        let pixel_budget_limited =
            self.budget_exhausted && self.frame_work_budget.writes_exhausted(self.pixel_writes);
        SoftwareRasterizerTelemetry {
            visited_nodes: self.visited_nodes,
            node_soft_limit: self.frame_work_budget.soft_node_visit_limit(),
            node_visit_limit: self.frame_work_budget.node_visit_limit(),
            node_budget_limited,
            budget_exhausted: self.budget_exhausted,
            max_virtual_depth: self.max_virtual_depth_reached,
            splat_count: self.splat_count,
            pixel_writes: self.pixel_writes,
            pixel_write_limit: self.frame_work_budget.pixel_write_limit(),
            pixel_budget_limited,
        }
    }

    /// Test seam and frame clear: atmosphere chooses the background; target
    /// storage owns color/depth/HZ reset mechanics.
    pub(super) fn clear(&mut self) {
        let background_radiance = if self.environment.outdoor {
            self.environment.sky_color
        } else {
            self.environment.fog_color
        };
        let background = encode_display_color(background_radiance)
            .map(|channel| (channel * 255.0).round().clamp(0.0, 255.0) as u8);
        self.target.clear(background);
    }

    /// Frame-policy wrapper around the pixel-only target operation. It keeps
    /// the exact in-loop write cap and renderer telemetry out of `target`.
    fn splat(&mut self, cx: f32, cy: f32, half: f32, z: f32, rgb: [u8; 3]) {
        self.write_splat_request(SquareSplat::new(cx, cy, half, z, rgb));
    }

    /// Writes a shaded voxel surface using its exact ray/plane depth. The
    /// emissive flag affects only exact coplanar reconstruction ties.
    fn surface_splat(
        &mut self,
        cx: f32,
        cy: f32,
        half: f32,
        depth: ProjectedSurfaceDepth,
        rgb: [u8; 3],
        emissive: bool,
    ) {
        let mut splat = SquareSplat::new(cx, cy, half, depth.representative_depth(), rgb)
            .with_surface_depth(depth);
        if emissive {
            splat = splat.with_equal_depth_priority();
        }
        self.write_splat_request(splat);
    }

    fn write_splat_request(&mut self, splat: SquareSplat) {
        if self.frame_work_budget.writes_exhausted(self.pixel_writes) {
            self.budget_exhausted = true;
            return;
        }
        self.splat_count += 1;

        let result = self.target.write_splat(
            splat,
            self.frame_work_budget
                .remaining_pixel_writes(self.pixel_writes),
            self.settings.toggles.hierarchical_z,
        );
        self.pixel_writes += result.pixel_writes;
        self.budget_exhausted |= result.budget_exhausted;
    }

    /// Exact pre-shading fine-depth rejection. This deliberately lives at the
    /// renderer boundary: traversal can avoid radiance and visibility work,
    /// while the pixel target remains the single owner of clipping/depth math.
    fn surface_splat_may_contribute(
        &self,
        cx: f32,
        cy: f32,
        half: f32,
        depth: ProjectedSurfaceDepth,
        emissive: bool,
    ) -> bool {
        if self.frame_work_budget.writes_exhausted(self.pixel_writes) {
            return false;
        }
        let mut splat = SquareSplat::new(cx, cy, half, depth.representative_depth(), [0; 3])
            .with_surface_depth(depth);
        if emissive {
            splat = splat.with_equal_depth_priority();
        }
        self.target.surface_splat_may_contribute(&splat)
    }

    /// Shades one exact box and writes its square splat. Retained as a narrow
    /// test seam; lighting itself remains in `shading`.
    #[cfg(test)]
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
    ) {
        self.shade_and_splat_with_frame_lighting(
            cam,
            chunks,
            FrameLighting::unshadowed(&[]),
            center,
            world_size,
            px,
            py,
            half_px,
            dist,
            base_color,
            light_level,
            is_emissive,
            crowded_siblings,
        );
    }

    /// Production shading path with the exact per-chunk fixture cluster and
    /// optional id-matched hero visibility context.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn shade_and_splat_with_frame_lighting(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        lighting: FrameLighting<'_>,
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
    ) {
        let linear_albedo = super::decode_srgb_albedo::decode_albedo(base_color);
        self.shade_predecoded_and_splat_with_frame_lighting(
            cam,
            chunks,
            lighting,
            center,
            world_size,
            px,
            py,
            half_px,
            ProjectedSurfaceDepth::constant(dist),
            linear_albedo,
            [light_level; 3],
            is_emissive.then(|| shading::emission_strength(base_color)),
            1,
            SurfaceExposure::ALL_FACES,
            crowded_siblings,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn shade_predecoded_and_splat_with_frame_lighting(
        &mut self,
        cam: &Camera,
        chunks: &[ChunkDraw],
        lighting: FrameLighting<'_>,
        center: [f32; 3],
        world_size: f32,
        px: f32,
        py: f32,
        half_px: f32,
        surface_depth: ProjectedSurfaceDepth,
        linear_albedo: [f32; 3],
        baked_irradiance: [f32; 3],
        emission_strength: Option<f32>,
        voxel_type: u32,
        exposure: SurfaceExposure,
        crowded_siblings: u32,
    ) {
        let surface = SplatSurface {
            center,
            world_size,
            dist: surface_depth.representative_depth(),
            base_color: [0.0; 3],
            baked_irradiance,
            voxel_type,
            is_emissive: emission_strength.is_some(),
            exposure,
            crowded_siblings,
        };
        let rgb = shading::shade_predecoded_with_frame_lighting(
            cam,
            &self.environment,
            &self.settings,
            &self.atlas,
            chunks,
            lighting,
            &surface,
            linear_albedo,
            emission_strength,
        );
        self.surface_splat(
            px,
            py,
            half_px,
            surface_depth,
            rgb,
            emission_strength.is_some(),
        );
    }
}

impl RendererPort for SoftwareRasterizer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        self.atlas = texels.to_vec();
        // MIP attributes are rebuilt on upload, never per frame.
        self.mips = build_mips(&self.atlas);
    }

    fn upload_atlas_rows(&mut self, first_row: u32, texels: &[u32]) -> bool {
        update_prepared_atlas_rows(
            &mut self.atlas,
            &mut self.mips,
            &mut self.mip_update_scratch,
            first_row,
            texels,
        )
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        self.render_cpu_frame(frame, chunks);
    }
}

#[cfg(test)]
mod atlas_row_upload_tests {
    use crate::application::atlas::ROW_TEXELS;
    use crate::application::ports::{ChunkDraw, FrameParams, RendererPort};

    use super::SoftwareRasterizer;

    const WORDS_PER_NODE: usize = 4;
    const AIR: [u32; WORDS_PER_NODE] = [1, 0, 0, 0];

    fn air_rows(rows: usize) -> Vec<u32> {
        AIR.repeat(ROW_TEXELS * rows)
    }

    #[test]
    fn renderer_port_row_upload_matches_a_full_reference_upload() {
        let initial = air_rows(2);
        let mut changed_row = air_rows(1);
        changed_row[1] = 1;
        changed_row[2] = 0x66aa44;
        changed_row[3] = 9 | (9 << 16) | (7 << 20) | (5 << 24);

        let mut incremental = SoftwareRasterizer::new(8, 8);
        incremental.upload_atlas(&initial);
        assert!(incremental.upload_atlas_rows(1, &changed_row));

        let mut expected_atlas = initial;
        let second_row_start = ROW_TEXELS * WORDS_PER_NODE;
        expected_atlas[second_row_start..].copy_from_slice(&changed_row);
        let mut reference = SoftwareRasterizer::new(8, 8);
        reference.upload_atlas(&expected_atlas);

        assert_eq!(incremental.atlas, reference.atlas);
        assert_eq!(incremental.mips, reference.mips);
        assert_eq!(incremental.mip_update_scratch.len(), ROW_TEXELS);

        let chunks = [ChunkDraw {
            origin: [0.0; 3],
            root_index: ROW_TEXELS as i32,
            world_size: 1.0,
            voxel_size: 1.0,
            svo_depth: 0,
        }];
        let frame = FrameParams {
            camera_pos: [0.5, 0.5, -2.0],
            yaw: std::f32::consts::PI,
            ..FrameParams::default()
        };
        incremental.draw(&frame, &chunks);
        reference.draw(&frame, &chunks);
        assert_eq!(incremental.framebuffer(), reference.framebuffer());
    }
}
