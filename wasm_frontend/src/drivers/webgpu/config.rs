//! Typed, target-independent configuration for every WebGPU strategy.
//!
//! The browser may obtain these values from query parameters and settings
//! controls, but no DOM or `wgpu` type crosses this boundary. A profile owns
//! quality and workload budgets. Capability switches cap the runtime toggle
//! set, so profiles cannot claim paths the selected strategy does not own.

use std::{error::Error, fmt, str::FromStr};

pub use crate::application::ports::RenderArtifactNeeds;
pub use crate::application::quality::{
    ParseQualityProfileError as ParseGpuQualityProfileError, QualityProfile as GpuQualityProfile,
};
use crate::application::render_settings::RenderToggles;

/// A production rendering strategy with WebGPU presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RendererKind {
    /// Indexed greedy meshes rendered with fixed-function depth testing.
    #[default]
    Surface,
    /// Instanced rectangles reconstructed from compact surface-face records.
    Splat,
    /// Fullscreen primary rays traversing serialized sparse voxel octrees.
    Raymarch,
    /// Software SVO rasterization uploaded through a WebGPU presentation pass.
    Cpu,
}

impl RendererKind {
    /// Stable iteration order for menus, diagnostics, and exhaustive tests.
    pub const ALL: [Self; 4] = [Self::Surface, Self::Splat, Self::Raymarch, Self::Cpu];

    /// Parses a renderer query value. Surrounding whitespace and ASCII case
    /// are ignored; aliases are deliberately rejected so URLs remain stable.
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        Self::ALL
            .into_iter()
            .find(|kind| value.eq_ignore_ascii_case(kind.query_value()))
    }

    /// Canonical value used by `?renderer=<value>` and persisted settings.
    pub const fn query_value(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::Splat => "splat",
            Self::Raymarch => "raymarch",
            Self::Cpu => "cpu",
        }
    }

    /// Human-readable name suitable for renderer menus and telemetry.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Surface => "WebGPU indexed surfaces",
            Self::Splat => "WebGPU face splats",
            Self::Raymarch => "WebGPU SVO raymarcher",
            Self::Cpu => "CPU reference / WebGPU present",
        }
    }

    /// Generation artifacts that must remain available for this strategy.
    pub const fn artifact_needs(self) -> RenderArtifactNeeds {
        match self {
            Self::Surface => RenderArtifactNeeds::SURFACE,
            Self::Splat => RenderArtifactNeeds::SPLAT,
            Self::Raymarch => RenderArtifactNeeds::RAYMARCH,
            Self::Cpu => RenderArtifactNeeds::CPU,
        }
    }

    /// Builds the selected strategy's complete profile at one quality tier.
    pub const fn profile(self, quality: GpuQualityProfile) -> RendererProfile {
        RendererProfile::for_renderer(self, quality)
    }
}

impl fmt::Display for RendererKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.query_value())
    }
}

impl FromStr for RendererKind {
    type Err = ParseRendererKindError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or(ParseRendererKindError)
    }
}

/// Returned when a renderer query value is not in [`RendererKind::ALL`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseRendererKindError;

impl fmt::Display for ParseRendererKindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("expected renderer `surface`, `splat`, `raymarch`, or `cpu`")
    }
}

impl Error for ParseRendererKindError {}

/// Runtime capabilities shared by renderer strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommonOptimizationSettings {
    /// Permit a strategy's conservative chunk visibility rejection.
    pub cull_invisible_chunks: bool,
    /// Permit output dithering in WebGPU strategies that implement it.
    pub dither_output: bool,
}

impl CommonOptimizationSettings {
    pub const fn optimized() -> Self {
        Self {
            cull_invisible_chunks: true,
            dither_output: true,
        }
    }

    /// Deterministic baseline used by offscreen correctness tests.
    pub const fn reference() -> Self {
        Self {
            cull_invisible_chunks: false,
            dither_output: false,
        }
    }
}

impl Default for CommonOptimizationSettings {
    fn default() -> Self {
        Self::optimized()
    }
}

/// Quality metadata shared by renderer profiles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonGpuConfig {
    pub quality: GpuQualityProfile,
    /// Initial backing-store multiplier for GPU-rendered strategies. The
    /// runtime governor composes with it; CPU uses its own resolution control
    /// and deliberately ignores this value.
    pub initial_render_scale: f32,
    /// Conservative world-space visibility and projection far distance.
    pub maximum_draw_distance: f32,
    pub optimizations: CommonOptimizationSettings,
}

impl CommonGpuConfig {
    pub const fn for_quality(quality: GpuQualityProfile) -> Self {
        match quality {
            GpuQualityProfile::Low => Self::low(),
            GpuQualityProfile::High => Self::high(),
        }
    }

    pub const fn low() -> Self {
        Self {
            quality: GpuQualityProfile::Low,
            initial_render_scale: 0.8,
            maximum_draw_distance: 110.0,
            optimizations: CommonOptimizationSettings::optimized(),
        }
    }

    pub const fn high() -> Self {
        Self {
            quality: GpuQualityProfile::High,
            initial_render_scale: 1.0,
            maximum_draw_distance: 110.0,
            optimizations: CommonOptimizationSettings::optimized(),
        }
    }
}

impl Default for CommonGpuConfig {
    fn default() -> Self {
        Self::low()
    }
}

/// Shared raster shadow-map quality settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShadowMapConfig {
    /// Width and height of the square depth texture.
    pub size: u32,
    /// Number of comparisons per receiver (`1` hard sample, `4` 2x2 PCF).
    pub filter_taps: u8,
}

impl ShadowMapConfig {
    pub const fn for_quality(quality: GpuQualityProfile) -> Self {
        match quality {
            GpuQualityProfile::Low => Self::low(),
            GpuQualityProfile::High => Self::high(),
        }
    }

    pub const fn low() -> Self {
        Self {
            size: 128,
            filter_taps: 1,
        }
    }

    pub const fn high() -> Self {
        Self {
            size: 256,
            filter_taps: 4,
        }
    }
}

/// Capability switches for the indexed renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceOptimizationSettings {
    /// Permit the hero-fixture shadow-map pass.
    pub hero_light_shadow_map: bool,
    /// Permit the optional compact baked-light scalar in surface shading.
    pub baked_surface_lighting: bool,
}

impl SurfaceOptimizationSettings {
    pub const fn optimized() -> Self {
        Self {
            hero_light_shadow_map: true,
            baked_surface_lighting: true,
        }
    }

    pub const fn reference() -> Self {
        Self {
            hero_light_shadow_map: true,
            baked_surface_lighting: false,
        }
    }
}

impl Default for SurfaceOptimizationSettings {
    fn default() -> Self {
        Self::optimized()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceProfile {
    pub common: CommonGpuConfig,
    pub shadow_map: ShadowMapConfig,
    pub optimizations: SurfaceOptimizationSettings,
}

impl SurfaceProfile {
    pub const fn for_quality(quality: GpuQualityProfile) -> Self {
        Self {
            common: CommonGpuConfig::for_quality(quality),
            shadow_map: ShadowMapConfig::for_quality(quality),
            optimizations: SurfaceOptimizationSettings::optimized(),
        }
    }

    pub const fn low() -> Self {
        Self::for_quality(GpuQualityProfile::Low)
    }

    pub const fn high() -> Self {
        Self::for_quality(GpuQualityProfile::High)
    }
}

impl Default for SurfaceProfile {
    fn default() -> Self {
        Self::low()
    }
}

/// Implemented CPU submission and shadow policies for the face-splat
/// renderer. Both visible and depth-only draws expand compact face records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplatOptimizationSettings {
    /// Sort resident chunks on the CPU before direct draw submission.
    pub cpu_front_to_back_submission: bool,
    /// Test compact face-cell ranges on the CPU and submit visible ranges.
    pub cpu_cell_range_culling: bool,
    /// Stop submitting whole far chunks after the profile face budget.
    pub enforce_face_budget: bool,
    /// Permit the hero-fixture face-expanded shadow-map pass.
    pub hero_light_shadow_map: bool,
    /// Permit the optional compact baked-light scalar in splat shading.
    pub baked_surface_lighting: bool,
}

impl SplatOptimizationSettings {
    pub const fn optimized() -> Self {
        Self {
            cpu_front_to_back_submission: true,
            cpu_cell_range_culling: true,
            enforce_face_budget: true,
            hero_light_shadow_map: true,
            baked_surface_lighting: true,
        }
    }

    pub const fn reference() -> Self {
        Self {
            cpu_front_to_back_submission: false,
            cpu_cell_range_culling: false,
            enforce_face_budget: false,
            hero_light_shadow_map: true,
            baked_surface_lighting: false,
        }
    }
}

impl Default for SplatOptimizationSettings {
    fn default() -> Self {
        Self::optimized()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplatProfile {
    pub common: CommonGpuConfig,
    pub shadow_map: ShadowMapConfig,
    /// Maximum visible face instances submitted in one frame.
    pub face_budget: u32,
    pub optimizations: SplatOptimizationSettings,
}

impl SplatProfile {
    pub const fn for_quality(quality: GpuQualityProfile) -> Self {
        let face_budget = match quality {
            GpuQualityProfile::Low => 40_000,
            GpuQualityProfile::High => 150_000,
        };
        Self {
            common: CommonGpuConfig::for_quality(quality),
            shadow_map: ShadowMapConfig::for_quality(quality),
            face_budget,
            optimizations: SplatOptimizationSettings::optimized(),
        }
    }

    pub const fn low() -> Self {
        Self::for_quality(GpuQualityProfile::Low)
    }

    pub const fn high() -> Self {
        Self::for_quality(GpuQualityProfile::High)
    }
}

impl Default for SplatProfile {
    fn default() -> Self {
        Self::low()
    }
}

/// Work-reduction capability policy for the SVO raymarcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaymarchOptimizationSettings {
    /// Traverse intersected chunk intervals from nearest to farthest.
    pub front_to_back_chunk_order: bool,
    /// Advance across the complete empty SVO leaf containing a sample.
    pub empty_space_skipping: bool,
    /// Permit finite SVO visibility traces to analytic emitter samples.
    pub direct_light_visibility: bool,
}

impl RaymarchOptimizationSettings {
    pub const fn optimized() -> Self {
        Self {
            front_to_back_chunk_order: true,
            empty_space_skipping: true,
            direct_light_visibility: true,
        }
    }

    pub const fn reference() -> Self {
        Self {
            front_to_back_chunk_order: false,
            empty_space_skipping: false,
            // Visibility changes the image, so the reference retains it.
            direct_light_visibility: true,
        }
    }
}

impl Default for RaymarchOptimizationSettings {
    fn default() -> Self {
        Self::optimized()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaymarchProfile {
    pub common: CommonGpuConfig,
    /// Capacity of the camera-visible chunk table (3x3 or 5x5 profile).
    pub maximum_resident_chunks: u32,
    /// Hard bound for one primary-ray traversal. Empty-leaf skipping changes
    /// how much world one step covers without weakening this safety budget.
    pub maximum_trace_steps: u32,
    /// Analytic rectangle-light visibility samples (center or 2x2 Gauss).
    pub area_light_visibility_samples: u8,
    pub optimizations: RaymarchOptimizationSettings,
}

impl RaymarchProfile {
    pub const fn for_quality(quality: GpuQualityProfile) -> Self {
        let (maximum_resident_chunks, maximum_trace_steps, area_light_visibility_samples) =
            match quality {
                GpuQualityProfile::Low => (9, 768, 1),
                GpuQualityProfile::High => (25, 1_536, 4),
            };
        Self {
            common: CommonGpuConfig::for_quality(quality),
            maximum_resident_chunks,
            maximum_trace_steps,
            area_light_visibility_samples,
            optimizations: RaymarchOptimizationSettings::optimized(),
        }
    }

    pub const fn low() -> Self {
        Self::for_quality(GpuQualityProfile::Low)
    }

    pub const fn high() -> Self {
        Self::for_quality(GpuQualityProfile::High)
    }
}

impl Default for RaymarchProfile {
    fn default() -> Self {
        Self::low()
    }
}

/// Software rendering settings whose final color buffer is presented by
/// WebGPU. World traversal and resolution remain under the CPU settings; the
/// shared profile records the selected quality decision and capability caps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuPresentationProfile {
    pub common: CommonGpuConfig,
}

impl CpuPresentationProfile {
    pub const fn for_quality(quality: GpuQualityProfile) -> Self {
        Self {
            common: CommonGpuConfig::for_quality(quality),
        }
    }

    pub const fn low() -> Self {
        Self::for_quality(GpuQualityProfile::Low)
    }

    pub const fn high() -> Self {
        Self::for_quality(GpuQualityProfile::High)
    }
}

impl Default for CpuPresentationProfile {
    fn default() -> Self {
        Self::low()
    }
}

/// Complete, type-safe configuration for one selected strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RendererProfile {
    Surface(SurfaceProfile),
    Splat(SplatProfile),
    Raymarch(RaymarchProfile),
    Cpu(CpuPresentationProfile),
}

impl RendererProfile {
    pub const fn for_renderer(renderer: RendererKind, quality: GpuQualityProfile) -> Self {
        match renderer {
            RendererKind::Surface => Self::Surface(SurfaceProfile::for_quality(quality)),
            RendererKind::Splat => Self::Splat(SplatProfile::for_quality(quality)),
            RendererKind::Raymarch => Self::Raymarch(RaymarchProfile::for_quality(quality)),
            RendererKind::Cpu => Self::Cpu(CpuPresentationProfile::for_quality(quality)),
        }
    }

    pub const fn renderer(self) -> RendererKind {
        match self {
            Self::Surface(_) => RendererKind::Surface,
            Self::Splat(_) => RendererKind::Splat,
            Self::Raymarch(_) => RendererKind::Raymarch,
            Self::Cpu(_) => RendererKind::Cpu,
        }
    }

    pub const fn common(&self) -> &CommonGpuConfig {
        match self {
            Self::Surface(profile) => &profile.common,
            Self::Splat(profile) => &profile.common,
            Self::Raymarch(profile) => &profile.common,
            Self::Cpu(profile) => &profile.common,
        }
    }

    pub const fn artifact_needs(self) -> RenderArtifactNeeds {
        self.renderer().artifact_needs()
    }

    /// GPU backing-store multiplier selected by this profile. CPU rendering
    /// returns `None` because its software-framebuffer scale is already a
    /// complete resolution policy and must not be multiplied a second time.
    pub const fn initial_gpu_render_scale(&self) -> Option<f32> {
        match self {
            Self::Surface(profile) => Some(profile.common.initial_render_scale),
            Self::Splat(profile) => Some(profile.common.initial_render_scale),
            Self::Raymarch(profile) => Some(profile.common.initial_render_scale),
            Self::Cpu(_) => None,
        }
    }

    /// Intersects the global runtime switchboard with this strategy's typed
    /// capability policy. The returned value contains no switches belonging
    /// exclusively to another strategy; retained shadow switches are the
    /// explicitly configured visibility capability for that strategy.
    pub fn apply_runtime_toggles(&self, requested: RenderToggles) -> RenderToggles {
        match self {
            Self::Surface(profile) => {
                let mut applied = apply_common_gpu_toggles(&profile.common, requested);
                applied.shadow_pass =
                    requested.shadow_pass && profile.optimizations.hero_light_shadow_map;
                applied.baked_lighting =
                    requested.baked_lighting && profile.optimizations.baked_surface_lighting;
                applied
            }
            Self::Splat(profile) => {
                let mut applied = apply_common_gpu_toggles(&profile.common, requested);
                applied.front_to_back =
                    requested.front_to_back && profile.optimizations.cpu_front_to_back_submission;
                applied.cell_culling =
                    requested.cell_culling && profile.optimizations.cpu_cell_range_culling;
                applied.face_budget =
                    requested.face_budget && profile.optimizations.enforce_face_budget;
                applied.shadow_pass =
                    requested.shadow_pass && profile.optimizations.hero_light_shadow_map;
                applied.baked_lighting =
                    requested.baked_lighting && profile.optimizations.baked_surface_lighting;
                applied
            }
            Self::Raymarch(profile) => {
                let mut applied = apply_common_gpu_toggles(&profile.common, requested);
                applied.front_to_back =
                    requested.front_to_back && profile.optimizations.front_to_back_chunk_order;
                applied.empty_space_skip =
                    requested.empty_space_skip && profile.optimizations.empty_space_skipping;
                applied.shadow_pass =
                    requested.shadow_pass && profile.optimizations.direct_light_visibility;
                applied
            }
            Self::Cpu(profile) => {
                let mut applied = RenderToggles::disabled();
                applied.hierarchical_z = requested.hierarchical_z;
                applied.front_to_back = requested.front_to_back;
                applied.mip_lod = requested.mip_lod;
                applied.flashlight_occlusion = requested.flashlight_occlusion;
                applied.deferred_shading = requested.deferred_shading;
                applied.ambient_occlusion = requested.ambient_occlusion;
                applied.baked_lighting = requested.baked_lighting;
                applied.distance_cull =
                    requested.distance_cull && profile.common.optimizations.cull_invisible_chunks;
                applied
            }
        }
    }
}

fn apply_common_gpu_toggles(common: &CommonGpuConfig, requested: RenderToggles) -> RenderToggles {
    let mut applied = RenderToggles::disabled();
    applied.distance_cull = requested.distance_cull && common.optimizations.cull_invisible_chunks;
    applied.dither = requested.dither && common.optimizations.dither_output;
    applied
}

impl Default for RendererProfile {
    fn default() -> Self {
        Self::Surface(SurfaceProfile::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_catalog_has_stable_unique_values_and_labels() {
        let expected = [
            (RendererKind::Surface, "surface", "WebGPU indexed surfaces"),
            (RendererKind::Splat, "splat", "WebGPU face splats"),
            (RendererKind::Raymarch, "raymarch", "WebGPU SVO raymarcher"),
            (RendererKind::Cpu, "cpu", "CPU reference / WebGPU present"),
        ];
        assert_eq!(RendererKind::ALL.len(), expected.len());
        for (index, (kind, query_value, label)) in expected.into_iter().enumerate() {
            assert_eq!(RendererKind::ALL[index], kind);
            assert_eq!(kind.query_value(), query_value);
            assert_eq!(kind.label(), label);
            assert_eq!(kind.to_string(), query_value);
            assert_eq!(query_value.parse(), Ok(kind));
            assert_eq!(query_value.to_ascii_uppercase().parse(), Ok(kind));
            assert_eq!(
                RendererKind::parse(&format!("  {query_value}  ")),
                Some(kind)
            );
        }
    }

    #[test]
    fn renderer_parser_rejects_legacy_and_ambiguous_values() {
        for invalid in ["", "gpu", "mesh", "software", "webgl", "surface-extra"] {
            assert_eq!(RendererKind::parse(invalid), None, "accepted {invalid:?}");
            assert_eq!(invalid.parse::<RendererKind>(), Err(ParseRendererKindError));
        }
        assert_eq!(RendererKind::default(), RendererKind::Surface);
    }

    #[test]
    fn quality_catalog_round_trips_all_values() {
        let expected = [
            (GpuQualityProfile::Low, "low", "Standard"),
            (GpuQualityProfile::High, "high", "High"),
        ];
        assert_eq!(GpuQualityProfile::ALL.len(), expected.len());
        for (index, (quality, query_value, label)) in expected.into_iter().enumerate() {
            assert_eq!(GpuQualityProfile::ALL[index], quality);
            assert_eq!(quality.query_value(), query_value);
            assert_eq!(quality.label(), label);
            assert_eq!(quality.to_string(), query_value);
            assert_eq!(query_value.parse(), Ok(quality));
            assert_eq!(
                GpuQualityProfile::parse(&query_value.to_ascii_uppercase()),
                Some(quality)
            );
        }
        assert_eq!(GpuQualityProfile::parse("ultra"), None);
        assert_eq!(GpuQualityProfile::default(), GpuQualityProfile::Low);
    }

    #[test]
    fn common_profiles_exhaustively_define_shared_quality() {
        let low = CommonGpuConfig::low();
        let high = CommonGpuConfig::high();
        assert_eq!(CommonGpuConfig::for_quality(GpuQualityProfile::Low), low);
        assert_eq!(CommonGpuConfig::for_quality(GpuQualityProfile::High), high);
        assert_eq!(CommonGpuConfig::default(), low);
        assert_eq!(low.quality, GpuQualityProfile::Low);
        assert_eq!(high.quality, GpuQualityProfile::High);
        assert_eq!(low.initial_render_scale, 0.8);
        assert_eq!(high.initial_render_scale, 1.0);
        assert_eq!(low.maximum_draw_distance, 110.0);
        assert_eq!(high.maximum_draw_distance, 110.0);
        assert_eq!(low.optimizations, CommonOptimizationSettings::optimized());
        assert_eq!(high.optimizations, CommonOptimizationSettings::optimized());
    }

    #[test]
    fn reference_common_settings_disable_every_optional_shared_path() {
        let settings = CommonOptimizationSettings::reference();
        assert!(!settings.cull_invisible_chunks);
        assert!(!settings.dither_output);
        assert_eq!(
            CommonOptimizationSettings::default(),
            CommonOptimizationSettings::optimized()
        );
    }

    #[test]
    fn surface_profiles_exhaustively_define_shadow_quality() {
        let low = SurfaceProfile::low();
        let high = SurfaceProfile::high();
        assert_eq!(low.common, CommonGpuConfig::low());
        assert_eq!(high.common, CommonGpuConfig::high());
        assert_eq!(
            low.shadow_map,
            ShadowMapConfig {
                size: 128,
                filter_taps: 1
            }
        );
        assert_eq!(
            high.shadow_map,
            ShadowMapConfig {
                size: 256,
                filter_taps: 4
            }
        );
        assert_eq!(low.optimizations, SurfaceOptimizationSettings::optimized());
        assert_eq!(high.optimizations, SurfaceOptimizationSettings::optimized());
        assert_eq!(SurfaceProfile::default(), low);

        let reference = SurfaceOptimizationSettings::reference();
        assert!(reference.hero_light_shadow_map);
        assert!(!reference.baked_surface_lighting);
    }

    #[test]
    fn splat_profiles_preserve_the_existing_workload_budgets() {
        let low = SplatProfile::low();
        let high = SplatProfile::high();
        assert_eq!(low.common, CommonGpuConfig::low());
        assert_eq!(high.common, CommonGpuConfig::high());
        assert_eq!(low.shadow_map, ShadowMapConfig::low());
        assert_eq!(high.shadow_map, ShadowMapConfig::high());
        assert_eq!(low.face_budget, 40_000);
        assert_eq!(high.face_budget, 150_000);
        assert_eq!(low.optimizations, SplatOptimizationSettings::optimized());
        assert_eq!(high.optimizations, SplatOptimizationSettings::optimized());
        assert_eq!(SplatProfile::default(), low);

        let reference = SplatOptimizationSettings::reference();
        assert!(!reference.cpu_front_to_back_submission);
        assert!(!reference.cpu_cell_range_culling);
        assert!(!reference.enforce_face_budget);
        assert!(reference.hero_light_shadow_map);
        assert!(!reference.baked_surface_lighting);
    }

    #[test]
    fn raymarch_profiles_match_streaming_footprints_and_light_sampling() {
        let low = RaymarchProfile::low();
        let high = RaymarchProfile::high();
        assert_eq!(low.common, CommonGpuConfig::low());
        assert_eq!(high.common, CommonGpuConfig::high());
        assert_eq!(low.maximum_resident_chunks, 9);
        assert_eq!(high.maximum_resident_chunks, 25);
        assert_eq!(low.maximum_trace_steps, 768);
        assert_eq!(high.maximum_trace_steps, 1_536);
        assert_eq!(low.area_light_visibility_samples, 1);
        assert_eq!(high.area_light_visibility_samples, 4);
        assert_eq!(low.optimizations, RaymarchOptimizationSettings::optimized());
        assert_eq!(
            high.optimizations,
            RaymarchOptimizationSettings::optimized()
        );
        assert_eq!(RaymarchProfile::default(), low);

        let reference = RaymarchOptimizationSettings::reference();
        assert!(!reference.front_to_back_chunk_order);
        assert!(!reference.empty_space_skipping);
        assert!(reference.direct_light_visibility);
    }

    #[test]
    fn cpu_profiles_only_configure_webgpu_presentation() {
        assert_eq!(CpuPresentationProfile::low().common, CommonGpuConfig::low());
        assert_eq!(
            CpuPresentationProfile::high().common,
            CommonGpuConfig::high()
        );
        assert_eq!(
            CpuPresentationProfile::default(),
            CpuPresentationProfile::low()
        );
    }

    #[test]
    fn every_renderer_and_quality_pair_builds_the_matching_typed_profile() {
        for renderer in RendererKind::ALL {
            for quality in GpuQualityProfile::ALL {
                let profile = renderer.profile(quality);
                assert_eq!(profile.renderer(), renderer);
                assert_eq!(profile.common().quality, quality);
                assert_eq!(profile.artifact_needs(), renderer.artifact_needs());
                match (renderer, profile) {
                    (RendererKind::Surface, RendererProfile::Surface(_))
                    | (RendererKind::Splat, RendererProfile::Splat(_))
                    | (RendererKind::Raymarch, RendererProfile::Raymarch(_))
                    | (RendererKind::Cpu, RendererProfile::Cpu(_)) => {}
                    _ => panic!("renderer/profile variant mismatch"),
                }
            }
        }
        assert_eq!(
            RendererProfile::default(),
            RendererProfile::Surface(SurfaceProfile::low())
        );
    }

    #[test]
    fn gpu_scale_is_profile_owned_while_cpu_scale_remains_independent() {
        for renderer in [
            RendererKind::Surface,
            RendererKind::Splat,
            RendererKind::Raymarch,
        ] {
            assert_eq!(
                renderer
                    .profile(GpuQualityProfile::Low)
                    .initial_gpu_render_scale(),
                Some(0.8)
            );
            assert_eq!(
                renderer
                    .profile(GpuQualityProfile::High)
                    .initial_gpu_render_scale(),
                Some(1.0)
            );
        }
        assert_eq!(
            RendererKind::Cpu
                .profile(GpuQualityProfile::Low)
                .initial_gpu_render_scale(),
            None
        );
    }

    #[test]
    fn runtime_toggles_are_capped_to_each_strategy_and_profile() {
        let requested = RenderToggles {
            baked_lighting: true,
            ..RenderToggles::default()
        };

        let surface =
            RendererProfile::Surface(SurfaceProfile::low()).apply_runtime_toggles(requested);
        assert_eq!(
            surface,
            RenderToggles {
                shadow_pass: true,
                distance_cull: true,
                dither: true,
                baked_lighting: true,
                ..RenderToggles::disabled()
            }
        );

        let splat = RendererProfile::Splat(SplatProfile::low()).apply_runtime_toggles(requested);
        assert_eq!(
            splat,
            RenderToggles {
                front_to_back: true,
                shadow_pass: true,
                cell_culling: true,
                face_budget: true,
                distance_cull: true,
                dither: true,
                baked_lighting: true,
                ..RenderToggles::disabled()
            }
        );

        let raymarch =
            RendererProfile::Raymarch(RaymarchProfile::low()).apply_runtime_toggles(requested);
        assert_eq!(
            raymarch,
            RenderToggles {
                front_to_back: true,
                empty_space_skip: true,
                shadow_pass: true,
                distance_cull: true,
                dither: true,
                ..RenderToggles::disabled()
            }
        );

        let cpu =
            RendererProfile::Cpu(CpuPresentationProfile::low()).apply_runtime_toggles(requested);
        assert_eq!(
            cpu,
            RenderToggles {
                hierarchical_z: true,
                front_to_back: true,
                mip_lod: true,
                flashlight_occlusion: true,
                deferred_shading: true,
                ambient_occlusion: true,
                distance_cull: true,
                baked_lighting: true,
                ..RenderToggles::disabled()
            }
        );
    }

    #[test]
    fn reference_profiles_disable_only_the_paths_they_control() {
        let requested = RenderToggles {
            baked_lighting: true,
            ..RenderToggles::default()
        };

        let mut splat = SplatProfile::low();
        splat.common.optimizations = CommonOptimizationSettings::reference();
        splat.optimizations = SplatOptimizationSettings::reference();
        let applied = RendererProfile::Splat(splat).apply_runtime_toggles(requested);
        assert!(!applied.front_to_back);
        assert!(!applied.cell_culling);
        assert!(!applied.face_budget);
        assert!(!applied.distance_cull);
        assert!(!applied.dither);
        assert!(!applied.baked_lighting);
        assert!(applied.shadow_pass, "reference lighting retains visibility");

        let mut raymarch = RaymarchProfile::low();
        raymarch.optimizations = RaymarchOptimizationSettings::reference();
        let applied = RendererProfile::Raymarch(raymarch).apply_runtime_toggles(requested);
        assert!(!applied.front_to_back);
        assert!(!applied.empty_space_skip);
        assert!(applied.shadow_pass, "reference lighting retains visibility");
    }

    #[test]
    fn artifact_contract_is_exhaustive_and_composable() {
        assert_eq!(
            RendererKind::Surface.artifact_needs(),
            RenderArtifactNeeds::SURFACE
        );
        assert_eq!(
            RendererKind::Splat.artifact_needs(),
            RenderArtifactNeeds::SPLAT
        );
        assert_eq!(
            RendererKind::Raymarch.artifact_needs(),
            RenderArtifactNeeds::RAYMARCH
        );
        assert_eq!(RendererKind::Cpu.artifact_needs(), RenderArtifactNeeds::CPU);

        let all = RendererKind::ALL
            .into_iter()
            .fold(RenderArtifactNeeds::default(), |needs, renderer| {
                needs.union(renderer.artifact_needs())
            });
        assert_eq!(all, RenderArtifactNeeds::ALL);
        assert!(RenderArtifactNeeds::SURFACE.needs_surface_extraction());
        assert!(RenderArtifactNeeds::SPLAT.needs_surface_extraction());
        assert!(!RenderArtifactNeeds::RAYMARCH.needs_surface_extraction());
        assert!(!RenderArtifactNeeds::CPU.needs_surface_extraction());
    }
}
