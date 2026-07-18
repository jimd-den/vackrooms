//! Literate WGSL composition.
//!
//! Each strategy file describes only its own data and pass. The shared file
//! is prepended once so camera, lighting, fog, and display equations cannot
//! drift between surface, splat, and raymarch pipelines.

use std::borrow::Cow;
use std::fmt::Write;

use vackrooms::domain::entities::voxel_grid::{VOXEL_MATERIAL_COUNT, material_emission_strength};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderProgram {
    Surface,
    SurfaceShadow,
    Splat,
    SplatShadow,
    Raymarch,
    CpuPresent,
    SupplyLabels,
}

impl ShaderProgram {
    pub fn label(self) -> &'static str {
        match self {
            Self::Surface => "webgpu.surface",
            Self::SurfaceShadow => "webgpu.surface-shadow",
            Self::Splat => "webgpu.splat",
            Self::SplatShadow => "webgpu.splat-shadow",
            Self::Raymarch => "webgpu.raymarch",
            Self::CpuPresent => "webgpu.cpu-present",
            Self::SupplyLabels => "webgpu.supply-labels",
        }
    }

    pub fn source(self) -> Cow<'static, str> {
        const COMMON: &str = include_str!("shaders/common.wgsl");
        const HERO_SHADOW_CONTRACT: &str = include_str!("shaders/hero_shadow_contract.wgsl");
        const RASTER_SHADOW: &str = include_str!("shaders/raster_shadow.wgsl");
        const RASTER_CHUNK: &str = include_str!("shaders/raster_chunk_contract.wgsl");
        const SHADOW_CASTER: &str = include_str!("shaders/shadow_caster.wgsl");
        const SURFACE_GEOMETRY: &str = include_str!("shaders/surface_geometry.wgsl");
        const SPLAT_GEOMETRY: &str = include_str!("shaders/splat_geometry.wgsl");
        match self {
            Self::Surface => Cow::Owned(
                [
                    COMMON,
                    HERO_SHADOW_CONTRACT,
                    RASTER_SHADOW,
                    RASTER_CHUNK,
                    SURFACE_GEOMETRY,
                    include_str!("shaders/surface.wgsl"),
                ]
                .concat(),
            ),
            Self::SurfaceShadow => Cow::Owned(
                [
                    HERO_SHADOW_CONTRACT,
                    SHADOW_CASTER,
                    RASTER_CHUNK,
                    SURFACE_GEOMETRY,
                    include_str!("shaders/surface_shadow.wgsl"),
                ]
                .concat(),
            ),
            Self::Splat => Cow::Owned(
                [
                    COMMON,
                    HERO_SHADOW_CONTRACT,
                    RASTER_SHADOW,
                    RASTER_CHUNK,
                    SPLAT_GEOMETRY,
                    include_str!("shaders/splat.wgsl"),
                ]
                .concat(),
            ),
            Self::SplatShadow => Cow::Owned(
                [
                    HERO_SHADOW_CONTRACT,
                    SHADOW_CASTER,
                    RASTER_CHUNK,
                    SPLAT_GEOMETRY,
                    include_str!("shaders/splat_shadow.wgsl"),
                ]
                .concat(),
            ),
            Self::Raymarch => {
                let mut source = String::from(COMMON);
                source.push_str(&material_emission_contract_wgsl());
                source.push_str(include_str!("shaders/raymarch.wgsl"));
                Cow::Owned(source)
            }
            Self::CpuPresent => Cow::Borrowed(include_str!("shaders/cpu_present.wgsl")),
            Self::SupplyLabels => {
                Cow::Owned([COMMON, include_str!("shaders/supply_labels.wgsl")].concat())
            }
        }
    }

    pub fn create_module(self, device: &wgpu::Device) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(self.label()),
            source: wgpu::ShaderSource::Wgsl(self.source()),
        })
    }
}

/// Generates the ray-visible emission switch directly from the domain
/// contract. Raster uploads carry the same information in their compact
/// visual word; the SVO already carries color but still needs an exact
/// material-to-strength mapping at shader creation time.
fn material_emission_contract_wgsl() -> String {
    let mut source = String::from(
        "\n// Generated from voxel_grid::material_emission_strength.\n\
         fn authored_emission_strength(material: u32) -> f32 {\n\
             switch material {\n",
    );
    for material in 0..VOXEL_MATERIAL_COUNT {
        let material = material as u8;
        let Some(strength) = material_emission_strength(material) else {
            continue;
        };
        writeln!(
            source,
            "        case {material}u: {{ return {strength:.6}; }}"
        )
        .expect("writing WGSL to a String cannot fail");
    }
    source.push_str(
        "        default: { return 0.0; }\n\
             }\n\
         }\n",
    );
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::rendering::{
        CACHED_DIFFUSE_FILL_GAIN, FLASHLIGHT_SPEC, INDOOR_AMBIENT_IRRADIANCE,
        OUTDOOR_AMBIENT_IRRADIANCE,
    };

    #[test]
    fn strategies_expose_named_entry_points() {
        for (program, entries) in [
            (
                ShaderProgram::Surface,
                &["surface_vertex", "surface_fragment"][..],
            ),
            (ShaderProgram::SurfaceShadow, &["surface_shadow_vertex"][..]),
            (
                ShaderProgram::Splat,
                &["splat_vertex", "splat_fragment"][..],
            ),
            (ShaderProgram::SplatShadow, &["splat_shadow_vertex"][..]),
            (
                ShaderProgram::Raymarch,
                &["raymarch_vertex", "raymarch_fragment"][..],
            ),
            (
                ShaderProgram::CpuPresent,
                &["cpu_present_vertex", "cpu_present_fragment"][..],
            ),
            (
                ShaderProgram::SupplyLabels,
                &["supply_label_vertex", "supply_label_fragment"][..],
            ),
        ] {
            let source = program.source();
            for entry in entries {
                assert!(source.contains(entry), "{} lost {entry}", program.label());
            }
        }
    }

    #[test]
    fn gpu_strategies_share_one_radiometric_vocabulary() {
        for program in [
            ShaderProgram::Surface,
            ShaderProgram::Splat,
            ShaderProgram::Raymarch,
            ShaderProgram::SupplyLabels,
        ] {
            let source = program.source();
            assert_eq!(source.matches("fn surface_radiance(").count(), 1);
            assert_eq!(source.matches("fn present_radiance(").count(), 1);
        }
    }

    #[test]
    fn ray_emission_switch_is_generated_from_the_domain_contract() {
        let contract = material_emission_contract_wgsl();
        for material in 0..VOXEL_MATERIAL_COUNT {
            let material = material as u8;
            let case = format!("case {material}u:");
            match material_emission_strength(material) {
                Some(strength) => {
                    assert!(contract.contains(&case), "missing material {material}");
                    assert!(contract.contains(&format!("return {strength:.6};")));
                }
                None => assert!(
                    !contract.contains(&case),
                    "invented emission for {material}"
                ),
            }
        }
        assert_eq!(
            ShaderProgram::Raymarch
                .source()
                .matches("fn authored_emission_strength(")
                .count(),
            1
        );
    }

    #[test]
    fn shared_direct_light_uses_finite_support_and_the_emitting_hemisphere() {
        let source = ShaderProgram::Surface.source();
        assert!(source.contains("fn finite_range_inverse_square("));
        assert!(source.contains("normalized_squared * normalized_squared"));
        assert!(source.contains("array<vec2<f32>, 4>("));
        assert!(source.contains("return max(surface_to_light.y, 0.0);"));
        assert!(!source.contains("max(-surface_to_light.y"));
    }

    #[test]
    fn direct_light_sample_primitives_are_reusable_by_visibility_strategies() {
        let source = ShaderProgram::Surface.source();
        for primitive in [
            "fn point_light_sample_irradiance(",
            "fn rectangle_gauss_sample_position(",
            "fn rectangle_light_sample_weight(",
            "fn rectangle_light_irradiance_from_integral(",
        ] {
            assert_eq!(source.matches(primitive).count(), 1, "missing {primitive}");
        }
    }

    #[test]
    fn splat_shadow_program_expands_faces_without_an_indexed_mesh() {
        let source = ShaderProgram::SplatShadow.source();
        assert!(source.contains("var<storage, read> packed_faces"));
        assert!(source.contains("@builtin(instance_index) face_index"));
        assert!(!source.contains("PackedSurfaceVertex"));
        assert!(!source.contains("frame.view_projection"));
    }

    #[test]
    fn raster_strategies_decode_uploaded_palette_words() {
        for program in [ShaderProgram::Surface, ShaderProgram::Splat] {
            let source = program.source();
            assert_eq!(source.matches("packed_srgb_to_linear(").count(), 2);
            assert!(!source.contains("fn material_color("));
        }
    }

    #[test]
    fn shared_ambient_and_diffuse_composition_match_the_reference_contract() {
        let source = ShaderProgram::Surface.source();
        assert!(source.contains(&format!(
            "vec3<f32>({:.3}, {:.3}, {:.3})",
            INDOOR_AMBIENT_IRRADIANCE[0],
            INDOOR_AMBIENT_IRRADIANCE[1],
            INDOOR_AMBIENT_IRRADIANCE[2]
        )));
        assert!(source.contains(&format!(
            "vec3<f32>({:.2}, {:.2}, {:.2})",
            OUTDOOR_AMBIENT_IRRADIANCE[0],
            OUTDOOR_AMBIENT_IRRADIANCE[1],
            OUTDOOR_AMBIENT_IRRADIANCE[2]
        )));
        assert!(source.contains(&format!(
            "const CACHED_DIFFUSE_FILL_GAIN: f32 = {CACHED_DIFFUSE_FILL_GAIN:.2};"
        )));
        assert!(source.contains("let ambient = ambient_irradiance() * ambient_visibility;"));
        assert!(
            source.contains(
                "let irradiance = ambient + cached_static + analytic_direct + flashlight;"
            )
        );
        assert!(source.contains("return albedo * irradiance * (1.0 / PI);"));
    }

    #[test]
    fn shared_flashlight_literals_match_the_cpu_spec() {
        let source = ShaderProgram::Surface.source();
        let inner_cosine = FLASHLIGHT_SPEC.inner_degrees.to_radians().cos();
        let outer_cosine = FLASHLIGHT_SPEC.outer_degrees.to_radians().cos();
        for expected in [
            format!("const FLASHLIGHT_INNER_COSINE: f32 = {inner_cosine:.9};"),
            format!("const FLASHLIGHT_OUTER_COSINE: f32 = {outer_cosine:.9};"),
            format!(
                "const FLASHLIGHT_RANGE: f32 = {:.1};",
                FLASHLIGHT_SPEC.range
            ),
            format!(
                "const FLASHLIGHT_INTENSITY: f32 = {:.1};",
                FLASHLIGHT_SPEC.intensity
            ),
            format!(
                "const FLASHLIGHT_MINIMUM_DISTANCE: f32 = {:.2};",
                FLASHLIGHT_SPEC.minimum_distance
            ),
            format!(
                "const FLASHLIGHT_FORWARD_OFFSET: f32 = {:.2};",
                FLASHLIGHT_SPEC.forward_offset
            ),
            format!(
                "const FLASHLIGHT_DOWNWARD_OFFSET: f32 = {:.2};",
                FLASHLIGHT_SPEC.downward_offset
            ),
            format!(
                "vec3<f32>({:.1}, {:.2}, {:.2})",
                FLASHLIGHT_SPEC.tint[0], FLASHLIGHT_SPEC.tint[1], FLASHLIGHT_SPEC.tint[2]
            ),
        ] {
            assert!(source.contains(&expected), "missing `{expected}`");
        }
        assert!(source.contains("frame.camera_forward.xyz * FLASHLIGHT_FORWARD_OFFSET"));
        assert!(source.contains("- vec3<f32>(0.0, FLASHLIGHT_DOWNWARD_OFFSET, 0.0)"));
        assert!(source.contains("let receiver = receiver_cosine(normal, -direction);"));
    }

    #[test]
    fn bake_flag_is_raster_only_and_raymarch_stays_strictly_analytic() {
        for program in [ShaderProgram::Surface, ShaderProgram::Splat] {
            let source = program.source();
            assert!(source.contains("if frame.lighting_options.y == 0u"));
            assert!(source.contains("cached_static_irradiance(baked)"));
        }

        let raymarch = ShaderProgram::Raymarch.source();
        assert!(!raymarch.contains("nearest.light_word &"));
        assert!(raymarch.contains("surface_radiance_from_analytic_direct("));
        assert!(raymarch.contains("            0.0,\n            1.0,"));
    }
}
