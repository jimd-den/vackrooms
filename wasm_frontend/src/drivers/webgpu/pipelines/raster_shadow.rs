//! Hero-fixture shadow resources shared by both raster strategies.
//!
//! The resource contract is intentionally independent from caster geometry:
//! indexed surfaces render their existing vertex/index buffers, while splats
//! expand their existing packed face records. This module owns only the
//! light selection, transform, depth target, and the two bind-group views of
//! one uniform buffer (depth-caster input and lit-receiver input).

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};

use crate::application::ports::{FrameParams, LightKind, LightSource};
use crate::application::rendering::multiply_column_major;
use crate::drivers::webgpu::config::ShadowMapConfig;

const SHADOW_NEAR: f32 = 0.03;
const MINIMUM_RECEIVER_BIAS: f32 = 0.0005;
const SLOPE_RECEIVER_BIAS: f32 = 0.0015;

/// Profile-owned raster shadow capability and quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeroShadowOptions {
    pub map: ShadowMapConfig,
    pub profile_enabled: bool,
}

impl HeroShadowOptions {
    pub const fn new(map: ShadowMapConfig, profile_enabled: bool) -> Self {
        Self {
            map,
            profile_enabled,
        }
    }
}

impl Default for HeroShadowOptions {
    fn default() -> Self {
        Self::new(ShadowMapConfig::low(), true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeroShadowSelection {
    gpu_light_index: u32,
    light: LightSource,
}

/// Matches the static-light prefix written by `collect_frame_lights`.
/// Runtime point lights are appended later and can never own a top-down map.
fn select_hero_shadow_light(frame: &FrameParams) -> Option<HeroShadowSelection> {
    let mut gpu_light_index = 0u32;
    for light in frame.active_scene_lights() {
        if !light.enabled {
            continue;
        }
        if light.kind != LightKind::Point {
            return Some(HeroShadowSelection {
                gpu_light_index,
                light: *light,
            });
        }
        gpu_light_index = gpu_light_index.saturating_add(1);
    }
    None
}

/// One record is consumed by both the caster and receiver pipelines.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuHeroShadow {
    light_view_projection: [f32; 16],
    /// xy: inverse map extent; z: minimum bias; w: slope-scaled bias.
    sampling: [f32; 4],
    /// x: enabled; y: absolute GPU-light index; z: taps; w: map extent.
    options: [u32; 4],
}

impl GpuHeroShadow {
    fn disabled(options: HeroShadowOptions) -> Self {
        let size = options.map.size.max(1);
        Self {
            light_view_projection: [0.0; 16],
            sampling: [
                1.0 / size as f32,
                1.0 / size as f32,
                MINIMUM_RECEIVER_BIAS,
                SLOPE_RECEIVER_BIAS,
            ],
            options: [0, 0, u32::from(options.map.filter_taps), size],
        }
    }

    fn enabled(options: HeroShadowOptions, hero: HeroShadowSelection) -> Self {
        let mut uniforms = Self::disabled(options);
        uniforms.light_view_projection = hero_light_view_projection(&hero.light);
        uniforms.options[0] = 1;
        uniforms.options[1] = hero.gpu_light_index;
        uniforms
    }
}

/// Texture and bindings common to one Surface or Splat pipeline instance.
pub(super) struct HeroShadowMap {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    caster_layout: wgpu::BindGroupLayout,
    receiver_layout: wgpu::BindGroupLayout,
    caster_bind_group: wgpu::BindGroup,
    receiver_bind_group: wgpu::BindGroup,
    options: HeroShadowOptions,
}

impl HeroShadowMap {
    pub(super) fn new(device: &wgpu::Device, options: HeroShadowOptions) -> Self {
        assert!(options.map.size > 0, "shadow-map extent must be non-zero");
        assert!(
            matches!(options.map.filter_taps, 1 | 4),
            "shadow-map filter taps must be 1 or 4"
        );

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("webgpu.hero-shadow.depth"),
            size: wgpu::Extent3d {
                width: options.map.size,
                height: options.map.size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("webgpu.hero-shadow.comparison-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("webgpu.hero-shadow.uniforms"),
            size: size_of::<GpuHeroShadow>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let caster_layout = create_caster_layout(device);
        let receiver_layout = create_receiver_layout(device);
        let caster_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("webgpu.hero-shadow.caster-bind-group"),
            layout: &caster_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let receiver_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("webgpu.hero-shadow.receiver-bind-group"),
            layout: &receiver_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        Self {
            _texture: texture,
            view,
            _sampler: sampler,
            uniform,
            caster_layout,
            receiver_layout,
            caster_bind_group,
            receiver_bind_group,
            options,
        }
    }

    /// Snapshots hero identity and runtime policy once before both passes.
    pub(super) fn prepare(
        &self,
        queue: &wgpu::Queue,
        frame: &FrameParams,
        runtime_enabled: bool,
    ) -> bool {
        let hero = select_hero_shadow_light(frame);
        let enabled = self.options.profile_enabled && runtime_enabled && hero.is_some();
        let uniforms = if enabled {
            GpuHeroShadow::enabled(self.options, hero.expect("enabled requires a hero"))
        } else {
            GpuHeroShadow::disabled(self.options)
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniforms));
        enabled
    }

    pub(super) fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub(super) fn caster_layout(&self) -> &wgpu::BindGroupLayout {
        &self.caster_layout
    }

    pub(super) fn receiver_layout(&self) -> &wgpu::BindGroupLayout {
        &self.receiver_layout
    }

    pub(super) fn caster_bind_group(&self) -> &wgpu::BindGroup {
        &self.caster_bind_group
    }

    pub(super) fn receiver_bind_group(&self) -> &wgpu::BindGroup {
        &self.receiver_bind_group
    }
}

fn create_caster_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu.hero-shadow.caster-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size_of::<GpuHeroShadow>() as u64),
            },
            count: None,
        }],
    })
}

fn create_receiver_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu.hero-shadow.receiver-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<GpuHeroShadow>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ],
    })
}

/// Column-major top-down fixture transform with WebGPU's `0..1` depth.
fn hero_light_view_projection(light: &LightSource) -> [f32; 16] {
    let emitter_half_diagonal =
        (light.half_size[0] * light.half_size[0] + light.half_size[1] * light.half_size[1]).sqrt();
    let extent = (light.radius + emitter_half_diagonal).max(SHADOW_NEAR + 0.01);
    let projection = webgpu_orthographic(-extent, extent, -extent, extent, SHADOW_NEAR, extent);
    let view = top_down_view(light.position);
    multiply_column_major(&projection, &view)
}

fn webgpu_orthographic(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [f32; 16] {
    let near = near.max(0.0);
    let far = far.max(near + 0.01);
    [
        2.0 / (right - left),
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 / (top - bottom),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / (near - far),
        0.0,
        -(right + left) / (right - left),
        -(top + bottom) / (top - bottom),
        near / (near - far),
        1.0,
    ]
}

/// Light looks down `-Y`; `-Z` is map up, matching ceiling rectangles in XZ.
fn top_down_view(eye: [f32; 3]) -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, // right = +X
        0.0, 0.0, 1.0, 0.0, // -forward = +Y
        0.0, -1.0, 0.0, 0.0, // up = -Z
        -eye[0], eye[2], -eye[1], 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn light(id: u64, kind: LightKind, enabled: bool) -> LightSource {
        LightSource {
            id,
            position: [4.0, 7.0, -2.0],
            half_size: [1.0, 0.5],
            color: [1.0; 3],
            radius: 12.0,
            intensity: 10.0,
            kind,
            flicker_mode: 0,
            enabled,
        }
    }

    fn project(matrix: &[f32; 16], point: [f32; 3]) -> [f32; 3] {
        let x = matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12];
        let y = matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13];
        let z = matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14];
        let w = matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];
        [x / w, y / w, z / w]
    }

    #[test]
    fn hero_selection_matches_the_filtered_gpu_light_prefix() {
        let frame = FrameParams {
            scene_lights: vec![
                light(1, LightKind::CeilingPanel, false),
                light(2, LightKind::Point, true),
                light(3, LightKind::Strip, true),
            ],
            ..FrameParams::default()
        };

        let selected = select_hero_shadow_light(&frame).expect("strip is a fixture");
        assert_eq!(selected.gpu_light_index, 1);
        assert_eq!(selected.light.id, 3);
    }

    #[test]
    fn point_and_dynamic_only_frames_have_no_top_down_hero() {
        let frame = FrameParams {
            scene_lights: vec![light(1, LightKind::Point, true)],
            ..FrameParams::default()
        };
        assert_eq!(select_hero_shadow_light(&frame), None);
    }

    #[test]
    fn top_down_projection_maps_near_and_far_to_webgpu_depth() {
        let light = light(7, LightKind::CeilingPanel, true);
        let matrix = hero_light_view_projection(&light);
        let extent = light.radius
            + (light.half_size[0] * light.half_size[0] + light.half_size[1] * light.half_size[1])
                .sqrt();
        let near = project(
            &matrix,
            [
                light.position[0],
                light.position[1] - SHADOW_NEAR,
                light.position[2],
            ],
        );
        let far = project(
            &matrix,
            [
                light.position[0],
                light.position[1] - extent,
                light.position[2],
            ],
        );
        assert!(near[0].abs() < 1e-6 && near[1].abs() < 1e-6);
        assert!(near[2].abs() < 1e-5, "near depth was {}", near[2]);
        assert!((far[2] - 1.0).abs() < 1e-5, "far depth was {}", far[2]);
    }

    #[test]
    fn hero_support_bounds_fit_the_orthographic_map() {
        let light = light(9, LightKind::CeilingPanel, true);
        let matrix = hero_light_view_projection(&light);
        let extent = light.radius
            + (light.half_size[0] * light.half_size[0] + light.half_size[1] * light.half_size[1])
                .sqrt();
        for point in [
            [
                light.position[0] - extent,
                light.position[1] - 1.0,
                light.position[2],
            ],
            [
                light.position[0] + extent,
                light.position[1] - 1.0,
                light.position[2],
            ],
            [
                light.position[0],
                light.position[1] - 1.0,
                light.position[2] - extent,
            ],
            [
                light.position[0],
                light.position[1] - 1.0,
                light.position[2] + extent,
            ],
        ] {
            let projected = project(&matrix, point);
            assert!(projected[0].abs() <= 1.0 + 1e-5);
            assert!(projected[1].abs() <= 1.0 + 1e-5);
        }
    }

    #[test]
    fn shadow_uniform_record_obeys_wgsl_alignment() {
        assert_eq!(size_of::<GpuHeroShadow>(), 96);
        assert_eq!(size_of::<GpuHeroShadow>() % 16, 0);
    }

    #[test]
    fn profile_extent_and_taps_reach_the_gpu_record_exactly() {
        for config in [ShadowMapConfig::low(), ShadowMapConfig::high()] {
            let uniforms = GpuHeroShadow::disabled(HeroShadowOptions::new(config, true));
            assert_eq!(uniforms.options[2], u32::from(config.filter_taps));
            assert_eq!(uniforms.options[3], config.size);
            assert_eq!(uniforms.sampling[0], 1.0 / config.size as f32);
            assert_eq!(uniforms.sampling[1], 1.0 / config.size as f32);
        }
    }
}
