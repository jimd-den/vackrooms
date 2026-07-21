//! Fullscreen SVO traversal pipeline and atlas lifecycle.

use std::num::NonZeroU64;

use wgpu::util::DeviceExt;

use crate::application::ports::{ChunkDraw, FrameParams};
use crate::application::render_settings::RenderToggles;
use crate::application::rendering::encode_display_color;

use crate::drivers::webgpu::frame_resources::FrameResources;
use crate::drivers::webgpu::gpu_types::{GpuRayChunk, GpuRayScene};
use crate::drivers::webgpu::shader::ShaderProgram;

const ATLAS_TEXELS_PER_ROW: u64 = 1024;
const WORDS_PER_NODE: u64 = 4;

/// Mutable profile values which affect ray-scene selection and secondary rays.
/// Keeping them separate from GPU object construction lets quality profiles be
/// changed without rebuilding the pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaymarchRuntimeOptions {
    maximum_draw_distance: f32,
    direct_light_visibility: bool,
    area_light_visibility_samples: u8,
}

impl RaymarchRuntimeOptions {
    pub fn new(
        maximum_draw_distance: f32,
        direct_light_visibility: bool,
        area_light_visibility_samples: u8,
    ) -> Self {
        Self {
            maximum_draw_distance: if maximum_draw_distance.is_finite()
                && maximum_draw_distance > 0.0
            {
                maximum_draw_distance
            } else {
                f32::INFINITY
            },
            direct_light_visibility,
            area_light_visibility_samples: if area_light_visibility_samples == 4 {
                4
            } else {
                1
            },
        }
    }
}

impl Default for RaymarchRuntimeOptions {
    fn default() -> Self {
        Self::new(f32::INFINITY, false, 1)
    }
}

pub struct RaymarchPipeline {
    pipeline: wgpu::RenderPipeline,
    scene_layout: wgpu::BindGroupLayout,
    atlas: wgpu::Buffer,
    chunks: wgpu::Buffer,
    scene: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    atlas_word_capacity: usize,
    atlas_initialized: bool,
    max_chunks: usize,
    runtime_options: RaymarchRuntimeOptions,
}

impl RaymarchPipeline {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        frame_layout: &wgpu::BindGroupLayout,
        max_chunks: usize,
    ) -> Self {
        let max_chunks = max_chunks.clamp(1, 25);
        let scene_layout = create_scene_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("webgpu.raymarch.pipeline-layout"),
            bind_group_layouts: &[Some(frame_layout), Some(&scene_layout)],
            immediate_size: 0,
        });
        let shader = ShaderProgram::Raymarch.create_module(device);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("webgpu.raymarch.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("raymarch_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("raymarch_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let atlas_word_capacity = WORDS_PER_NODE as usize;
        let atlas = create_atlas_buffer(device, &[0; WORDS_PER_NODE as usize]);
        let chunks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("webgpu.raymarch.chunks"),
            size: (max_chunks * size_of::<GpuRayChunk>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("webgpu.raymarch.scene"),
            size: size_of::<GpuRayScene>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = create_bind_group(device, &scene_layout, &atlas, &chunks, &scene);
        Self {
            pipeline,
            scene_layout,
            atlas,
            chunks,
            scene,
            bind_group,
            atlas_word_capacity,
            atlas_initialized: false,
            max_chunks,
            runtime_options: RaymarchRuntimeOptions::default(),
        }
    }

    pub fn configure(&mut self, options: RaymarchRuntimeOptions) {
        self.runtime_options = options;
    }

    pub fn upload_atlas(&mut self, device: &wgpu::Device, texels: &[u32]) {
        let words: &[u32] = if texels.is_empty() {
            &[0, 0, 0, 0]
        } else {
            texels
        };
        self.atlas = create_atlas_buffer(device, words);
        self.atlas_word_capacity = words.len();
        self.atlas_initialized = !texels.is_empty();
        self.bind_group = create_bind_group(
            device,
            &self.scene_layout,
            &self.atlas,
            &self.chunks,
            &self.scene,
        );
    }

    pub fn upload_atlas_rows(
        &mut self,
        queue: &wgpu::Queue,
        first_row: u32,
        texels: &[u32],
    ) -> bool {
        let row_words = (ATLAS_TEXELS_PER_ROW * WORDS_PER_NODE) as usize;
        if !self.atlas_initialized || texels.is_empty() || texels.len() % row_words != 0 {
            return false;
        }
        let first_word = first_row as usize * row_words;
        let Some(end_word) = first_word.checked_add(texels.len()) else {
            return false;
        };
        if end_word > self.atlas_word_capacity {
            return false;
        }
        queue.write_buffer(
            &self.atlas,
            (first_word * size_of::<u32>()) as u64,
            bytemuck::cast_slice(texels),
        );
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        frame_resources: &FrameResources,
        frame: &FrameParams,
        chunks: &[ChunkDraw],
        toggles: RenderToggles,
    ) {
        let chunks: Vec<_> = select_ray_chunks(
            chunks,
            frame.camera_pos,
            self.max_chunks,
            toggles.front_to_back,
            toggles.distance_cull,
            self.runtime_options.maximum_draw_distance,
        )
        .into_iter()
        .map(|(source_index, chunk)| pack_ray_chunk(source_index, chunk))
        .collect();
        if !chunks.is_empty() {
            queue.write_buffer(&self.chunks, 0, bytemuck::cast_slice(&chunks));
        }
        queue.write_buffer(
            &self.scene,
            0,
            bytemuck::bytes_of(&GpuRayScene::from_policy(
                chunks.len() as u32,
                toggles.empty_space_skip,
                toggles.front_to_back,
                direct_light_visibility_enabled(
                    self.runtime_options.direct_light_visibility,
                    toggles.shadow_pass,
                ),
                self.runtime_options.area_light_visibility_samples,
            )),
        );

        let environment = frame.environment;
        let clear_linear = if environment.outdoor {
            environment.sky_color
        } else {
            environment.fog_color
        };
        let clear = encode_display_color(clear_linear);
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: clear[0] as f64,
                    g: clear[1] as f64,
                    b: clear[2] as f64,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("webgpu.raymarch.pass"),
            color_attachments: &color_attachments,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame_resources.bind_group(), &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

const fn direct_light_visibility_enabled(profile_capability: bool, rt_shadows: bool) -> bool {
    profile_capability && rt_shadows
}

/// Selects the bounded resident set before packing it for the GPU. Disabled
/// policy preserves exact source order; enabled policy uses stable distance
/// to each complete chunk cube, so a large cube containing the camera wins
/// over a smaller cube whose origin merely happens to be nearby.
fn select_ray_chunks(
    chunks: &[ChunkDraw],
    camera_position: [f32; 3],
    maximum: usize,
    front_to_back: bool,
    distance_cull: bool,
    maximum_draw_distance: f32,
) -> Vec<(usize, &ChunkDraw)> {
    let mut selected: Vec<_> = chunks.iter().enumerate().collect();
    if distance_cull {
        let maximum_distance_squared = maximum_draw_distance * maximum_draw_distance;
        selected.retain(|(_, chunk)| {
            let distance_squared = distance_squared_to_chunk(camera_position, chunk);
            distance_squared.is_finite()
                && (maximum_distance_squared.is_infinite()
                    || distance_squared <= maximum_distance_squared)
        });
    }
    if front_to_back {
        selected.sort_by(|(left_index, left), (right_index, right)| {
            distance_squared_to_chunk(camera_position, left)
                .total_cmp(&distance_squared_to_chunk(camera_position, right))
                .then_with(|| left_index.cmp(right_index))
        });
    }
    selected.truncate(maximum);
    selected
}

fn distance_squared_to_chunk(camera_position: [f32; 3], chunk: &ChunkDraw) -> f32 {
    if camera_position.iter().any(|value| !value.is_finite())
        || chunk.origin.iter().any(|value| !value.is_finite())
        || !chunk.world_size.is_finite()
        || chunk.world_size <= 0.0
    {
        return f32::INFINITY;
    }

    let mut distance_squared = 0.0;
    for axis in 0..3 {
        let minimum = chunk.origin[axis];
        let maximum = minimum + chunk.world_size;
        if !maximum.is_finite() {
            return f32::INFINITY;
        }
        let delta = if camera_position[axis] < minimum {
            minimum - camera_position[axis]
        } else if camera_position[axis] > maximum {
            camera_position[axis] - maximum
        } else {
            0.0
        };
        distance_squared += delta * delta;
    }
    if distance_squared.is_finite() {
        distance_squared
    } else {
        f32::INFINITY
    }
}

fn pack_ray_chunk(source_index: usize, chunk: &ChunkDraw) -> GpuRayChunk {
    GpuRayChunk {
        origin_world_size: [
            chunk.origin[0],
            chunk.origin[1],
            chunk.origin[2],
            chunk.world_size,
        ],
        voxel_depth: [chunk.voxel_size, f32::from(chunk.svo_depth), 0.0, 0.0],
        // Keep the stable source ordinal so reordering cannot change which
        // overlapping chunk wins an exactly equal-distance hit.
        indices: [
            chunk.root_index.max(0) as u32,
            source_index.min(u32::MAX as usize) as u32,
            0,
            0,
        ],
    }
}

fn create_scene_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu.raymarch.scene-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<GpuRayChunk>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<GpuRayScene>() as u64),
                },
                count: None,
            },
        ],
    })
}

fn create_atlas_buffer(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("webgpu.raymarch.atlas"),
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    atlas: &wgpu::Buffer,
    chunks: &wgpu::Buffer,
    scene: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webgpu.raymarch.scene-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: atlas.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: chunks.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scene.as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(origin: [f32; 3], world_size: f32, root_index: i32) -> ChunkDraw {
        ChunkDraw {
            origin,
            root_index,
            world_size,
            voxel_size: 0.25,
            svo_depth: 6,
        }
    }

    #[test]
    fn front_to_back_selects_nearest_candidates_before_the_chunk_cap() {
        let chunks = [
            chunk([20.0, 0.0, 0.0], 1.0, 10),
            chunk([-1.0, -1.0, -1.0], 2.0, 20),
            chunk([5.0, 0.0, 0.0], 1.0, 30),
        ];
        let selected = select_ray_chunks(&chunks, [0.0; 3], 2, true, false, f32::INFINITY);
        assert_eq!(
            selected
                .iter()
                .map(|(index, chunk)| (*index, chunk.root_index))
                .collect::<Vec<_>>(),
            vec![(1, 20), (2, 30)]
        );

        let packed: Vec<_> = selected
            .into_iter()
            .map(|(index, chunk)| pack_ray_chunk(index, chunk))
            .collect();
        assert_eq!(packed[0].indices[..2], [20, 1]);
        assert_eq!(packed[1].indices[..2], [30, 2]);
    }

    #[test]
    fn disabled_front_to_back_preserves_source_order_and_truncation() {
        let chunks = [
            chunk([20.0, 0.0, 0.0], 1.0, 10),
            chunk([0.0, 0.0, 0.0], 1.0, 20),
            chunk([5.0, 0.0, 0.0], 1.0, 30),
        ];
        let selected = select_ray_chunks(&chunks, [0.0; 3], 2, false, false, f32::INFINITY);
        assert_eq!(
            selected
                .iter()
                .map(|(index, chunk)| (*index, chunk.root_index))
                .collect::<Vec<_>>(),
            vec![(0, 10), (1, 20)]
        );
    }

    #[test]
    fn distance_order_uses_complete_bounds_and_stable_source_ties() {
        let chunks = [
            chunk([-10.0, -10.0, -10.0], 20.0, 10),
            chunk([2.0, 0.0, 0.0], 1.0, 20),
            chunk([-3.0, 0.0, 0.0], 1.0, 30),
        ];
        let selected = select_ray_chunks(&chunks, [0.0; 3], 3, true, false, f32::INFINITY);
        assert_eq!(
            selected.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn distance_culling_filters_complete_bounds_before_the_resident_cap() {
        let chunks = [
            chunk([20.0, 0.0, 0.0], 1.0, 10),
            chunk([2.5, -1.0, -1.0], 2.0, 20),
            chunk([-1.0, -1.0, -1.0], 2.0, 30),
        ];
        let selected = select_ray_chunks(&chunks, [0.0; 3], 2, false, true, 3.0);
        assert_eq!(
            selected
                .iter()
                .map(|(index, chunk)| (*index, chunk.root_index))
                .collect::<Vec<_>>(),
            vec![(1, 20), (2, 30)]
        );
    }

    #[test]
    fn disabled_distance_culling_preserves_source_order() {
        let chunks = [
            chunk([20.0, 0.0, 0.0], 1.0, 10),
            chunk([0.0, 0.0, 0.0], 1.0, 20),
        ];
        let selected = select_ray_chunks(&chunks, [0.0; 3], 2, false, false, 1.0);
        assert_eq!(
            selected.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn runtime_options_validate_open_distance_and_area_sample_quality() {
        let defaults = RaymarchRuntimeOptions::default();
        assert!(defaults.maximum_draw_distance.is_infinite());
        assert!(!defaults.direct_light_visibility);
        assert_eq!(defaults.area_light_visibility_samples, 1);

        let configured = RaymarchRuntimeOptions::new(80.0, true, 4);
        assert_eq!(configured.maximum_draw_distance, 80.0);
        assert!(configured.direct_light_visibility);
        assert_eq!(configured.area_light_visibility_samples, 4);

        let invalid = RaymarchRuntimeOptions::new(f32::NAN, true, 2);
        assert!(invalid.maximum_draw_distance.is_infinite());
        assert_eq!(invalid.area_light_visibility_samples, 1);
    }

    #[test]
    fn direct_visibility_requires_both_profile_capability_and_runtime_toggle() {
        assert!(direct_light_visibility_enabled(true, true));
        assert!(!direct_light_visibility_enabled(true, false));
        assert!(!direct_light_visibility_enabled(false, true));
        assert!(!direct_light_visibility_enabled(false, false));
    }

    #[test]
    fn shader_keeps_skip_as_coarse_proof_and_dda_as_canonical_resolver() {
        let source = include_str!("../shaders/raymarch.wgsl");
        assert!(source.contains("fn trace_chunk_dda("));
        assert!(source.contains("fn trace_chunk_skipping_empty_leaves("));
        assert!(source.contains("let refinement_interval = BoxHit("));
        assert!(source.contains("return trace_chunk_dda("));
        assert!(source.contains("abs(times.x - nearest) <= tolerance"));
        assert!(source.contains("abs(times.y - nearest) <= tolerance"));
        assert!(source.contains("ray_policy_enabled(RAY_POLICY_EMPTY_SPACE_SKIP)"));
        assert!(source.contains("chunk.indices.y < nearest_source_index"));
    }

    /// The primary-ray loop must never trace a candidate chunk past the
    /// closest hit already found for this pixel — provably safe (a farther
    /// hit in that chunk could not win the nearest-hit comparison anyway),
    /// and the same clip-to-known-distance pattern `finite_segment_visible`
    /// already trusts for shadow rays against a light's own distance.
    #[test]
    fn primary_rays_clip_the_search_interval_to_the_closest_hit_so_far() {
        let source = include_str!("../shaders/raymarch.wgsl");
        let fragment = &source[source.find("@fragment").expect("fragment entry")..];
        assert!(fragment.contains("min(interval.exit, nearest.distance)"));
        assert!(fragment.contains("let hit = trace_chunk(local_origin, direction, chunk, clipped_interval)"));
    }

    #[test]
    fn shader_secondary_rays_are_finite_exact_and_lazily_evaluated() {
        let source = include_str!("../shaders/raymarch.wgsl");
        let visibility_start = source
            .find("fn trace_visibility_chunk_dda(")
            .expect("secondary DDA");
        let visibility_end = source.find("fn camera_ray(").expect("camera ray");
        let visibility = &source[visibility_start..visibility_end];

        assert!(source.contains("DIRECT_VISIBILITY_TRACE_BUDGET: u32 = 4096u"));
        assert!(!visibility.contains("frame.render_options.w"));
        assert!(!visibility.contains("ray_policy_enabled(RAY_POLICY_EMPTY_SPACE_SKIP)"));
        assert!(visibility.contains("fn trace_visibility_chunk_skipping_empty_leaves("));
        assert!(visibility.contains("let refined = trace_visibility_chunk_dda("));
        assert!(visibility.contains("let biased_origin = receiver_world + receiver_normal"));
        assert!(visibility.contains("let maximum_distance = endpoint_distance - endpoint_bias"));
        assert!(visibility.contains("for (var index = 0u; index < chunk_count; index += 1u)"));
        assert!(visibility.contains("let finite_exit = min(slab.exit, maximum_distance)"));
        assert!(visibility.contains("if trace.blocked {\n            return false;"));

        assert!(visibility.contains("point_light_sample_irradiance("));
        assert!(visibility.contains("rectangle_gauss_sample_position("));
        assert!(visibility.contains("rectangle_light_sample_weight("));
        assert!(visibility.contains("rectangle_light_irradiance_from_integral("));
        assert!(visibility.contains("let unoccluded = evaluate_rectangle_light("));
        assert!(visibility.contains("if !has_positive_rgb(sample_irradiance)"));

        let fragment = &source[source.find("@fragment").expect("fragment entry")..];
        let emissive_branch = fragment.find("if emits {").expect("emissive branch");
        let visibility_branch = fragment
            .find("else if ray_policy_enabled(RAY_POLICY_DIRECT_LIGHT_VISIBILITY)")
            .expect("visibility branch");
        assert!(emissive_branch < visibility_branch);
        assert!(!fragment.contains("let radiance = select("));
        assert!(fragment.contains("surface_radiance_from_analytic_direct("));
    }
}
