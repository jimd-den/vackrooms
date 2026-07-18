//! Indexed surface cache and pass.

use std::collections::BTreeMap;
use std::num::NonZeroU64;

use wgpu::util::DeviceExt;

use crate::application::ports::{FrameParams, SurfaceChunk, SurfaceChunkKey};
use crate::application::render_settings::RenderToggles;
use crate::application::rendering::encode_display_color;

use super::raster_shadow::{HeroShadowMap, HeroShadowOptions};
use super::visibility::{bounds_sphere, sphere_is_visible};
use crate::drivers::webgpu::frame_resources::FrameResources;
use crate::drivers::webgpu::gpu_types::{GpuChunkUniforms, GpuPackedSurfaceVertex};
use crate::drivers::webgpu::shader::ShaderProgram;

struct SurfaceChunkGpu {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    index_count: u32,
    origin: [f32; 3],
    bounds_max: [f32; 3],
    voxel_size: f32,
}

pub struct SurfacePipeline {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow: HeroShadowMap,
    chunk_layout: wgpu::BindGroupLayout,
    /// Stable key order is the reference submission order when f2b is off.
    chunks: BTreeMap<SurfaceChunkKey, SurfaceChunkGpu>,
    max_draw_distance: f32,
}

impl SurfacePipeline {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        frame_layout: &wgpu::BindGroupLayout,
        max_draw_distance: f32,
    ) -> Self {
        Self::new_with_shadow_options(
            device,
            target_format,
            frame_layout,
            max_draw_distance,
            HeroShadowOptions::default(),
        )
    }

    pub fn new_with_shadow_options(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        frame_layout: &wgpu::BindGroupLayout,
        max_draw_distance: f32,
        shadow_options: HeroShadowOptions,
    ) -> Self {
        let chunk_layout = create_chunk_layout(device, size_of::<GpuPackedSurfaceVertex>() as u64);
        let shadow = HeroShadowMap::new(device, shadow_options);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("webgpu.surface.pipeline-layout"),
            bind_group_layouts: &[
                Some(frame_layout),
                Some(&chunk_layout),
                Some(shadow.receiver_layout()),
            ],
            immediate_size: 0,
        });
        let shader = ShaderProgram::Surface.create_module(device);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("webgpu.surface.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("surface_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("surface_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let shadow_pipeline = create_surface_shadow_pipeline(device, &chunk_layout, &shadow);
        Self {
            pipeline,
            shadow_pipeline,
            shadow,
            chunk_layout,
            chunks: BTreeMap::new(),
            max_draw_distance,
        }
    }

    pub fn upload(&mut self, device: &wgpu::Device, chunks: &[SurfaceChunk<'_>]) {
        for &chunk in chunks {
            self.chunks.remove(&chunk.key);
            if chunk.mesh.vertices.is_empty() || chunk.mesh.indices.is_empty() {
                continue;
            }
            let vertices: Vec<_> = chunk
                .mesh
                .vertices
                .iter()
                .map(GpuPackedSurfaceVertex::from)
                .collect();
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("webgpu.surface.vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("webgpu.surface.indices"),
                contents: bytemuck::cast_slice(&chunk.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("webgpu.surface.chunk-uniforms"),
                size: size_of::<GpuChunkUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("webgpu.surface.chunk-bind-group"),
                layout: &self.chunk_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                ],
            });
            self.chunks.insert(
                chunk.key,
                SurfaceChunkGpu {
                    vertices: vertex_buffer,
                    indices: index_buffer,
                    uniforms: uniform_buffer,
                    bind_group,
                    index_count: chunk.mesh.indices.len().min(u32::MAX as usize) as u32,
                    origin: chunk.origin,
                    bounds_max: chunk.mesh.bounds.max,
                    voxel_size: chunk.mesh.voxel_scale,
                },
            );
        }
    }

    pub fn remove(&mut self, keys: &[SurfaceChunkKey]) {
        for key in keys {
            self.chunks.remove(key);
        }
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    pub fn draw(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        frame_resources: &FrameResources,
        frame: &FrameParams,
        toggles: RenderToggles,
        light_count: u32,
    ) {
        for chunk in self.chunks.values() {
            let uniforms = GpuChunkUniforms {
                origin_world_size: [chunk.origin[0], chunk.origin[1], chunk.origin[2], 0.0],
                bounds_voxel_size: [
                    chunk.bounds_max[0],
                    chunk.bounds_max[1],
                    chunk.bounds_max[2],
                    chunk.voxel_size,
                ],
                draw: [0, light_count, 0, 0],
            };
            queue.write_buffer(&chunk.uniforms, 0, bytemuck::bytes_of(&uniforms));
        }
        if self.shadow.prepare(queue, frame, toggles.shadow_pass) {
            self.draw_shadow_pass(encoder);
        }

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
            label: Some("webgpu.surface.pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame_resources.bind_group(), &[]);
        pass.set_bind_group(2, self.shadow.receiver_bind_group(), &[]);
        for chunk in self.chunks.values() {
            if toggles.distance_cull {
                let (center, radius) = bounds_sphere(chunk.origin, chunk.bounds_max);
                if !sphere_is_visible(center, radius, frame, self.max_draw_distance) {
                    continue;
                }
            }
            // Keep the storage buffer alive and make its ownership explicit,
            // even though vertex pulling references it through the bind group.
            let _keep_alive = &chunk.vertices;
            pass.set_bind_group(1, &chunk.bind_group, &[]);
            pass.set_index_buffer(chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..chunk.index_count, 0, 0..1);
        }
    }

    /// Every resident indexed surface casts. Camera visibility cannot reject
    /// an off-screen object whose shadow reaches an on-screen receiver.
    fn draw_shadow_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("webgpu.surface.hero-shadow-pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.shadow.view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, self.shadow.caster_bind_group(), &[]);
        for chunk in self.chunks.values() {
            let _keep_alive = &chunk.vertices;
            pass.set_bind_group(1, &chunk.bind_group, &[]);
            pass.set_index_buffer(chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..chunk.index_count, 0, 0..1);
        }
    }
}

fn create_surface_shadow_pipeline(
    device: &wgpu::Device,
    chunk_layout: &wgpu::BindGroupLayout,
    shadow: &HeroShadowMap,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("webgpu.surface.hero-shadow-pipeline-layout"),
        bind_group_layouts: &[Some(shadow.caster_layout()), Some(chunk_layout)],
        immediate_size: 0,
    });
    let shader = ShaderProgram::SurfaceShadow.create_module(device);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("webgpu.surface.hero-shadow-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("surface_shadow_vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 1.5,
                clamp: 0.0,
            },
        }),
        multisample: Default::default(),
        fragment: None,
        multiview_mask: None,
        cache: None,
    })
}

pub(super) fn create_chunk_layout(
    device: &wgpu::Device,
    element_size: u64,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu.raster-chunk-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<GpuChunkUniforms>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(element_size),
                },
                count: None,
            },
        ],
    })
}
