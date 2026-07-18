//! Face-splat cache and staged instanced pass.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use wgpu::util::DeviceExt;

use crate::application::ports::{FaceCellRange, FrameParams, SurfaceChunk, SurfaceChunkKey};
use crate::application::render_settings::RenderToggles;
use crate::application::rendering::encode_display_color;

use super::raster_shadow::{HeroShadowMap, HeroShadowOptions};
use super::surface::create_chunk_layout;
use super::visibility::{bounds_sphere, sphere_is_visible};
use crate::drivers::webgpu::frame_resources::FrameResources;
use crate::drivers::webgpu::gpu_types::{GpuChunkUniforms, GpuPackedFace};
use crate::drivers::webgpu::shader::ShaderProgram;

/// A capped face may be bucketed at a cell corner and extend another half
/// cell along both in-plane axes. This proven bound is deliberately larger
/// than the cell cube's own circumscribed sphere.
const CONSERVATIVE_CELL_RADIUS_SCALE: f32 = 1.5;

struct SplatChunkGpu {
    faces: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    face_count: u32,
    cells: Vec<FaceCellRange>,
    cell_size: f32,
    origin: [f32; 3],
    bounds_max: [f32; 3],
    voxel_size: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SplatFrameStats {
    pub faces_drawn: u32,
    pub faces_dropped: u32,
    pub cells_culled: u32,
    pub draw_calls: u32,
}

pub struct SplatPipeline {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow: HeroShadowMap,
    chunk_layout: wgpu::BindGroupLayout,
    /// Stable key order is the reference submission order when f2b is off.
    chunks: BTreeMap<SurfaceChunkKey, SplatChunkGpu>,
    max_draw_distance: f32,
    face_budget: u32,
    stats: SplatFrameStats,
}

impl SplatPipeline {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        frame_layout: &wgpu::BindGroupLayout,
        max_draw_distance: f32,
        face_budget: u32,
    ) -> Self {
        Self::new_with_shadow_options(
            device,
            target_format,
            frame_layout,
            max_draw_distance,
            face_budget,
            HeroShadowOptions::default(),
        )
    }

    pub fn new_with_shadow_options(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        frame_layout: &wgpu::BindGroupLayout,
        max_draw_distance: f32,
        face_budget: u32,
        shadow_options: HeroShadowOptions,
    ) -> Self {
        let chunk_layout = create_chunk_layout(device, size_of::<GpuPackedFace>() as u64);
        let shadow = HeroShadowMap::new(device, shadow_options);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("webgpu.splat.pipeline-layout"),
            bind_group_layouts: &[
                Some(frame_layout),
                Some(&chunk_layout),
                Some(shadow.receiver_layout()),
            ],
            immediate_size: 0,
        });
        let shader = ShaderProgram::Splat.create_module(device);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("webgpu.splat.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("splat_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
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
                entry_point: Some("splat_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let shadow_pipeline = create_splat_shadow_pipeline(device, &chunk_layout, &shadow);
        Self {
            pipeline,
            shadow_pipeline,
            shadow,
            chunk_layout,
            chunks: BTreeMap::new(),
            max_draw_distance,
            face_budget,
            stats: SplatFrameStats::default(),
        }
    }

    pub fn upload(&mut self, device: &wgpu::Device, chunks: &[SurfaceChunk<'_>]) {
        for &chunk in chunks {
            self.chunks.remove(&chunk.key);
            if chunk.mesh.faces.instances.is_empty() {
                continue;
            }
            let faces: Vec<_> = chunk
                .mesh
                .faces
                .instances
                .iter()
                .map(GpuPackedFace::from)
                .collect();
            let face_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("webgpu.splat.faces"),
                contents: bytemuck::cast_slice(&faces),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("webgpu.splat.chunk-uniforms"),
                size: size_of::<GpuChunkUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("webgpu.splat.chunk-bind-group"),
                layout: &self.chunk_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: face_buffer.as_entire_binding(),
                    },
                ],
            });
            self.chunks.insert(
                chunk.key,
                SplatChunkGpu {
                    faces: face_buffer,
                    uniforms: uniform_buffer,
                    bind_group,
                    face_count: faces.len().min(u32::MAX as usize) as u32,
                    cells: chunk.mesh.faces.cells.clone(),
                    cell_size: chunk.mesh.faces.cell_size,
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

    pub fn stats(&self) -> SplatFrameStats {
        self.stats
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
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

        let mut chunks: Vec<_> = self.chunks.values().collect();
        if toggles.front_to_back || toggles.face_budget {
            chunks.sort_by(|left, right| {
                distance_squared(left.origin, frame.camera_pos)
                    .partial_cmp(&distance_squared(right.origin, frame.camera_pos))
                    .unwrap_or(Ordering::Equal)
            });
        }
        self.stats = SplatFrameStats::default();

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
            label: Some("webgpu.splat.pass"),
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

        for chunk in chunks {
            if toggles.distance_cull {
                let (center, radius) = bounds_sphere(chunk.origin, chunk.bounds_max);
                if !sphere_is_visible(center, radius, frame, self.max_draw_distance) {
                    self.stats.faces_dropped =
                        self.stats.faces_dropped.saturating_add(chunk.face_count);
                    continue;
                }
            }
            if face_budget_is_exhausted(
                toggles.face_budget,
                self.stats.faces_drawn,
                self.face_budget,
            ) {
                self.stats.faces_dropped =
                    self.stats.faces_dropped.saturating_add(chunk.face_count);
                continue;
            }
            let _keep_alive = &chunk.faces;
            pass.set_bind_group(1, &chunk.bind_group, &[]);
            if toggles.cell_culling && !chunk.cells.is_empty() {
                for cell in &chunk.cells {
                    let center = [
                        chunk.origin[0] + (f32::from(cell.cell[0]) + 0.5) * chunk.cell_size,
                        chunk.origin[1] + (f32::from(cell.cell[1]) + 0.5) * chunk.cell_size,
                        chunk.origin[2] + (f32::from(cell.cell[2]) + 0.5) * chunk.cell_size,
                    ];
                    let radius = conservative_cell_radius(chunk.cell_size);
                    if !sphere_is_visible(center, radius, frame, self.max_draw_distance) {
                        self.stats.cells_culled = self.stats.cells_culled.saturating_add(1);
                        continue;
                    }
                    let end = cell.offset.saturating_add(cell.count).min(chunk.face_count);
                    if cell.offset < end {
                        pass.draw(0..4, cell.offset..end);
                        self.stats.faces_drawn =
                            self.stats.faces_drawn.saturating_add(end - cell.offset);
                        self.stats.draw_calls = self.stats.draw_calls.saturating_add(1);
                    }
                }
            } else {
                pass.draw(0..4, 0..chunk.face_count);
                self.stats.faces_drawn = self.stats.faces_drawn.saturating_add(chunk.face_count);
                self.stats.draw_calls = self.stats.draw_calls.saturating_add(1);
            }
        }
    }

    /// Every resident packed face casts, independently from camera cells and
    /// the visible-face budget. This path never materializes indexed geometry.
    fn draw_shadow_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("webgpu.splat.hero-shadow-pass"),
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
            let _keep_alive = &chunk.faces;
            pass.set_bind_group(1, &chunk.bind_group, &[]);
            pass.draw(0..4, 0..chunk.face_count);
        }
    }
}

fn create_splat_shadow_pipeline(
    device: &wgpu::Device,
    chunk_layout: &wgpu::BindGroupLayout,
    shadow: &HeroShadowMap,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("webgpu.splat.hero-shadow-pipeline-layout"),
        bind_group_layouts: &[Some(shadow.caster_layout()), Some(chunk_layout)],
        immediate_size: 0,
    });
    let shader = ShaderProgram::SplatShadow.create_module(device);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("webgpu.splat.hero-shadow-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("splat_shadow_vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
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

fn distance_squared(left: [f32; 3], right: [f32; 3]) -> f32 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    dx * dx + dy * dy + dz * dz
}

fn conservative_cell_radius(cell_size: f32) -> f32 {
    cell_size.max(0.0) * CONSERVATIVE_CELL_RADIUS_SCALE
}

fn face_budget_is_exhausted(enabled: bool, faces_drawn: u32, budget: u32) -> bool {
    enabled && faces_drawn >= budget
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splat_cell_bound_covers_capped_faces_beyond_the_cell_cube() {
        assert_eq!(conservative_cell_radius(4.0), 6.0);
        assert_eq!(conservative_cell_radius(-1.0), 0.0);
    }

    #[test]
    fn nearest_whole_chunk_may_cross_the_face_budget() {
        let budget = 10;
        assert!(!face_budget_is_exhausted(true, 0, budget));
        // A first chunk with 25 faces is submitted whole. Only subsequent
        // farther chunks observe that the soft budget has been exhausted.
        assert!(face_budget_is_exhausted(true, 25, budget));
        assert!(!face_budget_is_exhausted(false, 25, budget));
    }
}
