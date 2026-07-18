//! WebGPU presentation adapter for the deterministic CPU reference renderer.

use crate::adapters::cpu_splatter::{CpuRenderSettings, SoftwareRasterizer};
use crate::application::ports::{ChunkDraw, FrameParams, RendererPort};
use crate::drivers::webgpu::shader::ShaderProgram;

pub struct CpuPresentPipeline {
    rasterizer: SoftwareRasterizer,
    pipeline: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl CpuPresentPipeline {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("webgpu.cpu-present.texture-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("webgpu.cpu-present.pipeline-layout"),
            bind_group_layouts: &[Some(&texture_layout)],
            immediate_size: 0,
        });
        let shader = ShaderProgram::CpuPresent.create_module(device);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("webgpu.cpu-present.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("cpu_present_vertex"),
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
                entry_point: Some("cpu_present_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("webgpu.cpu-present.sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let (texture, bind_group) =
            create_texture_resources(device, &texture_layout, &sampler, width, height);
        Self {
            rasterizer: SoftwareRasterizer::new(width as usize, height as usize),
            pipeline,
            texture_layout,
            sampler,
            texture,
            bind_group,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.rasterizer.resize(width as usize, height as usize);
        (self.texture, self.bind_group) =
            create_texture_resources(device, &self.texture_layout, &self.sampler, width, height);
    }

    pub fn upload_atlas(&mut self, texels: &[u32]) {
        self.rasterizer.upload_atlas(texels);
    }

    pub fn upload_atlas_rows(&mut self, first_row: u32, texels: &[u32]) -> bool {
        self.rasterizer.upload_atlas_rows(first_row, texels)
    }

    pub fn telemetry_string(&self) -> String {
        let stats = self.rasterizer.telemetry();
        format!(
            "CPU/WebGPU V:{}/{}{} (D:{})/S:{}/P:{}K/{}K{}{}",
            stats.visited_nodes,
            stats.node_visit_limit,
            if stats.node_budget_limited { "N!" } else { "" },
            stats.max_virtual_depth,
            stats.splat_count,
            stats.pixel_writes / 1000,
            stats.pixel_write_limit / 1000,
            if stats.pixel_budget_limited { "P!" } else { "" },
            if stats.budget_exhausted { "!" } else { "" },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        frame: &FrameParams,
        chunks: &[ChunkDraw],
        settings: CpuRenderSettings,
    ) {
        self.rasterizer.settings = settings;
        self.rasterizer.draw(frame, chunks);
        queue.write_texture(
            self.texture.as_image_copy(),
            self.rasterizer.framebuffer(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("webgpu.cpu-present.pass"),
            color_attachments: &color_attachments,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn create_texture_resources(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("webgpu.cpu-present.frame"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webgpu.cpu-present.texture-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    (texture, bind_group)
}
