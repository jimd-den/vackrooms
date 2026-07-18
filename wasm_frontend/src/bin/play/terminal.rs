//! The native field terminal: the desktop twin of the browser's DOM HUD.
//!
//! Same design language as `static/index.html` — research-DOS survival
//! instrumentation, sparse text channels, square geometry, explicit units —
//! rendered as two overlay pipelines: flat alpha-blended rects, and text
//! quads sampling a glyph atlas rasterized at startup from the same
//! Bytesized face the web terminal loads from Google Fonts (vendored under
//! `assets/`, SIL OFL).

use std::collections::HashMap;

// ---- Palette: static/index.html :root tokens, RGBA on straight alpha. ----
pub const INK: [f32; 4] = [0.835, 0.859, 0.816, 1.0];
pub const QUIET: [f32; 4] = [0.498, 0.545, 0.510, 1.0];
pub const ROUTE: [f32; 4] = [0.843, 0.741, 0.439, 1.0];
pub const SYSTEM: [f32; 4] = [0.486, 0.682, 0.667, 1.0];
pub const HAZARD: [f32; 4] = [0.831, 0.353, 0.298, 1.0];
pub const CRITICAL: [f32; 4] = [0.941, 0.875, 0.812, 1.0];
pub const PANEL: [f32; 4] = [0.027, 0.039, 0.031, 0.62];
pub const LOADING_PLATE: [f32; 4] = [0.0, 0.0, 0.0, 0.55];

/// Glyphs are rasterized once at this pixel size; text draws at integer
/// fractions/multiples so the 8px grid of the face stays crisp.
const RASTER_PX: f32 = 32.0;
const FIRST_GLYPH: char = '!';
const LAST_GLYPH: char = '~';

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RectVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TextVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

const RECT_SHADER: &str = r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn rect_vertex(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn rect_fragment(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

const TEXT_SHADER: &str = r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@group(0) @binding(0) var glyphs: texture_2d<f32>;
@group(0) @binding(1) var glyph_sampler: sampler;

@vertex
fn text_vertex(
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uv;
    out.color = color;
    return out;
}

@fragment
fn text_fragment(in: VsOut) -> @location(0) vec4<f32> {
    let coverage = textureSample(glyphs, glyph_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
"#;

struct Glyph {
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    /// Pixel size of the rasterized bitmap at `RASTER_PX`.
    size: [f32; 2],
    /// Offset from the pen position (baseline-left) to the bitmap's
    /// top-left, in raster pixels; y grows downward.
    bearing: [f32; 2],
    advance: f32,
}

pub struct FieldTerminal {
    rect_pipeline: wgpu::RenderPipeline,
    text_pipeline: wgpu::RenderPipeline,
    _atlas: wgpu::Texture,
    text_bind_group: wgpu::BindGroup,
    rect_buffer: wgpu::Buffer,
    text_buffer: wgpu::Buffer,
    glyphs: HashMap<char, Glyph>,
    line_height: f32,
    space_advance: f32,
    rects: Vec<RectVertex>,
    texts: Vec<TextVertex>,
    width: f32,
    height: f32,
}

const RECT_CAPACITY: usize = 4096;
const TEXT_CAPACITY: usize = 16384;

impl FieldTerminal {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let font = fontdue::Font::from_bytes(
            &include_bytes!("../../../assets/Bytesized-Regular.ttf")[..],
            fontdue::FontSettings {
                scale: RASTER_PX,
                ..Default::default()
            },
        )
        .expect("parse vendored Bytesized font");

        // Fixed-grid atlas: one cell per printable ASCII glyph.
        let cell = RASTER_PX as usize + 8;
        let columns = 12usize;
        let count = LAST_GLYPH as usize - FIRST_GLYPH as usize + 1;
        let rows = count.div_ceil(columns);
        let atlas_w = columns * cell;
        let atlas_h = rows * cell;
        let mut pixels = vec![0u8; atlas_w * atlas_h];
        let mut glyphs = HashMap::new();
        for (index, code) in (FIRST_GLYPH as u32..=LAST_GLYPH as u32).enumerate() {
            let ch = char::from_u32(code).expect("printable ascii");
            let (metrics, bitmap) = font.rasterize(ch, RASTER_PX);
            let cx = (index % columns) * cell;
            let cy = (index / columns) * cell;
            for y in 0..metrics.height.min(cell) {
                for x in 0..metrics.width.min(cell) {
                    pixels[(cy + y) * atlas_w + cx + x] = bitmap[y * metrics.width + x];
                }
            }
            glyphs.insert(
                ch,
                Glyph {
                    uv_min: [cx as f32 / atlas_w as f32, cy as f32 / atlas_h as f32],
                    uv_max: [
                        (cx + metrics.width) as f32 / atlas_w as f32,
                        (cy + metrics.height) as f32 / atlas_h as f32,
                    ],
                    size: [metrics.width as f32, metrics.height as f32],
                    bearing: [
                        metrics.xmin as f32,
                        -(metrics.height as f32 + metrics.ymin as f32),
                    ],
                    advance: metrics.advance_width,
                },
            );
        }
        let space_advance = font
            .rasterize(' ', RASTER_PX)
            .0
            .advance_width
            .max(RASTER_PX * 0.28);
        let line_height = font
            .horizontal_line_metrics(RASTER_PX)
            .map(|m| m.new_line_size)
            .unwrap_or(RASTER_PX * 1.2);

        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("native.terminal.glyphs"),
            size: wgpu::Extent3d {
                width: atlas_w as u32,
                height: atlas_h as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            atlas.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_w as u32),
                rows_per_image: Some(atlas_h as u32),
            },
            wgpu::Extent3d {
                width: atlas_w as u32,
                height: atlas_h as u32,
                depth_or_array_layers: 1,
            },
        );
        // Nearest sampling: Bytesized is a pixel face; interpolation would
        // only smear the grid.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("native.terminal.sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let text_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native.terminal.text-layout"),
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
        let text_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native.terminal.text-bind"),
            layout: &text_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &atlas.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let rect_pipeline = build_pipeline(
            device,
            format,
            RECT_SHADER,
            "rect_vertex",
            "rect_fragment",
            &[],
            size_of::<RectVertex>() as u64,
            &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
        );
        let text_pipeline = build_pipeline(
            device,
            format,
            TEXT_SHADER,
            "text_vertex",
            "text_fragment",
            std::slice::from_ref(&text_layout),
            size_of::<TextVertex>() as u64,
            &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
        );

        let rect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native.terminal.rects"),
            size: (RECT_CAPACITY * size_of::<RectVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let text_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native.terminal.text"),
            size: (TEXT_CAPACITY * size_of::<TextVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            rect_pipeline,
            text_pipeline,
            _atlas: atlas,
            text_bind_group,
            rect_buffer,
            text_buffer,
            glyphs,
            line_height,
            space_advance,
            rects: Vec::new(),
            texts: Vec::new(),
            width: 1.0,
            height: 1.0,
        }
    }

    /// Starts a frame's overlay; all push coordinates are window pixels.
    pub fn begin(&mut self, width: u32, height: u32) {
        self.width = width.max(1) as f32;
        self.height = height.max(1) as f32;
        self.rects.clear();
        self.texts.clear();
    }

    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        let x0 = x / self.width * 2.0 - 1.0;
        let x1 = (x + w) / self.width * 2.0 - 1.0;
        let y0 = 1.0 - y / self.height * 2.0;
        let y1 = 1.0 - (y + h) / self.height * 2.0;
        for pos in [[x0, y0], [x0, y1], [x1, y1], [x0, y0], [x1, y1], [x1, y0]] {
            self.rects.push(RectVertex { pos, color });
        }
    }

    /// Draws one line; `px` is the on-screen glyph size (16/24/32 keep the
    /// pixel grid exact). Returns the pen x after the last glyph.
    pub fn text(&mut self, text: &str, x: f32, y_baseline: f32, px: f32, color: [f32; 4]) -> f32 {
        let scale = px / RASTER_PX;
        let mut pen = x;
        for ch in text.chars() {
            if ch == ' ' {
                pen += self.space_advance * scale;
                continue;
            }
            let Some(glyph) = self.glyphs.get(&ch) else {
                pen += self.space_advance * scale;
                continue;
            };
            let gx = pen + glyph.bearing[0] * scale;
            let gy = y_baseline + glyph.bearing[1] * scale;
            let gw = glyph.size[0] * scale;
            let gh = glyph.size[1] * scale;
            let x0 = gx / self.width * 2.0 - 1.0;
            let x1 = (gx + gw) / self.width * 2.0 - 1.0;
            let y0 = 1.0 - gy / self.height * 2.0;
            let y1 = 1.0 - (gy + gh) / self.height * 2.0;
            let (u0, v0) = (glyph.uv_min[0], glyph.uv_min[1]);
            let (u1, v1) = (glyph.uv_max[0], glyph.uv_max[1]);
            let quad = [
                ([x0, y0], [u0, v0]),
                ([x0, y1], [u0, v1]),
                ([x1, y1], [u1, v1]),
                ([x0, y0], [u0, v0]),
                ([x1, y1], [u1, v1]),
                ([x1, y0], [u1, v0]),
            ];
            for (pos, uv) in quad {
                self.texts.push(TextVertex { pos, uv, color });
            }
            pen += glyph.advance * scale;
        }
        pen
    }

    /// Measures a line's advance width at `px` without emitting quads.
    pub fn measure(&self, text: &str, px: f32) -> f32 {
        let scale = px / RASTER_PX;
        text.chars()
            .map(|ch| match self.glyphs.get(&ch) {
                Some(glyph) => glyph.advance * scale,
                None => self.space_advance * scale,
            })
            .sum()
    }

    pub fn line_height(&self, px: f32) -> f32 {
        self.line_height * (px / RASTER_PX)
    }

    pub fn draw(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        self.rects.truncate(RECT_CAPACITY);
        self.texts.truncate(TEXT_CAPACITY);
        if self.rects.is_empty() && self.texts.is_empty() {
            return;
        }
        if !self.rects.is_empty() {
            queue.write_buffer(&self.rect_buffer, 0, bytemuck::cast_slice(&self.rects));
        }
        if !self.texts.is_empty() {
            queue.write_buffer(&self.text_buffer, 0, bytemuck::cast_slice(&self.texts));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("native.terminal.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        if !self.rects.is_empty() {
            pass.set_pipeline(&self.rect_pipeline);
            pass.set_vertex_buffer(0, self.rect_buffer.slice(..));
            pass.draw(0..self.rects.len() as u32, 0..1);
        }
        if !self.texts.is_empty() {
            pass.set_pipeline(&self.text_pipeline);
            pass.set_bind_group(0, &self.text_bind_group, &[]);
            pass.set_vertex_buffer(0, self.text_buffer.slice(..));
            pass.draw(0..self.texts.len() as u32, 0..1);
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "one-shot pipeline builder")]
fn build_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    shader_source: &str,
    vertex_entry: &str,
    fragment_entry: &str,
    bind_layouts: &[wgpu::BindGroupLayout],
    stride: u64,
    attributes: &[wgpu::VertexAttribute],
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("native.terminal.shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> =
        bind_layouts.iter().map(Some).collect();
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("native.terminal.pipeline-layout"),
        bind_group_layouts: &layout_refs,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("native.terminal.pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(vertex_entry),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes,
            })],
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
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
