//! WebGPU renderer facade implementing the application's existing port while
//! the deeper scene-lifecycle port is migrated incrementally.

use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;

use crate::application::ports::{
    ChunkDraw, FrameParams, RenderArtifactNeeds, RendererPort, SurfaceChunk, SurfaceChunkKey,
};
use crate::drivers::webgpu::browser_context::BrowserGpuContext;
use crate::drivers::webgpu::config::{RendererKind, RendererProfile};
use crate::drivers::webgpu::frame_resources::FrameResources;
use crate::drivers::webgpu::gpu_types::{GpuFrameUniforms, collect_frame_lights};
use crate::drivers::webgpu::pipelines::{
    CpuPresentPipeline, HeroShadowOptions, RaymarchPipeline, RaymarchRuntimeOptions, SplatPipeline,
    SupplyLabelPipeline, SurfacePipeline,
};

enum Strategy {
    Surface(SurfacePipeline),
    Splat(SplatPipeline),
    Raymarch(RaymarchPipeline),
    Cpu(CpuPresentPipeline),
}

pub struct WebGpuRenderer {
    context: BrowserGpuContext,
    frame_resources: FrameResources,
    strategy: Strategy,
    supply_labels: SupplyLabelPipeline,
    profile: RendererProfile,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

/// Picks the strategy this adapter can certainly run well when the player
/// has not chosen one. A fallback adapter (`isFallbackAdapter`, surfaced by
/// wgpu as `DeviceType::Cpu`) or a software rasterizer means every
/// GPU-heavy path would grind; the CPU splatter is built for exactly that
/// case and asks the device for nothing but one texture blit per frame.
/// Real GPUs take the default surface strategy.
fn most_compatible_kind(info: &wgpu::AdapterInfo) -> RendererKind {
    let name = info.name.to_ascii_lowercase();
    let software = matches!(info.device_type, wgpu::DeviceType::Cpu)
        || name.contains("swiftshader")
        || name.contains("llvmpipe")
        || name.contains("lavapipe");
    if software {
        RendererKind::Cpu
    } else {
        RendererKind::Surface
    }
}

impl WebGpuRenderer {
    /// Composition-root entry: honors an explicit `?renderer=` choice, and
    /// otherwise selects the most compatible strategy for the adapter the
    /// browser actually granted.
    pub async fn new_auto(
        canvas: &HtmlCanvasElement,
        requested: Option<RendererKind>,
        quality: crate::drivers::webgpu::config::GpuQualityProfile,
    ) -> Result<Self, JsValue> {
        let context = BrowserGpuContext::new(canvas).await?;
        let kind = match requested {
            Some(kind) => kind,
            None => {
                let kind = most_compatible_kind(&context.adapter_info);
                if kind != RendererKind::Surface {
                    web_sys::console::info_1(
                        &format!(
                            "auto renderer: {} (adapter \"{}\" is a software/fallback device)",
                            kind.label(),
                            context.adapter_info.name
                        )
                        .into(),
                    );
                }
                kind
            }
        };
        Self::from_context(context, kind.profile(quality))
    }

    pub async fn new(
        canvas: &HtmlCanvasElement,
        profile: RendererProfile,
    ) -> Result<Self, JsValue> {
        let context = BrowserGpuContext::new(canvas).await?;
        Self::from_context(context, profile)
    }

    fn from_context(context: BrowserGpuContext, profile: RendererProfile) -> Result<Self, JsValue> {
        let frame_resources = FrameResources::new(&context.device);
        let strategy = match profile {
            RendererProfile::Surface(settings) => {
                Strategy::Surface(SurfacePipeline::new_with_shadow_options(
                    &context.device,
                    context.format(),
                    frame_resources.layout(),
                    settings.common.maximum_draw_distance,
                    HeroShadowOptions::new(
                        settings.shadow_map,
                        settings.optimizations.hero_light_shadow_map,
                    ),
                ))
            }
            RendererProfile::Splat(settings) => {
                Strategy::Splat(SplatPipeline::new_with_shadow_options(
                    &context.device,
                    context.format(),
                    frame_resources.layout(),
                    settings.common.maximum_draw_distance,
                    settings.face_budget,
                    HeroShadowOptions::new(
                        settings.shadow_map,
                        settings.optimizations.hero_light_shadow_map,
                    ),
                ))
            }
            RendererProfile::Raymarch(settings) => {
                let mut pipeline = RaymarchPipeline::new(
                    &context.device,
                    context.format(),
                    frame_resources.layout(),
                    settings.maximum_resident_chunks as usize,
                );
                pipeline.configure(RaymarchRuntimeOptions::new(
                    settings.common.maximum_draw_distance,
                    settings.optimizations.direct_light_visibility,
                    settings.area_light_visibility_samples,
                ));
                Strategy::Raymarch(pipeline)
            }
            RendererProfile::Cpu(_) => Strategy::Cpu(CpuPresentPipeline::new(
                &context.device,
                context.format(),
                context.width(),
                context.height(),
            )),
        };
        let supply_labels =
            SupplyLabelPipeline::new(&context.device, context.format(), frame_resources.layout());
        let (depth_texture, depth_view) =
            create_depth_target(&context.device, context.width(), context.height());
        web_sys::console::info_1(
            &format!(
                "WebGPU adapter: {} ({:?})",
                context.adapter_info.name, context.adapter_info.backend
            )
            .into(),
        );
        Ok(Self {
            context,
            frame_resources,
            strategy,
            supply_labels,
            profile,
            depth_texture,
            depth_view,
        })
    }

    pub fn kind(&self) -> RendererKind {
        self.profile.renderer()
    }

    pub fn label(&self) -> &'static str {
        self.kind().label()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.context.resize(width, height);
        if let Strategy::Cpu(pipeline) = &mut self.strategy {
            pipeline.resize(&self.context.device, width, height);
        }
        (self.depth_texture, self.depth_view) =
            create_depth_target(&self.context.device, width.max(1), height.max(1));
    }

    pub fn resolution_factor(&self) -> f64 {
        match self.profile.initial_gpu_render_scale() {
            Some(scale) => f64::from(scale),
            None => crate::get_cpu_settings().canvas_resolution_factor(),
        }
    }

    fn trace_budget(&self) -> u32 {
        match self.profile {
            RendererProfile::Raymarch(profile) => profile.maximum_trace_steps,
            _ => 1,
        }
    }

    fn render(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        if self.context.is_device_lost() {
            return;
        }
        let lights = collect_frame_lights(frame);
        let toggles = self
            .profile
            .apply_runtime_toggles(crate::get_render_toggles());
        let common = self.profile.common();
        let uniforms = GpuFrameUniforms::from_frame(
            frame,
            self.context.width(),
            self.context.height(),
            super::camera_state::fov_tan(),
            common.maximum_draw_distance,
            self.trace_budget(),
            lights.len(),
            toggles,
        );
        self.frame_resources.write(
            &self.context.device,
            &self.context.queue,
            &uniforms,
            &lights,
        );

        let (surface_texture, should_reconfigure) = match self.context.current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.context.reconfigure();
                match self.context.current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
                    wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
                    _ => return,
                }
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                web_sys::console::error_1(&"WebGPU surface validation error".into());
                return;
            }
        };
        let view = surface_texture.texture.create_view(&Default::default());
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vackrooms.webgpu.frame"),
                });
        match &mut self.strategy {
            Strategy::Surface(pipeline) => pipeline.draw(
                &self.context.queue,
                &mut encoder,
                &view,
                &self.depth_view,
                &self.frame_resources,
                frame,
                toggles,
                lights.len().min(u32::MAX as usize) as u32,
            ),
            Strategy::Splat(pipeline) => pipeline.draw(
                &self.context.queue,
                &mut encoder,
                &view,
                &self.depth_view,
                &self.frame_resources,
                frame,
                toggles,
                lights.len().min(u32::MAX as usize) as u32,
            ),
            Strategy::Raymarch(pipeline) => pipeline.draw(
                &self.context.queue,
                &mut encoder,
                &view,
                &self.frame_resources,
                frame,
                chunks,
                toggles,
            ),
            Strategy::Cpu(pipeline) => {
                let mut settings = crate::get_cpu_settings();
                settings.fov_tan = super::camera_state::fov_tan();
                settings.toggles = toggles;
                pipeline.draw(
                    &self.context.queue,
                    &mut encoder,
                    &view,
                    frame,
                    chunks,
                    settings,
                );
            }
        }
        if matches!(&self.strategy, Strategy::Surface(_) | Strategy::Splat(_)) {
            self.supply_labels.draw(
                &self.context.queue,
                &mut encoder,
                &view,
                &self.depth_view,
                &self.frame_resources,
                frame,
            );
        }
        self.context.queue.submit([encoder.finish()]);
        self.context.queue.present(surface_texture);
        if should_reconfigure {
            self.context.reconfigure();
        }
    }
}

impl RendererPort for WebGpuRenderer {
    fn artifact_needs(&self) -> RenderArtifactNeeds {
        self.profile.artifact_needs()
    }

    fn uses_surface_meshes(&self) -> bool {
        self.profile.artifact_needs().needs_surface_extraction()
    }

    fn upload_surfaces(&mut self, chunks: &[SurfaceChunk<'_>]) {
        match &mut self.strategy {
            Strategy::Surface(pipeline) => pipeline.upload(&self.context.device, chunks),
            Strategy::Splat(pipeline) => pipeline.upload(&self.context.device, chunks),
            _ => {}
        }
    }

    fn remove_surfaces(&mut self, keys: &[SurfaceChunkKey]) {
        match &mut self.strategy {
            Strategy::Surface(pipeline) => pipeline.remove(keys),
            Strategy::Splat(pipeline) => pipeline.remove(keys),
            _ => {}
        }
    }

    fn clear_surfaces(&mut self) {
        match &mut self.strategy {
            Strategy::Surface(pipeline) => pipeline.clear(),
            Strategy::Splat(pipeline) => pipeline.clear(),
            _ => {}
        }
    }

    fn cpu_telemetry_string(&self) -> Option<String> {
        match &self.strategy {
            Strategy::Cpu(pipeline) => Some(pipeline.telemetry_string()),
            Strategy::Splat(pipeline) => {
                let stats = pipeline.stats();
                Some(format!(
                    "WebGPU splat F:{}/-{} C:{} D:{}",
                    stats.faces_drawn, stats.faces_dropped, stats.cells_culled, stats.draw_calls
                ))
            }
            _ => None,
        }
    }

    fn upload_atlas(&mut self, texels: &[u32]) {
        match &mut self.strategy {
            Strategy::Raymarch(pipeline) => pipeline.upload_atlas(&self.context.device, texels),
            Strategy::Cpu(pipeline) => pipeline.upload_atlas(texels),
            _ => {}
        }
    }

    fn upload_atlas_rows(&mut self, first_row: u32, texels: &[u32]) -> bool {
        match &mut self.strategy {
            Strategy::Raymarch(pipeline) => {
                pipeline.upload_atlas_rows(&self.context.queue, first_row, texels)
            }
            Strategy::Cpu(pipeline) => pipeline.upload_atlas_rows(first_row, texels),
            _ => false,
        }
    }

    fn upload_label_atlas(&mut self, rgba: &[u8], width: u32, height: u32) {
        self.supply_labels.upload_atlas(
            &self.context.device,
            &self.context.queue,
            rgba,
            width,
            height,
        );
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        self.render(frame, chunks);
    }
}

fn create_depth_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("webgpu.main-depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}
