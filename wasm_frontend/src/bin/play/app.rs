//! Native desktop build of Vackrooms: a winit window on wgpu's Vulkan
//! backend, driving the same `Engine`, `LocalChunkSource`, and production
//! pipelines the browser uses. No browser, no WebGPU support required.
//!
//! ```sh
//! cargo run -p wasm_frontend --bin play --release
//! cargo run -p wasm_frontend --bin play --release -- --renderer=raymarch --seed=7
//! ```
//!
//! Flags: `--renderer=surface|splat|raymarch|cpu` (default surface),
//! `--preset=compat|full` (default compat: culling optimizations on, the
//! expensive effect passes — shadows, ambient occlusion, deferred — off,
//! because this build exists to test weak/varied hardware), `--quality=low|high`,
//! `--seed=N`. Controls: WASD + mouse (click to capture, Esc releases),
//! F flashlight, G flare, R drink, T eat.
//!
//! This file is the native twin of `drivers/webgpu/renderer.rs` +
//! `drivers/browser.rs`: those are wasm-only (web-sys types in their
//! signatures), so the window/present plumbing is mirrored here while every
//! pipeline, the engine, and generation stay shared.

use std::sync::Arc;
use std::time::Instant;

use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::spawn_point;
use wasm_frontend::adapters::cpu_splatter::CpuRenderSettings;
use wasm_frontend::adapters::local_chunk_source::LocalChunkSource;
use wasm_frontend::application::engine::{Engine, EngineConfig, InputFrame};
use wasm_frontend::application::player::MoveIntent;
use wasm_frontend::application::ports::{
    ChunkDraw, FrameParams, RenderArtifactNeeds, RendererPort, SurfaceChunk, SurfaceChunkKey,
};
use wasm_frontend::application::render_settings::RenderToggles;
use wasm_frontend::drivers::webgpu::camera_state;
use wasm_frontend::drivers::webgpu::config::{
    GpuQualityProfile, RendererKind, RendererProfile,
};
use wasm_frontend::drivers::webgpu::frame_resources::FrameResources;
use wasm_frontend::drivers::webgpu::gpu_types::{GpuFrameUniforms, collect_frame_lights};
use wasm_frontend::drivers::webgpu::pipelines::{
    CpuPresentPipeline, HeroShadowOptions, RaymarchPipeline, RaymarchRuntimeOptions, SplatPipeline,
    SupplyLabelPipeline, SurfacePipeline,
};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

/// Every start explores a fresh world; `--seed=N` reproduces one. The
/// std hasher is randomly keyed per process, which is all the entropy a
/// world seed needs without pulling in a rand dependency.
fn random_seed() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish() as u32
}

struct Options {
    renderer: RendererKind,
    quality: GpuQualityProfile,
    seed: u32,
    toggles: RenderToggles,
}

/// The compatibility preset every run starts from: all load-*reducing*
/// switches stay on, all load-*adding* effect passes stay off. `--preset=full`
/// restores the browser's default switchboard for parity testing.
fn compat_toggles() -> RenderToggles {
    RenderToggles {
        shadow_pass: false,
        ambient_occlusion: false,
        deferred_shading: false,
        flashlight_occlusion: false,
        ..RenderToggles::default()
    }
}

fn parse_options() -> Options {
    let mut options = Options {
        renderer: RendererKind::Surface,
        quality: GpuQualityProfile::default(),
        seed: random_seed(),
        toggles: compat_toggles(),
    };
    for argument in std::env::args().skip(1) {
        if let Some(value) = argument.strip_prefix("--renderer=") {
            options.renderer = value
                .parse()
                .unwrap_or_else(|_| panic!("unknown renderer {value:?}; use surface|splat|raymarch|cpu"));
        } else if let Some(value) = argument.strip_prefix("--preset=") {
            options.toggles = match value {
                "compat" => compat_toggles(),
                "full" => RenderToggles::default(),
                other => panic!("unknown preset {other:?}; use compat|full"),
            };
        } else if let Some(value) = argument.strip_prefix("--quality=") {
            options.quality = value
                .parse()
                .unwrap_or_else(|_| panic!("unknown quality {value:?}; use low|high"));
        } else if let Some(value) = argument.strip_prefix("--seed=") {
            options.seed = value
                .parse()
                .unwrap_or_else(|_| panic!("seed must be a number, got {value:?}"));
        } else {
            panic!(
                "unknown flag {argument:?}; supported: --renderer= --preset= --quality= --seed="
            );
        }
    }
    options
}

pub fn run() {
    let options = parse_options();
    let event_loop = EventLoop::new().expect("create winit event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        options,
        state: None,
    };
    event_loop.run_app(&mut app).expect("run event loop");
}

struct App {
    options: Options,
    state: Option<State>,
}

/// Everything alive once the window exists.
struct State {
    window: Arc<Window>,
    renderer: SharedNativeRenderer,
    engine: Engine,
    input: InputState,
    last_frame: Instant,
    hud_window_start: Instant,
    hud_frames: u32,
}

#[derive(Default)]
struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    flashlight: bool,
    drop_flare: bool,
    drink: bool,
    eat: bool,
    look_dx: f32,
    look_dy: f32,
    captured: bool,
}

impl InputState {
    fn take_frame(&mut self) -> InputFrame {
        let frame = InputFrame {
            intent: MoveIntent {
                forward: self.forward,
                backward: self.backward,
                left: self.left,
                right: self.right,
                turn_left: false,
                turn_right: false,
            },
            look_dx: self.look_dx,
            look_dy: self.look_dy,
            locked: self.captured,
            flashlight: self.flashlight,
            drop_flare: self.drop_flare,
            drink: self.drink,
            eat: self.eat,
        };
        self.look_dx = 0.0;
        self.look_dy = 0.0;
        self.drop_flare = false;
        self.drink = false;
        self.eat = false;
        frame
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Vackrooms")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
                )
                .expect("create window"),
        );
        let renderer = SharedNativeRenderer::new(NativeRenderer::new(
            Arc::clone(&window),
            RendererProfile::for_renderer(self.options.renderer, self.options.quality),
            self.options.toggles,
        ));

        // Same world assembly as the browser composition root, minus the
        // query string: low-spec generator profile, spawn on the main
        // corridor of region (0,0), facing east down its west leg.
        let generator = GeneratorConfig::low_spec();
        let spawn = spawn_point(self.options.seed);
        let engine_config = EngineConfig {
            chunk_size: generator.chunk_size,
            seed: self.options.seed,
            spawn: [spawn.x, 1.7, spawn.z],
            spawn_yaw: -std::f32::consts::FRAC_PI_2,
            ..EngineConfig::default()
        };
        let source = LocalChunkSource::new(SimpleNoiseProvider::new(), self.options.seed, generator);
        let engine = Engine::new(
            engine_config,
            Box::new(renderer.clone()),
            Box::new(source),
        );

        eprintln!(
            "renderer: {} | seed {} (reproduce with --seed={}) | click to capture the mouse; Esc releases",
            self.options.renderer.label(),
            self.options.seed,
            self.options.seed,
        );
        window.request_redraw();
        self.state = Some(State {
            window,
            renderer,
            engine,
            input: InputState::default(),
            last_frame: Instant::now(),
            hud_window_start: Instant::now(),
            hud_frames: 0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.input.captured {
                    let grabbed = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    state.window.set_cursor_visible(!grabbed);
                    state.input.captured = grabbed;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW | KeyCode::ArrowUp) => {
                        state.input.forward = pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyS | KeyCode::ArrowDown) => {
                        state.input.backward = pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyA | KeyCode::ArrowLeft) => {
                        state.input.left = pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyD | KeyCode::ArrowRight) => {
                        state.input.right = pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyF) if pressed => {
                        state.input.flashlight = !state.input.flashlight;
                    }
                    PhysicalKey::Code(KeyCode::KeyG) if pressed => {
                        state.input.drop_flare = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyR) if pressed => {
                        state.input.drink = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyT) if pressed => {
                        state.input.eat = true;
                    }
                    PhysicalKey::Code(KeyCode::Escape) if pressed => {
                        let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                        state.window.set_cursor_visible(true);
                        state.input.captured = false;
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32().min(0.1);
                state.last_frame = now;
                // Last frame's stats feed this frame's HUD: the engine draws
                // (and presents) inside tick, so the snapshot must be staged
                // before it runs. One frame of HUD latency is invisible.
                let stats = state.engine.stats();
                state.renderer.set_hud(HudSnapshot {
                    hydration: stats.hydration,
                    satiety: stats.satiety,
                    condition: stats.condition,
                    almond_bottles: stats.almond_bottles,
                    rations: stats.rations,
                    ready: stats.ready,
                    flashlight: state.input.flashlight,
                });
                let frame_input = state.input.take_frame();
                state.engine.tick(dt, &frame_input);

                // Minimal HUD in the title bar, once a second.
                state.hud_frames += 1;
                let elapsed = now - state.hud_window_start;
                if elapsed.as_secs_f32() >= 1.0 {
                    let fps = state.hud_frames as f32 / elapsed.as_secs_f32();
                    state.hud_frames = 0;
                    state.hud_window_start = now;
                    let stats = state.engine.stats();
                    state.window.set_title(&format!(
                        "Vackrooms — {:.0} fps | chunks {}/{} | hyd {:.0}% sat {:.0}% | {:.0}°C | {}水 {}食",
                        fps,
                        stats.fine_chunks,
                        stats.resident_chunks,
                        stats.hydration * 100.0,
                        stats.satiety * 100.0,
                        stats.ambient_c,
                        stats.almond_bottles,
                        stats.rations,
                    ));
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.input.captured {
                state.input.look_dx += dx as f32;
                state.input.look_dy += dy as f32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Native renderer: the desktop twin of `WebGpuRenderer`, presenting to a
// winit surface instead of a canvas. Pipelines and strategies are shared.
// ---------------------------------------------------------------------------

enum Strategy {
    Surface(SurfacePipeline),
    Splat(SplatPipeline),
    Raymarch(RaymarchPipeline),
    Cpu(CpuPresentPipeline),
}

/// Presenter-side HUD state, staged once per frame by the event loop.
#[derive(Default, Clone, Copy)]
struct HudSnapshot {
    hydration: f32,
    satiety: f32,
    condition: f32,
    almond_bottles: u32,
    rations: u32,
    ready: bool,
    flashlight: bool,
}

struct NativeRenderer {
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    frame_resources: FrameResources,
    strategy: Strategy,
    supply_labels: SupplyLabelPipeline,
    profile: RendererProfile,
    toggles: RenderToggles,
    depth_view: wgpu::TextureView,
    hud: HudOverlay,
    hud_snapshot: HudSnapshot,
}

impl NativeRenderer {
    fn new(window: Arc<Window>, profile: RendererProfile, toggles: RenderToggles) -> Self {
        let mut descriptor =
            wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone()));
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let surface = instance
            .create_surface(window.clone())
            .expect("create window surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .expect("no Vulkan adapter; install Mesa (Lavapipe works) or expose a Vulkan GPU");
        let info = adapter.get_info();
        eprintln!(
            "Vulkan adapter: {} ({}, {})",
            info.name, info.driver, info.driver_info
        );
        let limits = wgpu::Limits::default().or_worse_values_from(&adapter.limits());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vackrooms.native-play"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .expect("create Vulkan device");

        // Mirror the browser context: the shaders encode display color
        // themselves, so present through a non-sRGB view of the swapchain.
        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .expect("surface exposes a linear color format");
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let frame_resources = FrameResources::new(&device);
        let strategy = match profile {
            RendererProfile::Surface(settings) => {
                Strategy::Surface(SurfacePipeline::new_with_shadow_options(
                    &device,
                    format,
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
                    &device,
                    format,
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
                    &device,
                    format,
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
                &device,
                format,
                config.width,
                config.height,
            )),
        };
        let supply_labels = SupplyLabelPipeline::new(&device, format, frame_resources.layout());
        let depth_view = create_depth_view(&device, config.width, config.height);
        let hud = HudOverlay::new(&device, format);
        Self {
            _instance: instance,
            surface,
            device,
            queue,
            config,
            frame_resources,
            strategy,
            supply_labels,
            profile,
            toggles,
            depth_view,
            hud,
            hud_snapshot: HudSnapshot::default(),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.config.width && height == self.config.height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        if let Strategy::Cpu(pipeline) = &mut self.strategy {
            pipeline.resize(&self.device, width, height);
        }
        self.depth_view = create_depth_view(&self.device, width, height);
    }

    fn trace_budget(&self) -> u32 {
        match self.profile {
            RendererProfile::Raymarch(profile) => profile.maximum_trace_steps,
            _ => 1,
        }
    }

    fn render(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        let lights = collect_frame_lights(frame);
        let toggles = self.profile.apply_runtime_toggles(self.toggles);
        let common = self.profile.common();
        let uniforms = GpuFrameUniforms::from_frame(
            frame,
            self.config.width,
            self.config.height,
            camera_state::fov_tan(),
            common.maximum_draw_distance,
            self.trace_budget(),
            lights.len(),
            toggles,
        );
        self.frame_resources
            .write(&self.device, &self.queue, &uniforms, &lights);

        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = surface_texture.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vackrooms.native.frame"),
            });
        match &mut self.strategy {
            Strategy::Surface(pipeline) => pipeline.draw(
                &self.queue,
                &mut encoder,
                &view,
                &self.depth_view,
                &self.frame_resources,
                frame,
                toggles,
                lights.len().min(u32::MAX as usize) as u32,
            ),
            Strategy::Splat(pipeline) => pipeline.draw(
                &self.queue,
                &mut encoder,
                &view,
                &self.depth_view,
                &self.frame_resources,
                frame,
                toggles,
                lights.len().min(u32::MAX as usize) as u32,
            ),
            Strategy::Raymarch(pipeline) => pipeline.draw(
                &self.queue,
                &mut encoder,
                &view,
                &self.frame_resources,
                frame,
                chunks,
                toggles,
            ),
            Strategy::Cpu(pipeline) => {
                let mut settings = CpuRenderSettings::default();
                settings.fov_tan = camera_state::fov_tan();
                settings.toggles = toggles;
                pipeline.draw(&self.queue, &mut encoder, &view, frame, chunks, settings);
            }
        }
        if matches!(&self.strategy, Strategy::Surface(_) | Strategy::Splat(_)) {
            self.supply_labels.draw(
                &self.queue,
                &mut encoder,
                &view,
                &self.depth_view,
                &self.frame_resources,
                frame,
            );
        }
        self.hud.draw(
            &self.queue,
            &mut encoder,
            &view,
            &self.hud_snapshot,
            self.config.width,
            self.config.height,
        );
        self.queue.submit([encoder.finish()]);
        self.queue.present(surface_texture);
    }
}

// ---------------------------------------------------------------------------
// HUD overlay: the native stand-in for the browser's DOM field terminal.
// Flat alpha-blended quads — vitals bars, supply pips, flashlight tick,
// crosshair, and a dimming plate while the spawn chunk streams in.
// ---------------------------------------------------------------------------

const HUD_SHADER: &str = r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn hud_vertex(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn hud_fragment(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HudVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

/// Enough for the loading plate, bars, pips, and crosshair, with headroom.
const HUD_VERTEX_CAPACITY: usize = 1024;

struct HudOverlay {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
}

impl HudOverlay {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native.hud.shader"),
            source: wgpu::ShaderSource::Wgsl(HUD_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("native.hud.pipeline-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("native.hud.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("hud_vertex"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<HudVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
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
                entry_point: Some("hud_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native.hud.vertices"),
            size: (HUD_VERTEX_CAPACITY * size_of::<HudVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, vertices }
    }

    fn draw(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        snapshot: &HudSnapshot,
        width: u32,
        height: u32,
    ) {
        let vertices = build_hud_vertices(snapshot, width, height);
        if vertices.is_empty() {
            return;
        }
        queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(&vertices));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("native.hud.pass"),
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
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
}

/// Field-terminal palette on straight alpha. The scene shaders already
/// encode display color, so these are plain display-space values.
const HUD_BACKPLATE: [f32; 4] = [0.02, 0.04, 0.02, 0.62];
const HUD_HYDRATION: [f32; 4] = [0.42, 0.86, 0.94, 0.9];
const HUD_SATIETY: [f32; 4] = [0.89, 0.75, 0.35, 0.9];
const HUD_CONDITION: [f32; 4] = [0.91, 0.42, 0.34, 0.9];
const HUD_PIP_WATER: [f32; 4] = [0.62, 0.86, 0.98, 0.95];
const HUD_PIP_RATION: [f32; 4] = [0.93, 0.80, 0.45, 0.95];
const HUD_FLASHLIGHT: [f32; 4] = [0.98, 0.95, 0.75, 0.95];
const HUD_CROSSHAIR: [f32; 4] = [0.9, 0.9, 0.85, 0.55];
const HUD_LOADING_PLATE: [f32; 4] = [0.0, 0.0, 0.0, 0.55];

fn build_hud_vertices(snapshot: &HudSnapshot, width: u32, height: u32) -> Vec<HudVertex> {
    let mut vertices = Vec::with_capacity(HUD_VERTEX_CAPACITY);
    let width = width.max(1) as f32;
    let height = height.max(1) as f32;
    // Pixel-space rect -> NDC quad (two triangles).
    let mut rect = |x: f32, y: f32, w: f32, h: f32, color: [f32; 4]| {
        let x0 = x / width * 2.0 - 1.0;
        let x1 = (x + w) / width * 2.0 - 1.0;
        let y0 = 1.0 - y / height * 2.0;
        let y1 = 1.0 - (y + h) / height * 2.0;
        for pos in [
            [x0, y0],
            [x0, y1],
            [x1, y1],
            [x0, y0],
            [x1, y1],
            [x1, y0],
        ] {
            vertices.push(HudVertex { pos, color });
        }
    };

    if !snapshot.ready {
        rect(0.0, 0.0, width, height, HUD_LOADING_PLATE);
        return vertices;
    }

    // Vitals: three bars, bottom-left, brightest problem nearest the eye.
    let bar_w = 180.0;
    let bar_h = 10.0;
    let margin = 18.0;
    let gap = 6.0;
    let bars = [
        (snapshot.condition, HUD_CONDITION),
        (snapshot.satiety, HUD_SATIETY),
        (snapshot.hydration, HUD_HYDRATION),
    ];
    for (index, (value, color)) in bars.iter().enumerate() {
        let y = height - margin - bar_h - index as f32 * (bar_h + gap);
        rect(margin - 2.0, y - 2.0, bar_w + 4.0, bar_h + 4.0, HUD_BACKPLATE);
        rect(margin, y, bar_w * value.clamp(0.0, 1.0), bar_h, *color);
    }

    // Carried supplies as discrete pips above the bars, water then rations.
    let pip = 8.0;
    let pip_gap = 4.0;
    let pip_row_y = height - margin - 3.0 * (bar_h + gap) - pip - 4.0;
    for index in 0..snapshot.almond_bottles.min(12) {
        rect(
            margin + index as f32 * (pip + pip_gap),
            pip_row_y,
            pip,
            pip,
            HUD_PIP_WATER,
        );
    }
    for index in 0..snapshot.rations.min(12) {
        rect(
            margin + index as f32 * (pip + pip_gap),
            pip_row_y - pip - pip_gap,
            pip,
            pip,
            HUD_PIP_RATION,
        );
    }

    // Flashlight tick, bottom-right.
    if snapshot.flashlight {
        rect(width - margin - 26.0, height - margin - 10.0, 26.0, 10.0, HUD_FLASHLIGHT);
    }

    // Crosshair dot.
    rect(width * 0.5 - 2.0, height * 0.5 - 2.0, 4.0, 4.0, HUD_CROSSHAIR);

    vertices
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("vackrooms.native.depth"),
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
        })
        .create_view(&Default::default())
}

/// The engine owns its renderer as a boxed port while the event loop still
/// needs resize access, so the native renderer is shared the same way the
/// browser shares its canvas renderer.
#[derive(Clone)]
struct SharedNativeRenderer(std::rc::Rc<std::cell::RefCell<NativeRenderer>>);

impl SharedNativeRenderer {
    fn new(renderer: NativeRenderer) -> Self {
        Self(std::rc::Rc::new(std::cell::RefCell::new(renderer)))
    }

    fn resize(&self, width: u32, height: u32) {
        self.0.borrow_mut().resize(width, height);
    }

    fn set_hud(&self, snapshot: HudSnapshot) {
        self.0.borrow_mut().hud_snapshot = snapshot;
    }
}

impl RendererPort for SharedNativeRenderer {
    fn artifact_needs(&self) -> RenderArtifactNeeds {
        self.0.borrow().profile.artifact_needs()
    }

    fn uses_surface_meshes(&self) -> bool {
        self.artifact_needs().needs_surface_extraction()
    }

    fn upload_surfaces(&mut self, chunks: &[SurfaceChunk<'_>]) {
        let renderer = &mut *self.0.borrow_mut();
        match &mut renderer.strategy {
            Strategy::Surface(pipeline) => pipeline.upload(&renderer.device, chunks),
            Strategy::Splat(pipeline) => pipeline.upload(&renderer.device, chunks),
            _ => {}
        }
    }

    fn remove_surfaces(&mut self, keys: &[SurfaceChunkKey]) {
        match &mut self.0.borrow_mut().strategy {
            Strategy::Surface(pipeline) => pipeline.remove(keys),
            Strategy::Splat(pipeline) => pipeline.remove(keys),
            _ => {}
        }
    }

    fn clear_surfaces(&mut self) {
        match &mut self.0.borrow_mut().strategy {
            Strategy::Surface(pipeline) => pipeline.clear(),
            Strategy::Splat(pipeline) => pipeline.clear(),
            _ => {}
        }
    }

    fn upload_atlas(&mut self, texels: &[u32]) {
        let renderer = &mut *self.0.borrow_mut();
        match &mut renderer.strategy {
            Strategy::Raymarch(pipeline) => pipeline.upload_atlas(&renderer.device, texels),
            Strategy::Cpu(pipeline) => pipeline.upload_atlas(texels),
            _ => {}
        }
    }

    fn upload_atlas_rows(&mut self, first_row: u32, texels: &[u32]) -> bool {
        let renderer = &mut *self.0.borrow_mut();
        match &mut renderer.strategy {
            Strategy::Raymarch(pipeline) => {
                pipeline.upload_atlas_rows(&renderer.queue, first_row, texels)
            }
            Strategy::Cpu(pipeline) => pipeline.upload_atlas_rows(first_row, texels),
            _ => false,
        }
    }

    fn upload_label_atlas(&mut self, rgba: &[u8], width: u32, height: u32) {
        let renderer = &mut *self.0.borrow_mut();
        renderer
            .supply_labels
            .upload_atlas(&renderer.device, &renderer.queue, rgba, width, height);
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        self.0.borrow_mut().render(frame, chunks);
    }
}
