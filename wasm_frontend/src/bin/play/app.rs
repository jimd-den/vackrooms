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
use wasm_frontend::adapters::section_locator::SectionLocator;
use wasm_frontend::application::body::PulseSignal;
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
    locator: SectionLocator,
    toggles: RenderToggles,
    seed: u32,
    last_frame: Instant,
    hud_window_start: Instant,
    hud_frames: u32,
    fps: f32,
    section: String,
    show_config: bool,
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
            locator: SectionLocator::new(self.options.seed, generator),
            toggles: self.options.toggles,
            seed: self.options.seed,
            last_frame: Instant::now(),
            hud_window_start: Instant::now(),
            hud_frames: 0,
            fps: 0.0,
            section: String::new(),
            show_config: false,
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
                    PhysicalKey::Code(KeyCode::F1) if pressed => {
                        state.show_config = !state.show_config;
                    }
                    // Live effect-pass switches (the terminal's F1 panel
                    // documents them). These are per-frame parameters, so
                    // flipping them needs no pipeline rebuild.
                    PhysicalKey::Code(KeyCode::F5) if pressed => {
                        state.toggles.shadow_pass = !state.toggles.shadow_pass;
                        state.renderer.set_toggles(state.toggles);
                    }
                    PhysicalKey::Code(KeyCode::F6) if pressed => {
                        state.toggles.ambient_occlusion = !state.toggles.ambient_occlusion;
                        state.renderer.set_toggles(state.toggles);
                    }
                    PhysicalKey::Code(KeyCode::F7) if pressed => {
                        state.toggles.deferred_shading = !state.toggles.deferred_shading;
                        state.renderer.set_toggles(state.toggles);
                    }
                    PhysicalKey::Code(KeyCode::F8) if pressed => {
                        state.toggles.flashlight_occlusion = !state.toggles.flashlight_occlusion;
                        state.renderer.set_toggles(state.toggles);
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

                // Slow channels (fps, section locator) refresh once a second.
                state.hud_frames += 1;
                let elapsed = now - state.hud_window_start;
                if elapsed.as_secs_f32() >= 1.0 {
                    state.fps = state.hud_frames as f32 / elapsed.as_secs_f32();
                    state.hud_frames = 0;
                    state.hud_window_start = now;
                    let (level, pos) = {
                        let player = state.engine.player();
                        (state.engine.level(), player.position)
                    };
                    state.section = state.locator.describe(level, pos[0], pos[2]);
                }

                // Last frame's stats feed this frame's terminal: the engine
                // draws (and presents) inside tick, so the snapshot must be
                // staged before it runs. One frame of latency is invisible.
                let stats = state.engine.stats();
                state.renderer.set_hud(HudSnapshot {
                    hydration: stats.hydration,
                    satiety: stats.satiety,
                    condition: stats.condition,
                    almond_bottles: stats.almond_bottles,
                    rations: stats.rations,
                    ready: stats.ready,
                    flashlight: state.input.flashlight,
                    fps: state.fps,
                    fine_chunks: stats.fine_chunks,
                    resident_chunks: stats.resident_chunks,
                    atlas_nodes: stats.atlas_nodes,
                    bpm: stats.body.bpm,
                    signal: stats.body.signal,
                    section: state.section.clone(),
                    distance_m: stats.distance_m,
                    ambient_c: stats.ambient_c,
                    deaths: stats.deaths,
                    seed: state.seed,
                    show_config: state.show_config,
                    toggles: state.toggles,
                });
                let frame_input = state.input.take_frame();
                state.engine.tick(dt, &frame_input);
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

/// Presenter-side terminal state, staged once per frame by the event loop.
#[derive(Clone)]
struct HudSnapshot {
    hydration: f32,
    satiety: f32,
    condition: f32,
    almond_bottles: u32,
    rations: u32,
    ready: bool,
    flashlight: bool,
    fps: f32,
    fine_chunks: usize,
    resident_chunks: usize,
    atlas_nodes: usize,
    bpm: f32,
    signal: PulseSignal,
    section: String,
    distance_m: f32,
    ambient_c: f32,
    deaths: u32,
    seed: u32,
    show_config: bool,
    toggles: RenderToggles,
}

impl Default for HudSnapshot {
    fn default() -> Self {
        Self {
            hydration: 0.0,
            satiety: 0.0,
            condition: 0.0,
            almond_bottles: 0,
            rations: 0,
            ready: false,
            flashlight: false,
            fps: 0.0,
            fine_chunks: 0,
            resident_chunks: 0,
            atlas_nodes: 0,
            bpm: 0.0,
            signal: PulseSignal::Quiet,
            section: String::new(),
            distance_m: 0.0,
            ambient_c: 0.0,
            deaths: 0,
            seed: 0,
            show_config: false,
            toggles: RenderToggles::default(),
        }
    }
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
    terminal: crate::terminal::FieldTerminal,
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
        let terminal = crate::terminal::FieldTerminal::new(&device, &queue, format);
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
            terminal,
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
        let renderer_label = self.profile.renderer().label();
        draw_field_terminal(
            &mut self.terminal,
            &self.hud_snapshot,
            renderer_label,
            self.config.width,
            self.config.height,
        );
        self.terminal.draw(&self.queue, &mut encoder, &view);
        self.queue.submit([encoder.finish()]);
        self.queue.present(surface_texture);
    }
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

    /// Live effect-pass switches from the F5–F8 keys; per-frame parameters,
    /// no pipeline rebuild involved.
    fn set_toggles(&self, toggles: RenderToggles) {
        self.0.borrow_mut().toggles = toggles;
    }
}

// ---------------------------------------------------------------------------
// Field-terminal layout: the same channels as static/index.html, composed
// from the shared palette in `terminal`. All coordinates are window pixels.
// ---------------------------------------------------------------------------

fn draw_field_terminal(
    terminal: &mut crate::terminal::FieldTerminal,
    snapshot: &HudSnapshot,
    renderer_label: &str,
    width: u32,
    height: u32,
) {
    use crate::terminal::{
        CRITICAL, HAZARD, INK, LOADING_PLATE, PANEL, QUIET, ROUTE, SYSTEM,
    };
    terminal.begin(width, height);
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let px = 16.0;
    let margin = 18.0;

    if !snapshot.ready {
        terminal.rect(0.0, 0.0, w, h, LOADING_PLATE);
        let line = "GENERATING WORLD IN WEBASSEMBLY'S ABSENCE...";
        let lw = terminal.measure(line, 24.0);
        terminal.text(line, (w - lw) * 0.5, h * 0.5, 24.0, INK);
        let seed_line = format!("SEED {}", snapshot.seed);
        let sw = terminal.measure(&seed_line, px);
        terminal.text(&seed_line, (w - sw) * 0.5, h * 0.5 + 28.0, px, QUIET);
        return;
    }

    // Top-left: renderer / frame channels, key in quiet, value in ink.
    let mut y = margin + px;
    let pen = terminal.text("renderer ", margin, y, px, QUIET);
    let pen = terminal.text(renderer_label, pen, y, px, INK);
    let pen = terminal.text("   fps ", pen, y, px, QUIET);
    let pen = terminal.text(&format!("{:.0}", snapshot.fps), pen, y, px, INK);
    let pen = terminal.text("   frame ", pen, y, px, QUIET);
    terminal.text("Vulkan", pen, y, px, INK);
    y += terminal.line_height(px);
    let pen = terminal.text("chunks ", margin, y, px, QUIET);
    let pen = terminal.text(
        &format!("{}/{}", snapshot.fine_chunks, snapshot.resident_chunks),
        pen,
        y,
        px,
        INK,
    );
    let pen = terminal.text("   nodes ", pen, y, px, QUIET);
    let pen = terminal.text(&format!("{}", snapshot.atlas_nodes), pen, y, px, INK);
    let pen = terminal.text("   dist ", pen, y, px, QUIET);
    let pen = terminal.text(&format!("{:.0} m", snapshot.distance_m), pen, y, px, INK);
    let pen = terminal.text("   amb ", pen, y, px, QUIET);
    terminal.text(&format!("{:.0} C", snapshot.ambient_c), pen, y, px, INK);

    // Top-right: section locator, right-aligned (the web terminal's
    // top-right channel).
    if !snapshot.section.is_empty() {
        let sw = terminal.measure(&snapshot.section, px);
        terminal.text(&snapshot.section, w - margin - sw, margin + px, px, QUIET);
    }

    // Bottom-left: body trace, vitals bars, supply pips.
    let bar_w = 220.0;
    let bar_h = 12.0;
    let gap = 7.0;
    let bars_top = h - margin - 3.0 * (bar_h + gap) + gap;
    let signal_color = match snapshot.signal {
        PulseSignal::Quiet => QUIET,
        PulseSignal::Active => INK,
        PulseSignal::Strained => ROUTE,
        PulseSignal::Critical => HAZARD,
    };
    if snapshot.signal != PulseSignal::Quiet {
        terminal.text(
            &format!("{:.0} bpm", snapshot.bpm),
            margin,
            bars_top - 14.0,
            px,
            signal_color,
        );
    }
    let bars = [
        (snapshot.hydration, SYSTEM, "H2O"),
        (snapshot.satiety, ROUTE, "RAT"),
        (snapshot.condition, HAZARD, "BOD"),
    ];
    for (index, (value, color, tag)) in bars.iter().enumerate() {
        let by = bars_top + index as f32 * (bar_h + gap);
        terminal.text(tag, margin, by + bar_h - 1.0, 16.0, QUIET);
        let bx = margin + 46.0;
        terminal.rect(bx - 2.0, by - 2.0, bar_w + 4.0, bar_h + 4.0, PANEL);
        terminal.rect(bx, by, bar_w * value.clamp(0.0, 1.0), bar_h, *color);
    }
    // Discrete supply ticks above the bars, water then rations, like
    // #supply-rack.
    let pip = 9.0;
    let pip_gap = 5.0;
    let pip_y = bars_top - 34.0;
    for index in 0..snapshot.almond_bottles.min(12) {
        terminal.rect(
            margin + 46.0 + index as f32 * (pip + pip_gap),
            pip_y,
            pip,
            pip,
            SYSTEM,
        );
    }
    for index in 0..snapshot.rations.min(12) {
        terminal.rect(
            margin + 46.0 + index as f32 * (pip + pip_gap),
            pip_y - pip - pip_gap,
            pip,
            pip,
            ROUTE,
        );
    }

    // Bottom-right: torch state and the config-panel hint.
    let mut right_y = h - margin;
    let hint = "F1 CONFIG";
    let hw = terminal.measure(hint, px);
    terminal.text(hint, w - margin - hw, right_y, px, QUIET);
    right_y -= terminal.line_height(px);
    if snapshot.flashlight {
        let torch = "TORCH ON";
        let tw = terminal.measure(torch, px);
        terminal.text(torch, w - margin - tw, right_y, px, ROUTE);
    }
    if snapshot.deaths > 0 {
        let text = format!("SUCCUMBED x{}", snapshot.deaths);
        let tw = terminal.measure(&text, px);
        terminal.text(&text, w - margin - tw, right_y - terminal.line_height(px), px, HAZARD);
    }

    // Center mark: single restrained fixation point (see #reticle).
    terminal.rect(w * 0.5 - 2.0, h * 0.5 - 2.0, 4.0, 4.0, [0.9, 0.9, 0.85, 0.55]);

    // F1: configuration panel — the native stand-in for the settings menu,
    // documenting the flags and live switches.
    if snapshot.show_config {
        let pw = 560.0_f32.min(w - 2.0 * margin);
        let ph = 330.0_f32.min(h - 2.0 * margin);
        let px0 = (w - pw) * 0.5;
        let py0 = (h - ph) * 0.5;
        terminal.rect(px0, py0, pw, ph, [0.027, 0.039, 0.031, 0.92]);
        let mut ly = py0 + 34.0;
        let lx = px0 + 24.0;
        terminal.text("CONFIGURATION - FIELD TERMINAL", lx, ly, px, CRITICAL);
        ly += terminal.line_height(px) * 1.5;
        let on_off = |on: bool| if on { "ON" } else { "OFF" };
        let rows: Vec<(String, [f32; 4])> = vec![
            (format!("RENDERER   {renderer_label}   (--renderer=surface|splat|raymarch|cpu)"), INK),
            (format!("SEED       {}   (--seed=N reproduces this world)", snapshot.seed), INK),
            (format!("F5 SHADOWS          {}", on_off(snapshot.toggles.shadow_pass)), SYSTEM),
            (format!("F6 AMBIENT OCCL.    {}", on_off(snapshot.toggles.ambient_occlusion)), SYSTEM),
            (format!("F7 DEFERRED SHADING {}", on_off(snapshot.toggles.deferred_shading)), SYSTEM),
            (format!("F8 TORCH OCCLUSION  {}", on_off(snapshot.toggles.flashlight_occlusion)), SYSTEM),
            ("WASD MOVE   MOUSE LOOK   F TORCH   G FLARE".to_owned(), QUIET),
            ("R DRINK     T EAT        ESC RELEASE MOUSE".to_owned(), QUIET),
        ];
        for (row, color) in rows {
            terminal.text(&row, lx, ly, px, color);
            ly += terminal.line_height(px) * 1.15;
        }
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
