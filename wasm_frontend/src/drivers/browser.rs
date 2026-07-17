//! Browser runtime driver: DOM/event plumbing, the requestAnimationFrame
//! loop, pointer lock and the HUD.
//!
//! This module is the composition root's workhorse: it instantiates the
//! concrete adapters/drivers, hands them to `application::engine::Engine`,
//! and from then on only shuttles plain data across the port boundaries.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{
    Document, HtmlCanvasElement, HtmlElement, KeyboardEvent, MouseEvent, TouchEvent, Window,
};

use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;

use crate::adapters::input::InputCollector;
use crate::adapters::local_chunk_source::LocalChunkSource;
use crate::adapters::query_config::{generator_setup_from_query, query_flag, query_param};
use crate::adapters::section_locator::SectionLocator;
use crate::application::engine::{Engine, EngineConfig};
use crate::application::generation_worker_policy::parse_generation_worker_preference;
use crate::application::ports::{
    ChunkDraw, FrameParams, RendererPort, SurfaceChunk, SurfaceChunkKey,
};
use crate::drivers::console_telemetry::CONSOLE_TELEMETRY;
use crate::drivers::cpu_canvas::CpuCanvasRenderer;
use crate::drivers::splat_webgl::{SplatProfile, SplatRenderer};
use crate::drivers::surface_webgl::SurfaceRenderer;
use crate::drivers::webgl::WebGl2Renderer;

/// World seed shared with the native server so both render the same world.
const WORLD_SEED: u32 = 42;
/// Radius of the virtual joystick in CSS pixels (full deflection).
const JOYSTICK_RADIUS: f64 = 60.0;
/// Touch-look sensitivity relative to raw mouse pixels.
const TOUCH_LOOK_SCALE: f32 = 2.2;
/// HUD refresh cadence in frames.
const HUD_INTERVAL: u32 = 30;

/// The default renderer is indexed greedy surfaces. The old SVO raymarcher
/// remains available only as `?renderer=raymarch` for visual/reference
/// debugging; `?renderer=cpu` chooses the software fallback.
enum DriverRenderer {
    Surface(SurfaceRenderer),
    Splat(SplatRenderer),
    Raymarch(WebGl2Renderer),
    Cpu(CpuCanvasRenderer),
}

impl DriverRenderer {
    fn resize(&mut self, width: u32, height: u32) {
        match self {
            DriverRenderer::Surface(r) => r.resize(width, height),
            DriverRenderer::Splat(r) => r.resize(width, height),
            DriverRenderer::Raymarch(r) => r.resize(width, height),
            DriverRenderer::Cpu(r) => r.resize(width, height),
        }
    }

    /// Internal-resolution multiplier (further scaled by the adaptive
    /// governor). The CPU quality model owns its bounded fraction so the
    /// canvas and software framebuffer always have identical dimensions;
    /// CSS then presents that complete image over the viewport.
    fn resolution_factor(&self) -> f64 {
        match self {
            DriverRenderer::Surface(_) | DriverRenderer::Splat(_) | DriverRenderer::Raymarch(_) => {
                1.0
            }
            DriverRenderer::Cpu(_) => crate::get_cpu_settings().canvas_resolution_factor(),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            DriverRenderer::Surface(_) => "GPU surfaces",
            DriverRenderer::Splat(_) => "GPU face splats",
            DriverRenderer::Raymarch(_) => "GPU raymarch (debug)",
            DriverRenderer::Cpu(_) => "CPU splat",
        }
    }
}

/// Complete backing-store scale after the global governor/override and the
/// selected renderer's own bounded workload factor are composed.
fn effective_resolution_scale(renderer: &DriverRenderer, governor_scale: f32) -> f64 {
    let forced =
        f32::from_bits(crate::RENDER_SCALE_BITS.load(std::sync::atomic::Ordering::Relaxed));
    let global_scale = if forced > 0.0 {
        forced as f64
    } else {
        governor_scale as f64
    };
    global_scale * renderer.resolution_factor()
}

impl RendererPort for DriverRenderer {
    fn uses_surface_meshes(&self) -> bool {
        matches!(self, DriverRenderer::Surface(_) | DriverRenderer::Splat(_))
    }
    fn upload_surfaces(&mut self, chunks: &[SurfaceChunk<'_>]) {
        match self {
            DriverRenderer::Surface(r) => r.upload_surfaces(chunks),
            DriverRenderer::Splat(r) => r.upload_surfaces(chunks),
            _ => {}
        }
    }
    fn remove_surfaces(&mut self, keys: &[SurfaceChunkKey]) {
        match self {
            DriverRenderer::Surface(r) => r.remove_surfaces(keys),
            DriverRenderer::Splat(r) => r.remove_surfaces(keys),
            _ => {}
        }
    }
    fn clear_surfaces(&mut self) {
        match self {
            DriverRenderer::Surface(r) => r.clear_surfaces(),
            DriverRenderer::Splat(r) => r.clear_surfaces(),
            _ => {}
        }
    }
    fn gpu_frame_ms(&self) -> Option<f32> {
        match self {
            DriverRenderer::Surface(r) => r.gpu_frame_ms(),
            DriverRenderer::Splat(r) => r.gpu_frame_ms(),
            DriverRenderer::Raymarch(r) => r.gpu_frame_ms(),
            DriverRenderer::Cpu(_) => None,
        }
    }
    fn cpu_telemetry_string(&self) -> Option<String> {
        match self {
            DriverRenderer::Cpu(r) => r.cpu_telemetry_string(),
            DriverRenderer::Surface(r) => r.cpu_telemetry_string(),
            DriverRenderer::Splat(r) => r.cpu_telemetry_string(),
            DriverRenderer::Raymarch(r) => r.cpu_telemetry_string(),
        }
    }
    fn upload_atlas(&mut self, texels: &[u32]) {
        match self {
            DriverRenderer::Surface(r) => r.upload_atlas(texels),
            DriverRenderer::Splat(r) => r.upload_atlas(texels),
            DriverRenderer::Raymarch(r) => r.upload_atlas(texels),
            DriverRenderer::Cpu(r) => r.upload_atlas(texels),
        }
    }
    fn upload_atlas_rows(&mut self, first_row: u32, texels: &[u32]) -> bool {
        match self {
            DriverRenderer::Surface(_) | DriverRenderer::Splat(_) => false,
            DriverRenderer::Raymarch(r) => r.upload_atlas_rows(first_row, texels),
            DriverRenderer::Cpu(r) => r.upload_atlas_rows(first_row, texels),
        }
    }
    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        match self {
            DriverRenderer::Surface(r) => r.draw(frame, chunks),
            DriverRenderer::Splat(r) => r.draw(frame, chunks),
            DriverRenderer::Raymarch(r) => r.draw(frame, chunks),
            DriverRenderer::Cpu(r) => r.draw(frame, chunks),
        }
    }
    fn upload_label_atlas(&mut self, rgba: &[u8], width: u32, height: u32) {
        // Only the surface path draws label billboards today; the other
        // renderers keep the port's no-op default.
        if let DriverRenderer::Surface(r) = self {
            r.upload_label_atlas(rgba, width, height);
        }
    }
}

/// Engine owns its renderer behind the port; the driver also needs to call
/// `resize` on the concrete type. This thin adapter shares one renderer
/// between both without widening the port.
struct SharedRenderer(Rc<RefCell<DriverRenderer>>);

impl RendererPort for SharedRenderer {
    fn uses_surface_meshes(&self) -> bool {
        self.0.borrow().uses_surface_meshes()
    }
    fn upload_surfaces(&mut self, chunks: &[SurfaceChunk<'_>]) {
        self.0.borrow_mut().upload_surfaces(chunks);
    }
    fn remove_surfaces(&mut self, keys: &[SurfaceChunkKey]) {
        self.0.borrow_mut().remove_surfaces(keys);
    }
    fn clear_surfaces(&mut self) {
        self.0.borrow_mut().clear_surfaces();
    }
    fn gpu_frame_ms(&self) -> Option<f32> {
        self.0.borrow().gpu_frame_ms()
    }
    fn cpu_telemetry_string(&self) -> Option<String> {
        self.0.borrow().cpu_telemetry_string()
    }
    fn upload_atlas(&mut self, texels: &[u32]) {
        self.0.borrow_mut().upload_atlas(texels);
    }
    fn upload_atlas_rows(&mut self, first_row: u32, texels: &[u32]) -> bool {
        self.0.borrow_mut().upload_atlas_rows(first_row, texels)
    }
    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        self.0.borrow_mut().draw(frame, chunks);
    }
    fn upload_label_atlas(&mut self, rgba: &[u8], width: u32, height: u32) {
        self.0.borrow_mut().upload_label_atlas(rgba, width, height);
    }
}

/// The embedded product-label atlas: row 0 almond water, row 1 rations.
/// The rations row stays transparent until a rations logo lands in
/// `wasm_frontend/assets/` and gets blitted here.
fn build_label_atlas() -> Option<(Vec<u8>, u32, u32)> {
    const ALMOND_LABEL_PNG: &[u8] = include_bytes!("../../assets/almond_water_label.png");
    let label = image::load_from_memory(ALMOND_LABEL_PNG).ok()?.to_rgba8();
    let (width, row_height) = label.dimensions();
    let mut atlas = vec![0u8; (width * row_height * 2 * 4) as usize];
    atlas[..(width * row_height * 4) as usize].copy_from_slice(label.as_raw());
    Some((atlas, width, row_height * 2))
}

/// Touch-play session state. On touch devices there is no pointer lock:
/// tapping the overlay enters "touch play" directly, the left half of the
/// screen is a virtual joystick (drag from touch-down point), and the right
/// half is a look surface. On-screen buttons cover flashlight and menu.
#[derive(Default)]
struct TouchState {
    /// True while playing in touch mode.
    active: bool,
    move_id: Option<i32>,
    move_origin: (f64, f64),
    look_id: Option<i32>,
    look_last: (f64, f64),
}

/// Renderer selection: surfaces by default, `?renderer=splat` for the
/// instanced face-splat path (default candidate once parity/perf is
/// confirmed), `?renderer=raymarch` for the retained SVO debug path,
/// `?renderer=cpu` for software fallback.
fn create_renderer(canvas: &HtmlCanvasElement, query: &str) -> Result<DriverRenderer, JsValue> {
    let renderer_choice = query_param(query, "renderer");
    if renderer_choice == Some("cpu") {
        return Ok(DriverRenderer::Cpu(CpuCanvasRenderer::new(canvas)?));
    }
    if renderer_choice == Some("raymarch") {
        return WebGl2Renderer::new(canvas).map(DriverRenderer::Raymarch);
    }
    if renderer_choice == Some("splat") {
        let profile = if query_param(query, "spec") == Some("high") {
            SplatProfile::high()
        } else {
            SplatProfile::low()
        };
        match SplatRenderer::new(canvas, profile) {
            Ok(gpu) => return Ok(DriverRenderer::Splat(gpu)),
            Err(err) => {
                web_sys::console::warn_2(
                    &JsValue::from_str(
                        "splat renderer unavailable, falling back to surface meshes:",
                    ),
                    &err,
                );
            }
        }
    }
    match SurfaceRenderer::new(canvas) {
        Ok(gpu) => Ok(DriverRenderer::Surface(gpu)),
        Err(err) => {
            web_sys::console::warn_2(
                &JsValue::from_str(
                    "WebGL2 surface renderer unavailable, falling back to CPU splatting:",
                ),
                &err,
            );
            Ok(DriverRenderer::Cpu(CpuCanvasRenderer::new(canvas)?))
        }
    }
}

pub fn boot() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;

    let canvas: HtmlCanvasElement =
        element(&document, "view").or_else(|_| element(&document, "game-canvas"))?;
    let overlay: HtmlElement = element(&document, "overlay")?;
    let status_msg: HtmlElement = element(&document, "status-msg")?;
    let play_msg: HtmlElement = element(&document, "play-msg")?;

    // ?spec=high -> 20u chunks, 5x5 streaming radius, 0.1u voxels.
    // Default is the low-spec profile: 10u chunks, 3x3 radius, 0.2u voxels.
    // Generation controls: ?seed=… (number or any text) plus the density
    // knobs ?pillars= ?walls= ?atria= ?lights= (multipliers, default 1).
    let query = window.location().search().unwrap_or_default();
    web_sys::console::log_1(&format!("BOOTING ENGINE: query={}", query).into());
    // Renderer optimization switchboard (?rt_<name>=0|1); see
    // application::render_settings for the catalog of switches.
    crate::init_render_toggles(&query);
    // This resolver is also called inside every worker. Keeping the actual
    // GeneratorConfig behind one adapter prevents a rejected voxel override
    // from producing different worlds on the main and worker threads.
    let (resolved_seed, generator_config) = generator_setup_from_query(&query, WORLD_SEED);
    let high_spec = generator_config.chunk_size == GeneratorConfig::high_spec().chunk_size;
    // Spawn on the main corridor of region (0,0), looking east down its
    // west leg: the first frame is a lit, walled corridor receding into
    // fog — the player knows immediately that this is the Backrooms.
    let spawn_at = vackrooms::use_cases::region_plan::spawn_point(resolved_seed);
    let spawn = [spawn_at.x, 1.7, spawn_at.z];
    // Face east along +X.
    let spawn_yaw = -std::f32::consts::FRAC_PI_2;
    // ?level=34 boots straight into the grassland (debugging any level in
    // any renderer without waiting on a noclip roll).
    let initial_level = query_param(&query, "level")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);
    let engine_config = if high_spec {
        EngineConfig {
            chunk_size: generator_config.chunk_size,
            chunk_radius: 2,
            seed: resolved_seed,
            spawn,
            spawn_yaw,
            // 5x5 footprint: the outer ring (>= 20 units away) stays at
            // the coarse LOD, so high spec pays for ~9 fine chunks, not 25.
            fine_distance: 25.0,
            initial_level,
            ..EngineConfig::default()
        }
    } else {
        EngineConfig {
            chunk_size: generator_config.chunk_size,
            seed: resolved_seed,
            spawn,
            spawn_yaw,
            initial_level,
            ..EngineConfig::default()
        }
    };

    let renderer = Rc::new(RefCell::new(create_renderer(&canvas, &query)?));
    if let Some((label_rgba, label_w, label_h)) = build_label_atlas() {
        renderer
            .borrow_mut()
            .upload_label_atlas(&label_rgba, label_w, label_h);
    }
    if let Ok(hud_renderer) = element::<HtmlElement>(&document, "hud-renderer") {
        hud_renderer.set_text_content(Some(renderer.borrow().label()));
    }
    // Names the section the player is standing in (top-right HUD). The
    // locator re-derives the deterministic region plan, cached per region.
    let locator = Rc::new(RefCell::new(SectionLocator::new(
        resolved_seed,
        generator_config,
    )));
    // This pool parallelizes chunk generation/lighting/SVO construction,
    // never the renderer. The pure policy reserves the main thread and
    // clamps explicit requests to reported hardware. Zero selects the
    // synchronous source, which is also the worker-startup fallback.
    let worker_preference = parse_generation_worker_preference(&query);
    let hardware_concurrency = window.navigator().hardware_concurrency();
    let generation_worker_count = worker_preference.resolve(hardware_concurrency);
    let source: Box<dyn crate::application::ports::ChunkSourcePort> =
        if generation_worker_count == 0 {
            web_sys::console::log_1(&"chunk generation: synchronous main thread".into());
            Box::new(LocalChunkSource::with_telemetry(
                SimpleNoiseProvider::new(),
                resolved_seed,
                generator_config,
                &CONSOLE_TELEMETRY,
            ))
        } else {
            match crate::drivers::worker_source::WorkerChunkSource::new(
                &query,
                WORLD_SEED,
                generation_worker_count,
            ) {
                Ok(pool) => {
                    web_sys::console::log_1(
                        &format!("chunk generation: {} worker threads", pool.pool_size()).into(),
                    );
                    Box::new(pool)
                }
                Err(err) => {
                    web_sys::console::warn_2(
                        &JsValue::from_str(
                            "worker pool unavailable, falling back to in-thread generation:",
                        ),
                        &err,
                    );
                    Box::new(LocalChunkSource::with_telemetry(
                        SimpleNoiseProvider::new(),
                        resolved_seed,
                        generator_config,
                        &CONSOLE_TELEMETRY,
                    ))
                }
            }
        };
    let engine = Rc::new(RefCell::new(Engine::new(
        engine_config,
        Box::new(SharedRenderer(renderer.clone())),
        source,
    )));
    let input = Rc::new(RefCell::new(InputCollector::new()));
    let touch = Rc::new(RefCell::new(TouchState::default()));

    let is_capture = query_flag(&query, "capture");
    if is_capture {
        let mut cam_pos = spawn;
        if let Some(parts) = query_param(&query, "camera").map(|v| v.split(',').collect::<Vec<_>>())
            && parts.len() == 3
            && let (Ok(x), Ok(y), Ok(z)) = (
                parts[0].parse::<f32>(),
                parts[1].parse::<f32>(),
                parts[2].parse::<f32>(),
            )
        {
            cam_pos = [x, y, z];
        }
        let yaw_val = query_param(&query, "yaw")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.5708);
        let pitch_val = query_param(&query, "pitch")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(-0.05);
        engine
            .borrow_mut()
            .teleport_player(cam_pos, yaw_val, pitch_val);
    }

    if !is_capture {
        attach_input_listeners(&document, &canvas, &overlay, &input, &touch)?;
        attach_touch_listeners(&document, &canvas, &overlay, &input, &touch)?;
    }
    run_frame_loop(
        window, document, canvas, overlay, status_msg, play_msg, renderer, engine, input, locator,
    )
}

fn element<T: JsCast>(document: &Document, id: &str) -> Result<T, JsValue> {
    document
        .get_element_by_id(id)
        .ok_or_else(|| JsValue::from_str(&format!("missing #{id} element")))?
        .dyn_into::<T>()
        .map_err(|_| JsValue::from_str(&format!("#{id} has unexpected element type")))
}

fn attach_input_listeners(
    document: &Document,
    canvas: &HtmlCanvasElement,
    overlay: &HtmlElement,
    input: &Rc<RefCell<InputCollector>>,
    touch: &Rc<RefCell<TouchState>>,
) -> Result<(), JsValue> {
    // Keyboard: KeyboardEvent.code -> MoveIntent, mapped by the input adapter.
    for (event, pressed) in [("keydown", true), ("keyup", false)] {
        let input = input.clone();
        let closure = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
            // F3 toggles the anomaly debug overlay (and never reaches the
            // browser's own F3 find shortcut).
            if pressed && e.code() == "F3" {
                e.prevent_default();
                if !e.repeat() {
                    let on = crate::ANOMALY_DEBUG.load(std::sync::atomic::Ordering::Relaxed);
                    crate::ANOMALY_DEBUG.store(!on, std::sync::atomic::Ordering::Relaxed);
                }
                return;
            }
            if !e.repeat() {
                input.borrow_mut().key_event(&e.code(), pressed);
            }
        });
        document.add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Mouse look (deltas are ignored by the adapter unless pointer-locked).
    {
        let input = input.clone();
        let closure = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            input
                .borrow_mut()
                .mouse_delta(e.movement_x() as f32, e.movement_y() as f32);
        });
        document.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Click-to-play: the overlay requests pointer lock on the render canvas.
    {
        let canvas = canvas.clone();
        let closure = Closure::<dyn FnMut()>::new(move || {
            canvas.request_pointer_lock();
        });
        overlay.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Pointer-lock state drives both the input adapter and overlay
    // visibility — unless a touch session owns them.
    {
        let input = input.clone();
        let touch = touch.clone();
        let doc_for_closure = document.clone();
        let overlay = overlay.clone();
        let closure = Closure::<dyn FnMut()>::new(move || {
            if touch.borrow().active {
                return;
            }
            let locked = doc_for_closure.pointer_lock_element().is_some();
            input.borrow_mut().set_locked(locked);
            let _ = overlay
                .style()
                .set_property("display", if locked { "none" } else { "flex" });
        });
        document.add_event_listener_with_callback(
            "pointerlockchange",
            closure.as_ref().unchecked_ref(),
        )?;
        closure.forget();
    }

    Ok(())
}

/// Wires the touch controls: overlay tap-to-play, split-screen virtual
/// joystick + look surface on the canvas, and the on-screen flashlight and
/// menu buttons (`#btn-flashlight`, `#btn-menu`, inside `#touch-ui`).
fn attach_touch_listeners(
    document: &Document,
    canvas: &HtmlCanvasElement,
    overlay: &HtmlElement,
    input: &Rc<RefCell<InputCollector>>,
    touch: &Rc<RefCell<TouchState>>,
) -> Result<(), JsValue> {
    let touch_ui: Option<HtmlElement> = element(document, "touch-ui").ok();

    let set_touch_play = {
        let input = input.clone();
        let touch = touch.clone();
        let overlay = overlay.clone();
        let touch_ui = touch_ui.clone();
        Rc::new(move |on: bool| {
            touch.borrow_mut().active = on;
            let mut input = input.borrow_mut();
            input.set_locked(on);
            if !on {
                input.set_move_axes(0.0, 0.0);
            }
            let _ = overlay
                .style()
                .set_property("display", if on { "none" } else { "flex" });
            if let Some(ui) = &touch_ui {
                let _ = ui
                    .style()
                    .set_property("display", if on { "flex" } else { "none" });
            }
        })
    };

    // Tap the overlay -> enter touch play (the desktop path uses `click` +
    // pointer lock instead). prevent_default suppresses the synthetic click
    // that would otherwise also request pointer lock.
    {
        let enter = set_touch_play.clone();
        let closure = Closure::<dyn FnMut(TouchEvent)>::new(move |e: TouchEvent| {
            e.prevent_default();
            enter(true);
        });
        overlay.add_event_listener_with_callback("touchend", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Canvas touches: left 45% of the screen is the movement joystick,
    // the rest is the look surface.
    {
        let touch = touch.clone();
        let closure = Closure::<dyn FnMut(TouchEvent)>::new(move |e: TouchEvent| {
            let mut st = touch.borrow_mut();
            if !st.active {
                return;
            }
            e.prevent_default();
            let move_split = web_sys::window()
                .and_then(|w| w.inner_width().ok())
                .and_then(|v| v.as_f64())
                .unwrap_or(800.0)
                * 0.45;
            let changed = e.changed_touches();
            for i in 0..changed.length() {
                let Some(t) = changed.item(i) else { continue };
                let (x, y) = (t.client_x() as f64, t.client_y() as f64);
                if x < move_split && st.move_id.is_none() {
                    st.move_id = Some(t.identifier());
                    st.move_origin = (x, y);
                } else if st.look_id.is_none() {
                    st.look_id = Some(t.identifier());
                    st.look_last = (x, y);
                }
            }
        });
        canvas.add_event_listener_with_callback("touchstart", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    {
        let input = input.clone();
        let touch = touch.clone();
        let closure = Closure::<dyn FnMut(TouchEvent)>::new(move |e: TouchEvent| {
            let mut st = touch.borrow_mut();
            if !st.active {
                return;
            }
            e.prevent_default();
            let changed = e.changed_touches();
            for i in 0..changed.length() {
                let Some(t) = changed.item(i) else { continue };
                let (x, y) = (t.client_x() as f64, t.client_y() as f64);
                if st.move_id == Some(t.identifier()) {
                    let dx = ((x - st.move_origin.0) / JOYSTICK_RADIUS).clamp(-1.0, 1.0);
                    let dy = ((y - st.move_origin.1) / JOYSTICK_RADIUS).clamp(-1.0, 1.0);
                    // Screen-space up (negative dy) walks forward.
                    input.borrow_mut().set_move_axes(dx as f32, -dy as f32);
                } else if st.look_id == Some(t.identifier()) {
                    let (lx, ly) = st.look_last;
                    st.look_last = (x, y);
                    input.borrow_mut().mouse_delta(
                        (x - lx) as f32 * TOUCH_LOOK_SCALE,
                        (y - ly) as f32 * TOUCH_LOOK_SCALE,
                    );
                }
            }
        });
        canvas.add_event_listener_with_callback("touchmove", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    for event in ["touchend", "touchcancel"] {
        let input = input.clone();
        let touch = touch.clone();
        let closure = Closure::<dyn FnMut(TouchEvent)>::new(move |e: TouchEvent| {
            let mut st = touch.borrow_mut();
            if !st.active {
                return;
            }
            e.prevent_default();
            let changed = e.changed_touches();
            for i in 0..changed.length() {
                let Some(t) = changed.item(i) else { continue };
                if st.move_id == Some(t.identifier()) {
                    st.move_id = None;
                    input.borrow_mut().set_move_axes(0.0, 0.0);
                } else if st.look_id == Some(t.identifier()) {
                    st.look_id = None;
                }
            }
        });
        canvas.add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // On-screen buttons (optional elements; the page may omit them).
    if let Ok(btn) = element::<HtmlElement>(document, "btn-flare") {
        let input = input.clone();
        let closure = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
            e.prevent_default();
            e.stop_propagation();
            input.borrow_mut().queue_flare();
        });
        btn.add_event_listener_with_callback("touchend", closure.as_ref().unchecked_ref())?;
        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    if let Ok(btn) = element::<HtmlElement>(document, "btn-flashlight") {
        let input = input.clone();
        let closure = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
            e.prevent_default();
            e.stop_propagation();
            input.borrow_mut().toggle_flashlight();
        });
        btn.add_event_listener_with_callback("touchend", closure.as_ref().unchecked_ref())?;
        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    if let Ok(btn) = element::<HtmlElement>(document, "btn-menu") {
        let exit = set_touch_play.clone();
        let closure = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
            e.prevent_default();
            e.stop_propagation();
            exit(false);
        });
        btn.add_event_listener_with_callback("touchend", closure.as_ref().unchecked_ref())?;
        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_frame_loop(
    window: Window,
    _document: Document,
    canvas: HtmlCanvasElement,
    _overlay: HtmlElement,
    status_msg: HtmlElement,
    play_msg: HtmlElement,
    renderer: Rc<RefCell<DriverRenderer>>,
    engine: Rc<RefCell<Engine>>,
    input: Rc<RefCell<InputCollector>>,
    locator: Rc<RefCell<SectionLocator>>,
) -> Result<(), JsValue> {
    let hud_fps: HtmlElement = element(window.document().as_ref().unwrap(), "hud-fps")?;
    let hud_chunks: HtmlElement = element(window.document().as_ref().unwrap(), "hud-chunks")?;
    let hud_nodes: HtmlElement = element(window.document().as_ref().unwrap(), "hud-nodes")?;
    let hud_scale: HtmlElement = element(window.document().as_ref().unwrap(), "hud-scale")?;
    let hud_gpu: HtmlElement = element(window.document().as_ref().unwrap(), "hud-gpu")?;
    let hud_dist: Option<HtmlElement> =
        element(window.document().as_ref().unwrap(), "hud-dist").ok();
    let last_dist_text = Rc::new(RefCell::new(String::new()));
    // Top-right section readout and the anomaly debug overlay; both are
    // optional page elements so older/embedded shells keep working.
    let hud_section: Option<HtmlElement> =
        element(window.document().as_ref().unwrap(), "hud-section").ok();
    let last_section_text = Rc::new(RefCell::new(String::new()));
    let debug_overlay: Option<HtmlElement> =
        element(window.document().as_ref().unwrap(), "debug-overlay").ok();
    let debug_overlay_visible = Rc::new(Cell::new(false));

    // Survival vitals panel; optional so embedded shells keep working.
    let document = window.document().unwrap();
    let vitals_thirst: Option<HtmlElement> = element(&document, "vitals-thirst-fill").ok();
    let vitals_hunger: Option<HtmlElement> = element(&document, "vitals-hunger-fill").ok();
    let vitals_cond: Option<HtmlElement> = element(&document, "vitals-cond-fill").ok();
    let vitals_thirst_bar: Option<HtmlElement> = element(&document, "vitals-thirst").ok();
    let vitals_hunger_bar: Option<HtmlElement> = element(&document, "vitals-hunger").ok();
    let vitals_inv: Option<HtmlElement> = element(&document, "vitals-inv").ok();
    let death_overlay: Option<HtmlElement> = element(&document, "death-overlay").ok();
    let last_deaths = Rc::new(Cell::new(0u32));
    let death_shown_at = Rc::new(Cell::new(0.0f64));

    let query = window.location().search().unwrap_or_default();
    let is_capture = query_flag(&query, "capture");
    // Visual baselines get fifteen settled frames; renderer smoke tests only
    // need proof that a loaded scene completed a couple of real draws. The
    // shorter path keeps deliberately unoptimized reference renderers usable
    // under software WebGL in CI without weakening screenshot baselines.
    let capture_settle_frames = if query_flag(&query, "smoke") { 2 } else { 15 };
    let capture_frame_counter = Rc::new(Cell::new(0u32));

    let last_time = Rc::new(Cell::new(0.0f64));
    let frame_count = Rc::new(Cell::new(0u32));
    let hud_window_start = Rc::new(Cell::new(0.0f64));
    let was_ready = Rc::new(Cell::new(false));

    // Standard self-referential rAF closure pattern.
    let raf_handle: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let raf_handle_clone = raf_handle.clone();

    let loop_window = window.clone();
    *raf_handle.borrow_mut() = Some(Closure::new(move |time_ms: f64| {
        let dt = if is_capture {
            0.0
        } else if last_time.get() > 0.0 {
            ((time_ms - last_time.get()) / 1000.0) as f32
        } else {
            1.0 / 60.0
        };
        last_time.set(time_ms);

        // Adaptive resolution: size the backing store to
        // window * governor scale * renderer base factor. A non-zero
        // user override (settings menu) replaces the governor scale.
        {
            let scale = effective_resolution_scale(
                &renderer.borrow(),
                engine.borrow().stats().resolution_scale,
            );
            let target_w = (loop_window
                .inner_width()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(800.0)
                * scale)
                .max(1.0) as u32;
            let target_h = (loop_window
                .inner_height()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(600.0)
                * scale)
                .max(1.0) as u32;

            if canvas.width() != target_w || canvas.height() != target_h {
                canvas.set_width(target_w);
                canvas.set_height(target_h);
                renderer.borrow_mut().resize(target_w, target_h);
            }
        }

        let frame_input = input.borrow_mut().take_frame();
        engine.borrow_mut().tick(dt, &frame_input);

        // Loading overlay: flip to "click to play" once the spawn chunk is in.
        let stats = engine.borrow().stats();
        if stats.ready && !was_ready.get() {
            was_ready.set(true);
            let _ = status_msg.style().set_property("display", "none");
            let _ = play_msg.style().set_property("display", "block");
        }

        // HUD refresh at a fixed frame cadence.
        frame_count.set(frame_count.get() + 1);
        if frame_count.get() % HUD_INTERVAL == 0 {
            let elapsed = (time_ms - hud_window_start.get()) / 1000.0;
            if elapsed > 0.0 {
                let fps = (HUD_INTERVAL as f64 / elapsed).round();
                hud_fps.set_text_content(Some(&fps.to_string()));
            }
            hud_window_start.set(time_ms);
            hud_chunks.set_text_content(Some(&format!(
                "{}/{}",
                stats.fine_chunks, stats.resident_chunks
            )));
            hud_nodes.set_text_content(Some(&stats.atlas_nodes.to_string()));
            let effective_scale =
                effective_resolution_scale(&renderer.borrow(), stats.resolution_scale);
            hud_scale.set_text_content(Some(&format!("{:.0}%", effective_scale * 100.0)));
            if let Some(hud_dist) = &hud_dist {
                // Restrained rounding, and only touch the DOM on change.
                let text = format!("{:.0} m", stats.distance_m);
                let mut last = last_dist_text.borrow_mut();
                if *last != text {
                    hud_dist.set_text_content(Some(&text));
                    *last = text;
                }
            }
            let gpu = renderer.borrow().gpu_frame_ms();
            let cpu_telemetry = renderer.borrow().cpu_telemetry_string();
            if let Some(text) = cpu_telemetry {
                hud_gpu.set_text_content(Some(&text));
            } else {
                hud_gpu.set_text_content(Some(
                    &gpu.map(|ms| format!("{ms:.1} ms"))
                        .unwrap_or_else(|| "n/a".to_string()),
                ));
            }

            // Top-right section readout ("LEVEL 0 · MAIN CORRIDOR · ...").
            // The locator caches its region plan, so this is cheap except on
            // a region crossing; only touch the DOM when the text changes.
            if let Some(hud_section) = &hud_section {
                let (level, pos) = {
                    let engine_ref = engine.borrow();
                    (engine_ref.level(), engine_ref.player().position)
                };
                let text = locator.borrow_mut().describe(level, pos[0], pos[2]);
                let mut last = last_section_text.borrow_mut();
                if *last != text {
                    hud_section.set_text_content(Some(&text));
                    *last = text;
                }
            }

            // Survival vitals: bar widths + inventory/temperature line.
            if let Some(fill) = &vitals_thirst {
                let _ = fill
                    .style()
                    .set_property("width", &format!("{:.0}%", stats.hydration * 100.0));
            }
            if let Some(fill) = &vitals_hunger {
                let _ = fill
                    .style()
                    .set_property("width", &format!("{:.0}%", stats.satiety * 100.0));
            }
            if let Some(fill) = &vitals_cond {
                let _ = fill
                    .style()
                    .set_property("width", &format!("{:.0}%", stats.condition * 100.0));
            }
            if let Some(bar) = &vitals_thirst_bar {
                let _ = bar.set_class_name(if stats.hydration < 0.25 { "bar low" } else { "bar" });
            }
            if let Some(bar) = &vitals_hunger_bar {
                let _ = bar.set_class_name(if stats.satiety < 0.25 { "bar low" } else { "bar" });
            }
            if let Some(inv) = &vitals_inv {
                inv.set_text_content(Some(&format!(
                    "🥛 {} · 🥫 {} · {:.0}°C",
                    stats.almond_bottles, stats.rations, stats.ambient_c
                )));
            }
            // Death flash: appears on each new death, fades after ~2.5 s.
            if let Some(overlay) = &death_overlay {
                if stats.deaths > last_deaths.get() {
                    last_deaths.set(stats.deaths);
                    death_shown_at.set(time_ms);
                    let _ = overlay.set_class_name("shown");
                } else if time_ms - death_shown_at.get() > 2500.0
                    && !overlay.class_name().is_empty()
                {
                    let _ = overlay.set_class_name("");
                }
            }

            // Anomaly debug overlay (F3 or the settings switch).
            if let Some(overlay_el) = &debug_overlay {
                let on = crate::ANOMALY_DEBUG.load(std::sync::atomic::Ordering::Relaxed);
                if on {
                    overlay_el.set_text_content(Some(&engine.borrow().anomaly_debug_text()));
                }
                if on != debug_overlay_visible.get() {
                    debug_overlay_visible.set(on);
                    let _ = overlay_el
                        .style()
                        .set_property("display", if on { "block" } else { "none" });
                }
            }
        }

        // In capture mode, wait until the camera chunk is loaded, then allow
        // the requested number of frames to settle before exposing readiness.
        if is_capture {
            let engine_ref = engine.borrow();
            let pos = engine_ref.player().position;
            let cs = engine_ref.chunk_size();
            let cam_chunk = crate::application::streaming::chunk_key(
                (pos[0] / cs).floor() * cs,
                (pos[2] / cs).floor() * cs,
            );
            if frame_count.get() % 30 == 0 {
                web_sys::console::log_1(
                    &format!(
                        "[DEBUG CAPTURE] player pos: {:?}, cs: {}, cam_chunk: {:?}, resident: {}, ready: {}",
                        pos,
                        cs,
                        cam_chunk,
                        engine_ref.is_chunk_resident(cam_chunk),
                        engine_ref.stats().ready
                    )
                    .into(),
                );
            }
            if engine_ref.is_chunk_resident(cam_chunk) {
                let frames = capture_frame_counter.get();
                if frames < capture_settle_frames {
                    capture_frame_counter.set(frames + 1);
                } else {
                    if let Some(w) = web_sys::window() {
                        let _ = js_sys::Reflect::set(
                            &w,
                            &JsValue::from_str("__sceneReady"),
                            &JsValue::from_bool(true),
                        );
                    }
                }
            }
        }

        // Schedule next frame.
        if let Some(closure) = raf_handle_clone.borrow().as_ref() {
            let _ = loop_window.request_animation_frame(closure.as_ref().unchecked_ref());
        }
    }));

    if let Some(closure) = raf_handle.borrow().as_ref() {
        window.request_animation_frame(closure.as_ref().unchecked_ref())?;
    }
    // Keep the closure (and everything it captures) alive forever.
    std::mem::forget(raf_handle);
    Ok(())
}
