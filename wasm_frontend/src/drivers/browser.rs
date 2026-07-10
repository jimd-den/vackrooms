//! Browser runtime driver: DOM/event plumbing, the requestAnimationFrame
//! loop, pointer lock, the HUD and the 2D minimap.
//!
//! This module is the composition root's workhorse: it instantiates the
//! concrete adapters/drivers, hands them to `application::engine::Engine`,
//! and from then on only shuttles plain data across the port boundaries.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{
    CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, KeyboardEvent, MouseEvent,
    TouchEvent, Window,
};

use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;

use crate::adapters::input::InputCollector;
use crate::adapters::local_chunk_source::LocalChunkSource;
use crate::adapters::query_config::parse_generation_params;
use crate::application::engine::{Engine, EngineConfig};
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
/// Minimap pixels per world unit.
const MINIMAP_SCALE: f64 = 12.0;

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

    /// Base internal-resolution multiplier (further scaled by the adaptive
    /// governor). Splatting every pixel on the CPU is far more expensive
    /// than rasterizing a quad, so the CPU path renders much smaller and
    /// lets CSS upscale with `image-rendering: pixelated`.
    fn resolution_factor(&self) -> f64 {
        match self {
            DriverRenderer::Surface(_)
            | DriverRenderer::Splat(_)
            | DriverRenderer::Raymarch(_) => 1.0,
            DriverRenderer::Cpu(_) => 0.25,
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
            DriverRenderer::Raymarch(_) | DriverRenderer::Cpu(_) => None,
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
            // The CPU splatter rebuilds its mip pyramid from the whole
            // atlas, so it only supports full uploads.
            DriverRenderer::Cpu(_) => false,
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
    if query.contains("renderer=cpu") {
        return Ok(DriverRenderer::Cpu(CpuCanvasRenderer::new(canvas)?));
    }
    if query.contains("renderer=raymarch") {
        return WebGl2Renderer::new(canvas).map(DriverRenderer::Raymarch);
    }
    if query.contains("renderer=splat") {
        let profile = if query.contains("spec=high") {
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

    let canvas: HtmlCanvasElement = element(&document, "view")?;
    let minimap: HtmlCanvasElement = element(&document, "minimap")?;
    let overlay: HtmlElement = element(&document, "overlay")?;
    let status_msg: HtmlElement = element(&document, "status-msg")?;
    let play_msg: HtmlElement = element(&document, "play-msg")?;

    // ?spec=high -> 20u chunks, 5x5 streaming radius, 0.1u voxels.
    // Default is the low-spec profile: 10u chunks, 3x3 radius, 0.2u voxels.
    // Generation controls: ?seed=… (number or any text) plus the density
    // knobs ?pillars= ?walls= ?atria= ?lights= (multipliers, default 1).
    let query = window.location().search().unwrap_or_default();
    let gen_params = parse_generation_params(&query, WORLD_SEED);
    let high_spec = query.contains("spec=high");
    let (generator_config, engine_config) = if high_spec {
        (
            GeneratorConfig::high_spec().with_tuning(gen_params.tuning),
            EngineConfig {
                chunk_size: 20.0,
                chunk_radius: 2,
                seed: gen_params.seed,
                // 5x5 footprint: the outer ring (>= 20 units away) stays at
                // the coarse LOD, so high spec pays for ~9 fine chunks, not 25.
                fine_distance: 25.0,
                ..EngineConfig::default()
            },
        )
    } else {
        (
            GeneratorConfig::low_spec().with_tuning(gen_params.tuning),
            EngineConfig {
                seed: gen_params.seed,
                ..EngineConfig::default()
            },
        )
    };

    let renderer = Rc::new(RefCell::new(create_renderer(&canvas, &query)?));
    if let Ok(hud_renderer) = element::<HtmlElement>(&document, "hud-renderer") {
        hud_renderer.set_text_content(Some(renderer.borrow().label()));
    }
    // Chunk generation runs on a Web Worker pool so crossing a streaming
    // boundary never stalls the frame loop. `?workers=0` forces the old
    // synchronous in-thread source (also the fallback if workers fail).
    let source: Box<dyn crate::application::ports::ChunkSourcePort> = if query.contains("workers=0") {
        Box::new(LocalChunkSource::with_telemetry(
            SimpleNoiseProvider::new(),
            gen_params.seed,
            generator_config,
            &CONSOLE_TELEMETRY,
        ))
    } else {
        match crate::drivers::worker_source::WorkerChunkSource::new(&query, WORLD_SEED) {
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
                    gen_params.seed,
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

    attach_input_listeners(&document, &canvas, &overlay, &input, &touch)?;
    attach_touch_listeners(&document, &canvas, &overlay, &input, &touch)?;
    run_frame_loop(
        window, document, canvas, minimap, overlay, status_msg, play_msg, renderer, engine, input,
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
    minimap: HtmlCanvasElement,
    _overlay: HtmlElement,
    status_msg: HtmlElement,
    play_msg: HtmlElement,
    renderer: Rc<RefCell<DriverRenderer>>,
    engine: Rc<RefCell<Engine>>,
    input: Rc<RefCell<InputCollector>>,
) -> Result<(), JsValue> {
    let minimap_ctx: CanvasRenderingContext2d = minimap
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("no 2d context for minimap"))?
        .dyn_into()?;

    let hud_fps: HtmlElement = element(window.document().as_ref().unwrap(), "hud-fps")?;
    let hud_chunks: HtmlElement = element(window.document().as_ref().unwrap(), "hud-chunks")?;
    let hud_nodes: HtmlElement = element(window.document().as_ref().unwrap(), "hud-nodes")?;
    let hud_scale: HtmlElement = element(window.document().as_ref().unwrap(), "hud-scale")?;
    let hud_gpu: HtmlElement = element(window.document().as_ref().unwrap(), "hud-gpu")?;

    let last_time = Rc::new(Cell::new(0.0f64));
    let frame_count = Rc::new(Cell::new(0u32));
    let hud_window_start = Rc::new(Cell::new(0.0f64));
    let was_ready = Rc::new(Cell::new(false));

    // Standard self-referential rAF closure pattern.
    let raf_handle: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let raf_handle_clone = raf_handle.clone();

    let loop_window = window.clone();
    *raf_handle.borrow_mut() = Some(Closure::new(move |time_ms: f64| {
        let dt = if last_time.get() > 0.0 {
            ((time_ms - last_time.get()) / 1000.0) as f32
        } else {
            1.0 / 60.0
        };
        last_time.set(time_ms);

        // Adaptive resolution: size the backing store to
        // window * governor scale * renderer base factor. A non-zero
        // user override (settings menu) replaces the governor scale.
        {
            let forced =
                f32::from_bits(crate::RENDER_SCALE_BITS.load(std::sync::atomic::Ordering::Relaxed));
            let governor_scale = if forced > 0.0 {
                forced as f64
            } else {
                engine.borrow().stats().resolution_scale as f64
            };
            let scale = governor_scale * renderer.borrow().resolution_factor();
            let mut target_w = (loop_window
                .inner_width()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(800.0)
                * scale)
                .max(1.0) as u32;
            let mut target_h = (loop_window
                .inner_height()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(600.0)
                * scale)
                .max(1.0) as u32;

            if matches!(*renderer.borrow(), DriverRenderer::Cpu(_)) {
                let max_w = 480.0;
                let max_h = 270.0;
                let cap_scale = (max_w / target_w as f32).min(max_h as f32 / target_h as f32).min(1.0);
                target_w = (target_w as f32 * cap_scale).round() as u32;
                target_h = (target_h as f32 * cap_scale).round() as u32;
            }

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

        // The minimap is a pure convenience display: redrawing its dozens of
        // fill_rects every frame costs real CPU on low-end machines, so it
        // refreshes at a third of the frame rate.
        if frame_count.get() % 3 == 0 {
            draw_minimap(&minimap_ctx, &minimap, &engine.borrow());
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
            hud_scale.set_text_content(Some(&format!("{:.0}%", stats.resolution_scale * 100.0)));
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

/// Top-down radar: solid collision boxes around the player, plus a player
/// dot and view direction line. Pure presentation — reads engine state only.
fn draw_minimap(ctx: &CanvasRenderingContext2d, canvas: &HtmlCanvasElement, engine: &Engine) {
    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    let cx = w / 2.0;
    let cy = h / 2.0;
    let player = engine.player();

    ctx.clear_rect(0.0, 0.0, w, h);
    ctx.set_fill_style_str("rgba(0, 0, 0, 0.5)");
    ctx.fill_rect(0.0, 0.0, w, h);

    ctx.set_fill_style_str("#ff5555");
    let range = (w / 2.0) / MINIMAP_SCALE + 1.0;
    for b in engine.collision_world().boxes() {
        let rel_x = (b.min[0] - player.position[0]) as f64;
        let rel_z = (b.min[2] - player.position[2]) as f64;
        if rel_x.abs() > range || rel_z.abs() > range {
            continue;
        }
        let size_x = (b.max[0] - b.min[0]) as f64 * MINIMAP_SCALE;
        let size_z = (b.max[2] - b.min[2]) as f64 * MINIMAP_SCALE;
        ctx.fill_rect(
            cx + rel_x * MINIMAP_SCALE,
            cy + rel_z * MINIMAP_SCALE,
            size_x.max(1.0),
            size_z.max(1.0),
        );
    }

    // Player dot.
    ctx.set_fill_style_str("#00ff00");
    ctx.begin_path();
    let _ = ctx.arc(cx, cy, 4.0, 0.0, std::f64::consts::TAU);
    ctx.fill();

    // View direction.
    ctx.set_stroke_style_str("#00ff00");
    ctx.set_line_width(2.0);
    ctx.begin_path();
    ctx.move_to(cx, cy);
    let yaw = player.yaw as f64;
    ctx.line_to(cx + yaw.sin() * -14.0, cy + yaw.cos() * -14.0);
    ctx.stroke();
}
