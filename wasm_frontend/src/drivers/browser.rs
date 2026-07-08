//! Browser runtime driver: DOM/event plumbing, the requestAnimationFrame
//! loop, pointer lock, the HUD and the 2D minimap.
//!
//! This module is the composition root's workhorse: it instantiates the
//! concrete adapters/drivers, hands them to `application::engine::Engine`,
//! and from then on only shuttles plain data across the port boundaries.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, KeyboardEvent, MouseEvent,
    Window,
};

use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;

use crate::adapters::input::InputCollector;
use crate::adapters::local_chunk_source::LocalChunkSource;
use crate::application::engine::{Engine, EngineConfig};
use crate::application::ports::{ChunkDraw, FrameParams, RendererPort};
use crate::drivers::console_telemetry::CONSOLE_TELEMETRY;
use crate::drivers::cpu_canvas::CpuCanvasRenderer;
use crate::drivers::webgl::WebGl2Renderer;

/// World seed shared with the native server so both render the same world.
const WORLD_SEED: u32 = 42;
/// HUD refresh cadence in frames.
const HUD_INTERVAL: u32 = 30;
/// Minimap pixels per world unit.
const MINIMAP_SCALE: f64 = 12.0;

/// The two available renderer back ends. GPU raymarching is the default;
/// the CPU microvoxel splatter is selected by `?renderer=cpu` or used as an
/// automatic fallback when WebGL2 is unavailable.
enum DriverRenderer {
    Gpu(WebGl2Renderer),
    Cpu(CpuCanvasRenderer),
}

impl DriverRenderer {
    fn resize(&mut self, width: u32, height: u32) {
        match self {
            DriverRenderer::Gpu(r) => r.resize(width, height),
            DriverRenderer::Cpu(r) => r.resize(width, height),
        }
    }

    /// Base internal-resolution multiplier (further scaled by the adaptive
    /// governor). Splatting every pixel on the CPU is far more expensive
    /// than rasterizing a quad, so the CPU path renders much smaller and
    /// lets CSS upscale with `image-rendering: pixelated`.
    fn resolution_factor(&self) -> f64 {
        match self {
            DriverRenderer::Gpu(_) => 1.0,
            DriverRenderer::Cpu(_) => 0.25,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            DriverRenderer::Gpu(_) => "GPU raymarch",
            DriverRenderer::Cpu(_) => "CPU splat",
        }
    }
}

impl RendererPort for DriverRenderer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        match self {
            DriverRenderer::Gpu(r) => r.upload_atlas(texels),
            DriverRenderer::Cpu(r) => r.upload_atlas(texels),
        }
    }
    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        match self {
            DriverRenderer::Gpu(r) => r.draw(frame, chunks),
            DriverRenderer::Cpu(r) => r.draw(frame, chunks),
        }
    }
}

/// Engine owns its renderer behind the port; the driver also needs to call
/// `resize` on the concrete type. This thin adapter shares one renderer
/// between both without widening the port.
struct SharedRenderer(Rc<RefCell<DriverRenderer>>);

impl RendererPort for SharedRenderer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        self.0.borrow_mut().upload_atlas(texels);
    }
    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        self.0.borrow_mut().draw(frame, chunks);
    }
}

/// Renderer selection: explicit `?renderer=cpu`, otherwise WebGL2 with a
/// logged fallback to the CPU splatter if context creation fails.
fn create_renderer(canvas: &HtmlCanvasElement, query: &str) -> Result<DriverRenderer, JsValue> {
    if query.contains("renderer=cpu") {
        return Ok(DriverRenderer::Cpu(CpuCanvasRenderer::new(canvas)?));
    }
    match WebGl2Renderer::new(canvas) {
        Ok(gpu) => Ok(DriverRenderer::Gpu(gpu)),
        Err(err) => {
            web_sys::console::warn_2(
                &JsValue::from_str("WebGL2 unavailable, falling back to CPU splatting:"),
                &err,
            );
            Ok(DriverRenderer::Cpu(CpuCanvasRenderer::new(canvas)?))
        }
    }
}

pub fn boot() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let document = window.document().ok_or_else(|| JsValue::from_str("no document"))?;

    let canvas: HtmlCanvasElement = element(&document, "view")?;
    let minimap: HtmlCanvasElement = element(&document, "minimap")?;
    let overlay: HtmlElement = element(&document, "overlay")?;
    let status_msg: HtmlElement = element(&document, "status-msg")?;
    let play_msg: HtmlElement = element(&document, "play-msg")?;

    // ?spec=high -> 20u chunks, 5x5 streaming radius, 0.1u voxels.
    // Default is the low-spec profile: 10u chunks, 3x3 radius, 0.2u voxels.
    let query = window.location().search().unwrap_or_default();
    let high_spec = query.contains("spec=high");
    let (generator_config, engine_config) = if high_spec {
        (
            GeneratorConfig::high_spec(),
            EngineConfig { chunk_size: 20.0, chunk_radius: 2, ..EngineConfig::default() },
        )
    } else {
        (GeneratorConfig::low_spec(), EngineConfig::default())
    };

    let renderer = Rc::new(RefCell::new(create_renderer(&canvas, &query)?));
    if let Ok(hud_renderer) = element::<HtmlElement>(&document, "hud-renderer") {
        hud_renderer.set_text_content(Some(renderer.borrow().label()));
    }
    let source = LocalChunkSource::with_telemetry(
        SimpleNoiseProvider::new(),
        WORLD_SEED,
        generator_config,
        &CONSOLE_TELEMETRY,
    );
    let engine = Rc::new(RefCell::new(Engine::new(
        engine_config,
        Box::new(SharedRenderer(renderer.clone())),
        Box::new(source),
    )));
    let input = Rc::new(RefCell::new(InputCollector::new()));

    attach_input_listeners(&document, &canvas, &overlay, &input)?;
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

    // Pointer-lock state drives both the input adapter and overlay visibility.
    {
        let input = input.clone();
        let doc_for_closure = document.clone();
        let overlay = overlay.clone();
        let closure = Closure::<dyn FnMut()>::new(move || {
            let locked = doc_for_closure.pointer_lock_element().is_some();
            input.borrow_mut().set_locked(locked);
            let _ = overlay
                .style()
                .set_property("display", if locked { "none" } else { "flex" });
        });
        document
            .add_event_listener_with_callback("pointerlockchange", closure.as_ref().unchecked_ref())?;
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
        // window * governor scale * renderer base factor.
        {
            let scale = engine.borrow().stats().resolution_scale as f64
                * renderer.borrow().resolution_factor();
            let target_w = (loop_window.inner_width().ok().and_then(|v| v.as_f64()).unwrap_or(800.0)
                * scale)
                .max(1.0) as u32;
            let target_h = (loop_window.inner_height().ok().and_then(|v| v.as_f64()).unwrap_or(600.0)
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

        draw_minimap(&minimap_ctx, &minimap, &engine.borrow());

        // HUD refresh at a fixed frame cadence.
        frame_count.set(frame_count.get() + 1);
        if frame_count.get() % HUD_INTERVAL == 0 {
            let elapsed = (time_ms - hud_window_start.get()) / 1000.0;
            if elapsed > 0.0 {
                let fps = (HUD_INTERVAL as f64 / elapsed).round();
                hud_fps.set_text_content(Some(&fps.to_string()));
            }
            hud_window_start.set(time_ms);
            hud_chunks.set_text_content(Some(&stats.resident_chunks.to_string()));
            hud_nodes.set_text_content(Some(&stats.atlas_nodes.to_string()));
            hud_scale.set_text_content(Some(&format!("{:.0}%", stats.resolution_scale * 100.0)));
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
