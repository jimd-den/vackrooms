//! Browser runtime driver: DOM/event plumbing, the requestAnimationFrame
//! loop, pointer lock and the HUD.
//!
//! This module is the composition root's workhorse: it instantiates the
//! concrete adapters/drivers, hands them to `application::engine::Engine`,
//! and from then on only shuttles plain data across the port boundaries.

mod input_bindings;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{Document, HtmlCanvasElement, HtmlElement, Window};

use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;

use crate::adapters::input::InputCollector;
use crate::adapters::local_chunk_source::LocalChunkSource;
use crate::adapters::query_config::{
    generator_setup_for_quality, quality_profile_from_query, query_param,
};
use crate::adapters::section_locator::SectionLocator;
use crate::application::engine::{Engine, EngineConfig};
use crate::application::generation_worker_policy::parse_generation_worker_preference;
use crate::application::ports::{
    ChunkDraw, FrameParams, RenderArtifactNeeds, RendererPort, SurfaceChunk, SurfaceChunkKey,
};
use crate::drivers::console_telemetry::CONSOLE_TELEMETRY;
use crate::drivers::webgpu::config::{GpuQualityProfile, RendererKind};
use crate::drivers::webgpu::renderer::WebGpuRenderer;

use input_bindings::{TouchState, attach_input_listeners, attach_touch_listeners};

/// World seed shared with the native server so both render the same world.
const WORLD_SEED: u32 = 42;
/// HUD refresh cadence in frames.
const HUD_INTERVAL: u32 = 30;

type DriverRenderer = WebGpuRenderer;

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

/// Engine owns its renderer behind the port; the driver also needs to call
/// `resize` on the concrete type. This thin adapter shares one renderer
/// between both without widening the port.
struct SharedRenderer(Rc<RefCell<DriverRenderer>>);

impl RendererPort for SharedRenderer {
    fn artifact_needs(&self) -> RenderArtifactNeeds {
        self.0.borrow().artifact_needs()
    }

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

/// Renderer selection: surfaces by default, `?renderer=splat` for the
/// instanced face-splat path (default candidate once parity/perf is
/// confirmed), `?renderer=raymarch` for the retained SVO debug path, and
/// `?renderer=cpu` for deterministic software rasterization with WebGPU
/// presentation.
async fn create_renderer(
    canvas: &HtmlCanvasElement,
    query: &str,
    quality: GpuQualityProfile,
) -> Result<DriverRenderer, JsValue> {
    let kind = query_param(query, "renderer")
        .and_then(RendererKind::parse)
        .unwrap_or_default();
    WebGpuRenderer::new(canvas, kind.profile(quality)).await
}

pub async fn boot() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;

    let canvas: HtmlCanvasElement =
        element(&document, "view").or_else(|_| element(&document, "game-canvas"))?;
    let overlay: HtmlElement = element(&document, "overlay")?;
    let status_msg: HtmlElement = element(&document, "status-msg")?;
    let play_msg: HtmlElement = element(&document, "play-msg")?;

    // Preflight: every render path (including the CPU rasterizer's
    // presentation pass) requires WebGPU, and `canvas.getContext("webgpu")`
    // failing deep inside surface creation produces an unactionable error.
    // Checking `navigator.gpu` here turns that into concrete guidance.
    let has_webgpu = js_sys::Reflect::get(&window.navigator(), &JsValue::from_str("gpu"))
        .map(|gpu| !gpu.is_undefined() && !gpu.is_null())
        .unwrap_or(false);
    if !has_webgpu {
        let msg = "WebGPU is not available in this browser (navigator.gpu is missing). \
                   Use Chrome/Edge 113+ or Safari 18+, or in Firefox set \
                   dom.webgpu.enabled=true in about:config and restart.";
        status_msg.set_text_content(Some(msg));
        return Err(JsValue::from_str(msg));
    }

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
    let quality = quality_profile_from_query(&query);
    let (resolved_seed, generator_config) =
        generator_setup_for_quality(&query, WORLD_SEED, quality);
    let high_spec = quality == GpuQualityProfile::High;
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

    let renderer = Rc::new(RefCell::new(
        create_renderer(&canvas, &query, quality).await?,
    ));
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

    attach_input_listeners(&document, &canvas, &overlay, &input, &touch)?;
    attach_touch_listeners(&document, &canvas, &overlay, &input, &touch)?;
    run_frame_loop(
        window, document, canvas, overlay, status_msg, play_msg, renderer, engine, input, locator,
    )
}

/// Renders the engine's `§`-sectioned diagnostic report as aperture panels.
/// Body lines are escaped; only the fixed panel scaffolding is markup.
fn diagnostic_sections_html(report: &str) -> String {
    fn escape(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }
    let mut html = String::with_capacity(report.len() + 256);
    let mut open = false;
    for line in report.lines() {
        if let Some(title) = line.strip_prefix("§ ") {
            if open {
                html.push_str("</pre></section>");
            }
            html.push_str("<section class=\"diag-section\"><h4>");
            html.push_str(&escape(title));
            html.push_str("</h4><pre>");
            open = true;
        } else if open {
            html.push_str(&escape(line));
            html.push('\n');
        } else {
            // Preamble lines before the first section header.
            html.push_str("<section class=\"diag-section\"><pre>");
            html.push_str(&escape(line));
            html.push('\n');
            open = true;
        }
    }
    if open {
        html.push_str("</pre></section>");
    }
    html
}

fn element<T: JsCast>(document: &Document, id: &str) -> Result<T, JsValue> {
    document
        .get_element_by_id(id)
        .ok_or_else(|| JsValue::from_str(&format!("missing #{id} element")))?
        .dyn_into::<T>()
        .map_err(|_| JsValue::from_str(&format!("#{id} has unexpected element type")))
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
    let document = window.document().unwrap();
    // Engine metrics live inside the diagnostic aperture in the redesigned
    // shell; the ids are unchanged so older/embedded shells keep working.
    let hud_fps: HtmlElement = element(&document, "hud-fps")?;
    let hud_chunks: HtmlElement = element(&document, "hud-chunks")?;
    let hud_nodes: HtmlElement = element(&document, "hud-nodes")?;
    let hud_scale: HtmlElement = element(&document, "hud-scale")?;
    let hud_gpu: HtmlElement = element(&document, "hud-gpu")?;
    // Top-right section readout and the diagnostic aperture; both are
    // optional page elements so older/embedded shells keep working.
    let hud_section: Option<HtmlElement> = element(&document, "hud-section").ok();
    let last_section_text = Rc::new(RefCell::new(String::new()));
    let debug_overlay: Option<HtmlElement> = element(&document, "debug-overlay").ok();
    // The aperture wrapper also holds the always-in-DOM engine channels;
    // visibility toggles on it when present (older shells fall back to the
    // report element itself).
    let diagnostics_shell: Option<HtmlElement> = element(&document, "diagnostics").ok();
    let debug_overlay_visible = Rc::new(Cell::new(false));

    // Body trace: heart glyph container, pulse figure, step count, carried
    // supply ticks. All optional; text/attributes only change when the
    // presented value materially changes.
    let body_trace: Option<HtmlElement> = element(&document, "body-trace").ok();
    let hud_bpm: Option<HtmlElement> = element(&document, "hud-bpm").ok();
    let hud_steps: Option<HtmlElement> = element(&document, "hud-steps").ok();
    let supply_water: Option<HtmlElement> = element(&document, "supply-water").ok();
    let supply_ration: Option<HtmlElement> = element(&document, "supply-ration").ok();
    let last_steps = Rc::new(Cell::new(u64::MAX));
    let last_bpm_text = Rc::new(RefCell::new(String::new()));
    let last_signal = Rc::new(RefCell::new(String::new()));
    let last_beat_period_ms = Rc::new(Cell::new(0i32));
    let last_supplies = Rc::new(Cell::new((u32::MAX, u32::MAX)));

    // Route anchor near the reticle; hidden whenever no real door is known.
    let route_anchor: Option<HtmlElement> = element(&document, "route-anchor").ok();
    let route_target_el: Option<HtmlElement> = element(&document, "route-target").ok();
    let route_bearing_el: Option<HtmlElement> = element(&document, "route-bearing").ok();
    let route_range_el: Option<HtmlElement> = element(&document, "route-range").ok();
    let route_visible = Rc::new(Cell::new(false));
    let last_route_text = Rc::new(RefCell::new(String::new()));

    let death_overlay: Option<HtmlElement> = element(&document, "death-overlay").ok();
    let last_deaths = Rc::new(Cell::new(0u32));
    let death_shown_at = Rc::new(Cell::new(0.0f64));

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
            hud_window_start.set(time_ms);
            // Engine channels live inside the diagnostic aperture but stay
            // current even while it is closed.
            if elapsed > 0.0 {
                let fps = (HUD_INTERVAL as f64 / elapsed).round();
                hud_fps.set_text_content(Some(&fps.to_string()));
            }
            hud_chunks.set_text_content(Some(&format!(
                "{}/{}",
                stats.fine_chunks, stats.resident_chunks
            )));
            hud_nodes.set_text_content(Some(&stats.atlas_nodes.to_string()));
            let effective_scale =
                effective_resolution_scale(&renderer.borrow(), stats.resolution_scale);
            hud_scale.set_text_content(Some(&format!("{:.0}%", effective_scale * 100.0)));
            let cpu_telemetry = renderer.borrow().cpu_telemetry_string();
            if let Some(text) = cpu_telemetry {
                hud_gpu.set_text_content(Some(&text));
            } else {
                hud_gpu.set_text_content(Some("WebGPU"));
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

            // Accessibility switch is a page-level preference; forward it to
            // the engine on the same bounded cadence as everything else.
            engine.borrow_mut().set_assisted_consumption(
                crate::ASSISTED_CONSUMPTION.load(std::sync::atomic::Ordering::Relaxed),
            );

            // Body trace: the beat is driven entirely by CSS (period and
            // amplitude custom properties), so the DOM is only touched when
            // a presented value materially changes.
            let body = stats.body;
            if let Some(trace) = &body_trace {
                let signal = match body.signal {
                    crate::application::body::PulseSignal::Quiet => "quiet",
                    crate::application::body::PulseSignal::Active => "active",
                    crate::application::body::PulseSignal::Strained => "strained",
                    crate::application::body::PulseSignal::Critical => "critical",
                };
                if *last_signal.borrow() != signal {
                    let _ = trace.set_attribute("data-signal", signal);
                    *last_signal.borrow_mut() = signal.to_string();
                }
                let period_ms = (60_000.0 / body.bpm.max(30.0)) as i32;
                // Re-time the animation only on a meaningful shift (~4%).
                if (period_ms - last_beat_period_ms.get()).abs() > period_ms / 25 {
                    last_beat_period_ms.set(period_ms);
                    let _ = trace
                        .style()
                        .set_property("--beat-period", &format!("{period_ms}ms"));
                    let _ = trace.style().set_property(
                        "--beat-amp",
                        &format!("{:.2}", body.beat_intensity.clamp(0.0, 1.0)),
                    );
                }
            }
            if let Some(hud_bpm) = &hud_bpm {
                // The figure appears only when the body makes it relevant.
                let text = if body.signal == crate::application::body::PulseSignal::Quiet {
                    String::new()
                } else {
                    format!("{:.0}", body.bpm)
                };
                let mut last = last_bpm_text.borrow_mut();
                if *last != text {
                    hud_bpm.set_text_content(Some(&text));
                    *last = text;
                }
            }
            if let Some(hud_steps) = &hud_steps {
                if last_steps.get() != body.steps {
                    last_steps.set(body.steps);
                    hud_steps.set_text_content(Some(&body.steps.to_string()));
                }
            }
            // Carried supplies as discrete ticks, never numeric meters.
            if last_supplies.get() != (stats.almond_bottles, stats.rations) {
                last_supplies.set((stats.almond_bottles, stats.rations));
                if let Some(el) = &supply_water {
                    el.set_text_content(Some(&"▪".repeat(stats.almond_bottles as usize)));
                }
                if let Some(el) = &supply_ration {
                    el.set_text_content(Some(&"▪".repeat(stats.rations as usize)));
                }
            }

            // Route anchor: shown only while a real resident door is known.
            if let Some(anchor) = &route_anchor {
                match stats.route {
                    Some(route) => {
                        if !route_visible.get() {
                            route_visible.set(true);
                            let _ = anchor.style().set_property("display", "block");
                        }
                        let text = format!(
                            "{}|{:03}|{:.0}",
                            route.target_level, route.bearing_deg, route.range_m
                        );
                        let mut last = last_route_text.borrow_mut();
                        if *last != text {
                            *last = text;
                            if let Some(el) = &route_target_el {
                                el.set_text_content(Some(&format!(
                                    "THRESHOLD / L{}",
                                    route.target_level
                                )));
                            }
                            if let Some(el) = &route_bearing_el {
                                el.set_text_content(Some(&format!(
                                    "VECTOR {:03}°",
                                    route.bearing_deg
                                )));
                            }
                            if let Some(el) = &route_range_el {
                                el.set_text_content(Some(&format!("RANGE {:.0} m", route.range_m)));
                            }
                        }
                    }
                    None => {
                        if route_visible.get() {
                            route_visible.set(false);
                            last_route_text.borrow_mut().clear();
                            let _ = anchor.style().set_property("display", "none");
                        }
                    }
                }
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

            // Diagnostic aperture (F3 or the settings switch): the engine's
            // sectioned report rendered as structured panels. The section
            // split is presentation only; the text itself is authored by
            // `Engine::diagnostic_text` and stays natively testable.
            if let Some(overlay_el) = &debug_overlay {
                let on = crate::ANOMALY_DEBUG.load(std::sync::atomic::Ordering::Relaxed);
                if on {
                    let report = engine.borrow().diagnostic_text();
                    overlay_el.set_inner_html(&diagnostic_sections_html(&report));
                }
                if on != debug_overlay_visible.get() {
                    debug_overlay_visible.set(on);
                    let toggled = diagnostics_shell.as_ref().unwrap_or(overlay_el);
                    let _ = toggled
                        .style()
                        .set_property("display", if on { "block" } else { "none" });
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
