//! Browser input bindings for keyboard, pointer-lock mouse, and touch play.
//!
//! This module owns DOM listener lifetime setup only. Raw browser events are
//! translated into the platform-free [`InputCollector`], which remains the
//! application's input boundary.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{Document, HtmlCanvasElement, HtmlElement, KeyboardEvent, MouseEvent, TouchEvent};

use crate::adapters::input::InputCollector;

use super::element;

/// Radius of the virtual joystick in CSS pixels (full deflection).
const JOYSTICK_RADIUS: f64 = 60.0;
/// Touch-look sensitivity relative to raw mouse pixels.
const TOUCH_LOOK_SCALE: f32 = 2.2;

/// Touch-play session state. On touch devices there is no pointer lock:
/// tapping the overlay enters "touch play" directly, the left half of the
/// screen is a virtual joystick (drag from touch-down point), and the right
/// half is a look surface. On-screen buttons cover flashlight and menu.
#[derive(Default)]
pub(super) struct TouchState {
    /// True while playing in touch mode.
    active: bool,
    move_id: Option<i32>,
    move_origin: (f64, f64),
    look_id: Option<i32>,
    look_last: (f64, f64),
}

pub(super) fn attach_input_listeners(
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
pub(super) fn attach_touch_listeners(
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
    if let Ok(btn) = element::<HtmlElement>(document, "btn-drink") {
        let input = input.clone();
        let closure = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
            e.prevent_default();
            e.stop_propagation();
            input.borrow_mut().queue_drink();
        });
        btn.add_event_listener_with_callback("touchend", closure.as_ref().unchecked_ref())?;
        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    if let Ok(btn) = element::<HtmlElement>(document, "btn-eat") {
        let input = input.clone();
        let closure = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
            e.prevent_default();
            e.stop_propagation();
            input.borrow_mut().queue_eat();
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
