//! WebGL2 context creation and shader program linking — the one copy of
//! this plumbing for all drivers.

use js_sys::Reflect;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlProgram, WebGlShader};

/// Options for [`create_context`]. Raster paths want a depth buffer; the
/// fullscreen raymarcher resolves visibility itself, so depth and MSAA
/// would be wasted bandwidth on low-spec GPUs.
#[derive(Debug, Clone, Copy)]
pub struct ContextOptions {
    pub depth: bool,
}

/// Creates a WebGL2 context with the engine's standard attributes
/// (no antialiasing, opaque, high-performance GPU preference).
pub fn create_context(
    canvas: &HtmlCanvasElement,
    options: ContextOptions,
) -> Result<Gl, JsValue> {
    let attrs = js_sys::Object::new();
    Reflect::set(&attrs, &"antialias".into(), &false.into())?;
    Reflect::set(&attrs, &"depth".into(), &options.depth.into())?;
    Reflect::set(&attrs, &"stencil".into(), &false.into())?;
    Reflect::set(&attrs, &"alpha".into(), &false.into())?;
    Reflect::set(&attrs, &"powerPreference".into(), &"high-performance".into())?;
    canvas
        .get_context_with_context_options("webgl2", &attrs)?
        .ok_or_else(|| JsValue::from_str("WebGL2 is not supported on this browser"))?
        .dyn_into::<Gl>()
        .map_err(|_| JsValue::from_str("webgl2 context has an unexpected type"))
}

pub fn compile_shader(gl: &Gl, kind: u32, source: &str) -> Result<WebGlShader, JsValue> {
    let shader = gl
        .create_shader(kind)
        .ok_or_else(|| JsValue::from_str("failed to create shader object"))?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);
    if gl
        .get_shader_parameter(&shader, Gl::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        let log = gl.get_shader_info_log(&shader).unwrap_or_default();
        gl.delete_shader(Some(&shader));
        Err(JsValue::from_str(&format!("shader compile error: {log}")))
    }
}

pub fn link_program(gl: &Gl, vertex: &str, fragment: &str) -> Result<WebGlProgram, JsValue> {
    let vs = compile_shader(gl, Gl::VERTEX_SHADER, vertex)?;
    let fs = compile_shader(gl, Gl::FRAGMENT_SHADER, fragment)?;
    let program = gl
        .create_program()
        .ok_or_else(|| JsValue::from_str("failed to create shader program"))?;
    gl.attach_shader(&program, &vs);
    gl.attach_shader(&program, &fs);
    gl.link_program(&program);
    if gl
        .get_program_parameter(&program, Gl::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        let log = gl.get_program_info_log(&program).unwrap_or_default();
        gl.delete_program(Some(&program));
        Err(JsValue::from_str(&format!("shader link error: {log}")))
    }
}
