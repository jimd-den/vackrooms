//! Depth-only render target for the hero-light shadow map.

use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as Gl, WebGlFramebuffer, WebGlTexture};

pub struct ShadowTarget {
    pub fbo: WebGlFramebuffer,
    pub texture: WebGlTexture,
    pub size: i32,
}

/// Creates a `size`² DEPTH_COMPONENT16 texture bound to a framebuffer's
/// depth attachment. NEAREST filtering: the shaders do their own PCF.
pub fn create_shadow_target(gl: &Gl, size: i32) -> Result<ShadowTarget, JsValue> {
    let texture = gl
        .create_texture()
        .ok_or_else(|| JsValue::from_str("create shadow texture"))?;
    gl.bind_texture(Gl::TEXTURE_2D, Some(&texture));
    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        Gl::TEXTURE_2D,
        0,
        Gl::DEPTH_COMPONENT16 as i32,
        size,
        size,
        0,
        Gl::DEPTH_COMPONENT,
        Gl::UNSIGNED_SHORT,
        None,
    )?;
    gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
    gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
    gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
    gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);

    let fbo = gl
        .create_framebuffer()
        .ok_or_else(|| JsValue::from_str("create shadow FBO"))?;
    gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&fbo));
    gl.framebuffer_texture_2d(
        Gl::FRAMEBUFFER,
        Gl::DEPTH_ATTACHMENT,
        Gl::TEXTURE_2D,
        Some(&texture),
        0,
    );

    // A depth-only WebGL2 target has no color attachment. Explicitly route
    // both drawing and reading to NONE; leaving the defaults at
    // COLOR_ATTACHMENT0 makes strict mobile drivers report an incomplete
    // framebuffer and silently discard the shadow pass.
    let draw_buffers = js_sys::Array::new();
    draw_buffers.push(&wasm_bindgen::JsValue::from_f64(Gl::NONE as f64));
    gl.draw_buffers(draw_buffers.as_ref());
    gl.read_buffer(Gl::NONE);

    let status = gl.check_framebuffer_status(Gl::FRAMEBUFFER);
    if status != Gl::FRAMEBUFFER_COMPLETE {
        gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
        gl.delete_framebuffer(Some(&fbo));
        gl.delete_texture(Some(&texture));
        return Err(JsValue::from_str(&format!(
            "shadow framebuffer is incomplete (status 0x{status:04x})"
        )));
    }
    gl.bind_framebuffer(Gl::FRAMEBUFFER, None);

    Ok(ShadowTarget { fbo, texture, size })
}
