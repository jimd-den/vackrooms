//! Baked 3D light-volume upload, shared by the surface and splat drivers.
//!
//! The WebGPU migration removed baked volumes from generation (`rt_bake`
//! always defaulted off), but the GLSL programs still declare and bind the
//! sampler. Until the bake path returns, every chunk shares one black 1³
//! texel so the shaders stay valid and the toggle simply has no effect.

use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as Gl, WebGlTexture};

use crate::application::ports::SurfaceMeshPayload;

/// Uploads a chunk's 3D irradiance probe volume texture (or 1x1x1 black fallback).
///
/// Texture sampling uses `LINEAR` minification and magnification filters, with `CLAMP_TO_EDGE`
/// wrapping so boundary probe samples interpolate smoothly across adjacent chunk borders.
pub fn upload_light_volume(gl: &Gl, mesh: &SurfaceMeshPayload) -> Result<WebGlTexture, JsValue> {
    let texture = gl
        .create_texture()
        .ok_or_else(|| JsValue::from_str("create light texture"))?;
    gl.bind_texture(Gl::TEXTURE_3D, Some(&texture));
    gl.pixel_storei(Gl::UNPACK_ALIGNMENT, 1);

    let fallback = [0u8, 0u8, 0u8];
    let (width, height, depth, bytes) = if !mesh.light_volume_bytes.is_empty()
        && mesh.light_volume_dims[0] > 0
        && mesh.light_volume_dims[1] > 0
        && mesh.light_volume_dims[2] > 0
    {
        (
            mesh.light_volume_dims[0] as i32,
            mesh.light_volume_dims[1] as i32,
            mesh.light_volume_dims[2] as i32,
            mesh.light_volume_bytes.as_slice(),
        )
    } else {
        (1, 1, 1, fallback.as_slice())
    };

    gl.tex_image_3d_with_opt_u8_array(
        Gl::TEXTURE_3D,
        0,
        Gl::RGB8 as i32,
        width,
        height,
        depth,
        0,
        Gl::RGB,
        Gl::UNSIGNED_BYTE,
        Some(bytes),
    )?;
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_MIN_FILTER, Gl::LINEAR as i32);
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_MAG_FILTER, Gl::LINEAR as i32);
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_R, Gl::CLAMP_TO_EDGE as i32);
    gl.bind_texture(Gl::TEXTURE_3D, None);
    Ok(texture)
}
