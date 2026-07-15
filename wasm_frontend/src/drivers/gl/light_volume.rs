//! Baked 3D light-volume upload, shared by the surface and splat drivers.

use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as Gl, WebGlTexture};

use crate::application::ports::SurfaceMeshPayload;

/// Uploads a chunk's baked RGB light volume as a linearly-filtered 3D
/// texture (the fragment shaders sample it at world positions).
pub fn upload_light_volume(gl: &Gl, mesh: &SurfaceMeshPayload) -> Result<WebGlTexture, JsValue> {
    let texture = gl
        .create_texture()
        .ok_or_else(|| JsValue::from_str("create light texture"))?;
    gl.bind_texture(Gl::TEXTURE_3D, Some(&texture));
    // RGB rows are width*3 bytes — rarely 4-aligned, and the default
    // unpack alignment of 4 makes texImage3D reject the upload.
    gl.pixel_storei(Gl::UNPACK_ALIGNMENT, 1);
    gl.tex_image_3d_with_opt_u8_array(
        Gl::TEXTURE_3D,
        0,
        Gl::RGB8 as i32,
        mesh.light_volume_size[0] as i32,
        mesh.light_volume_size[1] as i32,
        mesh.light_volume_size[2] as i32,
        0,
        Gl::RGB,
        Gl::UNSIGNED_BYTE,
        Some(&mesh.light_volume),
    )?;
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_MIN_FILTER, Gl::LINEAR as i32);
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_MAG_FILTER, Gl::LINEAR as i32);
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
    gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_R, Gl::CLAMP_TO_EDGE as i32);
    gl.bind_texture(Gl::TEXTURE_3D, None);
    Ok(texture)
}
