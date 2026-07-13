//! RGBA32UI SVO node-atlas lifecycle.
//!
//! The serializer pads every atlas row to [`ATLAS_WIDTH`], which makes both a
//! full replacement and an incremental row overwrite exact typed-array copies.

use web_sys::{WebGl2RenderingContext as Gl, WebGlTexture};

/// Must match the raymarch shader's node-index decoder and the core
/// `OctreeGpuSerializer` row padding.
const ATLAS_WIDTH: i32 = 1024;

pub(super) struct AtlasTexture {
    texture: Option<WebGlTexture>,
    /// Allocated atlas height, used to reject out-of-bounds row patches.
    rows: i32,
}

impl AtlasTexture {
    pub(super) fn new() -> Self {
        Self {
            texture: None,
            rows: 0,
        }
    }

    /// Replaces the whole atlas. An empty stream intentionally releases the
    /// previous texture so level transitions cannot sample stale nodes.
    pub(super) fn replace(&mut self, gl: &Gl, texels: &[u32]) {
        if let Some(old) = self.texture.take() {
            gl.delete_texture(Some(&old));
        }
        self.rows = 0;
        if texels.is_empty() {
            return;
        }

        debug_assert_eq!(texels.len() % 4, 0, "atlas texels are RGBA u32 values");
        let texel_count = (texels.len() / 4) as i32;
        let rows = (texel_count + ATLAS_WIDTH - 1) / ATLAS_WIDTH;
        let texture = gl.create_texture().expect("create SVO atlas texture");

        gl.bind_texture(Gl::TEXTURE_2D, Some(&texture));
        gl.pixel_storei(Gl::UNPACK_ALIGNMENT, 4);
        let view = js_sys::Uint32Array::from(texels);
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            Gl::TEXTURE_2D,
            0,
            Gl::RGBA32UI as i32,
            ATLAS_WIDTH,
            rows,
            0,
            Gl::RGBA_INTEGER,
            Gl::UNSIGNED_INT,
            Some(&view),
        )
        .expect("SVO atlas texture upload failed");
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
        gl.bind_texture(Gl::TEXTURE_2D, None);

        self.texture = Some(texture);
        self.rows = rows;
    }

    /// Patches complete atlas rows without reallocating the texture.
    pub(super) fn update_rows(&mut self, gl: &Gl, first_row: u32, texels: &[u32]) -> bool {
        let row_stride = 4 * ATLAS_WIDTH as usize;
        let patch_rows = texels.len() / row_stride;
        if self.texture.is_none()
            || patch_rows == 0
            || texels.len() % row_stride != 0
            || first_row as i32 + patch_rows as i32 > self.rows
        {
            return false;
        }

        gl.bind_texture(Gl::TEXTURE_2D, self.texture.as_ref());
        let view = js_sys::Uint32Array::from(texels);
        let uploaded = gl
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
                Gl::TEXTURE_2D,
                0,
                0,
                first_row as i32,
                ATLAS_WIDTH,
                patch_rows as i32,
                Gl::RGBA_INTEGER,
                Gl::UNSIGNED_INT,
                Some(&view),
            )
            .is_ok();
        gl.bind_texture(Gl::TEXTURE_2D, None);
        uploaded
    }

    pub(super) fn bind(&self, gl: &Gl) {
        gl.bind_texture(Gl::TEXTURE_2D, self.texture.as_ref());
    }
}
