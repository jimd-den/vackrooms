//! Complete analytic-fixture upload shared by the surface and raymarch paths.
//!
//! A texture is used instead of fragment-uniform arrays for two reasons: the
//! physical scene is not silently truncated to a camera-ranked slot budget,
//! and the large raymarch chunk table no longer competes with lights for the
//! WebGL minimum uniform-vector allowance. Each light occupies three RGBA32F
//! texels in the layout decoded by `shaders/sample_scene_lights.rs`.

use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as Gl, WebGlTexture};

use crate::application::ports::LightSource;

const PREFERRED_ROW_TEXELS: i32 = 1024;
const TEXELS_PER_LIGHT: usize = 3;

pub struct SceneLightTexture {
    texture: WebGlTexture,
    width: i32,
    maximum_size: i32,
    staging: Vec<f32>,
    last_lights: Vec<LightSource>,
    uploaded: bool,
}

impl SceneLightTexture {
    pub fn create(gl: &Gl) -> Result<Self, JsValue> {
        let texture = gl
            .create_texture()
            .ok_or_else(|| JsValue::from_str("failed to create scene-light texture"))?;
        let maximum_size = gl
            .get_parameter(Gl::MAX_TEXTURE_SIZE)?
            .as_f64()
            .unwrap_or(0.0) as i32;
        if maximum_size < 1 {
            return Err(JsValue::from_str("WebGL reported no 2-D texture capacity"));
        }
        let width = PREFERRED_ROW_TEXELS.min(maximum_size);
        gl.bind_texture(Gl::TEXTURE_2D, Some(&texture));
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
        gl.bind_texture(Gl::TEXTURE_2D, None);
        Ok(Self {
            texture,
            width,
            maximum_size,
            staging: Vec::new(),
            last_lights: Vec::new(),
            uploaded: false,
        })
    }

    /// Replaces the immutable frame snapshot and returns its exact light count.
    pub fn upload(&mut self, gl: &Gl, lights: &[LightSource]) -> Result<i32, JsValue> {
        if self.uploaded && self.last_lights == lights {
            return Ok(lights.len().min(i32::MAX as usize) as i32);
        }
        let texel_count = lights.len().max(1) * TEXELS_PER_LIGHT;
        let rows = (texel_count as i32 + self.width - 1) / self.width;
        if rows > self.maximum_size || lights.len() > i32::MAX as usize {
            return Err(JsValue::from_str(
                "resident fixture list exceeds the WebGL scene-light texture",
            ));
        }

        self.staging.clear();
        self.staging
            .resize(self.width as usize * rows as usize * 4, 0.0);
        for (index, light) in lights.iter().enumerate() {
            let offset = index * TEXELS_PER_LIGHT * 4;
            self.staging[offset..offset + 3].copy_from_slice(&light.position);
            self.staging[offset + 3] = light.kind as u8 as f32;
            self.staging[offset + 4..offset + 7].copy_from_slice(&light.color);
            self.staging[offset + 7] = light.intensity;
            self.staging[offset + 8] = light.radius;
            self.staging[offset + 9] = light.half_size[0];
            self.staging[offset + 10] = light.half_size[1];
        }

        gl.bind_texture(Gl::TEXTURE_2D, Some(&self.texture));
        gl.pixel_storei(Gl::UNPACK_ALIGNMENT, 4);
        let view = js_sys::Float32Array::from(self.staging.as_slice());
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            Gl::TEXTURE_2D,
            0,
            Gl::RGBA32F as i32,
            self.width,
            rows,
            0,
            Gl::RGBA,
            Gl::FLOAT,
            Some(&view),
        )?;
        gl.bind_texture(Gl::TEXTURE_2D, None);
        self.last_lights.clear();
        self.last_lights.extend_from_slice(lights);
        self.uploaded = true;
        Ok(lights.len() as i32)
    }

    pub fn bind(&self, gl: &Gl) {
        gl.bind_texture(Gl::TEXTURE_2D, Some(&self.texture));
    }

    pub fn width(&self) -> i32 {
        self.width
    }
}
