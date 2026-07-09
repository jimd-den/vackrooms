//! WebGL2 renderer driver: the concrete implementation of the application's
//! [`RendererPort`]. Owns the GL context, shader program, fullscreen quad and
//! the RGBA32UI SVO node-atlas texture.

use wasm_bindgen::prelude::*;
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlProgram, WebGlShader, WebGlTexture,
    WebGlUniformLocation, WebGlVertexArrayObject,
};

use crate::application::atlas::MAX_CHUNKS;
use crate::application::ports::{ChunkDraw, FrameParams, RendererPort};
use crate::drivers::shaders::{FRAGMENT_SHADER, VERTEX_SHADER};

use std::cell::RefCell;

thread_local! {
    static FACE_WEIGHTS: RefCell<(f32, f32, f32, f32)> = RefCell::new((0.55, 1.0, 0.8, 0.7));
    /// tan(FOV/2). 0.767 = the 75-degree default.
    static FOV_TAN: RefCell<f32> = RefCell::new(0.767);
}

#[wasm_bindgen]
pub fn set_face_weights(top: f32, bottom: f32, x: f32, z: f32) {
    FACE_WEIGHTS.with(|w| {
        *w.borrow_mut() = (top, bottom, x, z);
    });
}

/// Sets the vertical field of view in degrees (clamped to 40–110).
#[wasm_bindgen]
pub fn set_fov(degrees: f32) {
    let half = degrees.clamp(40.0, 110.0).to_radians() * 0.5;
    FOV_TAN.with(|f| *f.borrow_mut() = half.tan());
}

/// Row width of the node atlas texture (texels). Must match both the shader's
/// `decodeNode` constant and the core `OctreeGpuSerializer` row padding.
const ATLAS_WIDTH: i32 = 1024;

/// Uniform locations resolved once at program link time.
struct Uniforms {
    camera_position: Option<WebGlUniformLocation>,
    cam_right: Option<WebGlUniformLocation>,
    cam_up: Option<WebGlUniformLocation>,
    cam_forward: Option<WebGlUniformLocation>,
    aspect: Option<WebGlUniformLocation>,
    fov_tan: Option<WebGlUniformLocation>,

    face_weight_top: Option<WebGlUniformLocation>,
    face_weight_bottom: Option<WebGlUniformLocation>,
    face_weight_x: Option<WebGlUniformLocation>,
    face_weight_z: Option<WebGlUniformLocation>,

    flashlight: Option<WebGlUniformLocation>,

    node_texture: Option<WebGlUniformLocation>,
    num_chunks: Option<WebGlUniformLocation>,
    chunk_origins: Option<WebGlUniformLocation>,
    chunk_root_indices: Option<WebGlUniformLocation>,
    chunk_world_sizes: Option<WebGlUniformLocation>,
}

pub struct WebGl2Renderer {
    gl: Gl,
    _program: WebGlProgram,
    vao: WebGlVertexArrayObject,
    uniforms: Uniforms,
    texture: Option<WebGlTexture>,
    /// Allocated height (rows) of the atlas texture, for bounds-checking
    /// partial row updates.
    texture_rows: i32,
    width: i32,
    height: i32,
    // Reused per-frame upload buffers so the draw path never allocates.
    origins_buf: Vec<f32>,
    roots_buf: Vec<i32>,
    sizes_buf: Vec<f32>,
}

impl WebGl2Renderer {
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, JsValue> {
        // No depth buffer and no MSAA: the fullscreen raymarcher resolves
        // visibility itself, so both would be wasted bandwidth on low-spec GPUs.
        let context_options = js_sys::Object::new();
        js_sys::Reflect::set(&context_options, &"antialias".into(), &false.into())?;
        js_sys::Reflect::set(&context_options, &"depth".into(), &false.into())?;

        let gl = canvas
            .get_context_with_context_options("webgl2", &context_options)?
            .ok_or_else(|| JsValue::from_str("WebGL2 is not supported on this browser"))?
            .dyn_into::<Gl>()?;

        let program = link_program(&gl, VERTEX_SHADER, FRAGMENT_SHADER)?;
        gl.use_program(Some(&program));

        let uniforms = Uniforms {
            camera_position: gl.get_uniform_location(&program, "uCameraPosition"),
            cam_right: gl.get_uniform_location(&program, "uCamRight"),
            cam_up: gl.get_uniform_location(&program, "uCamUp"),
            cam_forward: gl.get_uniform_location(&program, "uCamForward"),
            aspect: gl.get_uniform_location(&program, "uAspect"),
            fov_tan: gl.get_uniform_location(&program, "uFovTan"),

            face_weight_top: gl.get_uniform_location(&program, "uFaceWeightTop"),
            face_weight_bottom: gl.get_uniform_location(&program, "uFaceWeightBottom"),
            face_weight_x: gl.get_uniform_location(&program, "uFaceWeightX"),
            face_weight_z: gl.get_uniform_location(&program, "uFaceWeightZ"),

            flashlight: gl.get_uniform_location(&program, "uFlashlightEnabled"),

            node_texture: gl.get_uniform_location(&program, "uNodeTexture"),
            num_chunks: gl.get_uniform_location(&program, "uNumChunks"),
            chunk_origins: gl.get_uniform_location(&program, "uChunkOrigins"),
            chunk_root_indices: gl.get_uniform_location(&program, "uChunkRootIndices"),
            chunk_world_sizes: gl.get_uniform_location(&program, "uChunkWorldSizes"),
        };

        // Fullscreen quad as two triangles.
        let vao = gl
            .create_vertex_array()
            .ok_or_else(|| JsValue::from_str("failed to create VAO"))?;
        gl.bind_vertex_array(Some(&vao));

        let buffer = gl
            .create_buffer()
            .ok_or_else(|| JsValue::from_str("failed to create vertex buffer"))?;
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&buffer));
        let vertices: [f32; 12] = [
            -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
        ];
        let vertex_view = js_sys::Float32Array::from(vertices.as_slice());
        gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &vertex_view, Gl::STATIC_DRAW);

        let position_loc = gl.get_attrib_location(&program, "position") as u32;
        gl.enable_vertex_attrib_array(position_loc);
        gl.vertex_attrib_pointer_with_i32(position_loc, 2, Gl::FLOAT, false, 0, 0);
        gl.bind_vertex_array(None);

        let width = canvas.width() as i32;
        let height = canvas.height() as i32;

        Ok(Self {
            gl,
            _program: program,
            vao,
            uniforms,
            texture: None,
            texture_rows: 0,
            width,
            height,
            origins_buf: Vec::with_capacity(MAX_CHUNKS * 3),
            roots_buf: Vec::with_capacity(MAX_CHUNKS),
            sizes_buf: Vec::with_capacity(MAX_CHUNKS),
        })
    }

    /// Called by the browser driver when the canvas backing store changes
    /// (window resize or adaptive resolution step).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }
}

impl RendererPort for WebGl2Renderer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        let gl = &self.gl;

        if let Some(old) = self.texture.take() {
            gl.delete_texture(Some(&old));
        }
        if texels.is_empty() {
            return;
        }

        let total_texels = (texels.len() / 4) as i32;
        let height = (total_texels + ATLAS_WIDTH - 1) / ATLAS_WIDTH;

        let texture = gl.create_texture();
        gl.bind_texture(Gl::TEXTURE_2D, texture.as_ref());
        gl.pixel_storei(Gl::UNPACK_ALIGNMENT, 4);

        // The atlas is row-padded upstream, so the stream always fills
        // width * height * 4 exactly.
        let view = js_sys::Uint32Array::from(texels);
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            Gl::TEXTURE_2D,
            0,
            Gl::RGBA32UI as i32,
            ATLAS_WIDTH,
            height,
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

        self.texture = texture;
        self.texture_rows = height;
    }

    fn upload_atlas_rows(&mut self, first_row: u32, texels: &[u32]) -> bool {
        let row_stride = 4 * ATLAS_WIDTH as usize;
        let rows = texels.len() / row_stride;
        if self.texture.is_none()
            || rows == 0
            || texels.len() % row_stride != 0
            || first_row as i32 + rows as i32 > self.texture_rows
        {
            return false;
        }
        let gl = &self.gl;
        gl.bind_texture(Gl::TEXTURE_2D, self.texture.as_ref());
        let view = js_sys::Uint32Array::from(texels);
        let ok = gl
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
                Gl::TEXTURE_2D,
                0,
                0,
                first_row as i32,
                ATLAS_WIDTH,
                rows as i32,
                Gl::RGBA_INTEGER,
                Gl::UNSIGNED_INT,
                Some(&view),
            )
            .is_ok();
        gl.bind_texture(Gl::TEXTURE_2D, None);
        ok
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        let gl = &self.gl;

        gl.viewport(0, 0, self.width, self.height);
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.clear(Gl::COLOR_BUFFER_BIT);

        gl.uniform3f(
            self.uniforms.camera_position.as_ref(),
            frame.camera_pos[0],
            frame.camera_pos[1],
            frame.camera_pos[2],
        );
        // Camera basis from yaw/pitch, computed once per frame instead of
        // per pixel in the fragment shader.
        let (sp, cp) = frame.pitch.sin_cos();
        let (sy, cy) = frame.yaw.sin_cos();
        gl.uniform3f(self.uniforms.cam_right.as_ref(), cy, 0.0, -sy);
        gl.uniform3f(self.uniforms.cam_up.as_ref(), sp * sy, cp, sp * cy);
        gl.uniform3f(self.uniforms.cam_forward.as_ref(), -cp * sy, sp, -cp * cy);
        gl.uniform1f(
            self.uniforms.aspect.as_ref(),
            self.width as f32 / self.height.max(1) as f32,
        );
        FOV_TAN.with(|f| gl.uniform1f(self.uniforms.fov_tan.as_ref(), *f.borrow()));

        FACE_WEIGHTS.with(|w| {
            let (top, bottom, x, z) = *w.borrow();
            gl.uniform1f(self.uniforms.face_weight_top.as_ref(), top);
            gl.uniform1f(self.uniforms.face_weight_bottom.as_ref(), bottom);
            gl.uniform1f(self.uniforms.face_weight_x.as_ref(), x);
            gl.uniform1f(self.uniforms.face_weight_z.as_ref(), z);
        });

        gl.uniform1i(self.uniforms.flashlight.as_ref(), if frame.flashlight { 1 } else { 0 });

        let count = chunks.len().min(MAX_CHUNKS);
        gl.uniform1i(self.uniforms.num_chunks.as_ref(), count as i32);

        if count > 0 {
            self.origins_buf.clear();
            self.roots_buf.clear();
            self.sizes_buf.clear();
            for chunk in &chunks[..count] {
                self.origins_buf.extend_from_slice(&chunk.origin);
                self.roots_buf.push(chunk.root_index);
                self.sizes_buf.push(chunk.world_size);
            }
            gl.uniform3fv_with_f32_array(self.uniforms.chunk_origins.as_ref(), &self.origins_buf);
            gl.uniform1iv_with_i32_array(
                self.uniforms.chunk_root_indices.as_ref(),
                &self.roots_buf,
            );
            gl.uniform1fv_with_f32_array(self.uniforms.chunk_world_sizes.as_ref(), &self.sizes_buf);
        }

        gl.active_texture(Gl::TEXTURE0);
        gl.bind_texture(Gl::TEXTURE_2D, self.texture.as_ref());
        gl.uniform1i(self.uniforms.node_texture.as_ref(), 0);

        gl.bind_vertex_array(Some(&self.vao));
        gl.draw_arrays(Gl::TRIANGLES, 0, 6);
        gl.bind_vertex_array(None);
    }
}

fn compile_shader(gl: &Gl, kind: u32, source: &str) -> Result<WebGlShader, JsValue> {
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

fn link_program(gl: &Gl, vertex_src: &str, fragment_src: &str) -> Result<WebGlProgram, JsValue> {
    let vs = compile_shader(gl, Gl::VERTEX_SHADER, vertex_src)?;
    let fs = compile_shader(gl, Gl::FRAGMENT_SHADER, fragment_src)?;

    let program = gl
        .create_program()
        .ok_or_else(|| JsValue::from_str("failed to create program"))?;
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
        Err(JsValue::from_str(&format!("program link error: {log}")))
    }
}
