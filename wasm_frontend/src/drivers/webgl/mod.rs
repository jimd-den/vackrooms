//! Fullscreen SVO raymarch renderer.
//!
//! The module follows the lifetime of a frame instead of collecting the whole
//! renderer in one file:
//!
//! * [`atlas`] owns the integer SVO texture and its incremental row updates;
//! * [`draw`] uploads per-frame state and implements [`RendererPort`];
//! * this file owns long-lived program, quad, viewport, and tuning state.
//!
//! [`RendererPort`]: crate::application::ports::RendererPort

mod atlas;
mod draw;

use std::cell::RefCell;

use wasm_bindgen::prelude::*;
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram,
    WebGlUniformLocation, WebGlVertexArrayObject,
};

use crate::application::atlas::MAX_CHUNKS;
use crate::drivers::gl::program::{ContextOptions, create_context, link_program};
use crate::drivers::gl::timer::GpuFrameTimer;
use crate::drivers::shaders::raymarch;

use atlas::AtlasTexture;
use draw::ChunkUniformBuffers;

thread_local! {
    /// `tan(vertical_fov / 2)`. `0.767` is the 75-degree default.
    static FOV_TAN: RefCell<f32> = RefCell::new(0.767);
}

/// Sets the vertical field of view in degrees, clamped to a usable range.
#[wasm_bindgen]
pub fn set_fov(degrees: f32) {
    let half_angle = degrees.clamp(40.0, 110.0).to_radians() * 0.5;
    FOV_TAN.with(|fov| *fov.borrow_mut() = half_angle.tan());
}

/// Shared camera FOV state for the raymarcher, raster drivers, and CPU path.
pub(crate) fn fov_tan() -> f32 {
    FOV_TAN.with(|fov| *fov.borrow())
}

/// Uniform locations are resolved once, immediately after program linking.
struct Uniforms {
    camera_position: Option<WebGlUniformLocation>,
    cam_right: Option<WebGlUniformLocation>,
    cam_up: Option<WebGlUniformLocation>,
    cam_forward: Option<WebGlUniformLocation>,
    aspect: Option<WebGlUniformLocation>,
    fov_tan: Option<WebGlUniformLocation>,
    flashlight: Option<WebGlUniformLocation>,
    front_to_back: Option<WebGlUniformLocation>,
    empty_space_skip: Option<WebGlUniformLocation>,
    baked_lighting: Option<WebGlUniformLocation>,
    dither: Option<WebGlUniformLocation>,
    outdoor: Option<WebGlUniformLocation>,
    sky_color: Option<WebGlUniformLocation>,
    fog_color: Option<WebGlUniformLocation>,
    ambient_scale: Option<WebGlUniformLocation>,
    fog_density: Option<WebGlUniformLocation>,
    fog_start: Option<WebGlUniformLocation>,
    light_count: Option<WebGlUniformLocation>,
    light_positions: Option<WebGlUniformLocation>,
    light_colors: Option<WebGlUniformLocation>,
    light_params: Option<WebGlUniformLocation>,
    light_kinds: Option<WebGlUniformLocation>,
    dynamic_light_count: Option<WebGlUniformLocation>,
    dynamic_pos_radius: Option<WebGlUniformLocation>,
    dynamic_color_intensity: Option<WebGlUniformLocation>,
    node_texture: Option<WebGlUniformLocation>,
    num_chunks: Option<WebGlUniformLocation>,
    chunk_origins: Option<WebGlUniformLocation>,
    chunk_root_indices: Option<WebGlUniformLocation>,
    chunk_world_sizes: Option<WebGlUniformLocation>,
    chunk_voxel_sizes: Option<WebGlUniformLocation>,
    chunk_depths: Option<WebGlUniformLocation>,
}

impl Uniforms {
    fn resolve(gl: &Gl, program: &WebGlProgram) -> Self {
        let uniform = |name| gl.get_uniform_location(program, name);
        Self {
            camera_position: uniform("uCameraPosition"),
            cam_right: uniform("uCamRight"),
            cam_up: uniform("uCamUp"),
            cam_forward: uniform("uCamForward"),
            aspect: uniform("uAspect"),
            fov_tan: uniform("uFovTan"),
            flashlight: uniform("uFlashlightEnabled"),
            front_to_back: uniform("uFrontToBackEnabled"),
            empty_space_skip: uniform("uEmptySpaceSkipEnabled"),
            baked_lighting: uniform("uBakedLightingEnabled"),
            dither: uniform("uDitherEnabled"),
            outdoor: uniform("uOutdoor"),
            sky_color: uniform("uSkyColor"),
            fog_color: uniform("uFogColor"),
            ambient_scale: uniform("uAmbientScale"),
            fog_density: uniform("uFogDensity"),
            fog_start: uniform("uFogStart"),
            light_count: uniform("uLightCount"),
            light_positions: uniform("uLightPositions"),
            light_colors: uniform("uLightColors"),
            light_params: uniform("uLightParams"),
            light_kinds: uniform("uLightKinds"),
            dynamic_light_count: uniform("uDynamicLightCount"),
            dynamic_pos_radius: uniform("uDynamicPosRadius"),
            dynamic_color_intensity: uniform("uDynamicColorIntensity"),
            node_texture: uniform("uNodeTexture"),
            num_chunks: uniform("uNumChunks"),
            chunk_origins: uniform("uChunkOrigins"),
            chunk_root_indices: uniform("uChunkRootIndices"),
            chunk_world_sizes: uniform("uChunkWorldSizes"),
            chunk_voxel_sizes: uniform("uChunkVoxelSizes"),
            chunk_depths: uniform("uChunkDepths"),
        }
    }
}

/// Geometry for the one fullscreen draw. Retaining the buffer handle makes
/// its ownership explicit even though the VAO also references it in WebGL.
struct FullscreenQuad {
    vao: WebGlVertexArrayObject,
    _vertex_buffer: WebGlBuffer,
}

impl FullscreenQuad {
    fn create(gl: &Gl, program: &WebGlProgram) -> Result<Self, JsValue> {
        let vao = gl
            .create_vertex_array()
            .ok_or_else(|| JsValue::from_str("failed to create raymarch VAO"))?;
        let vertex_buffer = gl
            .create_buffer()
            .ok_or_else(|| JsValue::from_str("failed to create raymarch vertex buffer"))?;

        gl.bind_vertex_array(Some(&vao));
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&vertex_buffer));
        let vertices: [f32; 12] = [
            -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
        ];
        let vertex_view = js_sys::Float32Array::from(vertices.as_slice());
        gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &vertex_view, Gl::STATIC_DRAW);

        let position = gl.get_attrib_location(program, "position");
        if position < 0 {
            gl.bind_vertex_array(None);
            gl.bind_buffer(Gl::ARRAY_BUFFER, None);
            return Err(JsValue::from_str(
                "raymarch shader does not expose the position attribute",
            ));
        }
        let position = position as u32;
        gl.enable_vertex_attrib_array(position);
        gl.vertex_attrib_pointer_with_i32(position, 2, Gl::FLOAT, false, 0, 0);
        gl.bind_vertex_array(None);
        gl.bind_buffer(Gl::ARRAY_BUFFER, None);

        Ok(Self {
            vao,
            _vertex_buffer: vertex_buffer,
        })
    }
}

/// Long-lived state for the debug/reference fullscreen SVO raymarcher.
pub struct WebGl2Renderer {
    gl: Gl,
    program: WebGlProgram,
    quad: FullscreenQuad,
    uniforms: Uniforms,
    atlas: AtlasTexture,
    width: i32,
    height: i32,
    chunk_uniforms: ChunkUniformBuffers,
    timer: GpuFrameTimer,
}

impl WebGl2Renderer {
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, JsValue> {
        // Visibility is resolved in the fragment shader, so a fixed-function
        // depth attachment would consume memory without affecting the image.
        let gl = create_context(canvas, ContextOptions { depth: false })?;
        let program = link_program(&gl, raymarch::VERTEX_SHADER, &raymarch::fragment_source())?;
        let uniforms = Uniforms::resolve(&gl, &program);
        let quad = FullscreenQuad::create(&gl, &program)?;
        let timer = GpuFrameTimer::new(&gl);

        gl.use_program(Some(&program));

        Ok(Self {
            gl,
            program,
            quad,
            uniforms,
            atlas: AtlasTexture::new(),
            width: canvas.width() as i32,
            height: canvas.height() as i32,
            chunk_uniforms: ChunkUniformBuffers::with_capacity(MAX_CHUNKS),
            timer,
        })
    }

    /// Updates the canvas backing-store dimensions used by the viewport and
    /// camera aspect ratio.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }
}
