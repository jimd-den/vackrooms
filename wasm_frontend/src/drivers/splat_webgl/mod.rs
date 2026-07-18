//! Instanced face-splat renderer — the GPU microvoxel path.
//!
//! One instance is one error-selected, axis-aligned surface rectangle. The
//! vertex shader reconstructs its quad and evaluates lighting once per face;
//! the fragment shader is deliberately limited to fog, grain, and tone
//! mapping. Chunk uploads are incremental, so steady-state frames only cull
//! contiguous instance ranges and submit draws.
//!
//! The module is split by responsibility:
//!
//! * this file owns composition, profile settings, programs, and uniforms;
//! * [`resources`] owns per-chunk GPU buffers and textures;
//! * [`draw`] owns the frame pipeline and renderer-port implementation.
//!
//! Cross-renderer WebGL plumbing lives in [`crate::drivers::gl`], while GLSL
//! source composition lives in [`crate::drivers::shaders::splat`].

mod draw;
mod resources;

use std::collections::HashMap;

use wasm_bindgen::JsValue;
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlProgram, WebGlUniformLocation,
};

use crate::application::ports::SurfaceChunkKey;
use crate::drivers::gl::program::{ContextOptions, create_context, link_program};
use crate::drivers::gl::shadow_target::{ShadowTarget, create_shadow_target};
use crate::drivers::gl::timer::GpuFrameTimer;
use crate::drivers::shaders::{shadow, splat};

use resources::{InstanceAttribs, SplatChunk};

/// Hardware-profile knobs chosen once by the browser composition root.
///
/// These are quality/budget settings rather than correctness optimizations;
/// runtime optimization switches live in `RenderToggles`.
#[derive(Debug, Clone, Copy)]
pub struct SplatProfile {
    pub shadow_map_size: i32,
    /// `1` selects one shadow comparison; `4` selects four-tap PCF.
    pub shadow_taps: i32,
    /// Per-frame face budget. Once exhausted, farther chunks are dropped
    /// whole so nearby planes remain spatially coherent.
    pub face_budget: u32,
}

impl SplatProfile {
    pub fn low() -> Self {
        Self {
            shadow_map_size: 128,
            shadow_taps: 1,
            face_budget: 40_000,
        }
    }

    pub fn high() -> Self {
        Self {
            shadow_map_size: 256,
            shadow_taps: 4,
            face_budget: 150_000,
        }
    }
}

/// Main-program uniforms, resolved once after linking.
pub(crate) struct Uniforms {
    pub projection: Option<WebGlUniformLocation>,
    pub view: Option<WebGlUniformLocation>,
    pub chunk_origin: Option<WebGlUniformLocation>,
    pub chunk_size: Option<WebGlUniformLocation>,
    pub voxel_scale: Option<WebGlUniformLocation>,
    pub camera_position: Option<WebGlUniformLocation>,
    pub cam_forward: Option<WebGlUniformLocation>,
    pub flashlight: Option<WebGlUniformLocation>,
    pub dither: Option<WebGlUniformLocation>,
    pub light_volume: Option<WebGlUniformLocation>,
    pub light_count: Option<WebGlUniformLocation>,
    pub light_positions: Option<WebGlUniformLocation>,
    pub light_colors: Option<WebGlUniformLocation>,
    pub light_params: Option<WebGlUniformLocation>,
    pub core_count: Option<WebGlUniformLocation>,
    pub cores: Option<WebGlUniformLocation>,
    pub core_colors: Option<WebGlUniformLocation>,
    pub shadow_map: Option<WebGlUniformLocation>,
    pub light_view_proj: Option<WebGlUniformLocation>,
    pub shadowed_light_index: Option<WebGlUniformLocation>,
    pub shadow_taps: Option<WebGlUniformLocation>,
    pub outdoor: Option<WebGlUniformLocation>,
    pub fog_color: Option<WebGlUniformLocation>,
    pub ambient_scale: Option<WebGlUniformLocation>,
}

pub(crate) struct ShadowUniforms {
    pub light_view_proj: Option<WebGlUniformLocation>,
    pub chunk_origin: Option<WebGlUniformLocation>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FrameStats {
    pub faces_drawn: u32,
    pub draw_calls: u32,
    pub cells_culled: u32,
    pub faces_dropped: u32,
    pub chunks_drawn: u32,
}

/// WebGL2 face-splat renderer with one immutable resource page per resident
/// chunk. Mutable per-frame state is limited to the asynchronous timer and
/// telemetry counters.
pub struct SplatRenderer {
    pub(crate) gl: Gl,
    pub(crate) program: WebGlProgram,
    pub(crate) uniforms: Uniforms,
    pub(crate) attribs: InstanceAttribs,
    pub(crate) shadow_program: WebGlProgram,
    pub(crate) shadow_uniforms: ShadowUniforms,
    pub(crate) shadow: ShadowTarget,
    pub(crate) profile: SplatProfile,
    pub(crate) chunks: HashMap<SurfaceChunkKey, SplatChunk>,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) timer: GpuFrameTimer,
    pub(crate) stats: FrameStats,
}

impl SplatRenderer {
    pub fn new(canvas: &HtmlCanvasElement, profile: SplatProfile) -> Result<Self, JsValue> {
        let gl = create_context(canvas, ContextOptions { depth: true })?;

        let vertex_source = splat::vertex_source();
        let fragment_source = splat::fragment_source();
        let program = link_program(&gl, &vertex_source, &fragment_source)?;
        let uniforms = Uniforms {
            projection: gl.get_uniform_location(&program, "uProjection"),
            view: gl.get_uniform_location(&program, "uView"),
            chunk_origin: gl.get_uniform_location(&program, "uChunkOrigin"),
            chunk_size: gl.get_uniform_location(&program, "uChunkSize"),
            voxel_scale: gl.get_uniform_location(&program, "uVoxelScale"),
            camera_position: gl.get_uniform_location(&program, "uCameraPosition"),
            cam_forward: gl.get_uniform_location(&program, "uCamForward"),
            flashlight: gl.get_uniform_location(&program, "uFlashlightEnabled"),
            dither: gl.get_uniform_location(&program, "uDitherEnabled"),
            light_volume: gl.get_uniform_location(&program, "uLightVolume"),
            light_count: gl.get_uniform_location(&program, "uLightCount"),
            light_positions: gl.get_uniform_location(&program, "uLightPositions"),
            light_colors: gl.get_uniform_location(&program, "uLightColors"),
            light_params: gl.get_uniform_location(&program, "uLightParams"),
            core_count: gl.get_uniform_location(&program, "uCoreCount"),
            cores: gl.get_uniform_location(&program, "uCores"),
            core_colors: gl.get_uniform_location(&program, "uCoreColors"),
            shadow_map: gl.get_uniform_location(&program, "uShadowMap"),
            light_view_proj: gl.get_uniform_location(&program, "uLightViewProjection"),
            shadowed_light_index: gl.get_uniform_location(&program, "uShadowedLightIndex"),
            shadow_taps: gl.get_uniform_location(&program, "uShadowTaps"),
            outdoor: gl.get_uniform_location(&program, "uOutdoor"),
            fog_color: gl.get_uniform_location(&program, "uFogColor"),
            ambient_scale: gl.get_uniform_location(&program, "uAmbientScale"),
        };
        let attribs = InstanceAttribs::resolve(&gl, &program)?;

        let shadow_program = link_program(&gl, shadow::VERTEX_SHADER, shadow::FRAGMENT_SHADER)?;
        let shadow_uniforms = ShadowUniforms {
            light_view_proj: gl.get_uniform_location(&shadow_program, "uLightViewProjection"),
            chunk_origin: gl.get_uniform_location(&shadow_program, "uChunkOrigin"),
        };
        let shadow = create_shadow_target(&gl, profile.shadow_map_size)?;
        let timer = GpuFrameTimer::new(&gl);

        Ok(Self {
            gl,
            program,
            uniforms,
            attribs,
            shadow_program,
            shadow_uniforms,
            shadow,
            profile,
            chunks: HashMap::new(),
            width: canvas.width() as i32,
            height: canvas.height() as i32,
            timer,
            stats: FrameStats::default(),
        })
    }

    /// Updates the backing-store viewport dimensions after adaptive scaling
    /// or a browser resize. GPU resources are resolution-independent.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }
}
