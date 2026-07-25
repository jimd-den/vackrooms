//! Default WebGL2 renderer: indexed greedy meshes with fixed-function depth
//! visibility. The SVO remains in the chunk payload for collision and debug
//! traversal, but no visible fragment performs an octree walk.
//!
//! Split by responsibility:
//! * this file — the renderer struct, program setup, uniform locations;
//! * [`upload_world_surface_chunks`] — per-chunk GPU resource lifecycle (VAO/VBO/IBO + baked
//!   light volume), incremental upload/removal;
//! * [`draw_lit_world_surfaces`] — the per-frame `RendererPort::draw`: shadow pass, light
//!   selection, and the visible-mesh loop.
//!
//! Cross-driver plumbing (context creation, program linking, GPU timing,
//! matrices, light selection, shadow target) lives in [`crate::drivers::gl`].

#[path = "draw.rs"]
mod draw_lit_world_surfaces;
#[path = "resources.rs"]
mod upload_world_surface_chunks;

use std::collections::HashMap;

use wasm_bindgen::JsValue;
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlProgram, WebGlUniformLocation,
};

use crate::application::ports::SurfaceChunkKey;
use crate::drivers::gl::program::{ContextOptions, create_context, link_program};
use crate::drivers::gl::shadow_target::{ShadowTarget, create_shadow_target};
use crate::drivers::gl::timer::GpuFrameTimer;
use crate::drivers::gl::upload_scene_lights::SceneLightTexture;
use crate::drivers::shaders::{shadow, supply_labels, surface};

use upload_world_surface_chunks::GpuMesh;

/// Uniform locations resolved once at link time.
pub(crate) struct Uniforms {
    pub projection: Option<WebGlUniformLocation>,
    pub view: Option<WebGlUniformLocation>,
    pub chunk_origin: Option<WebGlUniformLocation>,
    pub camera_position: Option<WebGlUniformLocation>,
    pub cam_forward: Option<WebGlUniformLocation>,
    pub flashlight: Option<WebGlUniformLocation>,
    pub dither: Option<WebGlUniformLocation>,
    pub baked_lighting: Option<WebGlUniformLocation>,
    pub light_volume: Option<WebGlUniformLocation>,
    pub light_volume_origin: Option<WebGlUniformLocation>,
    pub voxel_size: Option<WebGlUniformLocation>,
    pub light_count: Option<WebGlUniformLocation>,
    pub light_first: Option<WebGlUniformLocation>,
    pub scene_light_texture: Option<WebGlUniformLocation>,
    pub scene_light_texture_width: Option<WebGlUniformLocation>,
    pub dynamic_light_count: Option<WebGlUniformLocation>,
    pub dynamic_pos_radius: Option<WebGlUniformLocation>,
    pub dynamic_color_intensity: Option<WebGlUniformLocation>,
    pub core_count: Option<WebGlUniformLocation>,
    pub cores: Option<WebGlUniformLocation>,
    pub core_colors: Option<WebGlUniformLocation>,
    pub shadow_map: Option<WebGlUniformLocation>,
    pub light_view_proj: Option<WebGlUniformLocation>,
    pub shadowed_light_index: Option<WebGlUniformLocation>,
    pub outdoor: Option<WebGlUniformLocation>,
    pub fog_color: Option<WebGlUniformLocation>,
    pub ambient_scale: Option<WebGlUniformLocation>,
    pub fog_density: Option<WebGlUniformLocation>,
    pub fog_start: Option<WebGlUniformLocation>,
}

pub(crate) struct ShadowUniforms {
    pub light_view_proj: Option<WebGlUniformLocation>,
    pub chunk_origin: Option<WebGlUniformLocation>,
}

/// Uniforms of the supply-label billboard pass.
pub(crate) struct LabelUniforms {
    pub projection: Option<WebGlUniformLocation>,
    pub view: Option<WebGlUniformLocation>,
    pub sprites: Option<WebGlUniformLocation>,
    pub cam_right: Option<WebGlUniformLocation>,
    pub half_size: Option<WebGlUniformLocation>,
    pub atlas: Option<WebGlUniformLocation>,
    pub camera_position: Option<WebGlUniformLocation>,
    pub fog_color: Option<WebGlUniformLocation>,
    pub fog_density: Option<WebGlUniformLocation>,
    pub fog_start: Option<WebGlUniformLocation>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SurfaceFrameStats {
    pub visible_chunks: usize,
    pub resident_chunks: usize,
    pub draw_calls: u32,
    pub triangles_drawn: u32,
    pub shadow_casters: u32,
}

/// Surface rasterizer with incremental chunk mesh uploads. One VAO/VBO/IBO
/// tuple per resident chunk keeps the draw path allocation-free.
pub struct SurfaceRenderer {
    pub(crate) gl: Gl,
    pub(crate) program: WebGlProgram,
    pub(crate) uniforms: Uniforms,
    pub(crate) meshes: HashMap<SurfaceChunkKey, GpuMesh>,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) timer: GpuFrameTimer,
    pub(crate) scene_lights: SceneLightTexture,
    pub(crate) shadow_program: WebGlProgram,
    pub(crate) shadow_uniforms: ShadowUniforms,
    pub(crate) shadow: ShadowTarget,
    pub(crate) label_program: WebGlProgram,
    pub(crate) label_uniforms: LabelUniforms,
    /// Present once the driver has decoded and shipped the label atlas.
    pub(crate) label_atlas: Option<web_sys::WebGlTexture>,
    pub(crate) stats: SurfaceFrameStats,
}

impl SurfaceRenderer {
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, JsValue> {
        let gl = create_context(canvas, ContextOptions { depth: true })?;

        let program = link_program(&gl, surface::VERTEX_SHADER, &surface::fragment_source())?;
        let uniforms = Uniforms {
            projection: gl.get_uniform_location(&program, "uProjection"),
            view: gl.get_uniform_location(&program, "uView"),
            chunk_origin: gl.get_uniform_location(&program, "uChunkOrigin"),
            camera_position: gl.get_uniform_location(&program, "uCameraPosition"),
            cam_forward: gl.get_uniform_location(&program, "uCamForward"),
            flashlight: gl.get_uniform_location(&program, "uFlashlightEnabled"),
            dither: gl.get_uniform_location(&program, "uDitherEnabled"),
            baked_lighting: gl.get_uniform_location(&program, "uBakedLightingEnabled"),
            light_volume: gl.get_uniform_location(&program, "uLightVolume"),
            light_volume_origin: gl.get_uniform_location(&program, "uLightVolumeOrigin"),
            voxel_size: gl.get_uniform_location(&program, "uVoxelSize"),
            light_count: gl.get_uniform_location(&program, "uLightCount"),
            light_first: gl.get_uniform_location(&program, "uLightFirst"),
            scene_light_texture: gl.get_uniform_location(&program, "uSceneLightTexture"),
            scene_light_texture_width: gl.get_uniform_location(&program, "uSceneLightTextureWidth"),
            dynamic_light_count: gl.get_uniform_location(&program, "uDynamicLightCount"),
            dynamic_pos_radius: gl.get_uniform_location(&program, "uDynamicPosRadius"),
            dynamic_color_intensity: gl.get_uniform_location(&program, "uDynamicColorIntensity"),
            core_count: gl.get_uniform_location(&program, "uCoreCount"),
            cores: gl.get_uniform_location(&program, "uCores"),
            core_colors: gl.get_uniform_location(&program, "uCoreColors"),
            shadow_map: gl.get_uniform_location(&program, "uShadowMap"),
            light_view_proj: gl.get_uniform_location(&program, "uLightViewProjection"),
            shadowed_light_index: gl.get_uniform_location(&program, "uShadowedLightIndex"),
            outdoor: gl.get_uniform_location(&program, "uOutdoor"),
            fog_color: gl.get_uniform_location(&program, "uFogColor"),
            ambient_scale: gl.get_uniform_location(&program, "uAmbientScale"),
            fog_density: gl.get_uniform_location(&program, "uFogDensity"),
            fog_start: gl.get_uniform_location(&program, "uFogStart"),
        };
        let timer = GpuFrameTimer::new(&gl);
        let scene_lights = SceneLightTexture::create(&gl)?;

        let shadow_program = link_program(&gl, shadow::VERTEX_SHADER, shadow::FRAGMENT_SHADER)?;
        let shadow_uniforms = ShadowUniforms {
            light_view_proj: gl.get_uniform_location(&shadow_program, "uLightViewProjection"),
            chunk_origin: gl.get_uniform_location(&shadow_program, "uChunkOrigin"),
        };
        let shadow = create_shadow_target(&gl, 128)?;

        let label_program = link_program(
            &gl,
            supply_labels::VERTEX_SHADER,
            &supply_labels::fragment_source(),
        )?;
        let label_uniforms = LabelUniforms {
            projection: gl.get_uniform_location(&label_program, "uProjection"),
            view: gl.get_uniform_location(&label_program, "uView"),
            sprites: gl.get_uniform_location(&label_program, "uSprites"),
            cam_right: gl.get_uniform_location(&label_program, "uCamRight"),
            half_size: gl.get_uniform_location(&label_program, "uHalfSize"),
            atlas: gl.get_uniform_location(&label_program, "uAtlas"),
            camera_position: gl.get_uniform_location(&label_program, "uCameraPosition"),
            fog_color: gl.get_uniform_location(&label_program, "uFogColor"),
            fog_density: gl.get_uniform_location(&label_program, "uFogDensity"),
            fog_start: gl.get_uniform_location(&label_program, "uFogStart"),
        };

        Ok(Self {
            gl,
            program,
            uniforms,
            meshes: HashMap::new(),
            width: canvas.width() as i32,
            height: canvas.height() as i32,
            timer,
            scene_lights,
            shadow_program,
            shadow_uniforms,
            shadow,
            label_program,
            label_uniforms,
            label_atlas: None,
            stats: SurfaceFrameStats::default(),
        })
    }

    /// Called by the browser driver when the canvas backing store changes
    /// (window resize or adaptive resolution step).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }
}
