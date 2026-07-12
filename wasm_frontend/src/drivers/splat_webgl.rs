//! Instanced face-splat renderer — the GPU microvoxel path. One instance is
//! one error-selected, axis-aligned surface rectangle (`PackedFaceInstance`);
//! the vertex shader rebuilds the quad and evaluates all lighting once per
//! face, flat, leaving a near-trivial fragment shader (fog + grain + dither).
//!
//! Selection is amortized: instance pages arrive prebuilt per chunk (same
//! lifecycle as meshes; steady-state frames upload nothing). Per frame the
//! CPU only culls chunks and cell ranges and issues instanced draws.
//!
//! The retained greedy mesh is drawn *only* into the hero-light shadow map:
//! both representations derive from the same grid at the same LOD, so the
//! shadow caster and the visible splats always agree.
//!
//! WebGL2 has no `baseInstance`, so drawing a sub-range of instances
//! re-points the instance attributes at `first * STRIDE` before the draw.

use std::collections::HashMap;

use js_sys::{Object, Reflect, Uint8Array, Uint32Array};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram, WebGlQuery,
    WebGlUniformLocation, WebGlVertexArrayObject,
};

use crate::application::ports::{
    ChunkDraw, FaceCellRange, FrameParams, LightSource, RendererPort, SurfaceChunk,
    SurfaceChunkKey,
};
use crate::drivers::shaders::{
    SHADOW_FRAGMENT_SHADER, SHADOW_VERTEX_SHADER, SPLAT_FRAGMENT_SHADER, SPLAT_VERTEX_SHADER,
};
use crate::drivers::surface_webgl::{
    MAX_DRAW_DISTANCE, camera_matrices, link_program, look_at_matrix_down, multiply_matrices,
    ortho_matrix,
};

/// EXT_disjoint_timer_query_webgl2 constants, absent from WebGL2 core.
const TIME_ELAPSED_EXT: u32 = 0x88BF;
const GPU_DISJOINT_EXT: u32 = 0x8FBB;
/// Bytes per `PackedFaceInstance` as uploaded.
const INSTANCE_STRIDE: i32 = 16;

/// Spec-profile knobs, chosen by the driver layer at construction.
#[derive(Debug, Clone, Copy)]
pub struct SplatProfile {
    pub shadow_map_size: i32,
    /// 1 = single shadow compare, 4 = PCF4.
    pub shadow_taps: i32,
    /// Per-frame face-instance budget; farthest chunks are dropped whole
    /// (they sit in fog) rather than thinning faces out of nearer planes.
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

struct Uniforms {
    projection: Option<WebGlUniformLocation>,
    view: Option<WebGlUniformLocation>,
    chunk_origin: Option<WebGlUniformLocation>,
    chunk_size: Option<WebGlUniformLocation>,
    voxel_scale: Option<WebGlUniformLocation>,
    camera_position: Option<WebGlUniformLocation>,
    flashlight: Option<WebGlUniformLocation>,
    light_volume: Option<WebGlUniformLocation>,
    light_count: Option<WebGlUniformLocation>,
    light_positions: Option<WebGlUniformLocation>,
    core_count: Option<WebGlUniformLocation>,
    cores: Option<WebGlUniformLocation>,
    core_colors: Option<WebGlUniformLocation>,
    light_colors: Option<WebGlUniformLocation>,
    light_params: Option<WebGlUniformLocation>,
    shadow_map: Option<WebGlUniformLocation>,
    light_view_proj: Option<WebGlUniformLocation>,
    shadowed_light_index: Option<WebGlUniformLocation>,
    shadow_taps: Option<WebGlUniformLocation>,
    outdoor: Option<WebGlUniformLocation>,
    fog_color: Option<WebGlUniformLocation>,
    ambient_scale: Option<WebGlUniformLocation>,
}

struct ShadowUniforms {
    light_view_proj: Option<WebGlUniformLocation>,
    chunk_origin: Option<WebGlUniformLocation>,
}

struct InstanceAttribs {
    pos: u32,
    extents: u32,
    meta: u32,
    flags: u32,
}

struct SplatChunk {
    instance_vao: WebGlVertexArrayObject,
    instance_buffer: WebGlBuffer,
    instance_count: i32,
    cells: Vec<FaceCellRange>,
    cell_size: f32,
    voxel_scale: f32,
    shadow_vao: WebGlVertexArrayObject,
    shadow_vertex_buffer: WebGlBuffer,
    shadow_index_buffer: WebGlBuffer,
    shadow_index_count: i32,
    origin: [f32; 3],
    bounds_max: [f32; 3],
    light_texture: web_sys::WebGlTexture,
    lights: Vec<LightSource>,
}

#[derive(Debug, Clone, Copy, Default)]
struct FrameStats {
    faces_drawn: u32,
    draw_calls: u32,
    cells_culled: u32,
    faces_dropped: u32,
    chunks_drawn: u32,
}

pub struct SplatRenderer {
    gl: Gl,
    program: WebGlProgram,
    uniforms: Uniforms,
    attribs: InstanceAttribs,
    shadow_program: WebGlProgram,
    shadow_uniforms: ShadowUniforms,
    shadow_fbo: web_sys::WebGlFramebuffer,
    shadow_texture: web_sys::WebGlTexture,
    profile: SplatProfile,
    chunks: HashMap<SurfaceChunkKey, SplatChunk>,
    width: i32,
    height: i32,
    timer_extension: Option<Object>,
    pending_timer: Option<WebGlQuery>,
    last_gpu_ms: Option<f32>,
    stats: FrameStats,
}

impl SplatRenderer {
    pub fn new(canvas: &HtmlCanvasElement, profile: SplatProfile) -> Result<Self, JsValue> {
        let options = js_sys::Object::new();
        Reflect::set(&options, &"antialias".into(), &false.into())?;
        Reflect::set(&options, &"depth".into(), &true.into())?;
        Reflect::set(&options, &"stencil".into(), &false.into())?;
        Reflect::set(&options, &"alpha".into(), &false.into())?;
        Reflect::set(
            &options,
            &"powerPreference".into(),
            &"high-performance".into(),
        )?;
        let gl = canvas
            .get_context_with_context_options("webgl2", &options)?
            .ok_or_else(|| JsValue::from_str("WebGL2 is not supported on this browser"))?
            .dyn_into::<Gl>()?;

        let program = link_program(&gl, SPLAT_VERTEX_SHADER, SPLAT_FRAGMENT_SHADER)?;
        let uniforms = Uniforms {
            projection: gl.get_uniform_location(&program, "uProjection"),
            view: gl.get_uniform_location(&program, "uView"),
            chunk_origin: gl.get_uniform_location(&program, "uChunkOrigin"),
            chunk_size: gl.get_uniform_location(&program, "uChunkSize"),
            voxel_scale: gl.get_uniform_location(&program, "uVoxelScale"),
            camera_position: gl.get_uniform_location(&program, "uCameraPosition"),
            flashlight: gl.get_uniform_location(&program, "uFlashlightEnabled"),
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
        let lookup = |name: &str| -> Result<u32, JsValue> {
            let location = gl.get_attrib_location(&program, name);
            if location < 0 {
                return Err(JsValue::from_str(&format!("splat shader lost {name}")));
            }
            Ok(location as u32)
        };
        let attribs = InstanceAttribs {
            pos: lookup("aFacePos")?,
            extents: lookup("aFaceExtents")?,
            meta: lookup("aFaceMeta")?,
            flags: lookup("aFaceFlags")?,
        };

        let shadow_program = link_program(&gl, SHADOW_VERTEX_SHADER, SHADOW_FRAGMENT_SHADER)?;
        let shadow_uniforms = ShadowUniforms {
            light_view_proj: gl.get_uniform_location(&shadow_program, "uLightViewProjection"),
            chunk_origin: gl.get_uniform_location(&shadow_program, "uChunkOrigin"),
        };

        let shadow_texture = gl
            .create_texture()
            .ok_or_else(|| JsValue::from_str("create shadow texture"))?;
        gl.bind_texture(Gl::TEXTURE_2D, Some(&shadow_texture));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            Gl::TEXTURE_2D,
            0,
            Gl::DEPTH_COMPONENT16 as i32,
            profile.shadow_map_size,
            profile.shadow_map_size,
            0,
            Gl::DEPTH_COMPONENT,
            Gl::UNSIGNED_SHORT,
            None,
        )?;
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);

        let shadow_fbo = gl
            .create_framebuffer()
            .ok_or_else(|| JsValue::from_str("create shadow FBO"))?;
        gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&shadow_fbo));
        gl.framebuffer_texture_2d(
            Gl::FRAMEBUFFER,
            Gl::DEPTH_ATTACHMENT,
            Gl::TEXTURE_2D,
            Some(&shadow_texture),
            0,
        );
        gl.bind_framebuffer(Gl::FRAMEBUFFER, None);

        let timer_extension = gl
            .get_extension("EXT_disjoint_timer_query_webgl2")
            .ok()
            .flatten();

        Ok(Self {
            gl,
            program,
            uniforms,
            attribs,
            shadow_program,
            shadow_uniforms,
            shadow_fbo,
            shadow_texture,
            profile,
            chunks: HashMap::new(),
            width: canvas.width() as i32,
            height: canvas.height() as i32,
            timer_extension,
            pending_timer: None,
            last_gpu_ms: None,
            stats: FrameStats::default(),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }

    fn destroy_chunk(&self, chunk: SplatChunk) {
        let gl = &self.gl;
        gl.delete_vertex_array(Some(&chunk.instance_vao));
        gl.delete_buffer(Some(&chunk.instance_buffer));
        gl.delete_vertex_array(Some(&chunk.shadow_vao));
        gl.delete_buffer(Some(&chunk.shadow_vertex_buffer));
        gl.delete_buffer(Some(&chunk.shadow_index_buffer));
        gl.delete_texture(Some(&chunk.light_texture));
    }

    /// Points the instance attributes at `first_instance` within the bound
    /// buffer. WebGL2 lacks `baseInstance`, so sub-range draws re-point.
    fn point_instance_attribs(&self, first_instance: u32) {
        let gl = &self.gl;
        let base = first_instance as i32 * INSTANCE_STRIDE;
        gl.vertex_attrib_pointer_with_i32(
            self.attribs.pos,
            3,
            Gl::UNSIGNED_SHORT,
            false,
            INSTANCE_STRIDE,
            base,
        );
        gl.vertex_attrib_pointer_with_i32(
            self.attribs.extents,
            2,
            Gl::UNSIGNED_BYTE,
            false,
            INSTANCE_STRIDE,
            base + 6,
        );
        gl.vertex_attrib_pointer_with_i32(
            self.attribs.meta,
            4,
            Gl::UNSIGNED_BYTE,
            false,
            INSTANCE_STRIDE,
            base + 8,
        );
        gl.vertex_attrib_pointer_with_i32(
            self.attribs.flags,
            1,
            Gl::UNSIGNED_BYTE,
            false,
            INSTANCE_STRIDE,
            base + 12,
        );
    }

    fn upload_surface(&mut self, chunk: SurfaceChunk<'_>) {
        if let Some(old) = self.chunks.remove(&chunk.key) {
            self.destroy_chunk(old);
        }
        let mesh = chunk.mesh;
        if mesh.faces.instances.is_empty() {
            return;
        }
        let gl = &self.gl;

        // Instance page: PackedFaceInstance serialized little-endian, 16 B.
        let mut packed = Vec::with_capacity(mesh.faces.instances.len() * INSTANCE_STRIDE as usize);
        for inst in &mesh.faces.instances {
            packed.extend_from_slice(&inst.position[0].to_le_bytes());
            packed.extend_from_slice(&inst.position[1].to_le_bytes());
            packed.extend_from_slice(&inst.position[2].to_le_bytes());
            packed.push(inst.extent_u);
            packed.push(inst.extent_v);
            packed.push(inst.normal_axis);
            packed.push(inst.material);
            packed.push(inst.baked_light);
            packed.push(inst.ao);
            packed.push(inst.flags);
            packed.extend_from_slice(&[0, 0, 0]);
        }
        let instance_vao = gl.create_vertex_array().expect("create splat VAO");
        let instance_buffer = gl.create_buffer().expect("create splat instance buffer");
        gl.bind_vertex_array(Some(&instance_vao));
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&instance_buffer));
        let bytes = Uint8Array::from(packed.as_slice());
        gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &bytes, Gl::STATIC_DRAW);
        for &location in [
            self.attribs.pos,
            self.attribs.extents,
            self.attribs.meta,
            self.attribs.flags,
        ]
        .iter()
        {
            gl.enable_vertex_attrib_array(location);
            gl.vertex_attrib_divisor(location, 1);
        }
        self.point_instance_attribs(0);
        gl.bind_vertex_array(None);

        // Shadow-caster mesh: the same 10-byte packed vertices the mesh
        // renderer uses, but only position (location 0) is bound — the
        // shadow program reads nothing else.
        let shadow_vao = gl.create_vertex_array().expect("create shadow VAO");
        let shadow_vertex_buffer = gl.create_buffer().expect("create shadow vertex buffer");
        let shadow_index_buffer = gl.create_buffer().expect("create shadow index buffer");
        gl.bind_vertex_array(Some(&shadow_vao));
        let mut vertex_bytes = Vec::with_capacity(mesh.vertices.len() * 10);
        for v in &mesh.vertices {
            vertex_bytes.extend_from_slice(&v.position[0].to_le_bytes());
            vertex_bytes.extend_from_slice(&v.position[1].to_le_bytes());
            vertex_bytes.extend_from_slice(&v.position[2].to_le_bytes());
            vertex_bytes.extend_from_slice(&[v.normal_axis, v.material, v.static_indirect, v.ao]);
        }
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&shadow_vertex_buffer));
        let vertex_view = Uint8Array::from(vertex_bytes.as_slice());
        gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &vertex_view, Gl::STATIC_DRAW);
        gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&shadow_index_buffer));
        let index_view = Uint32Array::from(mesh.indices.as_slice());
        gl.buffer_data_with_array_buffer_view(Gl::ELEMENT_ARRAY_BUFFER, &index_view, Gl::STATIC_DRAW);
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 3, Gl::UNSIGNED_SHORT, false, 10, 0);
        gl.bind_vertex_array(None);
        gl.bind_buffer(Gl::ARRAY_BUFFER, None);
        gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, None);

        let light_texture = gl.create_texture().expect("create light texture");
        gl.bind_texture(Gl::TEXTURE_3D, Some(&light_texture));
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
        )
        .expect("upload light volume");
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_MIN_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_MAG_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_R, Gl::CLAMP_TO_EDGE as i32);
        gl.bind_texture(Gl::TEXTURE_3D, None);

        self.chunks.insert(
            chunk.key,
            SplatChunk {
                instance_vao,
                instance_buffer,
                instance_count: mesh.faces.instances.len() as i32,
                cells: mesh.faces.cells.clone(),
                cell_size: mesh.faces.cell_size,
                voxel_scale: mesh.voxel_scale,
                shadow_vao,
                shadow_vertex_buffer,
                shadow_index_buffer,
                shadow_index_count: mesh.indices.len().min(i32::MAX as usize) as i32,
                origin: chunk.origin,
                bounds_max: mesh.bounds.max,
                light_texture,
                lights: mesh.lights.clone(),
            },
        );
    }

    fn poll_timer(&mut self) {
        let Some(query) = self.pending_timer.as_ref() else {
            return;
        };
        let available = self
            .gl
            .get_query_parameter(query, Gl::QUERY_RESULT_AVAILABLE)
            .as_bool()
            .unwrap_or(false);
        let disjoint = self.timer_extension.is_some()
            && self
                .gl
                .get_parameter(GPU_DISJOINT_EXT)
                .ok()
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
        if !available && !disjoint {
            return;
        }
        let query = self.pending_timer.take().expect("pending timer exists");
        if available && !disjoint {
            self.last_gpu_ms = self
                .gl
                .get_query_parameter(&query, Gl::QUERY_RESULT)
                .as_f64()
                .map(|nanos| (nanos as f32) * 1.0e-6);
        }
        self.gl.delete_query(Some(&query));
    }

    fn begin_timer(&mut self) -> bool {
        if self.timer_extension.is_none() || self.pending_timer.is_some() {
            return false;
        }
        let Some(query) = self.gl.create_query() else {
            return false;
        };
        self.gl.begin_query(TIME_ELAPSED_EXT, &query);
        self.pending_timer = Some(query);
        true
    }

    fn end_timer(&self, started: bool) {
        if started {
            self.gl.end_query(TIME_ELAPSED_EXT);
        }
    }
}

/// Conservative sphere visibility: inside draw distance and not entirely
/// behind the camera plane (same test the mesh renderer uses).
fn sphere_visible(center: [f32; 3], radius: f32, frame: &FrameParams) -> Option<f32> {
    let to = [
        center[0] - frame.camera_pos[0],
        center[1] - frame.camera_pos[1],
        center[2] - frame.camera_pos[2],
    ];
    let distance2 = to[0] * to[0] + to[1] * to[1] + to[2] * to[2];
    if distance2 > MAX_DRAW_DISTANCE * MAX_DRAW_DISTANCE {
        return None;
    }
    let (sp, cp) = frame.pitch.sin_cos();
    let (sy, cy) = frame.yaw.sin_cos();
    let forward = [-cp * sy, sp, -cp * cy];
    let dot = to[0] * forward[0] + to[1] * forward[1] + to[2] * forward[2];
    if dot >= -radius {
        Some(distance2)
    } else {
        None
    }
}

impl RendererPort for SplatRenderer {
    fn uses_surface_meshes(&self) -> bool {
        true
    }

    fn upload_surfaces(&mut self, chunks: &[SurfaceChunk<'_>]) {
        for &chunk in chunks {
            self.upload_surface(chunk);
        }
    }

    fn remove_surfaces(&mut self, keys: &[SurfaceChunkKey]) {
        for key in keys {
            if let Some(chunk) = self.chunks.remove(key) {
                self.destroy_chunk(chunk);
            }
        }
    }

    fn clear_surfaces(&mut self) {
        let chunks = std::mem::take(&mut self.chunks);
        for (_, chunk) in chunks {
            self.destroy_chunk(chunk);
        }
    }

    fn gpu_frame_ms(&self) -> Option<f32> {
        self.last_gpu_ms
    }

    fn cpu_telemetry_string(&self) -> Option<String> {
        let s = self.stats;
        Some(format!(
            "splat {} faces / {} calls / {} chunks / {} cells culled / {} dropped",
            s.faces_drawn, s.draw_calls, s.chunks_drawn, s.cells_culled, s.faces_dropped
        ))
    }

    fn upload_atlas(&mut self, _texels: &[u32]) {}

    fn draw(&mut self, frame: &FrameParams, _chunks: &[ChunkDraw]) {
        self.poll_timer();
        let timer_started = self.begin_timer();
        let mut stats = FrameStats::default();
        {
            let gl = &self.gl;
            gl.viewport(0, 0, self.width, self.height);
            // Unloaded space is literal black; the splat shader reaches the
            // same black before the draw limit so missing chunks have no
            // colored rectangle or visible render-distance wall. Outdoor
            // levels clear to the environment's bright sky instead.
            let env = frame.environment;
            if env.outdoor {
                gl.clear_color(env.sky_color[0], env.sky_color[1], env.sky_color[2], 1.0);
            } else {
                gl.clear_color(0.0, 0.0, 0.0, 1.0);
            }
            gl.clear_depth(1.0);
            gl.clear(Gl::COLOR_BUFFER_BIT | Gl::DEPTH_BUFFER_BIT);
            gl.enable(Gl::DEPTH_TEST);
            gl.depth_func(Gl::LEQUAL);
            gl.depth_mask(true);
            gl.enable(Gl::CULL_FACE);
            gl.cull_face(Gl::BACK);

            // Visible chunks, sorted near-to-far so the face budget drops
            // the farthest (fog-covered) chunks first.
            let mut visible: Vec<(&SplatChunk, f32)> = self
                .chunks
                .values()
                .filter_map(|chunk| {
                    let center = [
                        chunk.origin[0] + chunk.bounds_max[0] * 0.5,
                        chunk.origin[1] + chunk.bounds_max[1] * 0.5,
                        chunk.origin[2] + chunk.bounds_max[2] * 0.5,
                    ];
                    let radius = (chunk.bounds_max[0] * chunk.bounds_max[0]
                        + chunk.bounds_max[1] * chunk.bounds_max[1]
                        + chunk.bounds_max[2] * chunk.bounds_max[2])
                        .sqrt()
                        * 0.5;
                    sphere_visible(center, radius, frame).map(|d2| (chunk, d2))
                })
                .collect();
            visible
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Same light dedupe/importance selection as the mesh renderer.
            let mut unique_lights = HashMap::new();
            for (chunk, _) in &visible {
                for light in &chunk.lights {
                    if light.enabled {
                        unique_lights.insert(light.id, light);
                    }
                }
            }
            // Per-frame dynamic lights (flares) join the same selection but
            // never own the hero shadow: an ankle-height projector collapses
            // the top-down shadow frustum.
            let dynamic_sources: Vec<LightSource> = frame
                .active_dynamic_lights()
                .iter()
                .enumerate()
                .map(|(i, d)| LightSource {
                    id: u64::MAX - i as u64,
                    position: d.position,
                    half_size: [0.25, 0.25],
                    color: d.color,
                    radius: d.radius,
                    intensity: d.intensity,
                    flicker_mode: 0,
                    enabled: true,
                })
                .collect();
            let is_dynamic = |light: &LightSource| light.id > u64::MAX - 16;
            let mut lights_with_importance: Vec<(&LightSource, f32)> = unique_lights
                .values()
                .copied()
                .chain(dynamic_sources.iter())
                .map(|light| {
                    let dx = light.position[0] - frame.camera_pos[0];
                    let dy = light.position[1] - frame.camera_pos[1];
                    let dz = light.position[2] - frame.camera_pos[2];
                    let dist2 = dx * dx + dy * dy + dz * dz;
                    (light, light.radius * light.intensity.sqrt() / (dist2 + 0.1))
                })
                .collect();
            lights_with_importance
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let global_light_count = lights_with_importance.len().min(4);
            let mut light_positions = [0.0f32; 12];
            let mut light_colors = [0.0f32; 12];
            let mut light_params = [0.0f32; 16];
            for i in 0..global_light_count {
                let light = lights_with_importance[i].0;
                light_positions[i * 3..i * 3 + 3].copy_from_slice(&light.position);
                light_colors[i * 3..i * 3 + 3].copy_from_slice(&light.color);
                light_params[i * 4] = light.radius;
                light_params[i * 4 + 1] = light.intensity;
                light_params[i * 4 + 2] = light.half_size[0];
                light_params[i * 4 + 3] = light.half_size[1];
            }

            // Hero-light shadow pass: greedy meshes only. The splats derive
            // from the same grids, so caster and receiver always agree.
            let mut shadowed_light_index = -1;
            let mut light_view_proj = [0.0f32; 16];
            let hero_slot = lights_with_importance[..global_light_count]
                .iter()
                .position(|(light, _)| !is_dynamic(light));
            if let Some(hero_slot) = hero_slot {
                shadowed_light_index = hero_slot as i32;
                let hero = lights_with_importance[hero_slot].0;
                let range = hero.radius;
                // Keep the hero map focused enough for columns and partitions
                // to cast readable shadows at 128px on the Pi profile.
                let ortho_half = (range * 0.55).clamp(7.0, 13.0);
                let proj = ortho_matrix(-ortho_half, ortho_half, -ortho_half, ortho_half, 0.1, range * 1.2);
                let view = look_at_matrix_down(hero.position);
                light_view_proj = multiply_matrices(&proj, &view);

                gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&self.shadow_fbo));
                gl.viewport(0, 0, self.profile.shadow_map_size, self.profile.shadow_map_size);
                gl.color_mask(false, false, false, false);
                gl.clear_depth(1.0);
                gl.clear(Gl::DEPTH_BUFFER_BIT);
                gl.use_program(Some(&self.shadow_program));
                gl.uniform_matrix4fv_with_f32_array(
                    self.shadow_uniforms.light_view_proj.as_ref(),
                    false,
                    &light_view_proj,
                );
                for (chunk, _) in &visible {
                    if chunk.shadow_index_count == 0 {
                        continue;
                    }
                    gl.uniform3f(
                        self.shadow_uniforms.chunk_origin.as_ref(),
                        chunk.origin[0],
                        chunk.origin[1],
                        chunk.origin[2],
                    );
                    gl.bind_vertex_array(Some(&chunk.shadow_vao));
                    gl.draw_elements_with_i32(
                        Gl::TRIANGLES,
                        chunk.shadow_index_count,
                        Gl::UNSIGNED_INT,
                        0,
                    );
                }
                gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
                gl.viewport(0, 0, self.width, self.height);
                gl.color_mask(true, true, true, true);
            }

            // Main instanced splat pass.
            gl.use_program(Some(&self.program));
            let (projection, view) = camera_matrices(frame, self.width, self.height);
            gl.uniform_matrix4fv_with_f32_array(self.uniforms.projection.as_ref(), false, &projection);
            gl.uniform_matrix4fv_with_f32_array(self.uniforms.view.as_ref(), false, &view);
            gl.uniform3f(
                self.uniforms.camera_position.as_ref(),
                frame.camera_pos[0],
                frame.camera_pos[1],
                frame.camera_pos[2],
            );
            gl.uniform1i(
                self.uniforms.flashlight.as_ref(),
                if frame.flashlight { 1 } else { 0 },
            );
            gl.uniform1i(self.uniforms.light_count.as_ref(), global_light_count as i32);
            gl.uniform1i(self.uniforms.shadow_taps.as_ref(), self.profile.shadow_taps);
            gl.uniform1i(self.uniforms.outdoor.as_ref(), if env.outdoor { 1 } else { 0 });
            gl.uniform3f(
                self.uniforms.fog_color.as_ref(),
                env.fog_color[0],
                env.fog_color[1],
                env.fog_color[2],
            );
            gl.uniform1f(self.uniforms.ambient_scale.as_ref(), env.ambient_scale);

            // Flare cores: independent of the merged light slots so a flare
            // culled from the 4 shading lights still shows its ember.
            {
                let cores = frame.active_dynamic_lights();
                let mut core_vec = [0.0f32; 16];
                let mut core_col = [0.0f32; 12];
                for (i, c) in cores.iter().enumerate() {
                    core_vec[i * 4..i * 4 + 3].copy_from_slice(&c.position);
                    core_vec[i * 4 + 3] = c.intensity;
                    core_col[i * 3..i * 3 + 3].copy_from_slice(&c.color);
                }
                gl.uniform1i(self.uniforms.core_count.as_ref(), cores.len() as i32);
                gl.uniform4fv_with_f32_array(self.uniforms.cores.as_ref(), &core_vec);
                gl.uniform3fv_with_f32_array(self.uniforms.core_colors.as_ref(), &core_col);
            }
            // Always point the shadow sampler at unit 1 — left unset it
            // defaults to unit 0 and collides with the 3D light volume,
            // which voids every draw when no fixture lights are resident
            // (the grassland has none).
            gl.active_texture(Gl::TEXTURE1);
            gl.bind_texture(Gl::TEXTURE_2D, Some(&self.shadow_texture));
            gl.uniform1i(self.uniforms.shadow_map.as_ref(), 1);
            gl.uniform_matrix4fv_with_f32_array(
                self.uniforms.light_view_proj.as_ref(),
                false,
                &light_view_proj,
            );
            if global_light_count > 0 {
                gl.uniform3fv_with_f32_array(self.uniforms.light_positions.as_ref(), &light_positions);
                gl.uniform3fv_with_f32_array(self.uniforms.light_colors.as_ref(), &light_colors);
                gl.uniform4fv_with_f32_array(self.uniforms.light_params.as_ref(), &light_params);
            }
            gl.uniform1i(self.uniforms.shadowed_light_index.as_ref(), shadowed_light_index);

            for (chunk, _) in &visible {
                if stats.faces_drawn >= self.profile.face_budget {
                    stats.faces_dropped += chunk.instance_count as u32;
                    continue;
                }
                gl.uniform3f(
                    self.uniforms.chunk_origin.as_ref(),
                    chunk.origin[0],
                    chunk.origin[1],
                    chunk.origin[2],
                );
                gl.uniform3f(
                    self.uniforms.chunk_size.as_ref(),
                    chunk.bounds_max[0],
                    chunk.bounds_max[1],
                    chunk.bounds_max[2],
                );
                gl.uniform1f(self.uniforms.voxel_scale.as_ref(), chunk.voxel_scale);
                gl.active_texture(Gl::TEXTURE0);
                gl.bind_texture(Gl::TEXTURE_3D, Some(&chunk.light_texture));
                gl.uniform1i(self.uniforms.light_volume.as_ref(), 0);

                gl.bind_vertex_array(Some(&chunk.instance_vao));
                gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&chunk.instance_buffer));
                stats.chunks_drawn += 1;

                // Per-cell culling: cells are contiguous instance ranges, so
                // visible neighbors coalesce into one instanced draw. The
                // cell radius includes half a cell of slack because a capped
                // face's center can sit right on a cell border.
                let cell_radius = chunk.cell_size * 1.3;
                let mut run_start: Option<u32> = None;
                let mut run_count: u32 = 0;
                let flush =
                    |start: &mut Option<u32>, count: &mut u32, stats: &mut FrameStats| {
                        if let Some(first) = start.take() {
                            self.point_instance_attribs(first);
                            gl.draw_arrays_instanced(Gl::TRIANGLE_STRIP, 0, 4, *count as i32);
                            stats.faces_drawn += *count;
                            stats.draw_calls += 1;
                            *count = 0;
                        }
                    };
                for range in &chunk.cells {
                    let center = [
                        chunk.origin[0] + (range.cell[0] as f32 + 0.5) * chunk.cell_size,
                        chunk.origin[1] + (range.cell[1] as f32 + 0.5) * chunk.cell_size,
                        chunk.origin[2] + (range.cell[2] as f32 + 0.5) * chunk.cell_size,
                    ];
                    if sphere_visible(center, cell_radius, frame).is_some() {
                        if run_start.is_none() {
                            run_start = Some(range.offset);
                        }
                        run_count += range.count;
                    } else {
                        stats.cells_culled += 1;
                        flush(&mut run_start, &mut run_count, &mut stats);
                    }
                }
                flush(&mut run_start, &mut run_count, &mut stats);
            }
            gl.bind_vertex_array(None);
            gl.bind_buffer(Gl::ARRAY_BUFFER, None);
        }
        self.stats = stats;
        self.end_timer(timer_started);
    }
}
