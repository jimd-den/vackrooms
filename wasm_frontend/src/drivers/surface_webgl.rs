//! Default WebGL2 renderer: indexed greedy meshes with fixed-function depth
//! visibility. The SVO remains in the chunk payload for collision and debug
//! traversal, but no visible fragment performs an octree walk.

use std::collections::HashMap;

use js_sys::{Object, Reflect, Uint8Array, Uint32Array};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram, WebGlQuery,
    WebGlShader, WebGlUniformLocation, WebGlVertexArrayObject,
};

use crate::application::ports::{
    ChunkDraw, FrameParams, RendererPort, SurfaceChunk, SurfaceChunkKey,
};
use crate::drivers::shaders::{SURFACE_FRAGMENT_SHADER, SURFACE_VERTEX_SHADER};

/// EXT_disjoint_timer_query_webgl2 constants, absent from WebGL2 core.
const TIME_ELAPSED_EXT: u32 = 0x88BF;
const GPU_DISJOINT_EXT: u32 = 0x8FBB;
const VERTEX_STRIDE: i32 = 10;
pub(crate) const MAX_DRAW_DISTANCE: f32 = 100.0;

struct Uniforms {
    projection: Option<WebGlUniformLocation>,
    view: Option<WebGlUniformLocation>,
    chunk_origin: Option<WebGlUniformLocation>,
    camera_position: Option<WebGlUniformLocation>,
    flashlight: Option<WebGlUniformLocation>,
    light_volume: Option<WebGlUniformLocation>,
    chunk_size: Option<WebGlUniformLocation>,
    light_count: Option<WebGlUniformLocation>,
    light_positions: Option<WebGlUniformLocation>,
    light_colors: Option<WebGlUniformLocation>,
    light_params: Option<WebGlUniformLocation>,
    shadow_map: Option<WebGlUniformLocation>,
    light_view_proj: Option<WebGlUniformLocation>,
    shadowed_light_index: Option<WebGlUniformLocation>,
}

struct ShadowUniforms {
    light_view_proj: Option<WebGlUniformLocation>,
    chunk_origin: Option<WebGlUniformLocation>,
}

struct GpuMesh {
    vao: WebGlVertexArrayObject,
    vertex_buffer: WebGlBuffer,
    index_buffer: WebGlBuffer,
    index_count: i32,
    origin: [f32; 3],
    bounds_max: [f32; 3],
    light_texture: web_sys::WebGlTexture,
    lights: Vec<crate::application::ports::LightSource>,
}

/// Surface rasterizer with incremental chunk mesh uploads. One VAO/VBO/IBO
/// tuple per resident chunk keeps the draw path allocation-free.
pub struct SurfaceRenderer {
    gl: Gl,
    program: WebGlProgram,
    uniforms: Uniforms,
    meshes: HashMap<SurfaceChunkKey, GpuMesh>,
    width: i32,
    height: i32,
    timer_extension: Option<Object>,
    pending_timer: Option<WebGlQuery>,
    last_gpu_ms: Option<f32>,
    shadow_program: WebGlProgram,
    shadow_uniforms: ShadowUniforms,
    shadow_fbo: web_sys::WebGlFramebuffer,
    shadow_texture: web_sys::WebGlTexture,
}

impl SurfaceRenderer {
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, JsValue> {
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

        let program = link_program(&gl, SURFACE_VERTEX_SHADER, SURFACE_FRAGMENT_SHADER)?;
        let uniforms = Uniforms {
            projection: gl.get_uniform_location(&program, "uProjection"),
            view: gl.get_uniform_location(&program, "uView"),
            chunk_origin: gl.get_uniform_location(&program, "uChunkOrigin"),
            camera_position: gl.get_uniform_location(&program, "uCameraPosition"),
            flashlight: gl.get_uniform_location(&program, "uFlashlightEnabled"),
            light_volume: gl.get_uniform_location(&program, "uLightVolume"),
            chunk_size: gl.get_uniform_location(&program, "uChunkSize"),
            light_count: gl.get_uniform_location(&program, "uLightCount"),
            light_positions: gl.get_uniform_location(&program, "uLightPositions"),
            light_colors: gl.get_uniform_location(&program, "uLightColors"),
            light_params: gl.get_uniform_location(&program, "uLightParams"),
            shadow_map: gl.get_uniform_location(&program, "uShadowMap"),
            light_view_proj: gl.get_uniform_location(&program, "uLightViewProjection"),
            shadowed_light_index: gl.get_uniform_location(&program, "uShadowedLightIndex"),
        };
        let timer_extension = gl
            .get_extension("EXT_disjoint_timer_query_webgl2")
            .ok()
            .flatten();

        let shadow_program = link_program(&gl, crate::drivers::shaders::SHADOW_VERTEX_SHADER, crate::drivers::shaders::SHADOW_FRAGMENT_SHADER)?;
        let shadow_uniforms = ShadowUniforms {
            light_view_proj: gl.get_uniform_location(&shadow_program, "uLightViewProjection"),
            chunk_origin: gl.get_uniform_location(&shadow_program, "uChunkOrigin"),
        };

        let shadow_texture = gl.create_texture().unwrap();
        gl.bind_texture(Gl::TEXTURE_2D, Some(&shadow_texture));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            Gl::TEXTURE_2D, 0, Gl::DEPTH_COMPONENT16 as i32, 128, 128, 0,
            Gl::DEPTH_COMPONENT, Gl::UNSIGNED_SHORT, None
        ).unwrap();
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);

        let shadow_fbo = gl.create_framebuffer().unwrap();
        gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&shadow_fbo));
        gl.framebuffer_texture_2d(Gl::FRAMEBUFFER, Gl::DEPTH_ATTACHMENT, Gl::TEXTURE_2D, Some(&shadow_texture), 0);
        gl.bind_framebuffer(Gl::FRAMEBUFFER, None);

        Ok(Self {
            gl,
            program,
            uniforms,
            meshes: HashMap::new(),
            width: canvas.width() as i32,
            height: canvas.height() as i32,
            timer_extension,
            pending_timer: None,
            last_gpu_ms: None,
            shadow_program,
            shadow_uniforms,
            shadow_fbo,
            shadow_texture,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width as i32;
        self.height = height as i32;
    }

    fn destroy_mesh(&self, mesh: GpuMesh) {
        self.gl.delete_vertex_array(Some(&mesh.vao));
        self.gl.delete_buffer(Some(&mesh.vertex_buffer));
        self.gl.delete_buffer(Some(&mesh.index_buffer));
        self.gl.delete_texture(Some(&mesh.light_texture));
    }

    fn upload_surface(&mut self, chunk: SurfaceChunk<'_>) {
        if let Some(old) = self.meshes.remove(&chunk.key) {
            self.destroy_mesh(old);
        }
        if chunk.mesh.indices.is_empty() || chunk.mesh.vertices.is_empty() {
            return;
        }

        let gl = &self.gl;
        let vao = gl.create_vertex_array().expect("create surface VAO");
        let vertex_buffer = gl.create_buffer().expect("create surface vertex buffer");
        let index_buffer = gl.create_buffer().expect("create surface index buffer");
        gl.bind_vertex_array(Some(&vao));

        let mut packed = Vec::with_capacity(chunk.mesh.vertices.len() * VERTEX_STRIDE as usize);
        for v in &chunk.mesh.vertices {
            packed.extend_from_slice(&v.position[0].to_le_bytes());
            packed.extend_from_slice(&v.position[1].to_le_bytes());
            packed.extend_from_slice(&v.position[2].to_le_bytes());
            packed.extend_from_slice(&[v.normal_axis, v.material, v.static_indirect, v.ao]);
        }
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&vertex_buffer));
        let vertex_bytes = Uint8Array::from(packed.as_slice());
        gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &vertex_bytes, Gl::STATIC_DRAW);

        gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&index_buffer));
        let index_data = Uint32Array::from(chunk.mesh.indices.as_slice());
        gl.buffer_data_with_array_buffer_view(
            Gl::ELEMENT_ARRAY_BUFFER,
            &index_data,
            Gl::STATIC_DRAW,
        );

        self.bind_attributes();
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
            chunk.mesh.light_volume_size[0] as i32,
            chunk.mesh.light_volume_size[1] as i32,
            chunk.mesh.light_volume_size[2] as i32,
            0,
            Gl::RGB,
            Gl::UNSIGNED_BYTE,
            Some(&chunk.mesh.light_volume),
        ).unwrap();
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_MIN_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_MAG_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_3D, Gl::TEXTURE_WRAP_R, Gl::CLAMP_TO_EDGE as i32);
        gl.bind_texture(Gl::TEXTURE_3D, None);
 
        self.meshes.insert(
            chunk.key,
            GpuMesh {
                vao,
                vertex_buffer,
                index_buffer,
                index_count: chunk.mesh.indices.len().min(i32::MAX as usize) as i32,
                origin: chunk.origin,
                bounds_max: chunk.mesh.bounds.max,
                light_texture,
                lights: chunk.mesh.lights.clone(),
            },
        );
    }

    fn bind_attributes(&self) {
        let gl = &self.gl;
        for (name, size, ty, offset) in [
            ("aPosition", 3, Gl::UNSIGNED_SHORT, 0),
            ("aNormalAxis", 1, Gl::UNSIGNED_BYTE, 6),
            ("aMaterial", 1, Gl::UNSIGNED_BYTE, 7),
            ("aStaticIndirect", 1, Gl::UNSIGNED_BYTE, 8),
            ("aAo", 1, Gl::UNSIGNED_BYTE, 9),
        ] {
            let location = gl.get_attrib_location(&self.program, name);
            assert!(location >= 0, "surface shader lost {name}");
            let location = location as u32;
            gl.enable_vertex_attrib_array(location);
            gl.vertex_attrib_pointer_with_i32(location, size, ty, false, VERTEX_STRIDE, offset);
        }
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

    fn mesh_visible(mesh: &GpuMesh, frame: &FrameParams) -> bool {
        let center = [
            mesh.origin[0] + mesh.bounds_max[0] * 0.5,
            mesh.origin[1] + mesh.bounds_max[1] * 0.5,
            mesh.origin[2] + mesh.bounds_max[2] * 0.5,
        ];
        let to = [
            center[0] - frame.camera_pos[0],
            center[1] - frame.camera_pos[1],
            center[2] - frame.camera_pos[2],
        ];
        let distance2 = to[0] * to[0] + to[1] * to[1] + to[2] * to[2];
        if distance2 > MAX_DRAW_DISTANCE * MAX_DRAW_DISTANCE {
            return false;
        }
        let (sp, cp) = frame.pitch.sin_cos();
        let (sy, cy) = frame.yaw.sin_cos();
        let forward = [-cp * sy, sp, -cp * cy];
        let dot = to[0] * forward[0] + to[1] * forward[1] + to[2] * forward[2];
        let radius = (mesh.bounds_max[0] * mesh.bounds_max[0]
            + mesh.bounds_max[1] * mesh.bounds_max[1]
            + mesh.bounds_max[2] * mesh.bounds_max[2])
            .sqrt()
            * 0.5;
        dot >= -radius
    }
}

impl RendererPort for SurfaceRenderer {
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
            if let Some(mesh) = self.meshes.remove(key) {
                self.destroy_mesh(mesh);
            }
        }
    }

    fn clear_surfaces(&mut self) {
        let meshes = std::mem::take(&mut self.meshes);
        for (_, mesh) in meshes {
            self.destroy_mesh(mesh);
        }
    }

    fn gpu_frame_ms(&self) -> Option<f32> {
        self.last_gpu_ms
    }

    fn upload_atlas(&mut self, _texels: &[u32]) {}

    fn draw(&mut self, frame: &FrameParams, _chunks: &[ChunkDraw]) {
        self.poll_timer();
        let timer_started = self.begin_timer();
        {
            let gl = &self.gl;
            gl.viewport(0, 0, self.width, self.height);
            // Match the shader's fog color to hide ungenerated chunk boundaries
            gl.clear_color(0.15, 0.125, 0.055, 1.0);
            gl.clear_depth(1.0);
            gl.clear(Gl::COLOR_BUFFER_BIT | Gl::DEPTH_BUFFER_BIT);
            gl.enable(Gl::DEPTH_TEST);
            gl.depth_func(Gl::LEQUAL);
            gl.depth_mask(true);
            gl.enable(Gl::CULL_FACE);
            gl.cull_face(Gl::BACK);
            gl.use_program(Some(&self.program));

            let (projection, view) = camera_matrices(frame, self.width, self.height);
            gl.uniform_matrix4fv_with_f32_array(
                self.uniforms.projection.as_ref(),
                false,
                &projection,
            );
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

            // 1. Collect all visible meshes
            let mut visible_meshes = Vec::new();
            for mesh in self.meshes.values() {
                if Self::mesh_visible(mesh, frame) {
                    visible_meshes.push(mesh);
                }
            }

            // 2. Collect unique lights across all visible chunks, deduplicated by id
            let mut unique_lights = HashMap::new();
            for mesh in &visible_meshes {
                for light in &mesh.lights {
                    if light.enabled {
                        unique_lights.insert(light.id, light);
                    }
                }
            }

            // 3. Sort lights by importance near the player (radius / (distance^2 + 0.1))
            let mut lights_with_importance: Vec<(&crate::application::ports::LightSource, f32)> = unique_lights
                .values()
                .map(|&light| {
                    let dx = light.position[0] - frame.camera_pos[0];
                    let dy = light.position[1] - frame.camera_pos[1];
                    let dz = light.position[2] - frame.camera_pos[2];
                    let dist2 = dx * dx + dy * dy + dz * dz;
                    let importance = light.radius / (dist2 + 0.1);
                    (light, importance)
                })
                .collect();
            lights_with_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Select up to 4 lights for the global shader uniforms
            let global_light_count = lights_with_importance.len().min(4);
            let mut global_light_positions = [0.0; 12];
            let mut global_light_colors = [0.0; 12];
            let mut global_light_params = [0.0; 16];
            
            for i in 0..global_light_count {
                let light = lights_with_importance[i].0;
                global_light_positions[i * 3] = light.position[0];
                global_light_positions[i * 3 + 1] = light.position[1];
                global_light_positions[i * 3 + 2] = light.position[2];
                global_light_colors[i * 3] = light.color[0];
                global_light_colors[i * 3 + 1] = light.color[1];
                global_light_colors[i * 3 + 2] = light.color[2];
                global_light_params[i * 4] = light.radius;
                global_light_params[i * 4 + 1] = light.intensity;
                global_light_params[i * 4 + 2] = light.half_size[0];
                global_light_params[i * 4 + 3] = light.half_size[1];
            }

            let mut shadowed_light_index = -1;
            let mut light_view_proj = [0.0; 16];

            if global_light_count > 0 {
                // The top light is the hero light
                shadowed_light_index = 0;
                let hero_light = lights_with_importance[0].0;
                let lx = hero_light.position[0];
                let ly = hero_light.position[1];
                let lz = hero_light.position[2];

                // Derive shadow coverage and far depth from light range/height
                let range = hero_light.radius;
                let ortho_half = range * 0.75; // Focus the 128x128 resolution slightly
                let far_depth = range * 1.2;
                
                let proj = ortho_matrix(-ortho_half, ortho_half, -ortho_half, ortho_half, 0.1, far_depth);
                let view = look_at_matrix_down([lx, ly, lz]);
                light_view_proj = multiply_matrices(&proj, &view);

                // --- GLOBAL SHADOW PASS ---
                // Render ALL visible meshes into the shadow map
                gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&self.shadow_fbo));
                gl.viewport(0, 0, 128, 128);
                gl.color_mask(false, false, false, false);
                gl.clear_depth(1.0);
                gl.clear(Gl::DEPTH_BUFFER_BIT);
                gl.use_program(Some(&self.shadow_program));

                gl.uniform_matrix4fv_with_f32_array(self.shadow_uniforms.light_view_proj.as_ref(), false, &light_view_proj);
                
                for mesh in &visible_meshes {
                    gl.uniform3f(self.shadow_uniforms.chunk_origin.as_ref(), mesh.origin[0], mesh.origin[1], mesh.origin[2]);
                    gl.bind_vertex_array(Some(&mesh.vao));
                    gl.draw_elements_with_i32(Gl::TRIANGLES, mesh.index_count, Gl::UNSIGNED_INT, 0);
                }

                // Restore state for main pass
                gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
                gl.viewport(0, 0, self.width, self.height);
                gl.color_mask(true, true, true, true);
            }

            gl.use_program(Some(&self.program));

            // Upload camera position and flashlight status
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

            // Upload global lights (done once per frame)
            gl.uniform1i(self.uniforms.light_count.as_ref(), global_light_count as i32);
            if global_light_count > 0 {
                gl.uniform3fv_with_f32_array(self.uniforms.light_positions.as_ref(), &global_light_positions);
                gl.uniform3fv_with_f32_array(self.uniforms.light_colors.as_ref(), &global_light_colors);
                gl.uniform4fv_with_f32_array(self.uniforms.light_params.as_ref(), &global_light_params);

                gl.active_texture(Gl::TEXTURE1);
                gl.bind_texture(Gl::TEXTURE_2D, Some(&self.shadow_texture));
                gl.uniform1i(self.uniforms.shadow_map.as_ref(), 1);
                gl.uniform_matrix4fv_with_f32_array(self.uniforms.light_view_proj.as_ref(), false, &light_view_proj);
                gl.uniform1i(self.uniforms.shadowed_light_index.as_ref(), shadowed_light_index);
            } else {
                gl.uniform1i(self.uniforms.shadowed_light_index.as_ref(), -1);
            }

            for mesh in &visible_meshes {
                gl.uniform3f(
                    self.uniforms.chunk_origin.as_ref(),
                    mesh.origin[0],
                    mesh.origin[1],
                    mesh.origin[2],
                );
                gl.uniform3f(
                    self.uniforms.chunk_size.as_ref(),
                    mesh.bounds_max[0],
                    mesh.bounds_max[1],
                    mesh.bounds_max[2],
                );
                gl.active_texture(Gl::TEXTURE0);
                gl.bind_texture(Gl::TEXTURE_3D, Some(&mesh.light_texture));
                gl.uniform1i(self.uniforms.light_volume.as_ref(), 0);

                gl.bind_vertex_array(Some(&mesh.vao));
                gl.draw_elements_with_i32(Gl::TRIANGLES, mesh.index_count, Gl::UNSIGNED_INT, 0);
            }
            gl.bind_vertex_array(None);
        }
        self.end_timer(timer_started);
    }
}

pub(crate) fn camera_matrices(frame: &FrameParams, width: i32, height: i32) -> ([f32; 16], [f32; 16]) {
    let (sp, cp) = frame.pitch.sin_cos();
    let (sy, cy) = frame.yaw.sin_cos();
    let right = [cy, 0.0, -sy];
    let up = [sp * sy, cp, sp * cy];
    let forward = [-cp * sy, sp, -cp * cy];
    let p = frame.camera_pos;
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let view = [
        right[0],
        up[0],
        -forward[0],
        0.0,
        right[1],
        up[1],
        -forward[1],
        0.0,
        right[2],
        up[2],
        -forward[2],
        0.0,
        -dot(right, p),
        -dot(up, p),
        dot(forward, p),
        1.0,
    ];
    let near = 0.03;
    let far = MAX_DRAW_DISTANCE + 10.0;
    let f = 1.0 / crate::drivers::webgl::fov_tan();
    let aspect = width as f32 / height.max(1) as f32;
    let projection = [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        (far + near) / (near - far),
        -1.0,
        0.0,
        0.0,
        (2.0 * far * near) / (near - far),
        0.0,
    ];
    (projection, view)
}

pub(crate) fn ortho_matrix(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [f32; 16] {
    let mut m = [0.0; 16];
    m[0] = 2.0 / (right - left);
    m[5] = 2.0 / (top - bottom);
    m[10] = -2.0 / (far - near);
    m[12] = -(right + left) / (right - left);
    m[13] = -(top + bottom) / (top - bottom);
    m[14] = -(far + near) / (far - near);
    m[15] = 1.0;
    m
}

pub(crate) fn look_at_matrix_down(eye: [f32; 3]) -> [f32; 16] {
    // Looking straight down (-Y), up is -Z
    let forward = [0.0, -1.0, 0.0];
    let up = [0.0, 0.0, -1.0];
    let right = [1.0, 0.0, 0.0];
    
    let mut m = [0.0; 16];
    m[0] = right[0];
    m[1] = up[0];
    m[2] = -forward[0];
    
    m[4] = right[1];
    m[5] = up[1];
    m[6] = -forward[1];
    
    m[8] = right[2];
    m[9] = up[2];
    m[10] = -forward[2];
    
    m[12] = -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]);
    m[13] = -(up[0] * eye[0] + up[1] * eye[1] + up[2] * eye[2]);
    m[14] = -(-forward[0] * eye[0] + -forward[1] * eye[1] + -forward[2] * eye[2]);
    m[15] = 1.0;
    m
}

pub(crate) fn multiply_matrices(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut m = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            m[col * 4 + row] = 
                a[0 * 4 + row] * b[col * 4 + 0] +
                a[1 * 4 + row] * b[col * 4 + 1] +
                a[2 * 4 + row] * b[col * 4 + 2] +
                a[3 * 4 + row] * b[col * 4 + 3];
        }
    }
    m
}

pub(crate) fn compile_shader(gl: &Gl, kind: u32, source: &str) -> Result<WebGlShader, JsValue> {
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

pub(crate) fn link_program(gl: &Gl, vertex: &str, fragment: &str) -> Result<WebGlProgram, JsValue> {
    let vs = compile_shader(gl, Gl::VERTEX_SHADER, vertex)?;
    let fs = compile_shader(gl, Gl::FRAGMENT_SHADER, fragment)?;
    let program = gl
        .create_program()
        .ok_or_else(|| JsValue::from_str("failed to create shader program"))?;
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
        gl.delete_program(Some(&program));
        Err(JsValue::from_str(&format!("shader link error: {log}")))
    }
}
