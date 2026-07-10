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
const MAX_DRAW_DISTANCE: f32 = 100.0;

struct Uniforms {
    projection: Option<WebGlUniformLocation>,
    view: Option<WebGlUniformLocation>,
    chunk_origin: Option<WebGlUniformLocation>,
    camera_position: Option<WebGlUniformLocation>,
    flashlight: Option<WebGlUniformLocation>,
}

struct GpuMesh {
    vao: WebGlVertexArrayObject,
    vertex_buffer: WebGlBuffer,
    index_buffer: WebGlBuffer,
    index_count: i32,
    origin: [f32; 3],
    bounds_max: [f32; 3],
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
        };
        let timer_extension = gl
            .get_extension("EXT_disjoint_timer_query_webgl2")
            .ok()
            .flatten();

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
            packed.extend_from_slice(&[v.normal_axis, v.material, v.light, v.ao]);
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

        self.meshes.insert(
            chunk.key,
            GpuMesh {
                vao,
                vertex_buffer,
                index_buffer,
                index_count: chunk.mesh.indices.len().min(i32::MAX as usize) as i32,
                origin: chunk.origin,
                bounds_max: chunk.mesh.bounds.max,
            },
        );
    }

    fn bind_attributes(&self) {
        let gl = &self.gl;
        for (name, size, ty, offset) in [
            ("aPosition", 3, Gl::UNSIGNED_SHORT, 0),
            ("aNormalAxis", 1, Gl::UNSIGNED_BYTE, 6),
            ("aMaterial", 1, Gl::UNSIGNED_BYTE, 7),
            ("aLight", 1, Gl::UNSIGNED_BYTE, 8),
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
            gl.clear_color(0.018, 0.016, 0.009, 1.0);
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

            for mesh in self.meshes.values() {
                if !Self::mesh_visible(mesh, frame) {
                    continue;
                }
                gl.uniform3f(
                    self.uniforms.chunk_origin.as_ref(),
                    mesh.origin[0],
                    mesh.origin[1],
                    mesh.origin[2],
                );
                gl.bind_vertex_array(Some(&mesh.vao));
                gl.draw_elements_with_i32(Gl::TRIANGLES, mesh.index_count, Gl::UNSIGNED_INT, 0);
            }
            gl.bind_vertex_array(None);
        }
        self.end_timer(timer_started);
    }
}

fn camera_matrices(frame: &FrameParams, width: i32, height: i32) -> ([f32; 16], [f32; 16]) {
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

fn link_program(gl: &Gl, vertex: &str, fragment: &str) -> Result<WebGlProgram, JsValue> {
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
