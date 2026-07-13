//! Per-frame raymarch submission.
//!
//! Each helper uploads one coherent piece of frame state. The final
//! [`RendererPort::draw`] reads as the pipeline order: timer, target, program,
//! camera, lighting, chunks, atlas, fullscreen draw.

use web_sys::WebGl2RenderingContext as Gl;

use crate::application::atlas::MAX_CHUNKS;
use crate::application::ports::{ChunkDraw, FrameParams, RendererPort};

use super::{Uniforms, WebGl2Renderer, face_weights, fov_tan};

/// Reused uniform arrays. Their capacity is fixed at construction so steady
/// frame submission performs no heap allocations.
pub(super) struct ChunkUniformBuffers {
    origins: Vec<f32>,
    roots: Vec<i32>,
    sizes: Vec<f32>,
}

impl ChunkUniformBuffers {
    pub(super) fn with_capacity(chunks: usize) -> Self {
        Self {
            origins: Vec::with_capacity(chunks * 3),
            roots: Vec::with_capacity(chunks),
            sizes: Vec::with_capacity(chunks),
        }
    }
}

#[derive(Clone, Copy)]
struct CameraBasis {
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
}

fn camera_basis(frame: &FrameParams) -> CameraBasis {
    let (sin_pitch, cos_pitch) = frame.pitch.sin_cos();
    let (sin_yaw, cos_yaw) = frame.yaw.sin_cos();
    CameraBasis {
        right: [cos_yaw, 0.0, -sin_yaw],
        up: [sin_pitch * sin_yaw, cos_pitch, sin_pitch * cos_yaw],
        forward: [-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw],
    }
}

fn clear_target(gl: &Gl, frame: &FrameParams, width: i32, height: i32) {
    gl.viewport(0, 0, width, height);
    let environment = frame.environment;
    if environment.outdoor {
        gl.clear_color(
            environment.sky_color[0],
            environment.sky_color[1],
            environment.sky_color[2],
            1.0,
        );
    } else {
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
    }
    gl.clear(Gl::COLOR_BUFFER_BIT);
}

fn upload_camera(gl: &Gl, uniforms: &Uniforms, frame: &FrameParams, width: i32, height: i32) {
    gl.uniform3f(
        uniforms.camera_position.as_ref(),
        frame.camera_pos[0],
        frame.camera_pos[1],
        frame.camera_pos[2],
    );
    let basis = camera_basis(frame);
    gl.uniform3fv_with_f32_array(uniforms.cam_right.as_ref(), &basis.right);
    gl.uniform3fv_with_f32_array(uniforms.cam_up.as_ref(), &basis.up);
    gl.uniform3fv_with_f32_array(uniforms.cam_forward.as_ref(), &basis.forward);
    gl.uniform1f(
        uniforms.aspect.as_ref(),
        width as f32 / height.max(1) as f32,
    );
    gl.uniform1f(uniforms.fov_tan.as_ref(), fov_tan());
}

fn upload_face_shading(gl: &Gl, uniforms: &Uniforms) {
    let (top, bottom, x, z) = face_weights();
    gl.uniform1f(uniforms.face_weight_top.as_ref(), top);
    gl.uniform1f(uniforms.face_weight_bottom.as_ref(), bottom);
    gl.uniform1f(uniforms.face_weight_x.as_ref(), x);
    gl.uniform1f(uniforms.face_weight_z.as_ref(), z);
}

fn upload_environment(gl: &Gl, uniforms: &Uniforms, frame: &FrameParams) {
    let environment = frame.environment;
    gl.uniform1i(
        uniforms.outdoor.as_ref(),
        if environment.outdoor { 1 } else { 0 },
    );
    gl.uniform3fv_with_f32_array(uniforms.sky_color.as_ref(), &environment.sky_color);
    gl.uniform3fv_with_f32_array(uniforms.fog_color.as_ref(), &environment.fog_color);
    gl.uniform1f(uniforms.ambient_scale.as_ref(), environment.ambient_scale);
}

fn upload_dynamic_lights(gl: &Gl, uniforms: &Uniforms, frame: &FrameParams) {
    let lights = frame.active_dynamic_lights();
    let mut position_radius = [0.0f32; 16];
    let mut color_intensity = [0.0f32; 16];
    for (index, light) in lights.iter().enumerate() {
        let offset = index * 4;
        position_radius[offset..offset + 3].copy_from_slice(&light.position);
        position_radius[offset + 3] = light.radius;
        color_intensity[offset..offset + 3].copy_from_slice(&light.color);
        color_intensity[offset + 3] = light.intensity;
    }
    gl.uniform1i(uniforms.dynamic_light_count.as_ref(), lights.len() as i32);
    gl.uniform4fv_with_f32_array(uniforms.dynamic_pos_radius.as_ref(), &position_radius);
    gl.uniform4fv_with_f32_array(uniforms.dynamic_color_intensity.as_ref(), &color_intensity);
}

fn upload_chunks(
    gl: &Gl,
    uniforms: &Uniforms,
    buffers: &mut ChunkUniformBuffers,
    chunks: &[ChunkDraw],
) {
    let chunks = &chunks[..chunks.len().min(MAX_CHUNKS)];
    gl.uniform1i(uniforms.num_chunks.as_ref(), chunks.len() as i32);
    if chunks.is_empty() {
        return;
    }

    buffers.origins.clear();
    buffers.roots.clear();
    buffers.sizes.clear();
    for chunk in chunks {
        buffers.origins.extend_from_slice(&chunk.origin);
        buffers.roots.push(chunk.root_index);
        buffers.sizes.push(chunk.world_size);
    }
    gl.uniform3fv_with_f32_array(uniforms.chunk_origins.as_ref(), &buffers.origins);
    gl.uniform1iv_with_i32_array(uniforms.chunk_root_indices.as_ref(), &buffers.roots);
    gl.uniform1fv_with_f32_array(uniforms.chunk_world_sizes.as_ref(), &buffers.sizes);
}

impl RendererPort for WebGl2Renderer {
    fn gpu_frame_ms(&self) -> Option<f32> {
        self.timer.last_ms()
    }

    fn upload_atlas(&mut self, texels: &[u32]) {
        self.atlas.replace(&self.gl, texels);
    }

    fn upload_atlas_rows(&mut self, first_row: u32, texels: &[u32]) -> bool {
        self.atlas.update_rows(&self.gl, first_row, texels)
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        let toggles = crate::get_render_toggles();
        self.timer.poll(&self.gl);
        let timer_started = self.timer.begin(&self.gl, toggles.gpu_timer);

        {
            let gl = &self.gl;
            clear_target(gl, frame, self.width, self.height);
            gl.use_program(Some(&self.program));
            upload_camera(gl, &self.uniforms, frame, self.width, self.height);
            upload_face_shading(gl, &self.uniforms);
            gl.uniform1i(
                self.uniforms.flashlight.as_ref(),
                if frame.flashlight { 1 } else { 0 },
            );
            gl.uniform1i(
                self.uniforms.front_to_back.as_ref(),
                if toggles.front_to_back { 1 } else { 0 },
            );
            // OPTIMIZATION (rt_dither): optional noise hides color banding.
            gl.uniform1i(
                self.uniforms.dither.as_ref(),
                if toggles.dither { 1 } else { 0 },
            );
            upload_environment(gl, &self.uniforms, frame);
            upload_dynamic_lights(gl, &self.uniforms, frame);
            upload_chunks(gl, &self.uniforms, &mut self.chunk_uniforms, chunks);

            gl.active_texture(Gl::TEXTURE0);
            self.atlas.bind(gl);
            gl.uniform1i(self.uniforms.node_texture.as_ref(), 0);
            gl.bind_vertex_array(Some(&self.quad.vao));
            gl.draw_arrays(Gl::TRIANGLES, 0, 6);
            gl.bind_vertex_array(None);
        }

        self.timer.end(&self.gl, timer_started);
    }
}
