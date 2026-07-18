//! Per-frame raymarch submission.
//!
//! Each helper uploads one coherent piece of frame state. The final
//! [`RendererPort::draw`] reads as the pipeline order: timer, target, program,
//! camera, lighting, chunks, atlas, fullscreen draw.

use web_sys::WebGl2RenderingContext as Gl;

use crate::application::atlas::MAX_CHUNKS;
use crate::application::ports::{ChunkDraw, FrameParams, RendererPort};
use crate::application::rendering::encode_display_color;
use crate::drivers::gl::cluster_scene_lights::{LightRange, append_lights_reaching_bounds};
use crate::drivers::gl::upload_scene_lights::SceneLightTexture;

use super::{Uniforms, WebGl2Renderer, fov_tan};

/// Reused uniform arrays. Their capacity is fixed at construction so steady
/// frame submission performs no heap allocations.
pub(super) struct ChunkUniformBuffers {
    origins: Vec<f32>,
    roots: Vec<i32>,
    sizes: Vec<f32>,
    voxel_sizes: Vec<f32>,
    depths: Vec<i32>,
    light_first: Vec<i32>,
    light_counts: Vec<i32>,
}

impl ChunkUniformBuffers {
    pub(super) fn with_capacity(chunks: usize) -> Self {
        Self {
            origins: Vec::with_capacity(chunks * 3),
            roots: Vec::with_capacity(chunks),
            sizes: Vec::with_capacity(chunks),
            voxel_sizes: Vec::with_capacity(chunks),
            depths: Vec::with_capacity(chunks),
            light_first: Vec::with_capacity(chunks),
            light_counts: Vec::with_capacity(chunks),
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
    let atmosphere = if environment.outdoor {
        environment.sky_color
    } else {
        environment.fog_color
    };
    let clear = encode_display_color(atmosphere);
    gl.clear_color(clear[0], clear[1], clear[2], 1.0);
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

fn upload_environment(gl: &Gl, uniforms: &Uniforms, frame: &FrameParams) {
    let environment = frame.environment;
    gl.uniform1i(
        uniforms.outdoor.as_ref(),
        if environment.outdoor { 1 } else { 0 },
    );
    gl.uniform3fv_with_f32_array(uniforms.sky_color.as_ref(), &environment.sky_color);
    gl.uniform3fv_with_f32_array(uniforms.fog_color.as_ref(), &environment.fog_color);
    gl.uniform1f(uniforms.ambient_scale.as_ref(), environment.ambient_scale);
    gl.uniform1f(uniforms.fog_density.as_ref(), environment.fog_density);
    gl.uniform1f(uniforms.fog_start.as_ref(), environment.fog_start);
}

fn bind_scene_lights(gl: &Gl, uniforms: &Uniforms, texture: &SceneLightTexture) {
    gl.active_texture(Gl::TEXTURE1);
    texture.bind(gl);
    gl.uniform1i(uniforms.scene_light_texture.as_ref(), 1);
    gl.uniform1i(uniforms.scene_light_texture_width.as_ref(), texture.width());
}

fn cluster_scene_lights(
    frame: &FrameParams,
    chunks: &[ChunkDraw],
) -> (Vec<crate::application::ports::LightSource>, Vec<LightRange>) {
    let mut clustered = Vec::new();
    let ranges = chunks
        .iter()
        .take(MAX_CHUNKS)
        .map(|chunk| {
            let bounds_max = [
                chunk.origin[0] + chunk.world_size,
                chunk.origin[1] + chunk.world_size,
                chunk.origin[2] + chunk.world_size,
            ];
            append_lights_reaching_bounds(
                frame.active_scene_lights(),
                chunk.origin,
                bounds_max,
                &mut clustered,
            )
        })
        .collect();
    (clustered, ranges)
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
    light_ranges: &[LightRange],
) {
    let chunks = &chunks[..chunks.len().min(MAX_CHUNKS)];
    gl.uniform1i(uniforms.num_chunks.as_ref(), chunks.len() as i32);
    if chunks.is_empty() {
        return;
    }

    buffers.origins.clear();
    buffers.roots.clear();
    buffers.sizes.clear();
    buffers.voxel_sizes.clear();
    buffers.depths.clear();
    buffers.light_first.clear();
    buffers.light_counts.clear();
    for (index, chunk) in chunks.iter().enumerate() {
        buffers.origins.extend_from_slice(&chunk.origin);
        buffers.roots.push(chunk.root_index);
        buffers.sizes.push(chunk.world_size);
        buffers.voxel_sizes.push(chunk.voxel_size);
        buffers.depths.push(chunk.svo_depth as i32);
        let range = light_ranges
            .get(index)
            .copied()
            .unwrap_or(LightRange { first: 0, count: 0 });
        buffers.light_first.push(range.first);
        buffers.light_counts.push(range.count);
    }
    gl.uniform3fv_with_f32_array(uniforms.chunk_origins.as_ref(), &buffers.origins);
    gl.uniform1iv_with_i32_array(uniforms.chunk_root_indices.as_ref(), &buffers.roots);
    gl.uniform1fv_with_f32_array(uniforms.chunk_world_sizes.as_ref(), &buffers.sizes);
    gl.uniform1fv_with_f32_array(uniforms.chunk_voxel_sizes.as_ref(), &buffers.voxel_sizes);
    gl.uniform1iv_with_i32_array(uniforms.chunk_depths.as_ref(), &buffers.depths);
    gl.uniform1iv_with_i32_array(uniforms.chunk_light_first.as_ref(), &buffers.light_first);
    gl.uniform1iv_with_i32_array(uniforms.chunk_light_counts.as_ref(), &buffers.light_counts);
}

impl RendererPort for WebGl2Renderer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        self.atlas.replace(&self.gl, texels);
    }

    fn upload_atlas_rows(&mut self, first_row: u32, texels: &[u32]) -> bool {
        self.atlas.update_rows(&self.gl, first_row, texels)
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        let toggles = crate::get_render_toggles();
        self.timer.poll(&self.gl);
        let timer_started = self.timer.begin(&self.gl, false);
        let (clustered_lights, mut light_ranges) = cluster_scene_lights(frame, chunks);
        if self
            .scene_lights
            .upload(&self.gl, &clustered_lights)
            .is_err()
        {
            for range in &mut light_ranges {
                range.count = 0;
            }
        }

        {
            let gl = &self.gl;
            clear_target(gl, frame, self.width, self.height);
            gl.use_program(Some(&self.program));
            upload_camera(gl, &self.uniforms, frame, self.width, self.height);
            gl.uniform1i(
                self.uniforms.flashlight.as_ref(),
                if frame.flashlight { 1 } else { 0 },
            );
            gl.uniform1i(
                self.uniforms.front_to_back.as_ref(),
                if toggles.front_to_back { 1 } else { 0 },
            );
            gl.uniform1i(
                self.uniforms.empty_space_skip.as_ref(),
                if toggles.empty_space_skip { 1 } else { 0 },
            );
            gl.uniform1i(
                self.uniforms.direct_visibility.as_ref(),
                if toggles.shadow_pass { 1 } else { 0 },
            );
            // OPTIMIZATION (rt_dither): optional noise hides color banding.
            gl.uniform1i(
                self.uniforms.dither.as_ref(),
                if toggles.dither { 1 } else { 0 },
            );
            upload_environment(gl, &self.uniforms, frame);
            bind_scene_lights(gl, &self.uniforms, &self.scene_lights);
            upload_dynamic_lights(gl, &self.uniforms, frame);
            upload_chunks(
                gl,
                &self.uniforms,
                &mut self.chunk_uniforms,
                chunks,
                &light_ranges,
            );

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
