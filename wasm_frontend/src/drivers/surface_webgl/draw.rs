//! Per-frame indexed-surface pipeline.
//!
//! The submission order is deliberately visible in [`render_frame`]:
//!
//! 1. clear the target and establish fixed-function state;
//! 2. collect chunks accepted by the optional visibility optimization;
//! 3. select fixture and dynamic lights;
//! 4. optionally render the hero-light shadow map;
//! 5. bind uniforms that are constant for the frame;
//! 6. bind and draw each visible mesh.
//!
//! Each optional branch is labeled with its `rt_*` switch. Disabling an
//! optimization keeps the same pipeline and selects its reference path.

use web_sys::WebGl2RenderingContext as Gl;

use crate::application::ports::{
    ChunkDraw, FrameParams, RendererPort, SurfaceChunk, SurfaceChunkKey,
};
use crate::application::render_settings::RenderToggles;
use crate::drivers::gl::lights::{
    SelectedLights, dynamic_light_sources, flare_cores, select_lights,
};
use crate::drivers::gl::math::{
    camera_matrices, look_at_matrix_down, multiply_matrices, ortho_matrix,
};
use crate::drivers::gl::visibility::{MAX_DRAW_DISTANCE, chunk_bounding_sphere, sphere_visible};

use super::SurfaceRenderer;
use super::resources::GpuMesh;

/// Shadow uniforms always receive a complete state. The default disables the
/// lookup and supplies a deterministic matrix when no hero pass ran.
struct ShadowState {
    light_index: i32,
    light_view_projection: [f32; 16],
}

impl Default for ShadowState {
    fn default() -> Self {
        Self {
            light_index: -1,
            light_view_projection: [0.0; 16],
        }
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
        self.timer.last_ms()
    }

    fn upload_atlas(&mut self, _texels: &[u32]) {}

    fn draw(&mut self, frame: &FrameParams, _chunks: &[ChunkDraw]) {
        let toggles = crate::get_render_toggles();

        // OPTIMIZATION (rt_timer): asynchronous timing never waits for the
        // GPU. Disabling it prevents new queries while preserving rendering.
        self.timer.poll(&self.gl);
        let timer_started = self.timer.begin(&self.gl, toggles.gpu_timer);

        render_frame(self, frame, toggles);
        self.timer.end(&self.gl, timer_started);
    }
}

fn render_frame(renderer: &SurfaceRenderer, frame: &FrameParams, toggles: RenderToggles) {
    begin_main_frame(renderer, frame);

    let resident: Vec<&GpuMesh> = renderer.meshes.values().collect();
    let visible = collect_visible_meshes(&resident, frame, toggles.distance_cull);
    // OPTIMIZATION (rt_bake): the bake replaces static analytic fixtures; it
    // is never added on top of them. Runtime flares stay analytic in both
    // paths. This prevents the former direct-light double count.
    let lights = select_frame_lights(frame, !toggles.baked_lighting);
    let shadow = render_shadow_pass(renderer, &resident, &lights, toggles.shadow_pass);

    bind_frame_uniforms(
        renderer,
        frame,
        &lights,
        &shadow,
        toggles.dither,
        toggles.baked_lighting,
    );
    draw_visible_meshes(renderer, &visible);
}

/// Establishes every fixed-function state relied on by either raster pass.
fn begin_main_frame(renderer: &SurfaceRenderer, frame: &FrameParams) {
    let gl = &renderer.gl;
    gl.viewport(0, 0, renderer.width, renderer.height);

    // Match unloaded space to the level atmosphere so generated chunk edges
    // disappear into fog rather than exposing the canvas clear color.
    let environment = frame.environment;
    if environment.outdoor {
        gl.clear_color(
            environment.sky_color[0],
            environment.sky_color[1],
            environment.sky_color[2],
            1.0,
        );
    } else {
        gl.clear_color(0.15, 0.125, 0.055, 1.0);
    }
    gl.clear_depth(1.0);
    gl.clear(Gl::COLOR_BUFFER_BIT | Gl::DEPTH_BUFFER_BIT);
    gl.enable(Gl::DEPTH_TEST);
    gl.depth_func(Gl::LEQUAL);
    gl.depth_mask(true);
    gl.enable(Gl::CULL_FACE);
    gl.cull_face(Gl::BACK);
}

/// OPTIMIZATION (rt_cull): conservative distance and behind-camera sphere
/// rejection. The disabled reference path submits every resident mesh.
fn collect_visible_meshes<'a>(
    resident: &[&'a GpuMesh],
    frame: &FrameParams,
    culling_enabled: bool,
) -> Vec<&'a GpuMesh> {
    resident
        .iter()
        .copied()
        .filter(|mesh| {
            if !culling_enabled {
                return true;
            }
            let (center, radius) = chunk_bounding_sphere(mesh.origin, mesh.bounds_max);
            sphere_visible(center, radius, frame, MAX_DRAW_DISTANCE).is_some()
        })
        .collect()
}

/// Selects analytic fixtures when the reference path needs them and always
/// adds frame-local dynamic lights.
fn select_frame_lights(frame: &FrameParams, include_static_fixtures: bool) -> SelectedLights {
    let dynamic = dynamic_light_sources(frame);
    let fixtures = if include_static_fixtures {
        frame.active_scene_lights()
    } else {
        &[]
    };
    select_lights(fixtures.iter(), &dynamic, frame)
}

/// FEATURE (rt_shadows): renders every resident greedy mesh into the selected
/// fixture's top-down depth map. Camera culling must not remove an off-screen
/// caster whose shadow lands on a visible receiver. With the feature
/// disabled, the returned state makes the main shader skip shadow sampling.
fn render_shadow_pass(
    renderer: &SurfaceRenderer,
    casters: &[&GpuMesh],
    lights: &SelectedLights,
    enabled: bool,
) -> ShadowState {
    if !enabled || lights.hero_slot < 0 {
        return ShadowState::default();
    }

    let range = lights.hero_radius;
    let ortho_half = range * 0.75;
    let projection = ortho_matrix(
        -ortho_half,
        ortho_half,
        -ortho_half,
        ortho_half,
        0.1,
        range * 1.2,
    );
    let view = look_at_matrix_down(lights.hero_position);
    let light_view_projection = multiply_matrices(&projection, &view);

    let gl = &renderer.gl;
    gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&renderer.shadow.fbo));
    gl.viewport(0, 0, renderer.shadow.size, renderer.shadow.size);
    gl.color_mask(false, false, false, false);
    gl.clear_depth(1.0);
    gl.clear(Gl::DEPTH_BUFFER_BIT);
    gl.use_program(Some(&renderer.shadow_program));
    gl.uniform_matrix4fv_with_f32_array(
        renderer.shadow_uniforms.light_view_proj.as_ref(),
        false,
        &light_view_projection,
    );

    for mesh in casters {
        gl.uniform3f(
            renderer.shadow_uniforms.chunk_origin.as_ref(),
            mesh.origin[0],
            mesh.origin[1],
            mesh.origin[2],
        );
        gl.bind_vertex_array(Some(&mesh.vao));
        gl.draw_elements_with_i32(Gl::TRIANGLES, mesh.index_count, Gl::UNSIGNED_INT, 0);
    }

    // Restore the default target state required by the already-cleared main
    // pass. Depth/cull state is shared and remains intentionally unchanged.
    gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
    gl.viewport(0, 0, renderer.width, renderer.height);
    gl.color_mask(true, true, true, true);

    ShadowState {
        light_index: lights.hero_slot,
        light_view_projection,
    }
}

/// Binds state shared by every visible mesh exactly once per frame.
fn bind_frame_uniforms(
    renderer: &SurfaceRenderer,
    frame: &FrameParams,
    lights: &SelectedLights,
    shadow: &ShadowState,
    dither_enabled: bool,
    baked_lighting_enabled: bool,
) {
    let gl = &renderer.gl;
    let uniforms = &renderer.uniforms;
    gl.use_program(Some(&renderer.program));

    let (projection, view) =
        camera_matrices(frame, renderer.width, renderer.height, MAX_DRAW_DISTANCE);
    gl.uniform_matrix4fv_with_f32_array(uniforms.projection.as_ref(), false, &projection);
    gl.uniform_matrix4fv_with_f32_array(uniforms.view.as_ref(), false, &view);
    gl.uniform3f(
        uniforms.camera_position.as_ref(),
        frame.camera_pos[0],
        frame.camera_pos[1],
        frame.camera_pos[2],
    );

    let forward = camera_forward(frame);
    gl.uniform3fv_with_f32_array(uniforms.cam_forward.as_ref(), &forward);
    gl.uniform1i(
        uniforms.flashlight.as_ref(),
        if frame.flashlight { 1 } else { 0 },
    );
    gl.uniform1i(uniforms.dither.as_ref(), if dither_enabled { 1 } else { 0 });
    gl.uniform1i(
        uniforms.baked_lighting.as_ref(),
        if baked_lighting_enabled { 1 } else { 0 },
    );

    let environment = frame.environment;
    gl.uniform1i(
        uniforms.outdoor.as_ref(),
        if environment.outdoor { 1 } else { 0 },
    );
    gl.uniform3f(
        uniforms.fog_color.as_ref(),
        environment.fog_color[0],
        environment.fog_color[1],
        environment.fog_color[2],
    );
    gl.uniform1f(uniforms.ambient_scale.as_ref(), environment.ambient_scale);
    gl.uniform1f(uniforms.fog_density.as_ref(), environment.fog_density);
    gl.uniform1f(uniforms.fog_start.as_ref(), environment.fog_start);

    bind_light_uniforms(renderer, frame, lights, shadow);
}

fn camera_forward(frame: &FrameParams) -> [f32; 3] {
    let (sin_pitch, cos_pitch) = frame.pitch.sin_cos();
    let (sin_yaw, cos_yaw) = frame.yaw.sin_cos();
    [-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw]
}

fn bind_light_uniforms(
    renderer: &SurfaceRenderer,
    frame: &FrameParams,
    lights: &SelectedLights,
    shadow: &ShadowState,
) {
    let gl = &renderer.gl;
    let uniforms = &renderer.uniforms;
    gl.uniform1i(uniforms.light_count.as_ref(), lights.count as i32);
    if lights.count > 0 {
        gl.uniform3fv_with_f32_array(uniforms.light_positions.as_ref(), &lights.positions);
        gl.uniform3fv_with_f32_array(uniforms.light_colors.as_ref(), &lights.colors);
        gl.uniform4fv_with_f32_array(uniforms.light_params.as_ref(), &lights.params);
        gl.uniform1iv_with_i32_array(uniforms.light_kinds.as_ref(), &lights.kinds);
    }

    // Flare cores are independent from the selected shading-light slots, so
    // an unselected flare still renders its small emissive ember.
    let cores = flare_cores(frame);
    gl.uniform1i(uniforms.core_count.as_ref(), cores.count);
    gl.uniform4fv_with_f32_array(uniforms.cores.as_ref(), &cores.pos_intensity);
    gl.uniform3fv_with_f32_array(uniforms.core_colors.as_ref(), &cores.colors);

    // sampler3D and sampler2D may never alias, even when no hero is active:
    // WebGL rejects the draw if both default to texture unit zero.
    gl.active_texture(Gl::TEXTURE1);
    gl.bind_texture(Gl::TEXTURE_2D, Some(&renderer.shadow.texture));
    gl.uniform1i(uniforms.shadow_map.as_ref(), 1);
    gl.uniform_matrix4fv_with_f32_array(
        uniforms.light_view_proj.as_ref(),
        false,
        &shadow.light_view_projection,
    );
    gl.uniform1i(uniforms.shadowed_light_index.as_ref(), shadow.light_index);
}

fn draw_visible_meshes(renderer: &SurfaceRenderer, visible: &[&GpuMesh]) {
    for mesh in visible {
        draw_mesh(renderer, mesh);
    }
    renderer.gl.bind_vertex_array(None);
}

/// Binds the only per-mesh state: local transform bounds, baked light volume,
/// and indexed geometry.
fn draw_mesh(renderer: &SurfaceRenderer, mesh: &GpuMesh) {
    let gl = &renderer.gl;
    let uniforms = &renderer.uniforms;
    gl.uniform3f(
        uniforms.chunk_origin.as_ref(),
        mesh.origin[0],
        mesh.origin[1],
        mesh.origin[2],
    );
    gl.uniform1f(uniforms.voxel_size.as_ref(), mesh.voxel_size);
    gl.active_texture(Gl::TEXTURE0);
    gl.bind_texture(Gl::TEXTURE_3D, Some(&mesh.light_texture));
    gl.uniform1i(uniforms.light_volume.as_ref(), 0);

    gl.bind_vertex_array(Some(&mesh.vao));
    gl.draw_elements_with_i32(Gl::TRIANGLES, mesh.index_count, Gl::UNSIGNED_INT, 0);
}
