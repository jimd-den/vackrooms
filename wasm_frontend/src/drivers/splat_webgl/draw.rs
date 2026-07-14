//! Per-frame face-splat pipeline.
//!
//! The sequence is intentionally explicit and made of small stages:
//!
//! 1. clear and establish fixed-function state;
//! 2. select visible chunks near-to-far;
//! 3. select the bounded set of most important lights;
//! 4. optionally render the hero-light shadow map;
//! 5. bind frame uniforms once;
//! 6. draw whole instance pages or coalesced visible cell ranges.
//!
//! Every optional shortcut is labeled at its branch and controlled by the
//! shared `RenderToggles` switchboard.

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

use super::resources::SplatChunk;
use super::{FrameStats, SplatRenderer};

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
        self.timer.last_ms()
    }

    fn cpu_telemetry_string(&self) -> Option<String> {
        let stats = self.stats;
        Some(format!(
            "splat {} faces / {} calls / {} chunks / {} cells culled / {} dropped",
            stats.faces_drawn,
            stats.draw_calls,
            stats.chunks_drawn,
            stats.cells_culled,
            stats.faces_dropped,
        ))
    }

    fn upload_atlas(&mut self, _texels: &[u32]) {}

    fn draw(&mut self, frame: &FrameParams, _chunks: &[ChunkDraw]) {
        let toggles = crate::get_render_toggles();

        // OPTIMIZATION (rt_timer): asynchronous GPU timing. Polling never
        // blocks; disabling it prevents new timer queries from being issued.
        self.timer.poll(&self.gl);
        let timer_started = self.timer.begin(&self.gl, toggles.gpu_timer);

        self.stats = render_frame(self, frame, toggles);
        self.timer.end(&self.gl, timer_started);
    }
}

fn render_frame(
    renderer: &SplatRenderer,
    frame: &FrameParams,
    toggles: RenderToggles,
) -> FrameStats {
    begin_main_frame(renderer, frame);

    let resident: Vec<&SplatChunk> = renderer.chunks.values().collect();
    let visible = collect_visible_chunks(&resident, frame, toggles.distance_cull);
    let lights = select_frame_lights(frame);
    let shadow = render_shadow_pass(renderer, &resident, &lights, toggles.shadow_pass);

    bind_frame_uniforms(renderer, frame, &lights, &shadow, toggles.dither);
    draw_visible_chunks(
        renderer,
        &visible,
        frame,
        toggles.cell_culling,
        toggles.face_budget,
    )
}

fn begin_main_frame(renderer: &SplatRenderer, frame: &FrameParams) {
    let gl = &renderer.gl;
    gl.viewport(0, 0, renderer.width, renderer.height);

    let environment = frame.environment;
    if environment.outdoor {
        gl.clear_color(
            environment.sky_color[0],
            environment.sky_color[1],
            environment.sky_color[2],
            1.0,
        );
    } else {
        // Indoor fog reaches black before the draw limit, so unloaded space
        // and fogged geometry meet without a colored chunk boundary.
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
    }
    gl.clear_depth(1.0);
    gl.clear(Gl::COLOR_BUFFER_BIT | Gl::DEPTH_BUFFER_BIT);
    gl.enable(Gl::DEPTH_TEST);
    gl.depth_func(Gl::LEQUAL);
    gl.depth_mask(true);
    gl.enable(Gl::CULL_FACE);
    gl.cull_face(Gl::BACK);
}

/// OPTIMIZATION (rt_cull): conservative chunk sphere rejection. The
/// disabled reference path submits every resident chunk. Both paths retain
/// near-to-far ordering because the face budget deliberately drops whole
/// far chunks before nearby geometry.
fn collect_visible_chunks<'a>(
    resident: &[&'a SplatChunk],
    frame: &FrameParams,
    culling_enabled: bool,
) -> Vec<&'a SplatChunk> {
    let mut ranked: Vec<(&SplatChunk, f32)> = resident
        .iter()
        .copied()
        .filter_map(|chunk| {
            let (center, radius) = chunk_bounding_sphere(chunk.origin, chunk.bounds_max);
            let distance2 = if culling_enabled {
                sphere_visible(center, radius, frame, MAX_DRAW_DISTANCE)?
            } else {
                squared_distance(center, frame.camera_pos)
            };
            Some((chunk, distance2))
        })
        .collect();

    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    ranked.into_iter().map(|(chunk, _)| chunk).collect()
}

fn squared_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn select_frame_lights(frame: &FrameParams) -> SelectedLights {
    let dynamic = dynamic_light_sources(frame);
    select_lights(frame.active_scene_lights().iter(), &dynamic, frame)
}

/// FEATURE (rt_shadows): resident greedy meshes cast the hero-light shadow
/// map. Camera visibility is intentionally not reused here: off-screen
/// geometry can cast onto a visible receiver. Face instances and caster
/// meshes derive from the same grid and LOD, keeping silhouettes consistent.
fn render_shadow_pass(
    renderer: &SplatRenderer,
    casters: &[&SplatChunk],
    lights: &SelectedLights,
    enabled: bool,
) -> ShadowState {
    if !enabled || lights.hero_slot < 0 {
        return ShadowState::default();
    }

    let range = lights.hero_radius;
    let ortho_half = (range * 0.55).clamp(7.0, 13.0);
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

    for chunk in casters {
        if chunk.shadow_index_count == 0 {
            continue;
        }
        gl.uniform3f(
            renderer.shadow_uniforms.chunk_origin.as_ref(),
            chunk.origin[0],
            chunk.origin[1],
            chunk.origin[2],
        );
        gl.bind_vertex_array(Some(&chunk.shadow_vao));
        gl.draw_elements_with_i32(Gl::TRIANGLES, chunk.shadow_index_count, Gl::UNSIGNED_INT, 0);
    }

    gl.bind_framebuffer(Gl::FRAMEBUFFER, None);
    gl.viewport(0, 0, renderer.width, renderer.height);
    gl.color_mask(true, true, true, true);

    ShadowState {
        light_index: lights.hero_slot,
        light_view_projection,
    }
}

fn bind_frame_uniforms(
    renderer: &SplatRenderer,
    frame: &FrameParams,
    lights: &SelectedLights,
    shadow: &ShadowState,
    dither_enabled: bool,
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

    // The cone direction must use the same yaw/pitch basis as the view
    // matrix. Omitting this uniform collapses the spotlight around (0,0,0).
    let (sp, cp) = frame.pitch.sin_cos();
    let (sy, cy) = frame.yaw.sin_cos();
    gl.uniform3f(uniforms.cam_forward.as_ref(), -cp * sy, sp, -cp * cy);
    gl.uniform1i(
        uniforms.flashlight.as_ref(),
        if frame.flashlight { 1 } else { 0 },
    );
    gl.uniform1i(uniforms.dither.as_ref(), if dither_enabled { 1 } else { 0 });

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

    bind_light_uniforms(renderer, frame, lights, shadow);
}

fn bind_light_uniforms(
    renderer: &SplatRenderer,
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
    }

    let cores = flare_cores(frame);
    gl.uniform1i(uniforms.core_count.as_ref(), cores.count);
    gl.uniform4fv_with_f32_array(uniforms.cores.as_ref(), &cores.pos_intensity);
    gl.uniform3fv_with_f32_array(uniforms.core_colors.as_ref(), &cores.colors);

    // Keep samplers of different types on different units even when no hero
    // light is active; aliasing sampler3D and sampler2D invalidates the draw.
    gl.active_texture(Gl::TEXTURE1);
    gl.bind_texture(Gl::TEXTURE_2D, Some(&renderer.shadow.texture));
    gl.uniform1i(uniforms.shadow_map.as_ref(), 1);
    gl.uniform_matrix4fv_with_f32_array(
        uniforms.light_view_proj.as_ref(),
        false,
        &shadow.light_view_projection,
    );
    gl.uniform1i(uniforms.shadowed_light_index.as_ref(), shadow.light_index);
    gl.uniform1i(uniforms.shadow_taps.as_ref(), renderer.profile.shadow_taps);
}

fn draw_visible_chunks(
    renderer: &SplatRenderer,
    visible: &[&SplatChunk],
    frame: &FrameParams,
    cell_culling_enabled: bool,
    face_budget_enabled: bool,
) -> FrameStats {
    let gl = &renderer.gl;
    let mut stats = FrameStats::default();

    for chunk in visible {
        // OPTIMIZATION (rt_budget): once the profile budget is exhausted,
        // discard whole far chunks. The disabled reference path draws every
        // face that survived the independently controlled visibility tests.
        if face_budget_enabled && stats.faces_drawn >= renderer.profile.face_budget {
            stats.faces_dropped = stats
                .faces_dropped
                .saturating_add(chunk.instance_count.max(0) as u32);
            continue;
        }

        bind_chunk(renderer, chunk);
        stats.chunks_drawn += 1;

        // OPTIMIZATION (rt_cells): reject invisible cell ranges and merge
        // adjacent visible ranges. The disabled reference path submits the
        // complete chunk instance page in one draw.
        if cell_culling_enabled {
            draw_visible_cells(renderer, chunk, frame, &mut stats);
        } else {
            draw_instance_range(renderer, 0, chunk.instance_count.max(0) as u32, &mut stats);
        }
    }

    gl.bind_vertex_array(None);
    gl.bind_buffer(Gl::ARRAY_BUFFER, None);
    stats
}

fn bind_chunk(renderer: &SplatRenderer, chunk: &SplatChunk) {
    let gl = &renderer.gl;
    let uniforms = &renderer.uniforms;
    gl.uniform3f(
        uniforms.chunk_origin.as_ref(),
        chunk.origin[0],
        chunk.origin[1],
        chunk.origin[2],
    );
    gl.uniform3f(
        uniforms.chunk_size.as_ref(),
        chunk.bounds_max[0],
        chunk.bounds_max[1],
        chunk.bounds_max[2],
    );
    gl.uniform1f(uniforms.voxel_scale.as_ref(), chunk.voxel_scale);

    gl.active_texture(Gl::TEXTURE0);
    gl.bind_texture(Gl::TEXTURE_3D, Some(&chunk.light_texture));
    gl.uniform1i(uniforms.light_volume.as_ref(), 0);
    gl.bind_vertex_array(Some(&chunk.instance_vao));
    gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&chunk.instance_buffer));
}

fn draw_visible_cells(
    renderer: &SplatRenderer,
    chunk: &SplatChunk,
    frame: &FrameParams,
    stats: &mut FrameStats,
) {
    // A capped face can reach 1.5 cell widths from its bucket center: its
    // center may sit at a cell corner, then its two in-plane half-extents add
    // another half cell each. This exact conservative bound prevents faces
    // disappearing at the camera/distance planes.
    let cell_radius = chunk.cell_size * 1.5;
    let mut run_start = None;
    let mut run_count = 0u32;

    for range in &chunk.cells {
        let center = [
            chunk.origin[0] + (range.cell[0] as f32 + 0.5) * chunk.cell_size,
            chunk.origin[1] + (range.cell[1] as f32 + 0.5) * chunk.cell_size,
            chunk.origin[2] + (range.cell[2] as f32 + 0.5) * chunk.cell_size,
        ];
        if sphere_visible(center, cell_radius, frame, MAX_DRAW_DISTANCE).is_some() {
            if run_start.is_none() {
                run_start = Some(range.offset);
            }
            run_count = run_count.saturating_add(range.count);
        } else {
            stats.cells_culled += 1;
            flush_instance_run(renderer, &mut run_start, &mut run_count, stats);
        }
    }
    flush_instance_run(renderer, &mut run_start, &mut run_count, stats);
}

fn flush_instance_run(
    renderer: &SplatRenderer,
    start: &mut Option<u32>,
    count: &mut u32,
    stats: &mut FrameStats,
) {
    if let Some(first) = start.take() {
        draw_instance_range(renderer, first, *count, stats);
    }
    *count = 0;
}

fn draw_instance_range(renderer: &SplatRenderer, first: u32, count: u32, stats: &mut FrameStats) {
    if count == 0 {
        return;
    }
    renderer.attribs.point_at(&renderer.gl, first);
    renderer
        .gl
        .draw_arrays_instanced(Gl::TRIANGLE_STRIP, 0, 4, count.min(i32::MAX as u32) as i32);
    stats.faces_drawn = stats.faces_drawn.saturating_add(count);
    stats.draw_calls += 1;
}
