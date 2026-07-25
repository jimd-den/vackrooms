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
use crate::application::rendering::encode_display_color;
use crate::drivers::gl::cluster_scene_lights::{LightRange, append_lights_reaching_bounds};
use crate::drivers::gl::lights::flare_cores;
use crate::drivers::gl::math::{
    camera_matrices, look_at_matrix_down, multiply_matrices, ortho_matrix,
};
use crate::drivers::gl::visibility::{MAX_DRAW_DISTANCE, chunk_bounding_sphere, sphere_visible};

use super::SurfaceRenderer;
use super::upload_world_surface_chunks::GpuMesh;

/// Shadow uniforms always receive a complete state. The default disables the
/// lookup and supplies a deterministic matrix when no hero pass ran.
struct ShadowState {
    light_index: i32,
    light_view_projection: [f32; 16],
}

struct VisibleDraw {
    key: SurfaceChunkKey,
    lights: LightRange,
    hero_local_index: i32,
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

    fn cpu_telemetry_string(&self) -> Option<String> {
        let stats = self.stats;
        let gpu_ms = self.timer.last_ms().unwrap_or(0.0);
        Some(format!(
            "surface {} tri / {} calls / {}/{} chunks / {} shadow casters / GPU {:.2}ms",
            stats.triangles_drawn,
            stats.draw_calls,
            stats.visible_chunks,
            stats.resident_chunks,
            stats.shadow_casters,
            gpu_ms,
        ))
    }

    fn upload_atlas(&mut self, _texels: &[u32]) {}

    fn upload_label_atlas(&mut self, rgba: &[u8], width: u32, height: u32) {
        let gl = &self.gl;
        let texture = gl.create_texture();
        gl.active_texture(Gl::TEXTURE7);
        gl.bind_texture(Gl::TEXTURE_2D, texture.as_ref());
        let _ = gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            Gl::TEXTURE_2D,
            0,
            Gl::RGBA as i32,
            width as i32,
            height as i32,
            0,
            Gl::RGBA,
            Gl::UNSIGNED_BYTE,
            Some(rgba),
        );
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
        self.label_atlas = texture;
    }

    fn draw(&mut self, frame: &FrameParams, _chunks: &[ChunkDraw]) {
        let toggles = crate::get_render_toggles();

        // OPTIMIZATION (rt_timer): asynchronous timing never waits for the
        // GPU. Disabling it prevents new queries while preserving rendering.
        self.timer.poll(&self.gl);
        let timer_started = self.timer.begin(&self.gl, false);

        render_frame(self, frame, toggles);
        self.timer.end(&self.gl, timer_started);
    }
}

fn render_frame(renderer: &mut SurfaceRenderer, frame: &FrameParams, toggles: RenderToggles) {
    begin_main_frame(renderer, frame);
    let (mut visible, clustered_lights) =
        prepare_visible_draws(renderer, frame, toggles.distance_cull);
    if renderer
        .scene_lights
        .upload(&renderer.gl, &clustered_lights)
        .is_err()
    {
        for draw in &mut visible {
            draw.lights.count = 0;
            draw.hero_local_index = -1;
        }
    }

    let (shadow, shadow_casters, shadow_draw_calls) =
        render_shadow_pass(renderer, frame, toggles.shadow_pass);

    bind_frame_uniforms(
        renderer,
        frame,
        &shadow,
        toggles.dither,
        toggles.baked_lighting,
    );
    let (triangles_drawn, main_draw_calls) = draw_visible_meshes(renderer, &visible, &shadow);
    draw_supply_labels(renderer, frame);

    renderer.stats = super::SurfaceFrameStats {
        visible_chunks: visible.len(),
        resident_chunks: renderer.meshes.len(),
        draw_calls: shadow_draw_calls + main_draw_calls,
        triangles_drawn,
        shadow_casters,
    };
}

/// Camera-facing product labels over nearby supply pickups. A cutout pass:
/// depth-tested and depth-writing, so it needs no ordering against the
/// opaque world and never x-rays through walls.
fn draw_supply_labels(renderer: &SurfaceRenderer, frame: &FrameParams) {
    use crate::drivers::shaders::supply_labels;

    let Some(atlas) = renderer.label_atlas.as_ref() else {
        return;
    };
    if frame.supply_sprites.is_empty() {
        return;
    }
    let gl = &renderer.gl;
    let uniforms = &renderer.label_uniforms;
    gl.use_program(Some(&renderer.label_program));

    let (projection, view) =
        camera_matrices(frame, renderer.width, renderer.height, MAX_DRAW_DISTANCE);
    gl.uniform_matrix4fv_with_f32_array(uniforms.projection.as_ref(), false, &projection);
    gl.uniform_matrix4fv_with_f32_array(uniforms.view.as_ref(), false, &view);

    let count = frame
        .supply_sprites
        .len()
        .min(supply_labels::MAX_SPRITES);
    let mut sprites = [0.0f32; supply_labels::MAX_SPRITES * 4];
    for (i, sprite) in frame.supply_sprites.iter().take(count).enumerate() {
        sprites[i * 4] = sprite.position[0];
        sprites[i * 4 + 1] = sprite.position[1];
        sprites[i * 4 + 2] = sprite.position[2];
        sprites[i * 4 + 3] = sprite.atlas_row as f32;
    }
    gl.uniform4fv_with_f32_array(uniforms.sprites.as_ref(), &sprites);
    // Cylindrical billboard: turn with the player's yaw, stay upright.
    let right = [frame.yaw.cos(), 0.0, -frame.yaw.sin()];
    gl.uniform3fv_with_f32_array(uniforms.cam_right.as_ref(), &right);
    gl.uniform2f(
        uniforms.half_size.as_ref(),
        supply_labels::HALF_WIDTH,
        supply_labels::HALF_HEIGHT,
    );
    gl.uniform3f(
        uniforms.camera_position.as_ref(),
        frame.camera_pos[0],
        frame.camera_pos[1],
        frame.camera_pos[2],
    );
    let environment = frame.environment;
    gl.uniform3f(
        uniforms.fog_color.as_ref(),
        environment.fog_color[0],
        environment.fog_color[1],
        environment.fog_color[2],
    );
    gl.uniform1f(uniforms.fog_density.as_ref(), environment.fog_density);
    gl.uniform1f(uniforms.fog_start.as_ref(), environment.fog_start);

    gl.active_texture(Gl::TEXTURE7);
    gl.bind_texture(Gl::TEXTURE_2D, Some(atlas));
    gl.uniform1i(uniforms.atlas.as_ref(), 7);

    // Both label faces read; the quad expands entirely from gl_VertexID.
    gl.disable(Gl::CULL_FACE);
    gl.bind_vertex_array(None);
    gl.draw_arrays(Gl::TRIANGLES, 0, (count * 6) as i32);
    gl.enable(Gl::CULL_FACE);
}

/// Establishes every fixed-function state relied on by either raster pass.
fn begin_main_frame(renderer: &SurfaceRenderer, frame: &FrameParams) {
    let gl = &renderer.gl;
    gl.viewport(0, 0, renderer.width, renderer.height);

    // Match unloaded space to the level atmosphere so generated chunk edges
    // disappear into fog rather than exposing the canvas clear color.
    let environment = frame.environment;
    let atmosphere = if environment.outdoor {
        environment.sky_color
    } else {
        environment.fog_color
    };
    let clear = encode_display_color(atmosphere);
    gl.clear_color(clear[0], clear[1], clear[2], 1.0);
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
fn prepare_visible_draws(
    renderer: &SurfaceRenderer,
    frame: &FrameParams,
    culling_enabled: bool,
) -> (
    Vec<VisibleDraw>,
    Vec<crate::application::ports::LightSource>,
) {
    let mut draws = Vec::new();
    let mut clustered = Vec::new();
    let hero_id = frame.active_scene_lights().first().map(|light| light.id);
    for (&key, mesh) in &renderer.meshes {
        let visible = {
            if !culling_enabled {
                true
            } else {
                let (center, radius) = chunk_bounding_sphere(mesh.origin, mesh.bounds_max);
                sphere_visible(center, radius, frame, MAX_DRAW_DISTANCE).is_some()
            }
        };
        if !visible {
            continue;
        }
        let bounds_max = [
            mesh.origin[0] + mesh.bounds_max[0],
            mesh.origin[1] + mesh.bounds_max[1],
            mesh.origin[2] + mesh.bounds_max[2],
        ];
        let lights = append_lights_reaching_bounds(
            frame.active_scene_lights(),
            mesh.origin,
            bounds_max,
            &mut clustered,
        );
        let hero_local_index =
            if lights.count > 0 && hero_id == Some(clustered[lights.first as usize].id) {
                0
            } else {
                -1
            };
        draws.push(VisibleDraw {
            key,
            lights,
            hero_local_index,
        });
    }
    (draws, clustered)
}

/// FEATURE (rt_shadows): renders every resident greedy mesh into the selected
/// fixture's top-down depth map. Camera culling must not remove an off-screen
/// caster whose shadow lands on a visible receiver. With the feature
/// disabled, the returned state makes the main shader skip shadow sampling.
fn render_shadow_pass(
    renderer: &SurfaceRenderer,
    frame: &FrameParams,
    enabled: bool,
) -> (ShadowState, u32, u32) {
    let Some(hero) = frame.active_scene_lights().first() else {
        return (ShadowState::default(), 0, 0);
    };
    if !enabled {
        return (ShadowState::default(), 0, 0);
    }

    let range = hero.radius;
    let emitter_half_diagonal =
        (hero.half_size[0] * hero.half_size[0] + hero.half_size[1] * hero.half_size[1]).sqrt();
    let ortho_half = range + emitter_half_diagonal;
    let projection = ortho_matrix(
        -ortho_half,
        ortho_half,
        -ortho_half,
        ortho_half,
        0.1,
        range + emitter_half_diagonal,
    );
    let view = look_at_matrix_down(hero.position);
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

    let hero_min_x = hero.position[0] - ortho_half;
    let hero_max_x = hero.position[0] + ortho_half;
    let hero_min_z = hero.position[2] - ortho_half;
    let hero_max_z = hero.position[2] + ortho_half;
    let hero_min_y = hero.position[1] - range;
    let hero_max_y = hero.position[1] + 2.0;

    let mut casters_count = 0u32;
    let mut draw_calls = 0u32;

    for mesh in renderer.meshes.values() {
        let chunk_max_x = mesh.origin[0] + mesh.bounds_max[0];
        let chunk_max_y = mesh.origin[1] + mesh.bounds_max[1];
        let chunk_max_z = mesh.origin[2] + mesh.bounds_max[2];

        // Footprint AABB intersection test for hero-light caster culling
        if chunk_max_x < hero_min_x
            || mesh.origin[0] > hero_max_x
            || chunk_max_z < hero_min_z
            || mesh.origin[2] > hero_max_z
            || chunk_max_y < hero_min_y
            || mesh.origin[1] > hero_max_y
        {
            continue;
        }

        casters_count += 1;
        draw_calls += 1;

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

    (
        ShadowState {
            light_index: 0,
            light_view_projection,
        },
        casters_count,
        draw_calls,
    )
}

/// Binds state shared by every visible mesh exactly once per frame.
fn bind_frame_uniforms(
    renderer: &SurfaceRenderer,
    frame: &FrameParams,
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

    bind_light_uniforms(renderer, frame, shadow);
    bind_dynamic_light_uniforms(renderer, frame);
}

fn bind_dynamic_light_uniforms(renderer: &SurfaceRenderer, frame: &FrameParams) {
    let gl = &renderer.gl;
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
    gl.uniform1i(
        renderer.uniforms.dynamic_light_count.as_ref(),
        lights.len() as i32,
    );
    gl.uniform4fv_with_f32_array(
        renderer.uniforms.dynamic_pos_radius.as_ref(),
        &position_radius,
    );
    gl.uniform4fv_with_f32_array(
        renderer.uniforms.dynamic_color_intensity.as_ref(),
        &color_intensity,
    );
}

fn camera_forward(frame: &FrameParams) -> [f32; 3] {
    let (sin_pitch, cos_pitch) = frame.pitch.sin_cos();
    let (sin_yaw, cos_yaw) = frame.yaw.sin_cos();
    [-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw]
}

fn bind_light_uniforms(renderer: &SurfaceRenderer, frame: &FrameParams, shadow: &ShadowState) {
    let gl = &renderer.gl;
    let uniforms = &renderer.uniforms;
    gl.active_texture(Gl::TEXTURE2);
    renderer.scene_lights.bind(gl);
    gl.uniform1i(uniforms.scene_light_texture.as_ref(), 2);
    gl.uniform1i(
        uniforms.scene_light_texture_width.as_ref(),
        renderer.scene_lights.width(),
    );

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
}

fn draw_visible_meshes(
    renderer: &SurfaceRenderer,
    visible: &[VisibleDraw],
    shadow: &ShadowState,
) -> (u32, u32) {
    let mut triangles = 0u32;
    let mut draw_calls = 0u32;
    for draw in visible {
        let Some(mesh) = renderer.meshes.get(&draw.key) else {
            continue;
        };
        draw_mesh(renderer, mesh, draw, shadow);
        triangles += (mesh.index_count / 3) as u32;
        draw_calls += 1;
    }
    renderer.gl.bind_vertex_array(None);
    (triangles, draw_calls)
}

/// Binds the only per-mesh state: local transform bounds, baked light volume,
/// and indexed geometry.
fn draw_mesh(renderer: &SurfaceRenderer, mesh: &GpuMesh, draw: &VisibleDraw, shadow: &ShadowState) {
    let gl = &renderer.gl;
    let uniforms = &renderer.uniforms;
    gl.uniform1i(uniforms.light_first.as_ref(), draw.lights.first);
    gl.uniform1i(uniforms.light_count.as_ref(), draw.lights.count);
    gl.uniform1i(
        uniforms.shadowed_light_index.as_ref(),
        if shadow.light_index >= 0 {
            draw.hero_local_index
        } else {
            -1
        },
    );
    gl.uniform3f(
        uniforms.chunk_origin.as_ref(),
        mesh.origin[0],
        mesh.origin[1],
        mesh.origin[2],
    );
    gl.uniform1f(uniforms.voxel_size.as_ref(), mesh.voxel_size);
    gl.uniform3fv_with_f32_array(
        uniforms.light_volume_origin.as_ref(),
        &mesh.light_volume_origin,
    );
    gl.active_texture(Gl::TEXTURE0);
    gl.bind_texture(Gl::TEXTURE_3D, Some(&mesh.light_texture));
    gl.uniform1i(uniforms.light_volume.as_ref(), 0);

    gl.bind_vertex_array(Some(&mesh.vao));
    gl.draw_elements_with_i32(Gl::TRIANGLES, mesh.index_count, Gl::UNSIGNED_INT, 0);
}
