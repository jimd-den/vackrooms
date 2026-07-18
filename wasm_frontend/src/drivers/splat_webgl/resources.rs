//! Per-chunk GPU resources for the face-splat renderer.
//!
//! A chunk owns two representations derived from the same surface payload:
//!
//! * one packed instance page for the visible splat pass;
//! * one retained greedy mesh used only as the hero-light shadow caster.
//!
//! WebGL2 has no `baseInstance`, so a cell-range draw re-points the four
//! instance attributes at that range's byte offset before submission.

use js_sys::{Uint8Array, Uint32Array};
use wasm_bindgen::JsValue;
use web_sys::{WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram, WebGlVertexArrayObject};

use crate::application::ports::{FaceCellRange, SurfaceChunk};
use crate::drivers::gl::light_volume::upload_light_volume;

use super::SplatRenderer;

/// Bytes per packed face instance uploaded to WebGL.
pub(crate) const INSTANCE_STRIDE: i32 = 16;
const SHADOW_VERTEX_STRIDE: i32 = 10;

/// Attribute locations are resolved once because every chunk uses the same
/// packed instance layout.
pub(crate) struct InstanceAttribs {
    pos: u32,
    extents: u32,
    meta: u32,
    flags: u32,
}

impl InstanceAttribs {
    pub(crate) fn resolve(gl: &Gl, program: &WebGlProgram) -> Result<Self, JsValue> {
        let lookup = |name: &str| -> Result<u32, JsValue> {
            let location = gl.get_attrib_location(program, name);
            if location < 0 {
                return Err(JsValue::from_str(&format!("splat shader lost {name}")));
            }
            Ok(location as u32)
        };

        Ok(Self {
            pos: lookup("aFacePos")?,
            extents: lookup("aFaceExtents")?,
            meta: lookup("aFaceMeta")?,
            flags: lookup("aFaceFlags")?,
        })
    }

    fn enable(&self, gl: &Gl) {
        for location in [self.pos, self.extents, self.meta, self.flags] {
            gl.enable_vertex_attrib_array(location);
            gl.vertex_attrib_divisor(location, 1);
        }
    }

    /// Records attribute pointers for a range beginning at `first_instance`.
    /// The caller must have the chunk VAO and its instance buffer bound.
    pub(crate) fn point_at(&self, gl: &Gl, first_instance: u32) {
        let base = first_instance as i32 * INSTANCE_STRIDE;
        gl.vertex_attrib_pointer_with_i32(
            self.pos,
            3,
            Gl::UNSIGNED_SHORT,
            false,
            INSTANCE_STRIDE,
            base,
        );
        gl.vertex_attrib_pointer_with_i32(
            self.extents,
            2,
            Gl::UNSIGNED_BYTE,
            false,
            INSTANCE_STRIDE,
            base + 6,
        );
        gl.vertex_attrib_pointer_with_i32(
            self.meta,
            4,
            Gl::UNSIGNED_BYTE,
            false,
            INSTANCE_STRIDE,
            base + 8,
        );
        gl.vertex_attrib_pointer_with_i32(
            self.flags,
            1,
            Gl::UNSIGNED_BYTE,
            false,
            INSTANCE_STRIDE,
            base + 12,
        );
    }
}

/// All GPU and frame-selection data owned by one resident chunk.
pub(crate) struct SplatChunk {
    pub instance_vao: WebGlVertexArrayObject,
    pub instance_buffer: WebGlBuffer,
    pub instance_count: i32,
    pub cells: Vec<FaceCellRange>,
    pub cell_size: f32,
    pub voxel_scale: f32,
    pub shadow_vao: WebGlVertexArrayObject,
    pub shadow_vertex_buffer: WebGlBuffer,
    pub shadow_index_buffer: WebGlBuffer,
    pub shadow_index_count: i32,
    pub origin: [f32; 3],
    pub bounds_max: [f32; 3],
    pub light_texture: web_sys::WebGlTexture,
}

impl SplatRenderer {
    pub(crate) fn destroy_chunk(&self, chunk: SplatChunk) {
        let gl = &self.gl;
        gl.delete_vertex_array(Some(&chunk.instance_vao));
        gl.delete_buffer(Some(&chunk.instance_buffer));
        gl.delete_vertex_array(Some(&chunk.shadow_vao));
        gl.delete_buffer(Some(&chunk.shadow_vertex_buffer));
        gl.delete_buffer(Some(&chunk.shadow_index_buffer));
        gl.delete_texture(Some(&chunk.light_texture));
    }

    pub(crate) fn upload_surface(&mut self, chunk: SurfaceChunk<'_>) {
        if let Some(old) = self.chunks.remove(&chunk.key) {
            self.destroy_chunk(old);
        }
        if chunk.mesh.faces.instances.is_empty() {
            return;
        }

        let (instance_vao, instance_buffer) = self.upload_instance_page(chunk);
        let (shadow_vao, shadow_vertex_buffer, shadow_index_buffer) =
            self.upload_shadow_mesh(chunk);
        let light_texture =
            upload_light_volume(&self.gl, chunk.mesh).expect("upload splat light volume");

        self.chunks.insert(
            chunk.key,
            SplatChunk {
                instance_vao,
                instance_buffer,
                instance_count: chunk.mesh.faces.instances.len().min(i32::MAX as usize) as i32,
                cells: chunk.mesh.faces.cells.clone(),
                cell_size: chunk.mesh.faces.cell_size,
                voxel_scale: chunk.mesh.voxel_scale,
                shadow_vao,
                shadow_vertex_buffer,
                shadow_index_buffer,
                shadow_index_count: chunk.mesh.indices.len().min(i32::MAX as usize) as i32,
                origin: chunk.origin,
                bounds_max: chunk.mesh.bounds.max,
                light_texture,

            },
        );
    }

    fn upload_instance_page(
        &self,
        chunk: SurfaceChunk<'_>,
    ) -> (WebGlVertexArrayObject, WebGlBuffer) {
        let gl = &self.gl;
        let vao = gl.create_vertex_array().expect("create splat VAO");
        let buffer = gl.create_buffer().expect("create splat instance buffer");
        let packed = pack_instances(chunk);

        gl.bind_vertex_array(Some(&vao));
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&buffer));
        let bytes = Uint8Array::from(packed.as_slice());
        gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &bytes, Gl::STATIC_DRAW);
        self.attribs.enable(gl);
        self.attribs.point_at(gl, 0);
        gl.bind_vertex_array(None);
        gl.bind_buffer(Gl::ARRAY_BUFFER, None);

        (vao, buffer)
    }

    fn upload_shadow_mesh(
        &self,
        chunk: SurfaceChunk<'_>,
    ) -> (WebGlVertexArrayObject, WebGlBuffer, WebGlBuffer) {
        let gl = &self.gl;
        let vao = gl.create_vertex_array().expect("create splat shadow VAO");
        let vertex_buffer = gl
            .create_buffer()
            .expect("create splat shadow vertex buffer");
        let index_buffer = gl
            .create_buffer()
            .expect("create splat shadow index buffer");
        let packed_vertices = pack_shadow_vertices(chunk);

        gl.bind_vertex_array(Some(&vao));
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&vertex_buffer));
        let vertices = Uint8Array::from(packed_vertices.as_slice());
        gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &vertices, Gl::STATIC_DRAW);

        gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&index_buffer));
        let indices = Uint32Array::from(chunk.mesh.indices.as_slice());
        gl.buffer_data_with_array_buffer_view(Gl::ELEMENT_ARRAY_BUFFER, &indices, Gl::STATIC_DRAW);

        // The shared shadow shader declares `aPosition` at location zero.
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 3, Gl::UNSIGNED_SHORT, false, SHADOW_VERTEX_STRIDE, 0);
        gl.bind_vertex_array(None);
        gl.bind_buffer(Gl::ARRAY_BUFFER, None);
        gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, None);

        (vao, vertex_buffer, index_buffer)
    }
}

/// Serializes `PackedFaceInstance` explicitly so the GPU layout is stable
/// across Rust compiler versions and host alignment rules.
fn pack_instances(chunk: SurfaceChunk<'_>) -> Vec<u8> {
    let instances = &chunk.mesh.faces.instances;
    let mut packed = Vec::with_capacity(instances.len() * INSTANCE_STRIDE as usize);
    for instance in instances {
        packed.extend_from_slice(&instance.position[0].to_le_bytes());
        packed.extend_from_slice(&instance.position[1].to_le_bytes());
        packed.extend_from_slice(&instance.position[2].to_le_bytes());
        packed.push(instance.extent_u);
        packed.push(instance.extent_v);
        packed.push(instance.normal_axis);
        packed.push(instance.material);
        packed.push(instance.baked_light);
        packed.push(instance.ao);
        packed.push(instance.flags);
        packed.extend_from_slice(&[0, 0, 0]);
    }
    packed
}

/// Serializes the retained 10-byte greedy-mesh vertices. Only position is
/// consumed in the shadow pass, but retaining the canonical stride keeps the
/// caster byte-for-byte aligned with the visible surface representation.
fn pack_shadow_vertices(chunk: SurfaceChunk<'_>) -> Vec<u8> {
    let mut packed = Vec::with_capacity(chunk.mesh.vertices.len() * SHADOW_VERTEX_STRIDE as usize);
    for vertex in &chunk.mesh.vertices {
        packed.extend_from_slice(&vertex.position[0].to_le_bytes());
        packed.extend_from_slice(&vertex.position[1].to_le_bytes());
        packed.extend_from_slice(&vertex.position[2].to_le_bytes());
        packed.extend_from_slice(&[
            vertex.normal_axis,
            vertex.material,
            vertex.static_indirect,
            vertex.ao,
        ]);
    }
    packed
}
