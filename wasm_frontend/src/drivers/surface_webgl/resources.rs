//! Per-chunk GPU resources for the surface renderer: packed vertex/index
//! buffers plus the baked 3D light volume. Uploads are incremental — the
//! streaming engine calls in only new/refined/evicted chunks, so a steady
//! frame uploads nothing.

use js_sys::{Uint8Array, Uint32Array};
use web_sys::{WebGl2RenderingContext as Gl, WebGlBuffer, WebGlVertexArrayObject};

use crate::application::ports::SurfaceChunk;
use crate::drivers::gl::light_volume::upload_light_volume;

use super::SurfaceRenderer;

/// Bytes per packed vertex as uploaded (see `PackedVertex`).
pub(crate) const VERTEX_STRIDE: i32 = 10;

pub(crate) struct GpuMesh {
    pub vao: WebGlVertexArrayObject,
    pub vertex_buffer: WebGlBuffer,
    pub index_buffer: WebGlBuffer,
    pub index_count: i32,
    pub origin: [f32; 3],
    pub bounds_max: [f32; 3],
    pub voxel_size: f32,
    pub light_volume_origin: [f32; 3],
    pub light_texture: web_sys::WebGlTexture,
}

impl SurfaceRenderer {
    pub(crate) fn destroy_mesh(&self, mesh: GpuMesh) {
        self.gl.delete_vertex_array(Some(&mesh.vao));
        self.gl.delete_buffer(Some(&mesh.vertex_buffer));
        self.gl.delete_buffer(Some(&mesh.index_buffer));
        self.gl.delete_texture(Some(&mesh.light_texture));
    }

    pub(crate) fn upload_surface(&mut self, chunk: SurfaceChunk<'_>) {
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

        // PackedVertex serialized little-endian, 10 B: u16 position x3,
        // then normal_axis / material / static_indirect / ao bytes.
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

        let light_texture = upload_light_volume(gl, chunk.mesh).expect("upload light volume");

        // Baked volumes left generation with the WebGPU migration; the
        // placeholder texel makes padding moot, so the volume origin is
        // simply the chunk origin (see gl::light_volume).
        self.meshes.insert(
            chunk.key,
            GpuMesh {
                vao,
                vertex_buffer,
                index_buffer,
                index_count: chunk.mesh.indices.len().min(i32::MAX as usize) as i32,
                origin: chunk.origin,
                bounds_max: chunk.mesh.bounds.max,
                voxel_size: chunk.mesh.voxel_scale,
                light_volume_origin: chunk.origin,
                light_texture,
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
}
