//! Binary transport format for [`ChunkPayload`] between the generation
//! workers and the main thread. One chunk becomes one contiguous byte
//! buffer, so the browser can move it with a zero-copy `postMessage`
//! transfer instead of structured-cloning a deep object graph.
//!
//! The format is little-endian, length-prefixed, and versioned by a magic
//! header. Decode is defensive: any truncated or mismatched buffer yields
//! `None` and the driver simply drops that chunk (the streamer re-requests).
//! Platform-free and natively round-trip tested.

use crate::application::collision::Aabb;
use crate::application::ports::{
    ChunkPayload, FaceCellRange, FaceInstanceSet, LightSource, PackedFaceInstance, PackedVertex,
    SurfaceMeshPayload,
};

/// "VKC" + version 2. Version 2 adds authored light intensity.
const MAGIC: u32 = 0x564B_4302;

pub fn encode_chunk_payload(payload: &ChunkPayload) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        64 + payload.nodes.len() * 4
            + payload.surface.vertices.len() * 10
            + payload.surface.indices.len() * 4
            + payload.surface.light_volume.len()
            + payload.surface.faces.instances.len() * 16
            + payload.collision.len() * 24,
    );
    put_u32(&mut out, MAGIC);
    put_u32(&mut out, payload.root);
    put_f32(&mut out, payload.world_size);

    put_u32(&mut out, payload.nodes.len() as u32);
    for &n in &payload.nodes {
        put_u32(&mut out, n);
    }

    let s = &payload.surface;
    put_u32(&mut out, s.vertices.len() as u32);
    for v in &s.vertices {
        for c in v.position {
            put_u16(&mut out, c);
        }
        out.extend_from_slice(&[v.normal_axis, v.material, v.static_indirect, v.ao]);
    }
    put_u32(&mut out, s.indices.len() as u32);
    for &i in &s.indices {
        put_u32(&mut out, i);
    }
    put_aabb(&mut out, &s.bounds);
    out.push(s.lod);
    put_f32(&mut out, s.voxel_scale);
    for d in s.light_volume_size {
        put_u32(&mut out, d);
    }
    put_u32(&mut out, s.light_volume.len() as u32);
    out.extend_from_slice(&s.light_volume);

    put_u32(&mut out, s.lights.len() as u32);
    for l in &s.lights {
        out.extend_from_slice(&l.id.to_le_bytes());
        for c in l.position {
            put_f32(&mut out, c);
        }
        for c in l.half_size {
            put_f32(&mut out, c);
        }
        for c in l.color {
            put_f32(&mut out, c);
        }
        put_f32(&mut out, l.radius);
        put_f32(&mut out, l.intensity);
        out.push(l.flicker_mode);
        out.push(u8::from(l.enabled));
    }

    put_u32(&mut out, s.faces.instances.len() as u32);
    for inst in &s.faces.instances {
        for c in inst.position {
            put_u16(&mut out, c);
        }
        out.extend_from_slice(&[
            inst.extent_u,
            inst.extent_v,
            inst.normal_axis,
            inst.material,
            inst.baked_light,
            inst.ao,
            inst.flags,
        ]);
    }
    put_u32(&mut out, s.faces.cells.len() as u32);
    for cell in &s.faces.cells {
        out.extend_from_slice(&cell.cell);
        put_u32(&mut out, cell.offset);
        put_u32(&mut out, cell.count);
    }
    put_f32(&mut out, s.faces.cell_size);

    put_u32(&mut out, payload.collision.len() as u32);
    for aabb in &payload.collision {
        put_aabb(&mut out, aabb);
    }
    out
}

pub fn decode_chunk_payload(bytes: &[u8]) -> Option<ChunkPayload> {
    let mut r = Reader { bytes, pos: 0 };
    if r.u32()? != MAGIC {
        return None;
    }
    let root = r.u32()?;
    let world_size = r.f32()?;

    let node_count = r.len(4)?;
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(r.u32()?);
    }

    let vertex_count = r.len(10)?;
    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        vertices.push(PackedVertex {
            position: [r.u16()?, r.u16()?, r.u16()?],
            normal_axis: r.u8()?,
            material: r.u8()?,
            static_indirect: r.u8()?,
            ao: r.u8()?,
        });
    }
    let index_count = r.len(4)?;
    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        indices.push(r.u32()?);
    }
    let bounds = r.aabb()?;
    let lod = r.u8()?;
    let voxel_scale = r.f32()?;
    let light_volume_size = [r.u32()?, r.u32()?, r.u32()?];
    let volume_len = r.len(1)?;
    let light_volume = r.slice(volume_len)?.to_vec();

    let light_count = r.len(39)?;
    let mut lights = Vec::with_capacity(light_count);
    for _ in 0..light_count {
        lights.push(LightSource {
            id: r.u64()?,
            position: [r.f32()?, r.f32()?, r.f32()?],
            half_size: [r.f32()?, r.f32()?],
            color: [r.f32()?, r.f32()?, r.f32()?],
            radius: r.f32()?,
            intensity: r.f32()?,
            flicker_mode: r.u8()?,
            enabled: r.u8()? != 0,
        });
    }

    let instance_count = r.len(13)?;
    let mut instances = Vec::with_capacity(instance_count);
    for _ in 0..instance_count {
        instances.push(PackedFaceInstance {
            position: [r.u16()?, r.u16()?, r.u16()?],
            extent_u: r.u8()?,
            extent_v: r.u8()?,
            normal_axis: r.u8()?,
            material: r.u8()?,
            baked_light: r.u8()?,
            ao: r.u8()?,
            flags: r.u8()?,
            reserved: [0; 3],
        });
    }
    let cell_count = r.len(11)?;
    let mut cells = Vec::with_capacity(cell_count);
    for _ in 0..cell_count {
        cells.push(FaceCellRange {
            cell: [r.u8()?, r.u8()?, r.u8()?],
            offset: r.u32()?,
            count: r.u32()?,
        });
    }
    let cell_size = r.f32()?;

    let collision_count = r.len(24)?;
    let mut collision = Vec::with_capacity(collision_count);
    for _ in 0..collision_count {
        collision.push(r.aabb()?);
    }

    Some(ChunkPayload {
        root,
        nodes,
        world_size,
        surface: SurfaceMeshPayload {
            vertices,
            indices,
            bounds,
            lod,
            light_volume,
            light_volume_size,
            lights,
            faces: FaceInstanceSet {
                instances,
                cells,
                cell_size,
            },
            voxel_scale,
        },
        collision,
    })
}

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_aabb(out: &mut Vec<u8>, aabb: &Aabb) {
    for c in aabb.min {
        put_f32(out, c);
    }
    for c in aabb.max {
        put_f32(out, c);
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn slice(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let s = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.slice(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.slice(2)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.slice(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.slice(8)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.slice(4)?.try_into().ok()?))
    }
    fn aabb(&mut self) -> Option<Aabb> {
        Some(Aabb::new(
            [self.f32()?, self.f32()?, self.f32()?],
            [self.f32()?, self.f32()?, self.f32()?],
        ))
    }
    /// Reads a length prefix and sanity-bounds it: each element occupies at
    /// least `min_element_bytes`, so a corrupt length cannot force a huge
    /// allocation before the reads start failing.
    fn len(&mut self, min_element_bytes: usize) -> Option<usize> {
        let n = self.u32()? as usize;
        let remaining = self.bytes.len().saturating_sub(self.pos);
        if n.saturating_mul(min_element_bytes.max(1)) > remaining {
            return None;
        }
        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::local_chunk_source::LocalChunkSource;
    use crate::application::ports::ChunkSourcePort;
    use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
    use vackrooms::use_cases::generate_chunk::GeneratorConfig;

    #[test]
    fn generated_chunk_roundtrips_exactly() {
        let source =
            LocalChunkSource::new(SimpleNoiseProvider::new(), 42, GeneratorConfig::low_spec());
        for lod in [0u8, 1] {
            let payload = source.load(10.0, -10.0, 0, lod);
            let bytes = encode_chunk_payload(&payload);
            let decoded = decode_chunk_payload(&bytes).expect("decodes");
            assert_eq!(decoded.root, payload.root);
            assert_eq!(decoded.nodes, payload.nodes);
            assert_eq!(decoded.world_size, payload.world_size);
            assert_eq!(decoded.surface, payload.surface);
            assert_eq!(decoded.collision, payload.collision);
        }
    }

    #[test]
    fn truncated_buffers_decode_to_none() {
        let source =
            LocalChunkSource::new(SimpleNoiseProvider::new(), 42, GeneratorConfig::low_spec());
        let payload = source.load(0.0, 0.0, 0, 1);
        let bytes = encode_chunk_payload(&payload);
        for cut in [0, 1, 3, bytes.len() / 2, bytes.len() - 1] {
            assert!(decode_chunk_payload(&bytes[..cut]).is_none(), "cut={cut}");
        }
        assert!(decode_chunk_payload(&[0u8; 16]).is_none(), "bad magic");
    }
}
