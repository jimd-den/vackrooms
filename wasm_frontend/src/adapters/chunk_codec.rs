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
    ChunkPayload, FaceCellRange, FaceInstanceSet, LightKind, LightSource, PackedFaceInstance,
    PackedVertex, SurfaceMeshPayload,
};
use vackrooms::domain::entities::anomaly::{LevelExit, PitHazard, TraversalGate};
use vackrooms::domain::entities::supplies::SupplyItem;

/// "VKC" + version 6. Version 6 adds supply items and level exits.
const MAGIC: u32 = 0x564B_4306;

pub fn encode_chunk_payload(payload: &ChunkPayload) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        64 + payload.nodes.len() * 4
            + payload.surface.vertices.len() * 10
            + payload.surface.indices.len() * 4
            + payload.surface.faces.instances.len() * 16
            + payload.surface.light_volume_bytes.len()
            + payload.lights.len() * 51
            + payload.collision.len() * 24
            + payload.traversal_gates.len() * TraversalGate::WORDS * 4
            + payload.pit_hazards.len() * PitHazard::TRANSPORT_WORDS * 4,
    );
    put_u32(&mut out, MAGIC);
    put_u32(&mut out, payload.root);
    put_f32(&mut out, payload.world_size);
    put_f32(&mut out, payload.voxel_size);
    out.push(payload.svo_depth);

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
    // The volume occupies the original version-six slots, so restoring the
    // payload does not change later field offsets or require a format bump.
    for dimension in s.light_volume_dims {
        put_u32(&mut out, dimension);
    }
    out.push(0);
    put_u32(&mut out, s.light_volume_bytes.len() as u32);
    out.extend_from_slice(&s.light_volume_bytes);

    put_u32(&mut out, payload.lights.len() as u32);
    for l in &payload.lights {
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
        out.push(l.kind as u8);
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

    put_u32(&mut out, payload.traversal_gates.len() as u32);
    for gate in &payload.traversal_gates {
        for word in gate.to_words() {
            put_u32(&mut out, word);
        }
    }
    put_u32(&mut out, payload.pit_hazards.len() as u32);
    for hazard in &payload.pit_hazards {
        let mut words = Vec::with_capacity(PitHazard::TRANSPORT_WORDS);
        hazard.write_words(&mut words);
        for word in words {
            put_u32(&mut out, word);
        }
    }
    put_u32(&mut out, payload.supply_items.len() as u32);
    for item in &payload.supply_items {
        for word in item.to_words() {
            put_u32(&mut out, word);
        }
    }
    put_u32(&mut out, payload.level_exits.len() as u32);
    for exit in &payload.level_exits {
        for word in exit.to_words() {
            put_u32(&mut out, word);
        }
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
    let voxel_size = r.f32()?;
    let svo_depth = r.u8()?;

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
    let light_volume_dims = [r.u32()?, r.u32()?, r.u32()?];
    let _light_volume_padding = r.u8()?;
    let volume_len = r.len(1)?;
    let light_volume_bytes = r.slice(volume_len)?.to_vec();

    let light_count = r.len(51)?;
    let mut lights = Vec::with_capacity(light_count);
    for _ in 0..light_count {
        lights.push(LightSource {
            id: r.u64()?,
            position: [r.f32()?, r.f32()?, r.f32()?],
            half_size: [r.f32()?, r.f32()?],
            color: [r.f32()?, r.f32()?, r.f32()?],
            radius: r.f32()?,
            intensity: r.f32()?,
            kind: LightKind::from_u8(r.u8()?)?,
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

    let gate_count = r.len(TraversalGate::WORDS * 4)?;
    let mut traversal_gates = Vec::with_capacity(gate_count);
    for _ in 0..gate_count {
        let mut words = [0u32; TraversalGate::WORDS];
        for word in &mut words {
            *word = r.u32()?;
        }
        traversal_gates.push(TraversalGate::from_words(&words)?);
    }
    let hazard_count = r.len(PitHazard::TRANSPORT_WORDS * 4)?;
    let mut pit_hazards = Vec::with_capacity(hazard_count);
    for _ in 0..hazard_count {
        let mut words = [0u32; PitHazard::TRANSPORT_WORDS];
        for word in &mut words {
            *word = r.u32()?;
        }
        pit_hazards.push(PitHazard::from_transport_words(&words)?);
    }
    let supply_count = r.len(SupplyItem::WORDS * 4)?;
    let mut supply_items = Vec::with_capacity(supply_count);
    for _ in 0..supply_count {
        let mut words = [0u32; SupplyItem::WORDS];
        for word in &mut words {
            *word = r.u32()?;
        }
        supply_items.push(SupplyItem::from_words(&words)?);
    }
    let exit_count = r.len(LevelExit::WORDS * 4)?;
    let mut level_exits = Vec::with_capacity(exit_count);
    for _ in 0..exit_count {
        let mut words = [0u32; LevelExit::WORDS];
        for word in &mut words {
            *word = r.u32()?;
        }
        level_exits.push(LevelExit::from_words(&words)?);
    }

    Some(ChunkPayload {
        root,
        nodes,
        world_size,
        voxel_size,
        svo_depth,
        surface: SurfaceMeshPayload {
            vertices,
            indices,
            bounds,
            lod,
            faces: FaceInstanceSet {
                instances,
                cells,
                cell_size,
            },
            voxel_scale,
            light_volume_bytes,
            light_volume_dims,
        },
        lights,
        collision,
        traversal_gates,
        pit_hazards,
        supply_items,
        level_exits,
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
    use crate::application::ports::{ChunkSourcePort, RenderArtifactNeeds};
    use vackrooms::domain::entities::anomaly::RealitySnapshot;
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
            assert_eq!(decoded.lights, payload.lights);
            assert_eq!(decoded.collision, payload.collision);
        }
    }

    #[test]
    fn selected_artifacts_stay_omitted_across_worker_encoding() {
        let source =
            LocalChunkSource::new(SimpleNoiseProvider::new(), 42, GeneratorConfig::low_spec());
        let reality = RealitySnapshot::default();
        let surface =
            source.load_with_artifacts(0.0, 30.0, 0, 1, &reality, RenderArtifactNeeds::SURFACE);
        let raymarch =
            source.load_with_artifacts(0.0, 30.0, 0, 1, &reality, RenderArtifactNeeds::RAYMARCH);
        let all = source.load_with_artifacts(0.0, 30.0, 0, 1, &reality, RenderArtifactNeeds::ALL);

        let surface_bytes = encode_chunk_payload(&surface);
        let raymarch_bytes = encode_chunk_payload(&raymarch);
        let all_bytes = encode_chunk_payload(&all);
        let decoded_surface = decode_chunk_payload(&surface_bytes).expect("surface payload");
        let decoded_raymarch = decode_chunk_payload(&raymarch_bytes).expect("raymarch payload");

        assert!(decoded_surface.nodes.is_empty());
        assert!(!decoded_surface.surface.vertices.is_empty());
        assert!(decoded_surface.surface.faces.instances.is_empty());
        assert!(decoded_raymarch.surface.vertices.is_empty());
        assert!(decoded_raymarch.surface.faces.instances.is_empty());
        assert!(!decoded_raymarch.nodes.is_empty());
        assert_eq!(decoded_surface.lights, surface.lights);
        assert_eq!(decoded_raymarch.lights, raymarch.lights);
        assert!(surface_bytes.len() < all_bytes.len());
        assert!(raymarch_bytes.len() < all_bytes.len());
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

    #[test]
    fn anomaly_semantics_roundtrip_exactly() {
        use vackrooms::domain::entities::anomaly::{
            AnomalyKind, Axis2, AxisDirection, PitHazard, TraversalGate, TraversalGateKind,
            WorldBounds,
        };
        use vackrooms::domain::entities::position::Position;

        let source =
            LocalChunkSource::new(SimpleNoiseProvider::new(), 42, GeneratorConfig::low_spec());
        let mut payload = source.load(0.0, 0.0, 0, 1);
        payload.traversal_gates = vec![TraversalGate {
            id: u64::MAX - 3,
            instance_id: 0x1234_5678_90AB_CDEF,
            anomaly_kind: AnomalyKind::PillarExpanse,
            kind: TraversalGateKind::Remap,
            axis: Axis2::Z,
            plane: -12.4,
            span_min: 3.2,
            span_max: 19.6,
            forward: AxisDirection::Negative,
            affected_bounds: WorldBounds::new(-20.0, -30.0, 40.0, 50.0),
        }];
        payload.pit_hazards = vec![PitHazard {
            id: 77,
            instance_id: 88,
            center: Position::new(-1.2, 9.6),
            half_side: 0.6,
            depth: 2.4,
            recovery: Position::new(0.8, 10.8),
        }];
        payload.supply_items = vec![vackrooms::domain::entities::supplies::SupplyItem {
            id: 0xAA55_1234_5678_9ABC,
            kind: vackrooms::domain::entities::supplies::SupplyKind::AlmondWater,
            position: Position::new(33.4, -808.25),
            rest_y: 0.8,
        }];
        payload.level_exits = vec![vackrooms::domain::entities::anomaly::LevelExit {
            id: 0x0D00_E000_0000_0001,
            target_level: 1,
            center: Position::new(12.0, -8.0),
            half_extent: 0.7,
            arrival: Position::new(3.0, 3.0),
        }];
        let decoded = decode_chunk_payload(&encode_chunk_payload(&payload)).unwrap();
        assert_eq!(decoded.traversal_gates, payload.traversal_gates);
        assert_eq!(decoded.pit_hazards, payload.pit_hazards);
        assert_eq!(decoded.supply_items, payload.supply_items);
        assert_eq!(decoded.level_exits, payload.level_exits);
        assert_eq!(decoded, payload);
    }
}
