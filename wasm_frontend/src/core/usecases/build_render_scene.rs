//! Compile a dense reference-room description into the same serialized SVO
//! contract consumed by production renderers.

use crate::application::ports::ChunkDraw;
use crate::adapters::collect_emissive_lights::collect_emissive_lights;
use crate::core::domain::room::RoomScene;
use crate::core::ports::reference_renderer::RenderSceneSnapshot;
use vackrooms::adapters::octree_gpu_serializer::OctreeGpuSerializer;
use vackrooms::domain::entities::voxel_grid::VoxelGrid;
use vackrooms::domain::use_cases::build_octree::BuildOctreeUseCase;
use vackrooms::use_cases::bake_voxel_lighting::{VoxelLightingSettings, bake_voxel_lighting};

pub fn build_render_scene(scene: RoomScene) -> RenderSceneSnapshot {
    let (mut voxels, voxel_size, origin) = scene.into_parts();
    let svo_depth = cube_depth(&voxels);
    let world_size = voxels.width() as f32 * voxel_size;
    let scene_lights = collect_emissive_lights(&voxels, voxel_size, origin, 0);

    let lighting = VoxelLightingSettings::with_default_range(voxel_size)
        .expect("RoomScene guarantees a positive finite voxel size");
    bake_voxel_lighting(&mut voxels, lighting);

    let svo = BuildOctreeUseCase::new().execute(&voxels, svo_depth, world_size);
    let serialized = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
    RenderSceneSnapshot {
        atlas: serialized.texel_data,
        chunks: vec![ChunkDraw {
            origin,
            root_index: svo.root as i32,
            world_size,
            voxel_size,
            svo_depth: svo_depth as u8,
        }],
        scene_lights,
    }
}

fn cube_depth(voxels: &VoxelGrid) -> u32 {
    let edge = voxels.width();
    assert!(
        edge.is_power_of_two(),
        "reference room edge must be a power of two"
    );
    assert_eq!(voxels.height(), edge, "reference room must be cubic");
    assert_eq!(voxels.depth(), edge, "reference room must be cubic");
    edge.ilog2()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ports::LightKind;

    #[test]
    fn reference_snapshot_keeps_the_analytic_ceiling_fixture() {
        let snapshot = build_render_scene(RoomScene::single_ceiling_fixture());

        assert_eq!(snapshot.scene_lights.len(), 1);
        let fixture = snapshot.scene_lights[0];
        assert!(fixture.enabled);
        assert_eq!(fixture.kind, LightKind::CeilingPanel);
        assert!(fixture.intensity > 0.0);
        assert!(fixture.radius > 0.0);
    }
}
