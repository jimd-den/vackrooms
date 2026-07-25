//! Native unit tests for the CPU splatter. Everything here runs with plain
//! `cargo test` — no GPU, no browser.

use super::atlas::build_mips;
use super::camera::Camera;
use super::raycast::trace_svo;
use super::settings::{CpuRenderSettings, CpuShadowMode};
use super::shading::FrameLighting;
use super::*;
use crate::application::ports::{ChunkDraw, FrameParams, LightKind, LightSource, RendererPort};
use vackrooms::adapters::octree_gpu_serializer::OctreeGpuSerializer;
use vackrooms::domain::entities::sparse_voxel_octree::SparseVoxelOctree;

/// depth-2 SVO (4^3) with one solid voxel, serialized like production.
fn one_voxel_atlas(color: u32, light: u8) -> (Vec<u32>, u32) {
    let mut svo = SparseVoxelOctree::new(2, 4.0);
    svo.set(1, 1, 1, 1, color, [light; 3], 0);
    let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
    (gpu.texel_data, svo.root as u32)
}

fn frame_at(pos: [f32; 3], yaw: f32) -> FrameParams {
    FrameParams {
        camera_pos: pos,
        yaw,
        pitch: 0.0,
        flashlight: false,
        ..FrameParams::default()
    }
}

fn center_pixel(r: &SoftwareRasterizer) -> [u8; 4] {
    let idx = (r.height() / 2 * r.width() + r.width() / 2) * 4;
    let fb = r.framebuffer();
    [fb[idx], fb[idx + 1], fb[idx + 2], fb[idx + 3]]
}

#[test]
fn voxel_in_front_of_camera_covers_center_pixel() {
    let (atlas, root) = one_voxel_atlas(0xFF8040, 15);
    let mut r = SoftwareRasterizer::new(64, 64);
    r.upload_atlas(&atlas);

    // Voxel spans (1..2)^3; look at its center from -z (yaw = PI faces +z).
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];
    r.draw(&frame_at([1.5, 1.5, -2.0], std::f32::consts::PI), &chunks);

    let px = center_pixel(&r);
    assert!(px[0] > 0, "ambient-visible red channel, got {:?}", px);
    assert!(px[0] > px[2], "red voxel must stay reddish, got {:?}", px);

    // A corner pixel remains the indoor fog background rather than inheriting
    // the voxel's shaded color.
    let fb = r.framebuffer();
    assert_ne!(&fb[0..3], &px[0..3]);
}

#[test]
fn visible_voxel_renders_at_tiny_odd_and_non_tile_aligned_sizes() {
    let (atlas, root) = one_voxel_atlas(0xFF8040, 15);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];
    let frame = frame_at([1.5, 1.5, -2.0], std::f32::consts::PI);

    for (width, height) in [(1, 1), (3, 5), (7, 9), (15, 17), (31, 19), (127, 73)] {
        let mut renderer = SoftwareRasterizer::new(width, height);
        renderer.upload_atlas(&atlas);
        renderer.draw(&frame, &chunks);

        assert_eq!(renderer.width(), width, "width at {width}x{height}");
        assert_eq!(renderer.height(), height, "height at {width}x{height}");
        assert_eq!(renderer.framebuffer().len(), width * height * 4);
        assert!(
            center_pixel(&renderer)[0] > 0,
            "visible voxel disappeared at {width}x{height}"
        );
        assert!(
            renderer
                .framebuffer()
                .chunks_exact(4)
                .all(|pixel| pixel[3] == 255),
            "alpha was not initialized at {width}x{height}"
        );
        assert!(renderer.telemetry().pixel_write_limit >= width * height);
    }
}

#[test]
fn resize_rebuilds_projection_depth_hz_and_frame_budget_together() {
    let (atlas, root) = one_voxel_atlas(0xFF8040, 15);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];
    let frame = frame_at([1.5, 1.5, -2.0], std::f32::consts::PI);
    let mut renderer = SoftwareRasterizer::new(64, 64);
    renderer.upload_atlas(&atlas);

    let mut previous_write_limit = 0;
    for (width, height) in [(64, 64), (1, 3), (23, 11), (191, 107)] {
        renderer.resize(width, height);
        renderer.draw(&frame, &chunks);

        assert_eq!(renderer.framebuffer().len(), width * height * 4);
        assert!(
            center_pixel(&renderer)[0] > 0,
            "after resize to {width}x{height}"
        );
        let telemetry = renderer.telemetry();
        assert!(telemetry.pixel_write_limit >= width * height);
        if width * height > 64 * 64 {
            assert!(telemetry.pixel_write_limit > previous_write_limit);
        }
        previous_write_limit = telemetry.pixel_write_limit;
    }
}

#[test]
fn hierarchical_z_is_image_equivalent_at_non_multiple_of_eight_size() {
    let (atlas, root) = one_voxel_atlas(0xFF8040, 15);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];
    let frame = frame_at([1.5, 1.5, -2.0], std::f32::consts::PI);
    let render = |hierarchical_z| {
        let mut renderer = SoftwareRasterizer::new(53, 37);
        renderer.settings.toggles.hierarchical_z = hierarchical_z;
        renderer.upload_atlas(&atlas);
        renderer.draw(&frame, &chunks);
        renderer.framebuffer().to_vec()
    };

    assert_eq!(render(false), render(true));
}

#[test]
fn camera_facing_away_sees_nothing() {
    let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
    let mut r = SoftwareRasterizer::new(32, 32);
    r.upload_atlas(&atlas);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];
    // yaw = 0 looks toward -z; the voxel is at +z relative to the camera.
    r.draw(&frame_at([1.5, 1.5, -2.0], 0.0), &chunks);
    let background = &r.framebuffer()[0..3];
    assert!(
        r.framebuffer()
            .chunks_exact(4)
            .all(|pixel| &pixel[0..3] == background)
    );
    assert_eq!(r.telemetry().visited_nodes, 0);
}

#[test]
fn nearer_voxel_wins_depth_test() {
    // Two stacked chunks: a red voxel near, a white voxel behind it.
    let (red, red_root) = one_voxel_atlas(0xFF0000, 15);
    let (white, white_root) = one_voxel_atlas(0xFFFFFF, 15);
    let red_nodes = red.len() as u32 / 4;

    let mut atlas = red.clone();
    atlas.extend_from_slice(&white);

    let mut r = SoftwareRasterizer::new(64, 64);
    r.upload_atlas(&atlas);
    let chunks = [
        // Far chunk listed first to prove sorting/z-buffer handles order.
        ChunkDraw {
            origin: [0.0, 0.0, 6.0],
            root_index: (red_nodes + white_root) as i32,
            world_size: 4.0,
            voxel_size: 1.0,
            svo_depth: 2,
        },
        ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: red_root as i32,
            world_size: 4.0,
            voxel_size: 1.0,
            svo_depth: 2,
        },
    ];
    r.draw(&frame_at([1.5, 1.5, -2.0], std::f32::consts::PI), &chunks);

    let px = center_pixel(&r);
    assert!(
        px[0] > 0 && px[2] < px[0] / 2,
        "near red voxel must win: {:?}",
        px
    );
}

#[test]
fn depth_test_holds_even_without_front_to_back_sorting() {
    // Same scene as above with the ordering optimization off: the image
    // must not change (only the overdraw count may).
    let (red, red_root) = one_voxel_atlas(0xFF0000, 15);
    let (white, white_root) = one_voxel_atlas(0xFFFFFF, 15);
    let red_nodes = red.len() as u32 / 4;
    let mut atlas = red.clone();
    atlas.extend_from_slice(&white);

    let mut r = SoftwareRasterizer::new(64, 64);
    r.settings.toggles.front_to_back = false;
    r.settings.toggles.hierarchical_z = false;
    r.upload_atlas(&atlas);
    let chunks = [
        ChunkDraw {
            origin: [0.0, 0.0, 6.0],
            root_index: (red_nodes + white_root) as i32,
            world_size: 4.0,
            voxel_size: 1.0,
            svo_depth: 2,
        },
        ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: red_root as i32,
            world_size: 4.0,
            voxel_size: 1.0,
            svo_depth: 2,
        },
    ];
    r.draw(&frame_at([1.5, 1.5, -2.0], std::f32::consts::PI), &chunks);

    let px = center_pixel(&r);
    assert!(
        px[0] > 0 && px[2] < px[0] / 2,
        "depth test alone must keep the near voxel in front: {:?}",
        px
    );
}

#[test]
fn deferred_shading_rejection_is_byte_identical_to_shade_then_depth_test() {
    let (near, near_root) = one_voxel_atlas(0xCC3300, 0);
    let (far, far_root) = one_voxel_atlas(0xFFFFFF, 15);
    let near_nodes = near.len() as u32 / 4;
    let mut atlas = near;
    atlas.extend_from_slice(&far);
    let chunks = [
        ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: near_root as i32,
            world_size: 4.0,
            voxel_size: 1.0,
            svo_depth: 2,
        },
        ChunkDraw {
            origin: [0.0, 0.0, 6.0],
            root_index: (near_nodes + far_root) as i32,
            world_size: 4.0,
            voxel_size: 1.0,
            svo_depth: 2,
        },
    ];
    let frame = frame_at([1.5, 1.5, -2.0], std::f32::consts::PI);

    let render = |deferred_shading| {
        let mut renderer = SoftwareRasterizer::new(64, 64);
        renderer.settings.toggles.deferred_shading = deferred_shading;
        renderer.upload_atlas(&atlas);
        renderer.draw(&frame, &chunks);
        renderer.framebuffer().to_vec()
    };

    assert_eq!(render(false), render(true));
}

#[test]
fn mip_aggregation_works_for_root_last_node_order() {
    // BuildOctreeUseCase (the production chunk pipeline) pushes children
    // BEFORE their parent, so the root is the LAST node — the opposite
    // order of SparseVoxelOctree::set. build_mips must handle both.
    use vackrooms::domain::entities::voxel_grid::{VOXEL_WALL, VoxelGrid};
    use vackrooms::adapters::material_palette::DEFAULT_MATERIAL_PALETTE;
    use vackrooms::use_cases::build_octree::BuildOctreeUseCase;

    let mut grid = VoxelGrid::new(4, 4, 4);
    grid.set(0, 0, 0, VOXEL_WALL);
    grid.set(3, 3, 3, VOXEL_WALL);
    let svo = BuildOctreeUseCase::new(&DEFAULT_MATERIAL_PALETTE).execute(&grid, 2, 4.0);
    let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
    let mips = build_mips(&gpu.texel_data);

    let root = &mips[svo.root];
    assert!(
        root.occupancy > 0.0,
        "root mip must see its solid descendants, got occupancy 0"
    );
}

#[test]
fn mip_aggregation_averages_child_colors() {
    let mut svo = SparseVoxelOctree::new(1, 2.0);
    // Two solid children: pure red + pure blue -> average purple-ish.
    svo.set(0, 0, 0, 1, 0xFF0000, [10; 3], 0);
    svo.set(1, 1, 1, 1, 0x0000FF, [4; 3], 0);
    let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
    let mips = build_mips(&gpu.texel_data);

    let root = &mips[svo.root];
    assert!((root.occupancy - 2.0 / 8.0).abs() < 1e-6);
    assert!((root.linear_albedo[0] - 0.5).abs() < 1.0e-6);
    assert!((root.linear_albedo[2] - 0.5).abs() < 1.0e-6);
    assert_eq!(root.baked_rgb, [7.0; 3]);
}

#[test]
fn mip_bake_preserves_color_and_averages_instead_of_smearing_a_peak() {
    let mut svo = SparseVoxelOctree::new(1, 2.0);
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let baked = if [x, y, z] == [0, 0, 0] {
                    [15, 2, 1]
                } else {
                    [0, 0, 0]
                };
                svo.set(x, y, z, 1, 0x808080, baked, 0);
            }
        }
    }
    let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
    let root = build_mips(&gpu.texel_data)[svo.root];

    assert_eq!(root.baked_rgb, [15.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0]);
    assert!(root.baked_rgb[0] > root.baked_rgb[1]);
    assert!(root.baked_rgb[1] > root.baked_rgb[2]);
    assert!(
        root.baked_rgb[0] < 15.0,
        "one receiver must not light the whole MIP"
    );
}

#[test]
fn mip_cache_preserves_emissive_coverage_and_radiance() {
    use vackrooms::domain::entities::voxel_grid::VOXEL_LIGHT;

    let mut svo = SparseVoxelOctree::new(1, 2.0);
    svo.set(0, 1, 0, VOXEL_LIGHT, 0xFFF8D6, [15, 14, 11], 0);
    let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
    let root = build_mips(&gpu.texel_data)[svo.root];

    assert_eq!(root.emissive_coverage, 1.0 / 8.0);
    assert!(
        root.emitted_radiance_density
            .into_iter()
            .all(|value| value > 0.0)
    );
}

#[test]
fn distant_geometry_lods_to_single_splats() {
    let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
    let mut r = SoftwareRasterizer::new(64, 64);
    r.upload_atlas(&atlas);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];
    // Very far away: the voxel projects to well under a pixel, and the
    // root's occupancy (1/64) is below min_mip_occupancy -> no draw,
    // proving the LOD path (not the leaf path) handled it.
    r.settings.max_draw_distance = 1000.0;
    r.draw(&frame_at([1.5, 1.5, -400.0], std::f32::consts::PI), &chunks);
    assert_eq!(r.telemetry().splat_count, 0);
    assert_eq!(r.telemetry().pixel_writes, 0);
}

#[test]
fn exact_entry_face_changes_the_lambert_response_at_an_axis_boundary() {
    let mut r = SoftwareRasterizer::new(16, 16);
    let center = [0.0, 0.0, 0.0];

    // Case A: Camera Y is 1.01 (above the center, Y-dominant)
    let frame_a = FrameParams {
        camera_pos: [-1.0, 1.01, -0.1],
        yaw: 0.0,
        pitch: 0.0,
        flashlight: false,
        ..FrameParams::default()
    };
    let cam_a = Camera::new(&frame_a, 16, 16, &CpuRenderSettings::default());

    // Case B: Camera Y is 0.99 (below Case A, X-dominant)
    let frame_b = FrameParams {
        camera_pos: [-1.0, 0.99, -0.1],
        yaw: 0.0,
        pitch: 0.0,
        flashlight: false,
        ..FrameParams::default()
    };
    let cam_b = Camera::new(&frame_b, 16, 16, &CpuRenderSettings::default());

    r.clear();
    let fixture = [LightSource {
        id: 1,
        position: [0.0, 5.0, 0.0],
        half_size: [2.0, 2.0],
        color: [1.0, 0.9, 0.7],
        radius: 20.0,
        intensity: 10.0,
        kind: LightKind::CeilingPanel,
        flicker_mode: 0,
        enabled: true,
    }];

    r.shade_and_splat_with_frame_lighting(
        &cam_a,
        &[],
        FrameLighting::unshadowed(&fixture),
        center,
        8.0,                   // world_size
        8.0,                   // px — the center pixel both cases read
        8.0,                   // py
        1.0,                   // half_px
        1.0,                   // dist
        [100.0, 100.0, 100.0], // base color
        15.0,                  // light level
        false,                 // is_emissive
        1,                     // crowded siblings
    );
    let color_a = center_pixel(&r);

    r.clear();
    r.shade_and_splat_with_frame_lighting(
        &cam_b,
        &[],
        FrameLighting::unshadowed(&fixture),
        center,
        8.0,
        8.0,
        8.0,
        1.0,
        1.0,
        [100.0, 100.0, 100.0],
        15.0,
        false,
        1,
    );
    let color_b = center_pixel(&r);

    // The Y entry face sees the overhead panel. The neighboring X entry face
    // does not: this discontinuity is the exact AABB/Lambert result, not a
    // soft diagonal normal invented to hide a face boundary.
    assert!(
        color_a[0] > color_b[0] + 10,
        "expected exact face response: y-face={color_a:?}, x-face={color_b:?}"
    );
}

#[test]
fn equal_material_floor_splats_share_the_same_uniform_ambient() {
    let mut r = SoftwareRasterizer::new(16, 16);

    // Camera positioned above the floor plane (Y = 5.0)
    let frame = FrameParams {
        camera_pos: [1.0, 5.0, 1.0],
        yaw: 0.0,
        pitch: 0.0,
        flashlight: false,
        ..FrameParams::default()
    };
    let cam = Camera::new(&frame, 16, 16, &CpuRenderSettings::default());

    r.clear();
    r.shade_and_splat(
        &cam,
        &[],
        [0.5, 0.0, 0.5],
        8.0,
        8.0,
        8.0, // py
        1.0, // half_px
        5.0497,
        [100.0, 100.0, 100.0],
        15.0,
        false,
        1,
    );
    let color_1 = center_pixel(&r);

    r.clear();
    r.shade_and_splat(
        &cam,
        &[],
        [2.5, 0.0, 0.5],
        8.0,
        8.0,
        8.0, // py
        1.0, // half_px
        5.2440,
        [100.0, 100.0, 100.0],
        15.0,
        false,
        1,
    );
    let color_2 = center_pixel(&r);

    // Ambient irradiance is spatially uniform. Distance no longer invents a
    // per-tile brightness grid; analytic finite lights own spatial falloff.
    assert_eq!(color_1, color_2);
}

#[test]
fn test_splat_flat_shading_limitation() {
    let mut r = SoftwareRasterizer::new(16, 16);
    let frame = FrameParams {
        camera_pos: [1.0, 5.0, 1.0],
        yaw: 0.0,
        pitch: 0.0,
        flashlight: false,
        ..FrameParams::default()
    };
    let cam = Camera::new(&frame, 16, 16, &CpuRenderSettings::default());

    // Draw a single large splat (half_px = 4.0) centered at [8.0, 8.0]
    r.clear();
    r.shade_and_splat(
        &cam,
        &[],
        [0.5, 0.0, 0.5],
        8.0,
        8.0,
        8.0,
        4.0, // half_px
        5.0,
        [100.0, 100.0, 100.0],
        15.0,
        false,
        1,
    );

    let idx_left = (8 * r.width() + 5) * 4;
    let idx_right = (8 * r.width() + 10) * 4;
    let fb = r.framebuffer();
    let color_left = [fb[idx_left], fb[idx_left + 1], fb[idx_left + 2]];
    let color_right = [fb[idx_right], fb[idx_right + 1], fb[idx_right + 2]];

    // One splat is one flat color across its whole projected area.
    assert_eq!(
        color_left, color_right,
        "Expected a single splat to be flat-shaded (all its pixels have identical color)"
    );
}

#[test]
fn camera_inside_large_solid_leaf_terminates_under_budget() {
    const VIRTUAL_DEPTH_CAP: u8 = 3;

    let mut atlas = vec![0u32; 4];
    atlas[0] = 1; // Leaf
    atlas[1] = 1; // Solid voxel type
    atlas[2] = 0xFFFFFF; // White color
    atlas[3] = 15; // quantized diffuse fill

    let mut r = SoftwareRasterizer::new(64, 64);
    // A camera-plane box must subdivide for exact clipped coverage. Use the
    // minimum supported cap so even a completely visible eight-way walk has
    // a small, deterministic upper bound independent of culling details.
    r.settings.max_virtual_depth = VIRTUAL_DEPTH_CAP;
    r.upload_atlas(&atlas);

    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: 0,
        world_size: 16.0, // Very large leaf
        voxel_size: 16.0,
        svo_depth: 0,
    }];

    // Camera inside the bounding sphere of the root chunk.
    r.draw(&frame_at([8.0, 8.0, 8.0], 0.0), &chunks);

    let stats = r.telemetry();
    let complete_tree_visit_bound = (8usize.pow(u32::from(VIRTUAL_DEPTH_CAP) + 1) - 1) / (8 - 1);

    assert_eq!(
        stats.max_virtual_depth,
        usize::from(VIRTUAL_DEPTH_CAP),
        "camera-plane coverage should refine to the explicit virtual-depth cap"
    );
    assert!(
        complete_tree_visit_bound < stats.node_soft_limit,
        "test fixture's complete tree must fit below the soft work budget"
    );
    assert!(
        stats.visited_nodes <= complete_tree_visit_bound,
        "visited {} nodes, exceeding the depth-{VIRTUAL_DEPTH_CAP} complete-tree bound of {complete_tree_visit_bound}",
        stats.visited_nodes,
    );
    assert!(!stats.budget_exhausted);
}

#[test]
fn test_cpu_render_settings_default() {
    let settings = CpuRenderSettings::default();
    assert_eq!(settings, CpuQualityPreset::Balanced.settings());
    assert_eq!(settings.max_splat_radius_px, 8.0);
    assert_eq!(settings.max_virtual_depth, 5);
    assert_eq!(settings.shadows, CpuShadowMode::Off);
    assert!(settings.toggles.hierarchical_z, "optimizations default on");
}

#[test]
fn test_trace_svo_basic_intersect() {
    let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];

    // Ray passing through the center of the voxel (1.5, 1.5, 1.5).
    let hit = trace_svo(
        &atlas,
        &chunks,
        [1.5, 1.5, -1.0],
        [0.0, 0.0, 1.0],
        10.0,
        true,
    );
    assert!(hit.is_some());
    let hit_val = hit.unwrap();
    assert_eq!(hit_val.voxel_type, 1); // VOXEL_WALL
    assert!((hit_val.t - 2.0).abs() < 1e-4); // voxel starts at z=1.0

    // Ray missing the voxel.
    let miss = trace_svo(
        &atlas,
        &chunks,
        [0.5, 0.5, -1.0],
        [0.0, 0.0, 1.0],
        10.0,
        true,
    );
    assert!(miss.is_none());
}

/// REGRESSION (cone-light angles): a truly cardinal shadow ray must remain
/// cardinal. The old slab workaround changed every tiny/zero component to
/// +/-1e-4; by the time this +Z ray reached the voxel it had drifted across
/// x=2, selected the neighboring octant, and made the flashlight flicker off
/// as yaw crossed the axis.
#[test]
fn cardinal_secondary_ray_does_not_drift_across_an_octant_boundary() {
    let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];

    for x_direction in [-5.0e-7, -0.0, 0.0, 5.0e-7] {
        for front_to_back in [false, true] {
            let hit = trace_svo(
                &atlas,
                &chunks,
                [2.0 - 5.0e-5, 1.5, -1.0],
                [x_direction, 0.0, 1.0],
                10.0,
                front_to_back,
            )
            .expect("a near-cardinal +Z ray must stay inside the voxel's x slab");

            assert_eq!(hit.voxel_type, 1);
            assert!((hit.t - 2.0).abs() < 1.0e-4);
        }
    }
}

/// REGRESSION (cone light): a ray crossing an empty *lower-Y* masked-out
/// octant used to compute its exit plane from `center[1]` instead of the
/// octant floor, stall until the step budget, and report "no hit". The
/// flashlight then read as unoccluded through floors, and downward beam
/// rays randomly blacked out.
#[test]
fn trace_through_empty_lower_octant_still_finds_the_wall() {
    let mut svo = SparseVoxelOctree::new(2, 4.0);
    svo.set(1, 0, 1, 1, 0xFFFFFF, [15; 3], 0); // solid at cell (1,0,1)
    let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: svo.root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];

    // Start in the empty lower-half cell (0,0,1) and head +x with a
    // downward slope: the march must cross the empty octant and hit the
    // solid voxel spanning (1..2, 0..1, 1..2).
    let dir_len = (1.0f32 + 0.3 * 0.3).sqrt();
    let dir = [1.0 / dir_len, -0.3 / dir_len, 0.0];
    let hit = trace_svo(&gpu.texel_data, &chunks, [0.5, 0.9, 1.5], dir, 10.0, true);
    assert!(
        hit.is_some(),
        "downward ray must cross the empty octant and hit the voxel"
    );
    assert_eq!(hit.unwrap().voxel_type, 1);
}

/// REGRESSION (cone light): hits beyond `max_t` must be ignored — the old
/// code returned them, letting walls *behind* a lit surface "occlude" the
/// flashlight.
#[test]
fn hits_beyond_max_t_are_ignored() {
    let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];
    // The voxel face is at t = 2.0 along this ray; a trace capped at 1.0
    // must come back clear.
    let hit = trace_svo(
        &atlas,
        &chunks,
        [1.5, 1.5, -1.0],
        [0.0, 0.0, 1.0],
        1.0,
        true,
    );
    assert!(hit.is_none(), "hit at t=2.0 must not satisfy max_t=1.0");
}

#[test]
fn secondary_ray_reference_order_still_returns_the_nearest_chunk() {
    let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
    // Deliberately put the far chunk first: the rt_f2b-disabled path must
    // compare all hits instead of returning the first input-order hit.
    let chunks = [
        ChunkDraw {
            origin: [0.0, 0.0, 8.0],
            root_index: root as i32,
            world_size: 4.0,
            voxel_size: 1.0,
            svo_depth: 2,
        },
        ChunkDraw {
            origin: [0.0, 0.0, 0.0],
            root_index: root as i32,
            world_size: 4.0,
            voxel_size: 1.0,
            svo_depth: 2,
        },
    ];

    let origin = [1.5, 1.5, -1.0];
    let direction = [0.0, 0.0, 1.0];
    let reference = trace_svo(&atlas, &chunks, origin, direction, 20.0, false).unwrap();
    let optimized = trace_svo(&atlas, &chunks, origin, direction, 20.0, true).unwrap();

    assert!((reference.t - 2.0).abs() < 1.0e-4);
    assert_eq!(reference, optimized);
}

#[test]
fn test_draw_distance_culling() {
    let (atlas, root) = one_voxel_atlas(0xFFFFFF, 15);
    let mut r = SoftwareRasterizer::new(16, 16);
    r.upload_atlas(&atlas);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];
    // Draw with large draw distance
    r.settings.max_draw_distance = 100.0;
    let frame = frame_at([1.5, 1.5, -2.0], std::f32::consts::PI);
    r.draw(&frame, &chunks);
    assert!(r.telemetry().visited_nodes > 0);

    // Draw with tiny draw distance (should cull the chunk)
    r.settings.max_draw_distance = 0.1;
    r.draw(&frame, &chunks);
    assert_eq!(r.telemetry().visited_nodes, 0);
}

#[test]
fn test_flashlight_illuminates_voxel() {
    let (atlas, root) = one_voxel_atlas(0xFFFFFF, 0); // Unlit white voxel
    let mut r = SoftwareRasterizer::new(64, 64);
    r.upload_atlas(&atlas);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];

    // yaw = PI faces +z, toward the voxel (yaw = 0 would face away — the
    // old version of this test looked at empty space and compared black
    // to black).
    let mut frame_no_flash = frame_at([1.5, 1.5, -2.0], std::f32::consts::PI);
    frame_no_flash.flashlight = false;
    r.draw(&frame_no_flash, &chunks);
    let dark_px = center_pixel(&r);

    let mut frame_flash = frame_at([1.5, 1.5, -2.0], std::f32::consts::PI);
    frame_flash.flashlight = true;
    r.draw(&frame_flash, &chunks);
    let bright_px = center_pixel(&r);

    assert!(
        bright_px[0] > dark_px[0] + 50,
        "Flashlight should meaningfully illuminate the voxel (bright_px: {:?}, dark_px: {:?})",
        bright_px,
        dark_px
    );
}

/// The beam must NOT reach a surface behind a wall (needs the fixed
/// occlusion trace to pass together with `test_flashlight_illuminates_voxel`).
#[test]
fn flashlight_is_blocked_by_a_wall() {
    // Two-voxel column along z: near wall at cell z=1, far wall at z=2,
    // camera looking down the column. The near wall face is lit; if we
    // remove it the far wall would be lit — here we just assert the lit
    // face is the near one by depth.
    let mut svo = SparseVoxelOctree::new(2, 4.0);
    svo.set(1, 1, 1, 1, 0xFFFFFF, [0; 3], 0);
    svo.set(1, 1, 2, 1, 0xFFFFFF, [0; 3], 0);
    let gpu = OctreeGpuSerializer::serialize_to_gpu_data(&svo);
    let chunks = [ChunkDraw {
        origin: [0.0, 0.0, 0.0],
        root_index: svo.root as i32,
        world_size: 4.0,
        voxel_size: 1.0,
        svo_depth: 2,
    }];

    // The beam from the camera toward the far voxel's center is blocked by
    // the near voxel: an occlusion trace stopping short of the far surface
    // must report the near wall.
    let hit = trace_svo(
        &gpu.texel_data,
        &chunks,
        [1.5, 1.5, -0.5],
        [0.0, 0.0, 1.0],
        2.9,
        true,
    );
    assert!(hit.is_some(), "near wall must occlude");
    assert!(hit.unwrap().t < 2.0, "the NEAR wall is the occluder");
}
