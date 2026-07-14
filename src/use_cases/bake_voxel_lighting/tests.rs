use crate::domain::entities::voxel_grid::{
    VOXEL_AIR, VOXEL_CEILING, VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_RED_LIGHT, VOXEL_WALL, VoxelGrid,
};

use super::{VoxelLightingSettings, bake_voxel_lighting};

fn settings(voxel_size: f32, range: f32) -> VoxelLightingSettings {
    VoxelLightingSettings::new(voxel_size, range).expect("valid test lighting settings")
}

#[test]
fn downward_light_attenuates_symmetrically_in_world_space() {
    // Equal receivers one world unit above and below the source must agree.
    // The former implementation special-cased downward travel to lose no
    // energy, making the lower receiver substantially brighter.
    let mut grid = VoxelGrid::new(1, 7, 1);
    grid.set(0, 0, 0, VOXEL_FLOOR);
    grid.set(0, 3, 0, VOXEL_LIGHT);
    grid.set(0, 6, 0, VOXEL_CEILING);

    bake_voxel_lighting(&mut grid, settings(0.5, 4.0));

    let source = grid.get_light_rgb(0, 3, 0);
    let below = grid.get_light_rgb(0, 0, 0);
    let above = grid.get_light_rgb(0, 6, 0);
    assert_eq!(
        below, above,
        "attenuation must not depend on up versus down"
    );
    assert!(
        below[0] < source[0],
        "a remote floor cannot retain source intensity"
    );
    assert!(
        below[0] > 0,
        "the receiver remains inside the configured range"
    );
}

#[test]
fn a_wall_receives_light_but_does_not_transmit_it() {
    let mut grid = VoxelGrid::new(7, 1, 1);
    grid.set(0, 0, 0, VOXEL_LIGHT);
    grid.set(2, 0, 0, VOXEL_WALL);
    grid.set(6, 0, 0, VOXEL_WALL);

    bake_voxel_lighting(&mut grid, settings(0.5, 8.0));

    assert!(
        grid.get_light(2, 0, 0) > 0,
        "the near wall face is a receiver"
    );
    assert_eq!(
        grid.get_light_rgb(3, 0, 0),
        [0; 3],
        "no air path crosses the wall"
    );
    assert_eq!(
        grid.get_light_rgb(6, 0, 0),
        [0; 3],
        "a receiver behind the wall must stay dark"
    );
}

#[test]
fn a_ceiling_panel_emits_through_its_exposed_room_face() {
    let mut grid = VoxelGrid::new(1, 4, 1);
    grid.set(0, 0, 0, VOXEL_FLOOR);
    grid.set(0, 3, 0, VOXEL_LIGHT);

    bake_voxel_lighting(&mut grid, settings(0.5, 4.0));

    assert_eq!(grid.get_light_rgb(0, 3, 0), [15, 14, 11]);
    assert!(
        grid.get_light(0, 0, 0) > 0,
        "an embedded ceiling panel must illuminate the floor through its exposed lower face"
    );
}

#[test]
fn a_fully_buried_emitter_does_not_inject_light() {
    let mut grid = VoxelGrid::new(3, 3, 3);
    for z in 0..grid.depth() {
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                grid.set(x, y, z, VOXEL_WALL);
            }
        }
    }
    grid.set(1, 1, 1, VOXEL_LIGHT);

    bake_voxel_lighting(&mut grid, settings(0.5, 4.0));

    assert_eq!(grid.get_light_rgb(1, 1, 1), [15, 14, 11]);
    assert_eq!(
        grid.get_light_rgb(1, 1, 0),
        [0; 3],
        "emission may only enter the bake through an exposed face"
    );
}

fn equal_physical_gap_bake(voxel_size: f32) -> [u8; 3] {
    // A 0.2-unit floor slab, a 1.8-unit air gap, and a 0.2-unit panel slab.
    // Both resolutions describe the same physical surfaces.
    let cells_per_slab = (0.2 / voxel_size).round() as usize;
    let source_bottom = (2.0 / voxel_size).round() as usize;
    let height = source_bottom + cells_per_slab;
    let mut grid = VoxelGrid::new(1, height, 1);
    for y in 0..cells_per_slab {
        grid.set(0, y, 0, VOXEL_FLOOR);
    }
    for y in source_bottom..height {
        grid.set(0, y, 0, VOXEL_LIGHT);
    }

    bake_voxel_lighting(&mut grid, settings(voxel_size, 4.0));
    grid.get_light_rgb(0, cells_per_slab - 1, 0)
}

#[test]
fn equal_world_distances_match_at_point_one_and_point_two_voxels() {
    let fine = equal_physical_gap_bake(0.1);
    let coarse = equal_physical_gap_bake(0.2);
    assert_eq!(
        fine, coarse,
        "voxel resolution changed a physical light distance"
    );
    assert_ne!(fine, [0; 3]);
}

#[test]
fn rebaking_starts_from_a_cleared_light_field() {
    let mut grid = VoxelGrid::new(4, 1, 1);
    grid.set(0, 0, 0, VOXEL_LIGHT);
    grid.set(3, 0, 0, VOXEL_WALL);
    let bake_settings = settings(0.5, 4.0);
    bake_voxel_lighting(&mut grid, bake_settings);
    assert!(grid.get_light(3, 0, 0) > 0);

    grid.set(0, 0, 0, VOXEL_AIR);
    bake_voxel_lighting(&mut grid, bake_settings);
    for x in 0..grid.width() {
        assert_eq!(grid.get_light_rgb(x, 0, 0), [0; 3], "stale light at x={x}");
    }
}

#[test]
fn differently_colored_sources_add_at_a_shared_receiver() {
    let mut grid = VoxelGrid::new(7, 1, 1);
    grid.set(0, 0, 0, VOXEL_LIGHT);
    grid.set(3, 0, 0, VOXEL_WALL);
    grid.set(6, 0, 0, VOXEL_RED_LIGHT);

    bake_voxel_lighting(&mut grid, settings(0.5, 5.0));

    // Each source is one world unit from the receiver. Warm contributes
    // [12, 11.2, 8.8], red contributes [12, 2.4, 1.6], then the established
    // packed contract rounds and clamps the sum.
    assert_eq!(grid.get_light_rgb(3, 0, 0), [15, 14, 10]);
}

#[test]
fn invalid_physical_settings_are_rejected() {
    assert!(VoxelLightingSettings::new(0.0, 4.0).is_err());
    assert!(VoxelLightingSettings::new(f32::NAN, 4.0).is_err());
    assert!(VoxelLightingSettings::new(0.2, f32::INFINITY).is_err());
}
