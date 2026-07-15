use vackrooms::domain::entities::voxel_grid::{FACE_OCCLUDED_NEGATIVE_X, FACE_OCCLUDED_POSITIVE_Y};
use vackrooms::domain::entities::voxel_grid::{VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_WALL};

use super::{SurfaceExposure, SurfaceFace, exposure_for_octant, select_surface_normal};

#[test]
fn virtual_subdivision_inherits_only_the_real_parent_boundary() {
    let child = exposure_for_octant(SurfaceExposure::ALL_FACES, 0xff, 0);

    assert!(child.contains(SurfaceFace::NegativeX));
    assert!(child.contains(SurfaceFace::NegativeY));
    assert!(child.contains(SurfaceFace::NegativeZ));
    assert!(!child.contains(SurfaceFace::PositiveX));
    assert!(!child.contains(SurfaceFace::PositiveY));
    assert!(!child.contains(SurfaceFace::PositiveZ));
}

#[test]
fn a_missing_real_sibling_exposes_the_shared_octant_face() {
    let only_child_zero = 1;
    let child = exposure_for_octant(SurfaceExposure::NONE, only_child_zero, 0);

    assert!(child.contains(SurfaceFace::PositiveX));
    assert!(child.contains(SurfaceFace::PositiveY));
    assert!(child.contains(SurfaceFace::PositiveZ));
    assert!(!child.contains(SurfaceFace::NegativeX));
}

#[test]
fn parent_area_weights_survive_on_the_matching_child_boundary() {
    let mut parent = SurfaceExposure::NONE;
    parent.set_area(SurfaceFace::PositiveY, 96);
    let positive_y_child = exposure_for_octant(parent, 0xff, 2);
    let negative_y_child = exposure_for_octant(parent, 0xff, 0);

    assert_eq!(positive_y_child.area(SurfaceFace::PositiveY), 96);
    assert_eq!(negative_y_child.area(SurfaceFace::PositiveY), 0);
}

#[test]
fn floor_in_front_of_camera_keeps_its_exposed_up_normal() {
    let exposure = SurfaceExposure::from_area([255, 0, 0, 255, 0, 255]);
    let normal = select_surface_normal(VOXEL_FLOOR as u32, false, exposure, [0.0, -0.1, 5.0]);

    assert_eq!(normal, Some([0.0, 1.0, 0.0]));
}

#[test]
fn wall_can_only_select_a_face_the_geometry_exposes() {
    let exposure = SurfaceExposure::from_area([0, 255, 0, 0, 0, 0]);
    let normal = select_surface_normal(VOXEL_WALL as u32, false, exposure, [-9.0, 0.0, 1.0]);

    assert_eq!(normal, Some([1.0, 0.0, 0.0]));
}

#[test]
fn emissive_material_preserves_its_exposed_underside() {
    let exposure = SurfaceExposure::from_area([255, 255, 0, 0, 255, 255]);
    let buried = select_surface_normal(VOXEL_LIGHT as u32, true, exposure, [10.0, 3.0, 1.0]);
    assert_ne!(buried, Some([0.0, -1.0, 0.0]));

    let with_underside = SurfaceExposure::from_area([255, 255, 255, 0, 255, 255]);
    assert_eq!(
        select_surface_normal(VOXEL_LIGHT as u32, true, with_underside, [10.0, 3.0, 1.0],),
        Some([0.0, -1.0, 0.0])
    );
}

#[test]
fn child_area_converts_to_parent_face_units_without_disappearing() {
    let child = SurfaceExposure::from_area([1, 2, 3, 4, 254, 255]);
    assert_eq!(
        child.scaled_to_parent(),
        SurfaceExposure::from_area([0, 1, 1, 1, 64, 64])
    );
}

#[test]
fn atlas_neighbor_bits_decode_to_the_opposite_exposed_faces() {
    let exposure = SurfaceExposure::from_neighbor_occlusion(
        FACE_OCCLUDED_NEGATIVE_X | FACE_OCCLUDED_POSITIVE_Y,
    );

    assert!(!exposure.contains(SurfaceFace::NegativeX));
    assert!(exposure.contains(SurfaceFace::PositiveX));
    assert!(exposure.contains(SurfaceFace::NegativeY));
    assert!(!exposure.contains(SurfaceFace::PositiveY));
    assert!(exposure.contains(SurfaceFace::NegativeZ));
    assert!(exposure.contains(SurfaceFace::PositiveZ));
}
