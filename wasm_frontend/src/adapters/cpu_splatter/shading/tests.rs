//! Native contracts for the complete CPU shading sentence.

use crate::application::ports::{Environment, FrameParams, LightKind, LightSource};
use crate::application::rendering::{
    compose_fog, downward_rectangular_emitter_cosine, encode_display_color, fog_transmittance,
    lambertian_receiver_cosine,
};
use vackrooms::domain::entities::voxel_grid::{VOXEL_FLOOR, VOXEL_LIGHT, VOXEL_WALL};

use super::super::camera::Camera;
use super::super::settings::CpuRenderSettings;
use super::super::surface_geometry::SurfaceExposure;
use super::compose_environment_irradiance::ambient_irradiance;
use super::evaluate_emission::emitted_radiance;
use super::evaluate_fixture_irradiance::{
    fixture_irradiance, rectangle_light_irradiance, sample_geometry,
};
use super::present_radiance::present_radiance;
use super::surface_reconstruction::{
    camera_facing_aabb_normal, decode_albedo, representative_normal,
    representative_surface_position, srgb_channel_to_linear,
};
use super::{FrameLighting, SplatSurface, shade_with_frame_lighting};

fn camera() -> Camera {
    Camera::new(
        &FrameParams {
            camera_pos: [0.0, 1.0, 0.0],
            yaw: std::f32::consts::PI,
            ..FrameParams::default()
        },
        64,
        64,
        &CpuRenderSettings::default(),
    )
}

fn panel(id: u64, position: [f32; 3]) -> LightSource {
    LightSource {
        id,
        position,
        half_size: [0.5, 0.5],
        color: [1.0, 0.9, 0.7],
        radius: 20.0,
        intensity: 10.0,
        kind: LightKind::CeilingPanel,
        flicker_mode: 0,
        enabled: true,
    }
}

#[test]
fn srgb_is_decoded_before_lighting() {
    assert_eq!(decode_albedo([0.0, 0.0, 0.0]), [0.0; 3]);
    assert_eq!(decode_albedo([255.0, 255.0, 255.0]), [1.0; 3]);
    let middle = srgb_channel_to_linear(128.0);
    assert!((middle - 0.215_86).abs() < 1.0e-4);
}

#[test]
fn representative_light_sample_lies_on_the_cube_boundary() {
    let center = [2.0, 3.0, 4.0];
    for normal in [[0.0, 0.0, -1.0], [0.3, -0.4, -0.866_025_4]] {
        let point = representative_surface_position(center, 2.0, normal);
        let largest_offset = (0..3)
            .map(|axis| (point[axis] - center[axis]).abs())
            .fold(0.0_f32, f32::max);
        assert!((largest_offset - 1.0).abs() < 1.0e-6, "point={point:?}");
    }
}

#[test]
fn receiver_normal_is_one_exact_camera_facing_axis() {
    assert_eq!(
        camera_facing_aabb_normal([1.0, -1.01, 0.1]),
        Some([0.0, 1.0, 0.0])
    );
    assert_eq!(
        camera_facing_aabb_normal([1.0, -0.99, 0.1]),
        Some([-1.0, 0.0, 0.0])
    );
    assert_eq!(
        camera_facing_aabb_normal([0.1, 0.2, 4.0]),
        Some([0.0, 0.0, -1.0])
    );
}

#[test]
fn emissive_panel_keeps_its_underside_when_viewed_obliquely_from_below() {
    let panel = SplatSurface {
        center: [20.0, 3.0, 4.0],
        world_size: 0.1,
        dist: 20.0,
        base_color: [255.0, 248.0, 214.0],
        baked_irradiance: [0.0; 3],
        voxel_type: VOXEL_LIGHT as u32,
        is_emissive: true,
        exposure: SurfaceExposure::ALL_FACES,
        crowded_siblings: 1,
    };
    let below_oblique = [20.0, 3.0, 4.0];
    assert_eq!(
        representative_normal(&panel, below_oblique),
        Some([0.0, -1.0, 0.0])
    );
    assert!(
        emitted_radiance(
            &panel,
            decode_albedo(panel.base_color),
            representative_normal(&panel, below_oblique).unwrap(),
        )
        .is_some()
    );

    let above = [1.0, -3.0, 0.2];
    let above_normal = representative_normal(&panel, above).unwrap();
    assert!(emitted_radiance(&panel, decode_albedo(panel.base_color), above_normal).is_none());
}

#[test]
fn rectangle_emits_only_into_its_downward_hemisphere() {
    let light = panel(1, [0.0, 3.0, 0.0]);
    let floor = rectangle_light_irradiance([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], &light);
    let ceiling = rectangle_light_irradiance([0.0, 5.0, 0.0], [0.0, -1.0, 0.0], &light);
    assert!(floor[0] > 0.0);
    assert_eq!(ceiling, [0.0; 3]);
}

#[test]
fn single_normalization_cosines_match_the_shared_reference_helpers() {
    for normal in [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ] {
        for surface_to_light in [[1.0, 3.0, -2.0], [-4.0, -1.0, 0.5], [0.25, 8.0, 1.5]] {
            let (_, receiver, emitter) = sample_geometry(normal, surface_to_light).unwrap();
            let reference_receiver = lambertian_receiver_cosine(normal, surface_to_light);
            let reference_emitter =
                downward_rectangular_emitter_cosine(surface_to_light.map(|component| -component));
            assert!((receiver - reference_receiver).abs() <= 1.0e-6);
            assert!((emitter - reference_emitter).abs() <= 1.0e-6);
        }
    }
}

#[test]
fn hero_visibility_affects_only_the_named_fixture() {
    let hero = panel(10, [0.0, 3.0, 0.0]);
    let other = panel(11, [1.0, 3.0, 0.0]);
    let receiver = [0.0, 0.0, 0.0];
    let normal = [0.0, 1.0, 0.0];
    let other_only = fixture_irradiance(FrameLighting::unshadowed(&[other]), receiver, normal);
    let lights = [hero, other];
    let shadowed = fixture_irradiance(
        FrameLighting::with_hero_visibility(&lights, hero.id, 0.0),
        receiver,
        normal,
    );
    for channel in 0..3 {
        assert!((shadowed[channel] - other_only[channel]).abs() < 1.0e-6);
    }
}

#[test]
fn ambient_is_independent_of_bake_and_hero_visibility() {
    let environment = Environment::interior();
    let ambient = ambient_irradiance(&environment);
    assert!(ambient.into_iter().all(|channel| channel > 0.0));

    let surface = SplatSurface {
        center: [0.0, 0.0, 4.0],
        world_size: 1.0,
        dist: 4.0,
        base_color: [255.0; 3],
        baked_irradiance: [15.0; 3],
        voxel_type: VOXEL_WALL as u32,
        is_emissive: false,
        exposure: SurfaceExposure::ALL_FACES,
        crowded_siblings: 1,
    };
    let mut settings = CpuRenderSettings::default();
    settings.toggles.baked_lighting = false;
    let dark_bake = shade_with_frame_lighting(
        &camera(),
        &environment,
        &settings,
        &[],
        &[],
        FrameLighting::unshadowed(&[]),
        &surface,
    );
    assert!(dark_bake.into_iter().any(|channel| channel > 0));
}

#[test]
fn actual_scene_fixture_brightens_a_receiver() {
    let cam = camera();
    let environment = Environment::interior();
    let settings = CpuRenderSettings::default();
    let surface = SplatSurface {
        center: [0.0, 0.0, 4.0],
        world_size: 1.0,
        dist: 4.0,
        base_color: [221.0, 204.0, 102.0],
        baked_irradiance: [0.0; 3],
        voxel_type: VOXEL_WALL as u32,
        is_emissive: false,
        exposure: SurfaceExposure::ALL_FACES,
        crowded_siblings: 1,
    };
    // Camera-facing normal is -Z, so use a point light toward the camera.
    let light = LightSource {
        kind: LightKind::Point,
        position: [0.0, 0.0, 2.0],
        half_size: [0.0; 2],
        ..panel(2, [0.0, 0.0, 2.0])
    };
    let ambient_only = shade_with_frame_lighting(
        &cam,
        &environment,
        &settings,
        &[],
        &[],
        FrameLighting::unshadowed(&[]),
        &surface,
    );
    let lit = shade_with_frame_lighting(
        &cam,
        &environment,
        &settings,
        &[],
        &[],
        FrameLighting::unshadowed(&[light]),
        &surface,
    );
    assert!(
        lit[0] > ambient_only[0] + 20,
        "lit={lit:?}, ambient={ambient_only:?}"
    );
}

#[test]
fn floor_in_front_of_camera_receives_its_overhead_panel() {
    let cam = camera();
    let environment = Environment::interior();
    let settings = CpuRenderSettings::default();
    let surface = SplatSurface {
        center: [0.0, 0.0, 4.0],
        world_size: 1.0,
        dist: 4.0,
        base_color: [153.0, 136.0, 17.0],
        baked_irradiance: [0.0; 3],
        voxel_type: VOXEL_FLOOR as u32,
        is_emissive: false,
        // This interior floor patch has exactly one physical receiver face.
        exposure: SurfaceExposure::from_area([0, 0, 0, 255, 0, 0]),
        crowded_siblings: 1,
    };
    let camera_to_surface = [0.0, -1.0, 4.0];
    assert_eq!(
        representative_normal(&surface, camera_to_surface),
        Some([0.0, 1.0, 0.0])
    );

    let ambient_only = shade_with_frame_lighting(
        &cam,
        &environment,
        &settings,
        &[],
        &[],
        FrameLighting::unshadowed(&[]),
        &surface,
    );
    let overhead = panel(42, [0.0, 3.0, 4.0]);
    let lit = shade_with_frame_lighting(
        &cam,
        &environment,
        &settings,
        &[],
        &[],
        FrameLighting::unshadowed(&[overhead]),
        &surface,
    );

    assert!(
        lit[0] > ambient_only[0] + 20,
        "overhead panel must light the exposed floor: ambient={ambient_only:?}, lit={lit:?}"
    );
}

#[test]
fn fog_uses_euclidean_distance_and_linear_composition() {
    let environment = Environment {
        fog_start: 0.0,
        fog_density: 0.2,
        fog_color: [0.1, 0.2, 0.3],
        ..Environment::interior()
    };
    let radiance = [1.0, 0.5, 0.25];
    let distance = 7.0;
    let expected = encode_display_color(compose_fog(
        radiance,
        environment.fog_color,
        fog_transmittance(distance, environment.fog_start, environment.fog_density),
    ));
    let actual = present_radiance(radiance, distance, &environment);
    for channel in 0..3 {
        assert!((actual[channel] as f32 / 255.0 - expected[channel]).abs() <= 1.0 / 255.0);
    }
}
