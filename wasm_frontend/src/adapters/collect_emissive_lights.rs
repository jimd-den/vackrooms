//! Derive static direct lights from the emissive surfaces that are actually
//! present in a generated voxel grid.
//!
//! Architectural fixture metadata is useful while planning a scene, but it
//! is not authoritative after voxelization: anomalies, walls, and LOD
//! quantization can remove or reshape a fixture. This adapter therefore reads
//! the same halo grid that is meshed. A direct light exists exactly when a
//! downward-facing emissive surface exists.

use vackrooms::domain::entities::voxel_grid::{
    VOXEL_AIR, VOXEL_GLIMMER, VOXEL_LIGHT, VOXEL_RED_LIGHT, VoxelGrid, material_color_f32,
};
use vackrooms::use_cases::bake_voxel_lighting::DEFAULT_MAX_LIGHT_RANGE_WORLD_UNITS;

use crate::application::ports::{LightKind, LightSource, POSITION_FIXED_SCALE};

/// The direct-light spectrum comes from the authoritative visible-material
/// palette, decoded to linear RGB. The low-precision voxel bake has its own
/// deliberately coarser 0--15 spectrum and must not leak into this contract.
#[derive(Clone, Copy)]
struct EmissionProfile {
    linear_rgb: [f32; 3],
    /// Linear emitted-radiance scale used by the analytic area-light model.
    /// This deliberately matches `emittedRadiance`/`emittedVoxelRadiance`:
    /// normalizing a 0--15 bake sample to intensity 1 made a physical area
    /// integral roughly an order of magnitude too dark.
    radiance: f32,
    kind: LightKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ComponentKey {
    material: u8,
    min_world_fixed: [i64; 3],
    max_world_fixed: [i64; 3],
}

#[derive(Clone, Copy, Debug)]
struct ComponentBounds {
    material: u8,
    y: usize,
    min_x: usize,
    max_x_exclusive: usize,
    min_z: usize,
    max_z_exclusive: usize,
}

/// Collects non-overlapping world-space rectangles that exactly cover every
/// exposed emissive underside owned by the logical chunk interior.
///
/// Halo cells are deliberately excluded from ownership. Neighboring chunks
/// may both *inspect* those cells for meshing, but only the chunk containing a
/// cell emits its light rectangle. This prevents a boundary panel from being
/// counted twice. A deterministic greedy rectangle cover preserves L-shaped
/// or damaged fixtures without inventing emission over their empty holes.
pub fn collect_emissive_lights(
    halo_grid: &VoxelGrid,
    voxel_size: f32,
    halo_world_origin: [f32; 3],
    lateral_padding: usize,
) -> Vec<LightSource> {
    if !valid_transform(voxel_size, halo_world_origin)
        || halo_grid.width() <= lateral_padding * 2
        || halo_grid.depth() <= lateral_padding * 2
    {
        return Vec::new();
    }

    let mut visited = vec![false; halo_grid.width() * halo_grid.height() * halo_grid.depth()];
    let mut keyed_lights = Vec::new();

    // Explicit loop order plus a fixed neighbour order keeps component
    // discovery deterministic without depending on a hash collection.
    for y in 1..halo_grid.height() {
        for z in lateral_padding..halo_grid.depth() - lateral_padding {
            for x in lateral_padding..halo_grid.width() - lateral_padding {
                let index = grid_index(halo_grid, x, y, z);
                if visited[index] || !is_exposed_underside(halo_grid, x, y, z) {
                    continue;
                }

                let component =
                    collect_owned_rectangle(halo_grid, x, y, z, lateral_padding, &mut visited);
                let key = component_key(component, voxel_size, halo_world_origin);
                keyed_lights.push((
                    key,
                    light_from_component(component, key, voxel_size, halo_world_origin),
                ));
            }
        }
    }

    keyed_lights.sort_unstable_by_key(|(key, _)| *key);
    keyed_lights.dedup_by(|(left, _), (right, _)| left == right);
    keyed_lights.into_iter().map(|(_, light)| light).collect()
}

fn valid_transform(voxel_size: f32, origin: [f32; 3]) -> bool {
    voxel_size.is_finite() && voxel_size > 0.0 && origin.into_iter().all(f32::is_finite)
}

fn is_exposed_underside(grid: &VoxelGrid, x: usize, y: usize, z: usize) -> bool {
    emission_profile(grid.get(x, y, z)).is_some() && grid.get(x, y - 1, z) == VOXEL_AIR
}

/// Finds the widest first row, then extends only while every cell in that row
/// remains the same exposed material. Marking exactly that rectangle makes
/// subsequent passes cover any remainder without overlaps or phantom area.
fn collect_owned_rectangle(
    grid: &VoxelGrid,
    start_x: usize,
    y: usize,
    start_z: usize,
    lateral_padding: usize,
    visited: &mut [bool],
) -> ComponentBounds {
    let material = grid.get(start_x, y, start_z);
    let owned_x_end = grid.width() - lateral_padding;
    let owned_z_end = grid.depth() - lateral_padding;

    let mut max_x_exclusive = start_x;
    while max_x_exclusive < owned_x_end
        && rectangle_cell_matches(grid, max_x_exclusive, y, start_z, material, visited)
    {
        max_x_exclusive += 1;
    }

    let mut max_z_exclusive = start_z + 1;
    while max_z_exclusive < owned_z_end
        && (start_x..max_x_exclusive)
            .all(|x| rectangle_cell_matches(grid, x, y, max_z_exclusive, material, visited))
    {
        max_z_exclusive += 1;
    }

    for z in start_z..max_z_exclusive {
        for x in start_x..max_x_exclusive {
            visited[grid_index(grid, x, y, z)] = true;
        }
    }

    ComponentBounds {
        material,
        y,
        min_x: start_x,
        max_x_exclusive,
        min_z: start_z,
        max_z_exclusive,
    }
}

fn rectangle_cell_matches(
    grid: &VoxelGrid,
    x: usize,
    y: usize,
    z: usize,
    material: u8,
    visited: &[bool],
) -> bool {
    !visited[grid_index(grid, x, y, z)]
        && grid.get(x, y, z) == material
        && is_exposed_underside(grid, x, y, z)
}

fn grid_index(grid: &VoxelGrid, x: usize, y: usize, z: usize) -> usize {
    y * grid.width() * grid.depth() + z * grid.width() + x
}

fn component_key(component: ComponentBounds, voxel_size: f32, origin: [f32; 3]) -> ComponentKey {
    let min_world = [
        origin[0] + component.min_x as f32 * voxel_size,
        origin[1] + component.y as f32 * voxel_size,
        origin[2] + component.min_z as f32 * voxel_size,
    ];
    let max_world = [
        origin[0] + component.max_x_exclusive as f32 * voxel_size,
        min_world[1],
        origin[2] + component.max_z_exclusive as f32 * voxel_size,
    ];
    ComponentKey {
        material: component.material,
        min_world_fixed: min_world.map(world_to_fixed),
        max_world_fixed: max_world.map(world_to_fixed),
    }
}

fn world_to_fixed(value: f32) -> i64 {
    (value * POSITION_FIXED_SCALE).round() as i64
}

fn light_from_component(
    component: ComponentBounds,
    key: ComponentKey,
    voxel_size: f32,
    origin: [f32; 3],
) -> LightSource {
    let width = (component.max_x_exclusive - component.min_x) as f32 * voxel_size;
    let depth = (component.max_z_exclusive - component.min_z) as f32 * voxel_size;
    let profile = emission_profile(component.material)
        .expect("components are seeded only from emissive materials");
    let kind = if profile.kind == LightKind::Emergency {
        LightKind::Emergency
    } else if width > depth * 2.0 || depth > width * 2.0 {
        LightKind::Strip
    } else {
        LightKind::CeilingPanel
    };

    LightSource {
        id: stable_id(key),
        position: [
            origin[0] + (component.min_x + component.max_x_exclusive) as f32 * voxel_size * 0.5,
            origin[1] + component.y as f32 * voxel_size,
            origin[2] + (component.min_z + component.max_z_exclusive) as f32 * voxel_size * 0.5,
        ],
        half_size: [width * 0.5, depth * 0.5],
        color: profile.linear_rgb,
        radius: DEFAULT_MAX_LIGHT_RANGE_WORLD_UNITS,
        intensity: profile.radiance,
        kind,
        flicker_mode: 0,
        enabled: true,
    }
}

fn emission_profile(material: u8) -> Option<EmissionProfile> {
    let (radiance, kind) = match material {
        VOXEL_LIGHT => (10.0, LightKind::CeilingPanel),
        VOXEL_RED_LIGHT => (8.0, LightKind::CeilingPanel),
        VOXEL_GLIMMER => (0.9, LightKind::Emergency),
        _ => return None,
    };
    Some(EmissionProfile {
        linear_rgb: material_color_f32(material).map(srgb_channel_to_linear),
        radiance,
        kind,
    })
}

fn srgb_channel_to_linear(channel: f32) -> f32 {
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

/// FNV-1a over the full fixed-point world-space identity. Hash collisions are
/// not used for local deduplication (the complete key is), while the compact
/// `u64` remains convenient for scene-wide halo deduplication.
fn stable_id(key: ComponentKey) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for byte in std::iter::once(key.material).chain(
        key.min_world_fixed
            .into_iter()
            .chain(key.max_world_fixed)
            .flat_map(i64::to_le_bytes),
    ) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use vackrooms::domain::entities::voxel_grid::VOXEL_CEILING;

    fn assert_near(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }

    #[test]
    fn one_panel_becomes_one_world_space_area_light() {
        let mut grid = VoxelGrid::new(6, 4, 5);
        for z in 1..=2 {
            for x in 2..=3 {
                grid.set(x, 3, z, VOXEL_LIGHT);
            }
        }

        let lights = collect_emissive_lights(&grid, 0.5, [10.0, 0.0, -4.0], 0);

        assert_eq!(lights.len(), 1);
        let light = lights[0];
        assert_eq!(light.position, [11.5, 1.5, -3.0]);
        assert_eq!(light.half_size, [0.5, 0.5]);
        assert_eq!(light.kind, LightKind::CeilingPanel);
        let expected_color = material_color_f32(VOXEL_LIGHT).map(srgb_channel_to_linear);
        for (actual, expected) in light.color.into_iter().zip(expected_color) {
            assert_near(actual, expected);
        }
        assert_near(light.radius, DEFAULT_MAX_LIGHT_RANGE_WORLD_UNITS);
        assert_near(light.intensity, 10.0);
    }

    #[test]
    fn disconnected_emissive_surfaces_remain_distinct_fixtures() {
        let mut grid = VoxelGrid::new(8, 3, 4);
        grid.set(1, 2, 1, VOXEL_LIGHT);
        grid.set(5, 2, 1, VOXEL_LIGHT);

        let first = collect_emissive_lights(&grid, 1.0, [0.0; 3], 0);
        let second = collect_emissive_lights(&grid, 1.0, [0.0; 3], 0);

        assert_eq!(first, second, "collection order must be deterministic");
        assert_eq!(first.len(), 2);
        assert_ne!(first[0].id, first[1].id);
    }

    #[test]
    fn component_id_is_stable_across_overlapping_halo_windows() {
        let mut left_window = VoxelGrid::new(8, 4, 5);
        let mut right_window = VoxelGrid::new(8, 4, 5);
        for z in 1..=2 {
            for left_x in 3..=4 {
                left_window.set(left_x, 3, z, VOXEL_RED_LIGHT);
                right_window.set(left_x - 2, 3, z, VOXEL_RED_LIGHT);
            }
        }

        let left = collect_emissive_lights(&left_window, 0.5, [10.0, 0.0, -2.0], 0);
        let right = collect_emissive_lights(&right_window, 0.5, [11.0, 0.0, -2.0], 0);

        assert_eq!(left.len(), 1);
        assert_eq!(right.len(), 1);
        assert_eq!(left[0].id, right[0].id);
        assert_eq!(left[0].position, right[0].position);
        assert_eq!(left[0].half_size, right[0].half_size);
    }

    #[test]
    fn neighboring_halos_own_disjoint_halves_of_a_boundary_panel() {
        // Logical interiors are [0,4) and [4,8). The two-cell panel crosses
        // that boundary; each halo can see both cells, but owns only one.
        let mut left_window = VoxelGrid::new(6, 3, 4);
        left_window.set(4, 2, 2, VOXEL_LIGHT); // world [3,4)
        left_window.set(5, 2, 2, VOXEL_LIGHT); // halo: world [4,5)

        let mut right_window = VoxelGrid::new(6, 3, 4);
        right_window.set(0, 2, 2, VOXEL_LIGHT); // halo: world [3,4)
        right_window.set(1, 2, 2, VOXEL_LIGHT); // world [4,5)

        let left = collect_emissive_lights(&left_window, 1.0, [-1.0, 0.0, -1.0], 1);
        let right = collect_emissive_lights(&right_window, 1.0, [3.0, 0.0, -1.0], 1);

        assert_eq!(left.len(), 1);
        assert_eq!(right.len(), 1);
        assert_eq!(left[0].position[0], 3.5);
        assert_eq!(right[0].position[0], 4.5);
        assert_eq!(left[0].half_size[0] + right[0].half_size[0], 1.0);
        assert_ne!(left[0].id, right[0].id);
    }

    #[test]
    fn damaged_l_shaped_fixture_does_not_emit_over_its_hole() {
        let mut grid = VoxelGrid::new(4, 3, 4);
        grid.set(1, 2, 1, VOXEL_LIGHT);
        grid.set(2, 2, 1, VOXEL_LIGHT);
        grid.set(1, 2, 2, VOXEL_LIGHT);

        let lights = collect_emissive_lights(&grid, 1.0, [0.0; 3], 0);
        let total_area = lights
            .iter()
            .map(|light| 4.0 * light.half_size[0] * light.half_size[1])
            .sum::<f32>();

        assert_eq!(lights.len(), 2, "the hole prevents one bounding rectangle");
        assert_eq!(total_area, 3.0, "emissive area must be exactly conserved");
    }

    #[test]
    fn buried_emissive_voxel_does_not_create_a_direct_light() {
        let mut grid = VoxelGrid::new(3, 3, 3);
        grid.set(1, 1, 1, VOXEL_LIGHT);
        grid.set(1, 0, 1, VOXEL_CEILING);
        grid.set(0, 1, 1, VOXEL_CEILING);
        grid.set(2, 1, 1, VOXEL_CEILING);
        grid.set(1, 1, 0, VOXEL_CEILING);
        grid.set(1, 1, 2, VOXEL_CEILING);
        grid.set(1, 2, 1, VOXEL_CEILING);

        assert!(collect_emissive_lights(&grid, 1.0, [0.0; 3], 0).is_empty());
    }

    #[test]
    fn emissive_floor_material_is_not_a_downward_fixture() {
        let mut grid = VoxelGrid::new(3, 2, 3);
        grid.set(1, 0, 1, VOXEL_LIGHT);

        assert!(collect_emissive_lights(&grid, 1.0, [0.0; 3], 0).is_empty());
    }

    #[test]
    fn elongated_and_emergency_emitters_keep_their_semantics() {
        let mut grid = VoxelGrid::new(8, 3, 5);
        for x in 1..=4 {
            grid.set(x, 2, 1, VOXEL_LIGHT);
        }
        grid.set(6, 2, 3, VOXEL_GLIMMER);

        let lights = collect_emissive_lights(&grid, 1.0, [0.0; 3], 0);

        assert_eq!(lights.len(), 2);
        assert!(lights.iter().any(|light| light.kind == LightKind::Strip));
        let emergency = lights
            .iter()
            .find(|light| light.kind == LightKind::Emergency)
            .unwrap();
        let expected_color = material_color_f32(VOXEL_GLIMMER).map(srgb_channel_to_linear);
        for (actual, expected) in emergency.color.into_iter().zip(expected_color) {
            assert_near(actual, expected);
        }
        assert_near(emergency.intensity, 0.9);
    }
}
