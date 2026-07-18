//! Level 1 generator tests: determinism, seam tiling, walkability,
//! sector variety, and the supply/exit exports.

use super::generate::{ARRIVAL_POINT, stamp_supply_marker};
use super::*;
use crate::domain::entities::anomaly::RealitySnapshot;
use crate::domain::entities::supplies::SupplyKind;
use crate::domain::entities::voxel_grid::{
    VOXEL_AIR, VOXEL_ALMOND_WATER, VOXEL_CONCRETE_WALL, VOXEL_METAL_DOOR, VOXEL_TILE_FLOOR,
    VoxelGrid,
};
use crate::domain::entities::supplies::SupplyItem;
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::level_generator::{LEVEL_BACKROOMS, LevelGenerator};
use crate::use_cases::ports::NoiseProvider;

struct FlatNoise;
impl NoiseProvider for FlatNoise {
    fn evaluate_2d(&self, _seed: u32, _pos: Position) -> f32 {
        0.0
    }
}

fn config() -> GeneratorConfig {
    GeneratorConfig::low_spec().with_level(1)
}

fn generate(chunk_x: f32, chunk_z: f32) -> VoxelGrid {
    HabitableLevel.generate(Position::new(chunk_x, chunk_z), 42, config(), &FlatNoise)
}

/// Column stride used by walkability scans. Coarser than the 0.4u plan
/// lattice would skip whole wall bands; see the fabric test-stride note.
const SCAN_STEP: f32 = 0.4;

#[test]
fn generation_is_deterministic() {
    let a = generate(96.0, -64.0);
    let b = generate(96.0, -64.0);
    for z in 0..a.depth() {
        for x in 0..a.width() {
            for y in 0..a.height() {
                assert_eq!(a.get(x, y, z), b.get(x, y, z));
            }
        }
    }
    assert_eq!(a.supply_items, b.supply_items);
}

#[test]
fn every_sector_keeps_the_ground_plane_walkable() {
    // Flood-fill reachability over a 96u sector-sized window: from the
    // arrival plaza, most open floor must be mutually reachable. Openings
    // are >= 1 unit by construction (doorways 1.4u, lanes, bays).
    let seed = 42;
    let window = 96.0f32;
    let cells = (window / SCAN_STEP) as usize;
    let open = |ix: usize, iz: usize| -> bool {
        let wx = ix as f32 * SCAN_STEP - window * 0.5;
        let wz = iz as f32 * SCAN_STEP - window * 0.5;
        let c = HabitableLevel::plan_column(seed, wx, wz);
        !c.solid && c.floor_units < 0.5
    };
    let mut visited = vec![false; cells * cells];
    let start = (cells / 2, cells / 2);
    assert!(open(start.0, start.1), "arrival plaza must be open");
    let mut stack = vec![start];
    visited[start.1 * cells + start.0] = true;
    let mut reached = 0usize;
    while let Some((x, z)) = stack.pop() {
        reached += 1;
        let neighbors = [
            (x.wrapping_sub(1), z),
            (x + 1, z),
            (x, z.wrapping_sub(1)),
            (x, z + 1),
        ];
        for (nx, nz) in neighbors {
            if nx < cells && nz < cells && !visited[nz * cells + nx] && open(nx, nz) {
                visited[nz * cells + nx] = true;
                stack.push((nx, nz));
            }
        }
    }
    let total_open = (0..cells * cells)
        .filter(|i| open(i % cells, i / cells))
        .count();
    assert!(
        reached * 100 >= total_open * 85,
        "{reached} of {total_open} open columns reachable from the plaza"
    );
}

#[test]
fn sectors_are_deterministic_and_all_four_exist_nearby() {
    let mut seen = std::collections::HashSet::new();
    for cz in -6..=6 {
        for cx in -6..=6 {
            let wx = cx as f32 * SECTOR_CELL + 10.0;
            let wz = cz as f32 * SECTOR_CELL + 10.0;
            assert_eq!(Sector::at(42, wx, wz), Sector::at(42, wx, wz));
            seen.insert(Sector::at(42, wx, wz));
        }
    }
    assert_eq!(seen.len(), 4, "all four sectors within 13x13 cells");
    assert_eq!(Sector::at(42, 1.0, 1.0), Sector::Aquila, "arrival cell");
}

#[test]
fn arrival_chunk_exports_the_return_door_and_marker_geometry() {
    let grid = generate(0.0, -10.0);
    assert_eq!(grid.level_exits.len(), 1);
    let exit = grid.level_exits[0];
    assert_eq!(exit.target_level, LEVEL_BACKROOMS);
    assert!(exit.contains(2.0, -2.0));
    assert!(
        !exit.contains(ARRIVAL_POINT.0, ARRIVAL_POINT.1),
        "arriving from Level 0 must not immediately re-trigger the door"
    );

    let mut door_voxels = 0;
    let mut wall_voxels = 0;
    for z in 0..grid.depth() {
        for x in 0..grid.width() {
            for y in 0..grid.height() {
                match grid.get(x, y, z) {
                    VOXEL_METAL_DOOR => door_voxels += 1,
                    VOXEL_CONCRETE_WALL => wall_voxels += 1,
                    _ => {}
                }
            }
        }
    }
    assert!(door_voxels > 0, "door panel must be stamped into the grid");
    assert!(wall_voxels > 0, "door frame and sector walls exist");
}

#[test]
fn supplies_export_and_respect_consumed_reality() {
    // Scan a few chunks; Gild density guarantees finds quickly.
    let mut found: Option<(f32, f32, SupplyItem)> = None;
    'outer: for cz in 0..12 {
        for cx in 0..12 {
            let (ox, oz) = (cx as f32 * 10.0 + 40.0, cz as f32 * 10.0 + 40.0);
            let grid = generate(ox, oz);
            if let Some(item) = grid.supply_items.first().copied() {
                found = Some((ox, oz, item));
                break 'outer;
            }
        }
    }
    let (ox, oz, item) = found.expect("supplies must exist near the origin");
    assert!(matches!(
        item.kind,
        SupplyKind::AlmondWater | SupplyKind::Ration
    ));

    // Consuming the item removes both the export and its voxel marker.
    // Count the marker's body material (bottles and tins differ).
    let body_material = match item.kind {
        SupplyKind::AlmondWater => VOXEL_ALMOND_WATER,
        SupplyKind::Ration => VOXEL_METAL_DOOR,
    };
    let consumed_reality = RealitySnapshot::empty().with_supply_consumed(item.id);
    let after = HabitableLevel.generate_with_reality(
        Position::new(ox, oz),
        42,
        config(),
        &FlatNoise,
        &consumed_reality,
    );
    assert!(after.supply_items.iter().all(|s| s.id != item.id));
    let count_markers = |grid: &VoxelGrid| {
        let mut markers = 0;
        for z in 0..grid.depth() {
            for x in 0..grid.width() {
                for y in 0..grid.height() {
                    if grid.get(x, y, z) == body_material {
                        markers += 1;
                    }
                }
            }
        }
        markers
    };
    let markers_before = count_markers(&generate(ox, oz));
    let markers_after = count_markers(&after);
    assert!(markers_before > 0, "unconsumed marker must be visible");
    assert!(
        markers_after < markers_before,
        "consumed marker must vanish"
    );
}

#[test]
fn marker_stamp_is_small_shaped_and_sits_on_its_rest_height() {
    let mut grid = VoxelGrid::new(50, 29, 50);
    let item = SupplyItem {
        id: 9,
        kind: SupplyKind::AlmondWater,
        position: Position::new(5.0, 5.0),
        rest_y: 0.8,
    };
    stamp_supply_marker(&mut grid, Position::new(0.0, 0.0), 0.2, &item);
    let base = (0.8 / 0.2) as usize + 1;
    assert_eq!(grid.get(25, base, 25), VOXEL_ALMOND_WATER, "milky body");
    assert_eq!(
        grid.get(25, base + 1, 25),
        crate::domain::entities::voxel_grid::VOXEL_PIPE,
        "dark screw cap tops the bottle"
    );
    assert_eq!(grid.get(25, base + 3, 25), VOXEL_AIR, "bottle stays small");

    // Rations read as a squat tin, not a bottle.
    let tin = SupplyItem {
        id: 10,
        kind: SupplyKind::Ration,
        position: Position::new(3.0, 3.0),
        rest_y: 0.0,
    };
    stamp_supply_marker(&mut grid, Position::new(0.0, 0.0), 0.2, &tin);
    assert_eq!(
        grid.get(15, 1, 15),
        VOXEL_METAL_DOOR,
        "tin body uses the blue-grey metal"
    );
    assert_eq!(grid.get(15, 3, 15), VOXEL_AIR, "tin stays squat");
}

#[test]
fn aquila_reads_as_a_parking_lot() {
    // Inside the arrival sector: concrete pillars on the 9.6 bay, and the
    // floor of the plaza is tile.
    let plaza = HabitableLevel::plan_column(42, 1.0, 1.0);
    assert_eq!(plaza.floor_material, VOXEL_TILE_FLOOR);
    assert!(!plaza.solid);

    let pillar = HabitableLevel::plan_column(42, 9.6, 9.6);
    assert!(pillar.solid, "bay-node pillar");
    assert_eq!(pillar.wall_material, VOXEL_CONCRETE_WALL);
}
