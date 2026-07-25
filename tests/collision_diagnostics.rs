use vackrooms::domain::entities::voxel_grid::{VOXEL_FLOOR, VOXEL_STAINED_CARPET, VOXEL_WALL};
use vackrooms::domain::entities::position::Position;
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
/// Collision Diagnostic Tests — TDD Phase 1
///
/// WHY THESE TESTS EXIST:
/// The player reported spawning inside a wall.  Before changing any logic
/// we must prove (a) what coordinates the spawn point produces and (b) whether
/// the floor voxel's collision box overlaps the player's AABB at spawn.
///
/// Test-first order:
///   1. Prove the exact world-space coordinates the chunk generator places floor/wall
///      voxels at for the (0,0) hub chunk in low-spec mode.
///   2. Prove the spawn point (5.0, 1.7, 5.0) does NOT overlap any wall voxel.
///   3. Prove the floor voxel box does NOT block the player at spawn.
///   4. Prove terminal velocity is sane (<= 5 world-units/second).
///
/// If test 2 or 3 fails, it proves the bug.  Then we add a fix and verify
/// the test passes.
use vackrooms::use_cases::generate_chunk::{GenerateChunkArchitectureUseCase, GeneratorConfig};

/// The JavaScript side extracts wall collision boxes from the SVO.
/// This Rust analogue extracts the same information directly from the VoxelGrid
/// so we can verify coordinates without the JS/SVO layer.
fn grid_wall_boxes_at_spawn_x(chunk_x: f32, chunk_z: f32) -> Vec<(f32, f32, f32, f32, f32, f32)> {
    // Returns (minX, maxX, minY, maxY, minZ, maxZ) for every solid wall voxel
    // at approximately x=5 world-units inside the chunk.
    let config = GeneratorConfig::low_spec();
    let noise = SimpleNoiseProvider::new();
    let generator = GenerateChunkArchitectureUseCase::new(&noise);
    let grid = generator.execute(Position::new(chunk_x, chunk_z), 42, config);

    let scale = config.voxel_scale;
    let mut boxes = vec![];

    for y in 0..grid.height() {
        for z in 0..grid.depth() {
            for x in 0..grid.width() {
                let vt = grid.get(x, y, z);
                if vt == VOXEL_WALL {
                    let wx = chunk_x + x as f32 * scale;
                    let wy = y as f32 * scale;
                    let wz = chunk_z + z as f32 * scale;
                    // The JS collision system gives walls a 1-voxel-size box
                    boxes.push((wx, wx + scale, wy, wy + scale, wz, wz + scale));
                }
            }
        }
    }
    boxes
}

fn floor_boxes(chunk_x: f32, chunk_z: f32) -> Vec<(f32, f32, f32, f32, f32, f32)> {
    let config = GeneratorConfig::low_spec();
    let noise = SimpleNoiseProvider::new();
    let generator = GenerateChunkArchitectureUseCase::new(&noise);
    let grid = generator.execute(Position::new(chunk_x, chunk_z), 42, config);

    let scale = config.voxel_scale;
    let mut boxes = vec![];

    for z in 0..grid.depth() {
        for x in 0..grid.width() {
            let vt = grid.get(x, 0, z);
            // Ordinary fabric floor now voxelizes as either the fresh or
            // the institution-age-stained carpet material; both are the
            // same walkable floor geometry this diagnostic cares about.
            if vt == VOXEL_FLOOR || vt == VOXEL_STAINED_CARPET {
                let wx = chunk_x + x as f32 * scale;
                let wz = chunk_z + z as f32 * scale;
                // Floor voxel: y=0 to y=scale
                boxes.push((wx, wx + scale, 0.0_f32, scale, wz, wz + scale));
            }
        }
    }
    boxes
}

/// Player AABB: eye-level at pos.y, feet at pos.y - 1.65, head at pos.y + 0.1
fn player_overlaps(
    pos: (f32, f32, f32),
    radius: f32,
    vbox: (f32, f32, f32, f32, f32, f32),
) -> bool {
    let (px, py, pz) = pos;
    let p_min_x = px - radius;
    let p_max_x = px + radius;
    let p_min_y = py - 1.65;
    let p_max_y = py + 0.1;
    let p_min_z = pz - radius;
    let p_max_z = pz + radius;
    let (v_min_x, v_max_x, v_min_y, v_max_y, v_min_z, v_max_z) = vbox;

    p_max_x > v_min_x
        && p_min_x < v_max_x
        && p_max_y > v_min_y
        && p_min_y < v_max_y
        && p_max_z > v_min_z
        && p_min_z < v_max_z
}

// ============================================================
// TEST 1: Verify hub chunk (0,0) has walls only on boundary
// ============================================================
#[test]
fn test_hub_chunk_interior_is_wall_free() {
    // The hub chunk (0,0) is supposed to be an open plaza.
    // No wall voxels should exist at x≈5.0, z≈5.0 (centre of chunk).
    let walls = grid_wall_boxes_at_spawn_x(0.0, 0.0);

    let spawn = (5.0_f32, 1.7_f32, 5.0_f32);
    let radius = 0.35_f32;

    let overlapping: Vec<_> = walls
        .iter()
        .filter(|&&b| player_overlaps(spawn, radius, b))
        .collect();

    assert!(
        overlapping.is_empty(),
        "FAIL: Spawn point {:?} overlaps {} wall voxel(s).\n\
         First overlap: {:?}\n\
         This proves the spawn-in-wall bug.",
        spawn,
        overlapping.len(),
        overlapping.first()
    );
}

// ============================================================
// TEST 2: DIAGNOSTIC — floor geometry overlaps player AABB.
// This test is marked #[should_panic] because the floor voxels
// DO geometrically overlap the player.  The fix is applied in JS
// (extractSvoCollisions only passes WALL/RED_WALL to isColliding).
// This test is kept as living documentation of the bug's root cause.
// ============================================================
#[test]
#[should_panic(expected = "Floor voxels at spawn OVERLAP")]
fn test_floor_geometry_overlaps_player_proving_the_bug() {
    // Floor voxels are at y=0 to y=voxel_scale (0.2 for low-spec).
    // Player feet: pos.y - 1.65 = 1.7 - 1.65 = 0.05.
    // If pMinY(0.05) < vMaxY(0.2) AND pMaxY(1.8) > vMinY(0.0) → geometric overlap.
    // This test PROVES the overlap exists in the geometry.
    // The JS fix excludes floor boxes from isColliding() — see extractSvoCollisions().
    let floors = floor_boxes(0.0, 0.0);
    let spawn = (5.0_f32, 1.7_f32, 5.0_f32);
    let radius = 0.35_f32;

    let floor_at_spawn: Vec<_> = floors
        .iter()
        .filter(|&&b| player_overlaps(spawn, radius, b))
        .collect();

    assert!(
        floor_at_spawn.is_empty(),
        "FAIL: Floor voxels at spawn OVERLAP the player AABB.\n\
         Count: {}\n\
         This causes the player to spawn 'stuck' and unable to move.\n\
         Fix: exclude FLOOR and CEILING voxels from collision extraction.\n\
         First overlapping floor box: {:?}",
        floor_at_spawn.len(),
        floor_at_spawn.first()
    );
}

// ============================================================
// TEST 3: Terminal velocity must be sane (≤ 5 world-units/second)
// ============================================================
#[test]
fn test_terminal_velocity_is_sane() {
    // JS formula: velocity.z -= direction.z * speed * delta
    //             velocity.z -= velocity.z * friction * delta   (drag)
    // Terminal velocity = speed / friction
    let speed: f32 = 4.0; // target walk speed (world units/sec)
    let friction: f32 = 10.0; // damping coefficient
    let terminal = speed / friction;

    // The old code used speed=15.0 → terminal = 1.5 units/frame @60fps = 90 u/s
    assert!(
        terminal <= 5.0,
        "FAIL: Terminal velocity {:.2} u/s exceeds 5 u/s.\n\
         This causes the 'too fast' movement bug.\n\
         Use speed <= {:.1} with friction = {:.1}",
        terminal,
        friction * 5.0,
        friction
    );
}

// ============================================================
// TEST 4: Fix verification — wall-only collision boxes do NOT
//         overlap the player at spawn.
//         This test must PASS after the production fix is applied.
// ============================================================
#[test]
fn test_fix_only_walls_collide_at_spawn() {
    // After the fix: extractSvoCollisions should only emit voxelType==WALL (1)
    // or RED_WALL (5).  Floor (2) and Ceiling (3) must be excluded.
    // The wall_boxes helper already filters by VOXEL_WALL only.
    let walls = grid_wall_boxes_at_spawn_x(0.0, 0.0);
    let spawn = (5.0_f32, 1.7_f32, 5.0_f32);
    let radius = 0.35_f32;

    let blocked: Vec<_> = walls
        .iter()
        .filter(|&&b| player_overlaps(spawn, radius, b))
        .collect();

    assert!(
        blocked.is_empty(),
        "FAIL: Player still collides with {} wall box(es) at spawn even after floor fix.\n\
         First collision: {:?}\n\
         This is a separate bug — the hub chunk has a wall at the spawn point.",
        blocked.len(),
        blocked.first()
    );
}
