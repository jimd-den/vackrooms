//! Native reference implementation of the GPU fragment shader's SVO
//! ray-marcher (`shaders.rs::raymarchSVO`), marched through a real merged
//! multi-chunk atlas and validated against ground-truth SVO queries.
//!
//! Purpose: the GPU path cannot be unit-tested, so this test ports the GLSL
//! traversal 1:1 and asserts that every reported hit corresponds to an
//! actually-solid voxel in the source octree. Phantom hits (the "random
//! floating squares" artifact) fail the test with a diagnostic dump.

use vackrooms::domain::use_cases::build_octree::BuildOctreeUseCase;
use vackrooms::entities::models::Position;
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::{GenerateChunkArchitectureUseCase, GeneratorConfig};

use wasm_frontend::adapters::local_chunk_source::LocalChunkSource;
use wasm_frontend::application::atlas::build_atlas;
use wasm_frontend::application::ports::ChunkSourcePort;
use wasm_frontend::application::streaming::LoadedChunk;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // `t`/`voxel_type` exist for the Debug dump on failure
struct Hit {
    chunk: usize,
    t: f32,
    /// Leaf box in chunk-local space.
    box_min: [f32; 3],
    box_max: [f32; 3],
    voxel_type: u32,
}

/// 1:1 Rust port of the GLSL `raymarchSVO` (same constants, same epsilons).
fn raymarch_svo(
    atlas: &[u32],
    ro: [f32; 3],
    rd: [f32; 3],
    chunk_root_idx: usize,
    t_entry: f32,
    world_size: f32,
) -> Option<(f32, [f32; 3], [f32; 3], u32)> {
    let mut t = t_entry;
    let mut p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];

    let mut stack: Vec<(usize, [f32; 3], [f32; 3])> = Vec::new();
    let mut current_node = chunk_root_idx;
    let mut cmin = [0.0f32; 3];
    let mut cmax = [world_size; 3];

    for _ in 0..160 {
        let tx = current_node * 4;
        let (is_leaf, a, b) = (atlas[tx] == 1, atlas[tx + 1], atlas[tx + 2]);

        if is_leaf {
            if a != 0 {
                return Some((t, cmin, cmax, a));
            }
            // Empty-space skip to this leaf's exit plane.
            let exit = |lo: f32, hi: f32, o: f32, d: f32| (if d > 0.0 { hi } else { lo } - o) / d;
            let tx = exit(cmin[0], cmax[0], ro[0], rd[0]);
            let ty = exit(cmin[1], cmax[1], ro[1], rd[1]);
            let tz = exit(cmin[2], cmax[2], ro[2], rd[2]);
            let t_exit = tx.min(ty).min(tz);
            t = t_exit;
            p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];
            // Snap every tied axis past its exit plane (a ray exactly
            // through a corner advances diagonally; leaving an axis ON its
            // plane stalls the march instead).
            if (t_exit - tx).abs() < 1e-4 { p[0] = (if rd[0] > 0.0 { cmax[0] } else { cmin[0] }) + (if rd[0] > 0.0 { 0.001 } else { -0.001 }); }
            if (t_exit - ty).abs() < 1e-4 { p[1] = (if rd[1] > 0.0 { cmax[1] } else { cmin[1] }) + (if rd[1] > 0.0 { 0.001 } else { -0.001 }); }
            if (t_exit - tz).abs() < 1e-4 { p[2] = (if rd[2] > 0.0 { cmax[2] } else { cmin[2] }) + (if rd[2] > 0.0 { 0.001 } else { -0.001 }); }
            // Multi-level pop, mirroring the shader: the skip may cross
            // boundaries shared by several ancestor boxes at once.
            while outside(p, cmin, cmax) {
                let Some(f) = stack.pop() else { return None };
                current_node = f.0;
                cmin = f.1;
                cmax = f.2;
            }
        } else {
            let center = [
                (cmin[0] + cmax[0]) * 0.5,
                (cmin[1] + cmax[1]) * 0.5,
                (cmin[2] + cmax[2]) * 0.5,
            ];
            let ox = (p[0] >= center[0]) as usize;
            let oy = (p[1] >= center[1]) as usize;
            let oz = (p[2] >= center[2]) as usize;
            let child_idx = (oz << 2) | (oy << 1) | ox;

            if b & (1 << child_idx) != 0 {
                if stack.len() < 8 {
                    stack.push((current_node, cmin, cmax));
                }
                if ox == 1 { cmin[0] = center[0] } else { cmax[0] = center[0] }
                if oy == 1 { cmin[1] = center[1] } else { cmax[1] = center[1] }
                if oz == 1 { cmin[2] = center[2] } else { cmax[2] = center[2] }
                current_node = a as usize + child_idx;
            } else {
                let omin = [
                    if ox == 1 { center[0] } else { cmin[0] },
                    if oy == 1 { center[1] } else { cmin[1] },
                    if oz == 1 { center[2] } else { cmin[2] },
                ];
                let omax = [
                    if ox == 1 { cmax[0] } else { center[0] },
                    if oy == 1 { cmax[1] } else { center[1] },
                    if oz == 1 { cmax[2] } else { center[2] },
                ];
                let exit =
                    |lo: f32, hi: f32, o: f32, d: f32| (if d > 0.0 { hi } else { lo } - o) / d;
                let tx = exit(omin[0], omax[0], ro[0], rd[0]);
                let ty = exit(omin[1], omax[1], ro[1], rd[1]);
                let tz = exit(omin[2], omax[2], ro[2], rd[2]);
                let t_exit = tx.min(ty).min(tz);
                t = t_exit;
                p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];
                if (t_exit - tx).abs() < 1e-4 { p[0] = (if rd[0] > 0.0 { omax[0] } else { omin[0] }) + (if rd[0] > 0.0 { 0.001 } else { -0.001 }); }
                if (t_exit - ty).abs() < 1e-4 { p[1] = (if rd[1] > 0.0 { omax[1] } else { omin[1] }) + (if rd[1] > 0.0 { 0.001 } else { -0.001 }); }
                if (t_exit - tz).abs() < 1e-4 { p[2] = (if rd[2] > 0.0 { omax[2] } else { omin[2] }) + (if rd[2] > 0.0 { 0.001 } else { -0.001 }); }
                while outside(p, cmin, cmax) {
                    let Some(f) = stack.pop() else { return None };
                    current_node = f.0;
                    cmin = f.1;
                    cmax = f.2;
                }
            }
        }
    }
    None
}

fn outside(p: [f32; 3], min: [f32; 3], max: [f32; 3]) -> bool {
    p[0] < min[0] || p[0] > max[0] || p[1] < min[1] || p[1] > max[1] || p[2] < min[2]
        || p[2] > max[2]
}

/// Mirrors the shader's `main()`: slab-test all chunks, sort by entry, march
/// nearest first, return the first hit.
fn march_scene(
    atlas: &[u32],
    draws: &[wasm_frontend::application::ports::ChunkDraw],
    ro: [f32; 3],
    rd: [f32; 3],
) -> Option<Hit> {
    let safe = |v: f32| if v.abs() < 1e-4 { 1e-4f32.copysign(v) } else { v };
    // The shader passes safe_rd (not raw rd) into raymarchSVO too.
    let rd = [safe(rd[0]), safe(rd[1]), safe(rd[2])];
    let inv = [1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2]];

    let mut hits: Vec<(usize, f32)> = Vec::new();
    for (i, d) in draws.iter().enumerate() {
        let lro = [ro[0] - d.origin[0], ro[1] - d.origin[1], ro[2] - d.origin[2]];
        let mut t_entry = f32::MIN;
        let mut t_exit = f32::MAX;
        for a in 0..3 {
            let t1 = (0.0 - lro[a]) * inv[a];
            let t2 = (d.world_size - lro[a]) * inv[a];
            t_entry = t_entry.max(t1.min(t2));
            t_exit = t_exit.min(t1.max(t2));
        }
        if t_entry < t_exit && t_exit > 0.0 {
            hits.push((i, t_entry.max(0.0)));
        }
    }
    hits.sort_by(|a, b| a.1.total_cmp(&b.1));

    for (i, t_min) in hits {
        let d = &draws[i];
        let lro = [ro[0] - d.origin[0], ro[1] - d.origin[1], ro[2] - d.origin[2]];
        if let Some((t, bmin, bmax, vt)) =
            raymarch_svo(atlas, lro, rd, d.root_index as usize, t_min, d.world_size)
        {
            return Some(Hit { chunk: i, t, box_min: bmin, box_max: bmax, voxel_type: vt });
        }
    }
    None
}

/// Every hit the reference ray-marcher reports must be a genuinely solid
/// region of the source octree — otherwise the renderer shows phantom boxes.
#[test]
fn multi_chunk_raymarch_reports_no_phantom_geometry() {
    let seed = 42;
    let config = GeneratorConfig::low_spec();
    let noise = SimpleNoiseProvider::new();
    let source = LocalChunkSource::new(SimpleNoiseProvider::new(), seed, config);

    // 3x3 resident set around spawn, same as the engine with radius 1.
    let mut chunks = Vec::new();
    for dz in -1..=1 {
        for dx in -1..=1 {
            let (ox, oz) = (dx as f32 * 10.0, dz as f32 * 10.0);
            chunks.push(LoadedChunk { origin: (ox, oz), payload: source.load(ox, oz, 0) });
        }
    }
    let build = build_atlas(chunks.iter());

    // Ground truth: rebuild each chunk's SVO exactly like LocalChunkSource.
    let svos: Vec<_> = chunks
        .iter()
        .map(|c| {
            let generator = GenerateChunkArchitectureUseCase::new(&noise);
            let grid = generator.execute(
                Position::new(c.origin.0, c.origin.1),
                seed,
                config,
            );
            BuildOctreeUseCase::new().execute(&grid, config.svo_depth(), config.svo_world_size())
        })
        .collect();

    let ro = [5.0f32, 1.7, 5.0]; // spawn eye position
    let mut phantoms = Vec::new();
    let mut hit_count = 0usize;

    // Dense direction sweep: 64 yaw steps x 17 pitch steps.
    for yi in 0..64 {
        for pi in -8..=8i32 {
            let yaw = yi as f32 / 64.0 * std::f32::consts::TAU;
            let pitch = pi as f32 / 8.0 * 1.2;
            let (sy, cy) = yaw.sin_cos();
            let (sp, cp) = pitch.sin_cos();
            let rd = [-cp * sy, sp, -cp * cy];

            let Some(hit) = march_scene(&build.texels, &build.draws, ro, rd) else { continue };
            hit_count += 1;

            // The leaf box center, in voxel coordinates of the owning chunk.
            let scale = config.voxel_scale;
            let c = [
                (hit.box_min[0] + hit.box_max[0]) * 0.5,
                (hit.box_min[1] + hit.box_max[1]) * 0.5,
                (hit.box_min[2] + hit.box_max[2]) * 0.5,
            ];
            let (vx, vy, vz) =
                ((c[0] / scale) as u32, (c[1] / scale) as u32, (c[2] / scale) as u32);
            let truth = svos[hit.chunk].get(vx, vy, vz);
            let solid = truth.map(|(vt, _, _)| vt != 0).unwrap_or(false);
            if !solid {
                phantoms.push((yaw, pitch, hit));
            }
        }
    }

    assert!(hit_count > 200, "sanity: the sweep must actually hit walls, got {hit_count}");
    assert!(
        phantoms.is_empty(),
        "{} phantom hits out of {}; first 5: {:#?}",
        phantoms.len(),
        hit_count,
        &phantoms[..phantoms.len().min(5)]
    );
}

/// The world is a closed slab (floor everywhere at y=0, ceiling at the top),
/// so every ray with meaningful up/down pitch MUST hit something. A miss is
/// a traversal crack — the "black bars in walls" artifact caused by popping
/// only one stack level when a skip crosses several octree levels at once.
#[test]
fn pitched_rays_never_escape_through_cracks() {
    let seed = 42;
    let config = GeneratorConfig::low_spec();
    let source = LocalChunkSource::new(SimpleNoiseProvider::new(), seed, config);

    let mut chunks = Vec::new();
    for dz in -1..=1 {
        for dx in -1..=1 {
            let (ox, oz) = (dx as f32 * 10.0, dz as f32 * 10.0);
            chunks.push(LoadedChunk { origin: (ox, oz), payload: source.load(ox, oz, 0) });
        }
    }
    let build = build_atlas(chunks.iter());

    let mut misses = 0usize;
    let mut total = 0usize;
    for px in 0..8 {
        for pz in 0..8 {
            let ro = [1.0137 + px as f32 * 1.11, 1.7, 0.9713 + pz as f32 * 1.19];
            for yi in 0..128 {
                for pi in [-40i32, -25, 25, 40] {
                    let yaw = (yi as f32 + 0.173) / 128.0 * std::f32::consts::TAU;
                    let pitch = pi as f32 / 60.0 * 1.2;
                    let (sy, cy) = yaw.sin_cos();
                    let (sp, cp) = pitch.sin_cos();
                    let rd = [-cp * sy, sp, -cp * cy];
                    total += 1;
                    if march_scene(&build.texels, &build.draws, ro, rd).is_none() {
                        misses += 1;
                    }
                }
            }
        }
    }
    assert_eq!(misses, 0, "{misses} of {total} pitched rays escaped through cracks");
}
