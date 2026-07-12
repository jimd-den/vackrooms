//! # wasm_frontend — browser client for the vackrooms voxel engine
//!
//! This crate is the *outer rings* of the engine's Clean Architecture when it
//! runs in a browser. The dependency rule points strictly inward:
//!
//! ```text
//! drivers (web-sys, WebGL2, DOM)        <- wasm32 only
//!    |
//! adapters (input mapping, SVO atlas, local chunk source)
//!    |
//! application (player physics, streaming policy, frame orchestration, ports)
//!    |
//! vackrooms core (entities + use cases: VoxelGrid, SVO, generation, lighting)
//! ```
//!
//! * `application` and `adapters` are platform-agnostic and unit-tested with
//!   plain `cargo test` on the host.
//! * `drivers` is the only module that talks to the browser, and it is only
//!   compiled for `wasm32`.
//! * The `#[wasm_bindgen(start)]` entry point below is the composition root:
//!   it wires concrete drivers into the application's ports.

pub mod adapters;
pub mod application;

#[cfg(target_arch = "wasm32")]
pub mod drivers;

#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

#[cfg(target_arch = "wasm32")]
pub static DOOM_CONTROLS: AtomicBool = AtomicBool::new(false);

/// Mouse look inversion (Y axis).
#[cfg(target_arch = "wasm32")]
pub static INVERT_Y: AtomicBool = AtomicBool::new(false);

/// Mouse sensitivity multiplier, stored as f32 bits. 1.0 = default.
#[cfg(target_arch = "wasm32")]
pub static MOUSE_SENSITIVITY_BITS: AtomicU32 = AtomicU32::new(0x3F80_0000); // 1.0f32

/// Manual internal-resolution override as f32 bits; 0.0 = automatic
/// (adaptive governor).
#[cfg(target_arch = "wasm32")]
pub static RENDER_SCALE_BITS: AtomicU32 = AtomicU32::new(0);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_doom_controls(enabled: bool) {
    DOOM_CONTROLS.store(enabled, Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_invert_y(enabled: bool) {
    INVERT_Y.store(enabled, Ordering::Relaxed);
}

/// Mouse/touch look sensitivity multiplier (clamped to 0.1–5.0).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_mouse_sensitivity(multiplier: f32) {
    let m = if multiplier.is_finite() {
        multiplier.clamp(0.1, 5.0)
    } else {
        1.0
    };
    MOUSE_SENSITIVITY_BITS.store(m.to_bits(), Ordering::Relaxed);
}

/// Forces the internal render resolution scale (0.25–1.0), or restores the
/// adaptive governor when `scale` is 0.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_render_scale(scale: f32) {
    let s = if scale.is_finite() && scale > 0.0 {
        scale.clamp(0.25, 1.0)
    } else {
        0.0
    };
    RENDER_SCALE_BITS.store(s.to_bits(), Ordering::Relaxed);
}

/// CPU render settings: scale override.
#[cfg(target_arch = "wasm32")]
pub static CPU_SCALE_BITS: AtomicU32 = AtomicU32::new(0x3F80_0000); // 1.0f32
/// CPU render settings: lod cutoff.
#[cfg(target_arch = "wasm32")]
pub static CPU_LOD_CUTOFF_BITS: AtomicU32 = AtomicU32::new(0x3F80_0000); // 1.0f32
/// CPU render settings: max splat half px.
#[cfg(target_arch = "wasm32")]
pub static CPU_MAX_SPLAT_HALF_BITS: AtomicU32 = AtomicU32::new(0x4140_0000); // 12.0f32
/// CPU render settings: max draw distance.
#[cfg(target_arch = "wasm32")]
pub static CPU_MAX_DRAW_DISTANCE_BITS: AtomicU32 = AtomicU32::new(0x42C0_0000); // 96.0f32
/// CPU render settings: shadow mode (0 = Off, 1 = Hero).
#[cfg(target_arch = "wasm32")]
pub static CPU_SHADOWS: AtomicU32 = AtomicU32::new(0);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_cpu_scale(scale: f32) {
    let s = if scale.is_finite() {
        scale.clamp(0.25, 1.0)
    } else {
        1.0
    };
    CPU_SCALE_BITS.store(s.to_bits(), Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_cpu_lod_cutoff(cutoff: f32) {
    let c = if cutoff.is_finite() {
        cutoff.clamp(0.25, 2.0)
    } else {
        1.0
    };
    CPU_LOD_CUTOFF_BITS.store(c.to_bits(), Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_cpu_max_splat_half(half: f32) {
    let h = if half.is_finite() {
        half.clamp(2.0, 16.0)
    } else {
        12.0
    };
    CPU_MAX_SPLAT_HALF_BITS.store(h.to_bits(), Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_cpu_max_draw_distance(dist: f32) {
    let d = if dist.is_finite() {
        dist.clamp(16.0, 256.0)
    } else {
        96.0
    };
    CPU_MAX_DRAW_DISTANCE_BITS.store(d.to_bits(), Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_cpu_shadows(mode: u32) {
    CPU_SHADOWS.store(mode, Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
pub fn get_cpu_settings() -> crate::adapters::cpu_splatter::CpuRenderSettings {
    let scale = f32::from_bits(CPU_SCALE_BITS.load(Ordering::Relaxed));
    let lod_cutoff = f32::from_bits(CPU_LOD_CUTOFF_BITS.load(Ordering::Relaxed));
    let max_splat_half = f32::from_bits(CPU_MAX_SPLAT_HALF_BITS.load(Ordering::Relaxed));
    let max_draw_dist = f32::from_bits(CPU_MAX_DRAW_DISTANCE_BITS.load(Ordering::Relaxed));
    let shadow_mode = CPU_SHADOWS.load(Ordering::Relaxed);

    crate::adapters::cpu_splatter::CpuRenderSettings {
        internal_scale: scale,
        lod_cutoff_px: lod_cutoff,
        max_splat_half_px: max_splat_half,
        max_draw_distance: max_draw_dist,
        shadows: if shadow_mode == 1 {
            crate::adapters::cpu_splatter::CpuShadowMode::Hero
        } else {
            crate::adapters::cpu_splatter::CpuShadowMode::Off
        },
        ..crate::adapters::cpu_splatter::CpuRenderSettings::default()
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn get_blueprint_svg(seed: u32, rx: i32, rz: i32, voxel_scale: f32, size_world: f32) -> String {
    let noise = vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider::new();
    let plan = vackrooms::use_cases::region_plan::generate_region_plan(
        seed,
        vackrooms::entities::models::Position::new(rx as f32 * size_world, rz as f32 * size_world),
        size_world,
        &vackrooms::use_cases::generate_chunk::GeneratorConfig::low_spec(),
        &noise,
    );
    let chunk_size = if voxel_scale <= 0.1 { 20.0 } else { 10.0 };
    let mut chunk_bounds = Vec::new();
    let steps = (size_world / chunk_size).round() as i32;
    for i in 0..steps {
        for j in 0..steps {
            let min_x = rx as f32 * size_world + i as f32 * chunk_size;
            let min_z = rz as f32 * size_world + j as f32 * chunk_size;
            let chunk_coord_x = (min_x / chunk_size).round() as i64;
            let chunk_coord_z = (min_z / chunk_size).round() as i64;
            chunk_bounds.push(vackrooms::adapters::blueprint_renderer::ChunkBounds {
                min_x,
                min_z,
                max_x: min_x + chunk_size,
                max_z: min_z + chunk_size,
                label: format!("[{}, {}]", chunk_coord_x, chunk_coord_z),
            });
        }
    }

    let options = vackrooms::adapters::blueprint_renderer::BlueprintOptions {
        pixels_per_unit: 10.0,
        show_chunk_grid: true,
        show_fixtures: true,
        show_structure: true,
        seed,
        level: 0,
        voxel_scale,
    };
    vackrooms::adapters::blueprint_renderer::render_region_blueprint_svg(
        &plan,
        &chunk_bounds,
        options,
    )
}

/// Compact geometry feed for the canvas debug map. This deliberately returns
/// planning primitives rather than SVG so the page can redraw cheaply while
/// panning, zooming, or changing overlays.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn get_debug_region_json(seed: u32, region_x: i32, region_z: i32) -> String {
    use std::fmt::Write;
    use vackrooms::domain::entities::architecture::SpaceProgram;

    let noise = vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider::new();
    let size = vackrooms::use_cases::region_plan::REGION_SIZE;
    let origin =
        vackrooms::entities::models::Position::new(region_x as f32 * size, region_z as f32 * size);
    let config = vackrooms::use_cases::generate_chunk::GeneratorConfig::low_spec();
    let plan = vackrooms::use_cases::region_plan::generate_region_plan(
        seed, origin, size, &config, &noise,
    );
    let mut out = String::with_capacity(8192);
    write!(
        out,
        "{{\"origin\":[{:.1},{:.1}],\"size\":{:.1},\"walls\":[",
        origin.x, origin.z, size
    )
    .unwrap();
    let mut first = true;
    for assembly in &plan.assemblies {
        let vertices = &assembly.footprint.vertices;
        for pair in vertices
            .iter()
            .zip(vertices.iter().cycle().skip(1))
            .take(vertices.len())
        {
            if !first {
                out.push(',');
            }
            first = false;
            write!(
                out,
                "[{:.1},{:.1},{:.1},{:.1}]",
                pair.0.0, pair.0.1, pair.1.0, pair.1.1
            )
            .unwrap();
        }
    }
    out.push_str("],\"corridors\":[");
    first = true;
    for corridor in &plan.corridors {
        for pair in corridor.path.windows(2) {
            if !first {
                out.push(',');
            }
            first = false;
            write!(
                out,
                "[{:.1},{:.1},{:.1},{:.1},{:.1}]",
                pair[0].x, pair[0].z, pair[1].x, pair[1].z, corridor.width
            )
            .unwrap();
        }
    }
    out.push_str("],\"lights\":[");
    first = true;
    for assembly in &plan.assemblies {
        for fixture in &assembly.fixtures {
            if !first {
                out.push(',');
            }
            first = false;
            write!(
                out,
                "[{:.1},{:.1},{:.1},{:.1},{}]",
                fixture.at.x, fixture.at.z, fixture.half_x, fixture.half_z, fixture.lit
            )
            .unwrap();
        }
    }
    out.push_str("],\"rooms\":[");
    first = true;
    for assembly in &plan.assemblies {
        let b = assembly.footprint.bounds();
        if !first {
            out.push(',');
        }
        first = false;
        let kind = match assembly.program {
            SpaceProgram::MainCorridor => "corridor",
            SpaceProgram::AbandonedExpansion => "abandoned",
            _ => "room",
        };
        write!(
            out,
            "[{:.1},{:.1},{:.1},{:.1},\"{}\"]",
            b.0, b.1, b.2, b.3, kind
        )
        .unwrap();
    }
    out.push_str("],\"anomalies\":[");
    first = true;
    for anomaly in &plan.anomalies {
        if !first {
            out.push(',');
        }
        first = false;
        let b = anomaly.footprint.bounds();
        write!(
            out,
            "[\"{:?}\",\"{:016x}\",{:.1},{:.1},{:.1},{:.1},{}]",
            anomaly.kind,
            anomaly.id,
            b.min_x,
            b.min_z,
            b.max_x,
            b.max_z,
            anomaly.gates.len(),
        )
        .unwrap();
    }
    out.push_str("]}");
    out
}

/// Debug payload for one generated chunk: a compact material slice plus the
/// plan objects that explain why those voxels were placed.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn get_debug_chunk_json(seed: u32, chunk_x: f32, chunk_z: f32, voxel_scale: f32) -> String {
    use std::fmt::Write;
    use vackrooms::domain::entities::architecture::SpaceProgram;
    use vackrooms::domain::entities::voxel_grid::{
        VOXEL_CEILING, VOXEL_FLOOR, VOXEL_GRASS, VOXEL_LIGHT, VOXEL_RED_LIGHT, VOXEL_RED_WALL,
        VOXEL_TREE, VOXEL_WALL, VOXEL_WATER,
    };

    let noise = vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider::new();
    let config = if voxel_scale <= 0.1 {
        vackrooms::use_cases::generate_chunk::GeneratorConfig::high_spec()
    } else {
        vackrooms::use_cases::generate_chunk::GeneratorConfig::low_spec()
    };
    let chunk_origin = vackrooms::entities::models::Position::new(chunk_x, chunk_z);
    let generator =
        vackrooms::use_cases::generate_chunk::GenerateChunkArchitectureUseCase::new(&noise);
    let grid = generator.execute(chunk_origin, seed, config.clone());
    let region_size = vackrooms::use_cases::region_plan::REGION_SIZE;
    let region_origin = vackrooms::entities::models::Position::new(
        (chunk_x / region_size).floor() * region_size,
        (chunk_z / region_size).floor() * region_size,
    );
    let plan = vackrooms::use_cases::region_plan::generate_region_plan(
        seed,
        region_origin,
        region_size,
        &config,
        &noise,
    );
    let width = grid.width().saturating_sub(2);
    let depth = grid.depth().saturating_sub(2);
    let mut out = String::with_capacity(width * depth + 4096);
    write!(
        out,
        "{{\"origin\":[{:.1},{:.1}],\"scale\":{:.3},\"width\":{},\"depth\":{},\"cells\":\"",
        chunk_x, chunk_z, voxel_scale, width, depth
    )
    .unwrap();
    // One top-down material per column. Priority keeps red lights and walls
    // visible while still exposing floor/ceiling coverage in the same slice.
    for z in 0..depth {
        for x in 0..width {
            let mut material = b'.';
            for y in 0..grid.height() {
                material = match grid.get(x + 1, y, z + 1) {
                    VOXEL_RED_LIGHT => b'R',
                    VOXEL_LIGHT => b'L',
                    VOXEL_RED_WALL => b'X',
                    VOXEL_WALL | VOXEL_TREE => b'#',
                    VOXEL_CEILING => {
                        if material == b'.' {
                            b'^'
                        } else {
                            material
                        }
                    }
                    VOXEL_FLOOR | VOXEL_GRASS | VOXEL_WATER => {
                        if material == b'.' {
                            b'_'
                        } else {
                            material
                        }
                    }
                    _ => material,
                };
            }
            out.push(material as char);
        }
    }
    out.push_str("\",\"corridors\":[");
    let mut first = true;
    for corridor in &plan.corridors {
        for pair in corridor.path.windows(2) {
            if !first {
                out.push(',');
            }
            first = false;
            write!(
                out,
                "[{:.2},{:.2},{:.2},{:.2},{:.2},{}]",
                (pair[0].x - chunk_x) / voxel_scale,
                (pair[0].z - chunk_z) / voxel_scale,
                (pair[1].x - chunk_x) / voxel_scale,
                (pair[1].z - chunk_z) / voxel_scale,
                corridor.width / voxel_scale,
                if corridor.spine_kind == SpaceProgram::MainCorridor {
                    0
                } else {
                    1
                }
            )
            .unwrap();
        }
    }
    out.push_str("],\"boxes\":[");
    first = true;
    let mut box_out = |kind: &str, label: &str, bounds: (f32, f32, f32, f32), color: u8| {
        if !first {
            out.push(',');
        }
        first = false;
        write!(
            out,
            "[\"{}\",\"{}\",{:.2},{:.2},{:.2},{:.2},{}]",
            kind,
            label,
            (bounds.0 - chunk_x) / voxel_scale,
            (bounds.1 - chunk_z) / voxel_scale,
            (bounds.2 - bounds.0) / voxel_scale,
            (bounds.3 - bounds.1) / voxel_scale,
            color
        )
        .unwrap();
    };
    for assembly in &plan.assemblies {
        let b = assembly.footprint.bounds();
        let label = match assembly.program {
            SpaceProgram::AbandonedExpansion => "ABANDONED",
            _ => "ASSEMBLY",
        };
        box_out("assembly", label, b, 0);
        for space in &assembly.spaces {
            box_out("space", "SUB-ROOM", space.footprint.bounds(), 0);
        }
        for ceiling in &assembly.ceiling_zones {
            box_out("ceiling", "CEILING ZONE", ceiling.area.bounds(), 0);
        }
        let s = &assembly.structure;
        let b = assembly.footprint.bounds();
        box_out(
            "structure",
            "COLUMN GRID",
            (b.0 + s.phase.0, b.1 + s.phase.1, b.2, b.3),
            0,
        );
        for opening in &assembly.entrances {
            let half = opening.width * 0.5;
            let b = if opening.through_x_wall {
                (
                    opening.center.x - half,
                    opening.center.z - 0.12,
                    opening.center.x + half,
                    opening.center.z + 0.12,
                )
            } else {
                (
                    opening.center.x - 0.12,
                    opening.center.z - half,
                    opening.center.x + 0.12,
                    opening.center.z + half,
                )
            };
            box_out("portal", "PORTAL", b, 0);
        }
    }
    for anomaly in &plan.anomalies {
        let b = anomaly.footprint.bounds();
        let label = match anomaly.kind {
            vackrooms::domain::entities::anomaly::AnomalyKind::PillarExpanse => "PILLAR EXPANSE",
            vackrooms::domain::entities::anomaly::AnomalyKind::BlackoutExpanse => "BLACKOUT",
            vackrooms::domain::entities::anomaly::AnomalyKind::PitLattice => "PIT LATTICE",
            vackrooms::domain::entities::anomaly::AnomalyKind::RedRoom => "RED LOOP",
        };
        box_out("anomaly", label, (b.min_x, b.min_z, b.max_x, b.max_z), 1);
    }
    out.push_str("],\"fixtures\":[");
    first = true;
    for assembly in &plan.assemblies {
        for fixture in &assembly.fixtures {
            if !first {
                out.push(',');
            }
            first = false;
            write!(
                out,
                "[{:.2},{:.2},{:.2},{:.2},{}]",
                (fixture.at.x - chunk_x) / voxel_scale,
                (fixture.at.z - chunk_z) / voxel_scale,
                fixture.half_x / voxel_scale,
                fixture.half_z / voxel_scale,
                fixture.lit
            )
            .unwrap();
        }
    }
    out.push_str("]}");
    out
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn get_chunk_blueprint_svg(
    seed: u32,
    chunk_x: f32,
    chunk_z: f32,
    voxel_scale: f32,
    layer: String,
) -> String {
    let noise = vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider::new();
    let chunk_pos = vackrooms::entities::models::Position::new(chunk_x, chunk_z);

    let config = if voxel_scale <= 0.1 {
        vackrooms::use_cases::generate_chunk::GeneratorConfig::high_spec()
    } else {
        vackrooms::use_cases::generate_chunk::GeneratorConfig::low_spec()
    };

    let generator =
        vackrooms::use_cases::generate_chunk::GenerateChunkArchitectureUseCase::new(&noise);
    let grid = generator.execute(chunk_pos, seed, config);

    let region_size = vackrooms::use_cases::region_plan::REGION_SIZE;
    let rx = (chunk_x / region_size).floor() * region_size;
    let rz = (chunk_z / region_size).floor() * region_size;
    let semantics = vackrooms::use_cases::region_plan::generate_region_plan(
        seed,
        vackrooms::entities::models::Position::new(rx, rz),
        region_size,
        &config,
        &noise,
    );

    let slice = match layer.as_str() {
        "floor" => vackrooms::adapters::blueprint_renderer::BlueprintSlice::FloorPlan,
        "wall" => vackrooms::adapters::blueprint_renderer::BlueprintSlice::WallPlan,
        "ceiling" => vackrooms::adapters::blueprint_renderer::BlueprintSlice::CeilingPlan,
        _ => vackrooms::adapters::blueprint_renderer::BlueprintSlice::Composite,
    };

    vackrooms::adapters::blueprint_renderer::render_voxel_chunk_svg(
        &grid,
        chunk_pos,
        &config,
        slice,
        Some(&semantics),
    )
}

/// Fast single-chunk voxel blueprint with semantic overlays.
///
/// Why fast? This renders only one 50×50 (or 200×200 for high-spec) voxel
/// grid instead of stitching N×N chunks, so it completes in < 5 ms inside
/// WASM.  The browser can call this on every keypress without delay.
///
/// Arguments (all passed from JavaScript):
/// * `seed`          – World seed.
/// * `chunk_x`       – Chunk origin X in world units (chunk_index * chunk_size).
/// * `chunk_z`       – Chunk origin Z in world units.
/// * `voxel_scale`   – Metres per voxel (0.2 default, 0.1 high-spec).
/// * `pixels_per_voxel` – SVG pixels per cell (8–12 is comfortable).
/// * `show_grid`     – Draw hairline grid lines.
/// * `show_semantics`– Draw corridor / room / door overlays.
/// * `show_ceiling`  – Include ceiling and light voxels.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn get_chunk_voxel_blueprint_svg(
    seed: u32,
    chunk_x: f32,
    chunk_z: f32,
    voxel_scale: f32,
    pixels_per_voxel: f32,
    show_grid: bool,
    show_semantics: bool,
    show_ceiling: bool,
) -> String {
    let noise = vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider::new();
    let chunk_pos = vackrooms::entities::models::Position::new(chunk_x, chunk_z);

    let config = if voxel_scale <= 0.1 {
        vackrooms::use_cases::generate_chunk::GeneratorConfig::high_spec()
    } else {
        vackrooms::use_cases::generate_chunk::GeneratorConfig::low_spec()
    };

    // Generate the voxel grid for this single chunk.
    let generator =
        vackrooms::use_cases::generate_chunk::GenerateChunkArchitectureUseCase::new(&noise);
    let grid = generator.execute(chunk_pos, seed, config.clone());

    // Generate the region plan covering this chunk (for semantic overlays).
    let region_size = vackrooms::use_cases::region_plan::REGION_SIZE;
    let rx = (chunk_x / region_size).floor() * region_size;
    let rz = (chunk_z / region_size).floor() * region_size;
    let plan = vackrooms::use_cases::region_plan::generate_region_plan(
        seed,
        vackrooms::entities::models::Position::new(rx, rz),
        region_size,
        &config,
        &noise,
    );

    let options = vackrooms::adapters::chunk_voxel_renderer::VoxelBlueprintOptions {
        pixels_per_voxel: pixels_per_voxel.max(4.0).min(20.0),
        show_grid,
        show_voxels: true,
        show_semantics,
        show_labels: show_semantics, // labels only make sense when boxes are visible
        show_ceiling,
    };

    vackrooms::adapters::chunk_voxel_renderer::render_chunk_voxel_blueprint_svg(
        &grid,
        &plan,
        chunk_pos,
        voxel_scale,
        options,
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn get_large_voxel_blueprint_svg(
    seed: u32,
    rx_val: i32,
    rz_val: i32,
    voxel_scale: f32,
    size_world: f32,
    layer: String,
) -> String {
    let noise = vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider::new();

    let config = if voxel_scale <= 0.1 {
        vackrooms::use_cases::generate_chunk::GeneratorConfig::high_spec()
    } else {
        vackrooms::use_cases::generate_chunk::GeneratorConfig::low_spec()
    };

    let chunk_size = config.chunk_size;
    let steps = (size_world / chunk_size).round() as i32;

    // Generate the region plan
    let rx_origin = rx_val as f32 * size_world;
    let rz_origin = rz_val as f32 * size_world;
    let semantics = vackrooms::use_cases::region_plan::generate_region_plan(
        seed,
        vackrooms::entities::models::Position::new(rx_origin, rz_origin),
        size_world,
        &config,
        &noise,
    );

    // Generate all chunks and combine them
    let generator =
        vackrooms::use_cases::generate_chunk::GenerateChunkArchitectureUseCase::new(&noise);

    let chunk_w = (chunk_size / voxel_scale).round() as usize;
    let chunk_d = (chunk_size / voxel_scale).round() as usize;

    let total_w = steps as usize * chunk_w;
    let total_d = steps as usize * chunk_d;

    let mut combined_cells = vec![0u8; total_w * total_d];

    let slice = match layer.as_str() {
        "floor" => vackrooms::adapters::blueprint_renderer::BlueprintSlice::FloorPlan,
        "wall" => vackrooms::adapters::blueprint_renderer::BlueprintSlice::WallPlan,
        "ceiling" => vackrooms::adapters::blueprint_renderer::BlueprintSlice::CeilingPlan,
        _ => vackrooms::adapters::blueprint_renderer::BlueprintSlice::Composite,
    };

    for i in 0..steps {
        for j in 0..steps {
            let chunk_x = rx_origin + i as f32 * chunk_size;
            let chunk_z = rz_origin + j as f32 * chunk_size;
            let grid = generator.execute(
                vackrooms::entities::models::Position::new(chunk_x, chunk_z),
                seed,
                config.clone(),
            );

            // Copy chunk voxels to combined grid
            for cz in 0..chunk_d {
                for cx in 0..chunk_w {
                    let vx = cx + 1;
                    let vz = cz + 1;

                    let mut has_wall = false;
                    let mut has_red_wall = false;
                    let mut has_light = false;
                    let mut has_red_light = false;
                    let mut has_floor = false;
                    let mut has_ceiling = false;
                    let mut has_tree = false;
                    let mut has_grass = false;
                    let mut has_water = false;

                    for y in 0..grid.height() {
                        match grid.get(vx, y, vz) {
                            1 => has_wall = true,
                            5 => has_red_wall = true,
                            4 => has_light = true,
                            9 => has_red_light = true,
                            2 => has_floor = true,
                            3 => has_ceiling = true,
                            8 => has_tree = true,
                            6 => has_grass = true,
                            7 => has_water = true,
                            _ => {}
                        }
                    }

                    let cell_type = match slice {
                        vackrooms::adapters::blueprint_renderer::BlueprintSlice::FloorPlan => {
                            if has_wall || has_red_wall || has_tree {
                                if has_red_wall { 2 } else { 1 }
                            } else if has_floor || has_grass || has_water {
                                if has_water {
                                    7
                                } else if has_grass {
                                    6
                                } else {
                                    5
                                }
                            } else {
                                0
                            }
                        }
                        vackrooms::adapters::blueprint_renderer::BlueprintSlice::WallPlan => {
                            if has_wall || has_red_wall || has_tree {
                                if has_red_wall { 2 } else { 1 }
                            } else {
                                0
                            }
                        }
                        vackrooms::adapters::blueprint_renderer::BlueprintSlice::CeilingPlan => {
                            if has_light || has_red_light {
                                if has_red_light { 4 } else { 3 }
                            } else if has_ceiling {
                                9
                            } else {
                                0
                            }
                        }
                        vackrooms::adapters::blueprint_renderer::BlueprintSlice::Composite => {
                            if has_wall || has_red_wall || has_tree {
                                if has_red_wall { 2 } else { 1 }
                            } else if has_light || has_red_light {
                                if has_red_light { 4 } else { 3 }
                            } else if has_floor || has_grass || has_water {
                                if has_water {
                                    7
                                } else if has_grass {
                                    6
                                } else {
                                    5
                                }
                            } else {
                                0
                            }
                        }
                    };

                    let target_x = i as usize * chunk_w + cx;
                    let target_z = j as usize * chunk_d + cz;
                    let idx = target_z * total_w + target_x;
                    combined_cells[idx] = cell_type;
                }
            }
        }
    }

    vackrooms::adapters::blueprint_renderer::render_large_voxel_blueprint_svg(
        &combined_cells,
        total_w,
        total_d,
        rx_origin,
        rz_origin,
        voxel_scale,
        size_world,
        slice,
        Some(&semantics),
        false,
    )
}

#[cfg(target_arch = "wasm32")]
mod entry {
    use wasm_bindgen::prelude::*;

    /// Composition root. Runs automatically when the wasm module is
    /// instantiated by `static/index.html`.
    ///
    /// `static/worker.js` instantiates this same module inside a Web Worker
    /// with `#[wasm_bindgen(start)]` skipped (`init` is passed
    /// `{ skip_start: true }` is not available for start fns, so instead the
    /// worker checks for a missing DOM and bails out here).
    #[wasm_bindgen(start)]
    pub fn start() -> Result<(), JsValue> {
        console_error_panic_hook::set_once();
        // Inside a Web Worker there is no `window`/DOM: this module was
        // loaded as the generation worker, which drives itself through
        // `worker_init`/`worker_generate` instead of `boot`.
        if web_sys::window().is_none() {
            return Ok(());
        }
        crate::drivers::browser::boot()
    }
}

/// Generation-worker entry points. A worker is a second instance of this
/// same wasm module: `worker_init` builds a chunk source from the *same*
/// URL query the main thread used (identical world by construction), and
/// `worker_generate` runs the full chunk pipeline — architectural
/// generation, BFS lighting, greedy mesh + face instances, SVO build and
/// serialization, collision extraction — returning one transferable byte
/// buffer (see `adapters::chunk_codec`).
#[cfg(target_arch = "wasm32")]
mod worker_entry {
    use std::cell::RefCell;

    use wasm_bindgen::prelude::*;

    use crate::adapters::chunk_codec::encode_chunk_payload;
    use crate::adapters::local_chunk_source::LocalChunkSource;
    use crate::adapters::query_config::generator_setup_from_query;
    use crate::application::ports::ChunkSourcePort;
    use vackrooms::domain::entities::anomaly::RealitySnapshot;
    use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;

    thread_local! {
        static SOURCE: RefCell<Option<LocalChunkSource<SimpleNoiseProvider>>> =
            const { RefCell::new(None) };
    }

    /// `default_seed` must match the main thread's `WORLD_SEED`.
    #[wasm_bindgen]
    pub fn worker_init(query: &str, default_seed: u32) {
        let (seed, config) = generator_setup_from_query(query, default_seed);
        SOURCE.with(|s| {
            *s.borrow_mut() = Some(LocalChunkSource::new(
                SimpleNoiseProvider::new(),
                seed,
                config,
            ));
        });
    }

    #[wasm_bindgen]
    pub fn worker_generate(
        _request_id: u32,
        origin_x: f32,
        origin_z: f32,
        level: u32,
        lod: u8,
        reality_words: Vec<u32>,
    ) -> Vec<u8> {
        let reality = RealitySnapshot::from_words(&reality_words)
            .expect("worker_generate received an invalid reality snapshot");
        SOURCE.with(|s| {
            let source = s.borrow();
            let source = source
                .as_ref()
                .expect("worker_generate called before worker_init");
            encode_chunk_payload(
                &source.load_with_reality(origin_x, origin_z, level, lod, &reality),
            )
        })
    }
}
