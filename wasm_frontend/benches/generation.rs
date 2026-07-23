//! Size-parameterized benchmarks for the chunk-generation entry point real
//! callers use (`LocalChunkSource::load`), targeting two complexity risks
//! flagged in the generation-pipeline audit:
//!
//! - `chunk_by_voxel_size`: a single chunk's generation cost as `voxel_size`
//!   shrinks (chunk_size fixed). `ColumnField::sample` calls the per-column
//!   planning stack `(axis_voxels+2)^2` times, so this should show
//!   roughly-quadratic growth as `axis_voxels = chunk_size/voxel_size` grows.
//! - `neighborhood_redundancy`: total cost of loading a `(2r+1)^2`-chunk
//!   neighborhood at r = 0, 1, 2. `LocalChunkSource` has no cross-call cache
//!   (confirmed: no cache field on the struct), so `generate_region_plan` is
//!   re-derived from scratch for every chunk even when several chunks share
//!   one 80-unit region. If that's costing real time, per-chunk cost
//!   (total / chunk_count) should stay roughly flat as the neighborhood
//!   grows rather than dropping — chunks sharing a region would otherwise
//!   amortize the shared planning cost.
//!
//! `cargo bench -p wasm_frontend --bench generation`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::spawn_point;
use wasm_frontend::adapters::local_chunk_source::LocalChunkSource;
use wasm_frontend::application::ports::ChunkSourcePort;

fn chunk_by_voxel_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_by_voxel_size");
    for voxel_scale in [0.4f32, 0.2, 0.1] {
        let mut config = GeneratorConfig::low_spec();
        config.voxel_scale = voxel_scale;
        let source = LocalChunkSource::new(SimpleNoiseProvider::new(), 42, config);
        let spawn = spawn_point(42);
        let origin_x = (spawn.x / config.chunk_size).floor() * config.chunk_size;
        let origin_z = (spawn.z / config.chunk_size).floor() * config.chunk_size;
        let axis_voxels = (config.chunk_size / voxel_scale).round() as u32;
        group.bench_with_input(
            BenchmarkId::new("axis_voxels", axis_voxels),
            &(origin_x, origin_z),
            |b, &(x, z)| {
                b.iter(|| source.load(x, z, 0, 0));
            },
        );
    }
    group.finish();
}

fn neighborhood_redundancy(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighborhood_redundancy");
    let config = GeneratorConfig::low_spec();
    let source = LocalChunkSource::new(SimpleNoiseProvider::new(), 42, config);
    let spawn = spawn_point(42);
    let origin_x = (spawn.x / config.chunk_size).floor() * config.chunk_size;
    let origin_z = (spawn.z / config.chunk_size).floor() * config.chunk_size;
    for radius in [0i32, 1, 2] {
        let chunk_count = (2 * radius + 1).pow(2);
        group.bench_with_input(
            BenchmarkId::new("radius", format!("{radius}_{chunk_count}chunks")),
            &radius,
            |b, &radius| {
                b.iter(|| {
                    for dz in -radius..=radius {
                        for dx in -radius..=radius {
                            let x = origin_x + dx as f32 * config.chunk_size;
                            let z = origin_z + dz as f32 * config.chunk_size;
                            criterion::black_box(source.load(x, z, 0, 0));
                        }
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // Full neighborhoods at radius 2 (25 chunks) take real wall-clock time;
    // cap sample count so the suite finishes in a reasonable window.
    config = Criterion::default().sample_size(20);
    targets = chunk_by_voxel_size, neighborhood_redundancy
}
criterion_main!(benches);
