# Vackrooms

A voxel-only engine in Rust, organized with Clean Architecture and compiled
to WebAssembly. Procedural world planning, finite-range voxel irradiance,
sparse voxel octree (SVO) construction, chunk streaming, and sliding player
collision remain deterministic CPU work. One WebGPU facade presents four
runtime-selectable strategies: indexed surfaces, compact face splats, SVO
raymarching, and a CPU reference image uploaded through WebGPU.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design (layer map, ports,
data flow, SVO node encoding, streaming, and rendering pipelines).

## How it works

Each chunk starts with the same authoritative pipeline, on the main thread or
in a generation worker. The selected renderer controls which derived upload
artifacts are retained:

1. **Procedural generation** (`src/use_cases/generate_chunk.rs` +
   `*_level.rs`) — a `LevelGenerator` fills a `VoxelGrid` deterministically
   from `(chunk position, seed, config)`. Level 0 is the office backrooms;
   level 34 is an open grassland. Both tile seamlessly because geometry is
   derived from world-space coordinates, never chunk-local ones.
2. **Lighting preparation** — exposed emissive faces seed an air-only,
   finite-range RGB diffuse field; solids receive it but never propagate it.
   Runtime panels are separately derived from exposed emissive undersides and
   remain analytic area lights in every renderer strategy.
3. **SVO build + optional serialization** — the lit grid is packed into a
   sparse voxel octree for collision and traversal. Recursive cubes wholly
   beyond a non-power-of-two grid are pruned directly to the canonical air
   leaf. Raymarch and CPU strategies serialize nodes as a row-padded,
   four-`u32` stream; raymarch uploads it to GPU storage while the CPU
   reference decodes the same words (`src/adapters/octree_gpu_serializer.rs`).
4. **Streaming** (`wasm_frontend/src/application/streaming.rs`) — chunks
   around the player are requested, LOD'd by distance, and evicted as the
   player moves; a coarse-first pass keeps distant chunks cheap.
5. **Collision** — player movement is axis-separated (X and Z tested and
   moved independently against a `CollisionWorld`), so hitting a wall
   diagonally slides along it instead of stopping dead.
6. **Render** — one `WebGpuRenderer` port adapter delegates resident chunks to
   one of four composable strategies (see below).

Generation is deliberately CPU-authoritative: architectural graph decisions,
reality epochs, and seeded noise must produce the same chunk regardless of
GPU or worker scheduling. The GPU instead performs renderer-derived work that
does not redefine the world: vertex pulling, face-to-quad expansion,
depth-tested supply-label billboard expansion, and per-pixel SVO traversal.

A standing gameplay quirk: leaning into a wall for a few seconds has a
chance to "noclip" the player between level 0 (backrooms) and level 34
(grassland) — see `update_noclip` in
`wasm_frontend/src/application/engine.rs`.

### Renderer backends

Chosen with `?renderer=` (see [Usage](#usage)):

| Value | WebGPU strategy | Notes |
|---|---|---|
| `surface` *(default)* | indexed surfaces | Greedy-mesh indices plus compact vertices. WGSL pulls and decodes vertices from a storage buffer; hardware depth resolves visibility. |
| `splat` | face splats | Compact face records live in a storage buffer. The vertex stage expands each record into four corners; CPU draw submission applies the configured cell and face budgets. |
| `raymarch` | SVO raymarcher | A fullscreen triangle drives fragment-stage traversal of the merged SVO storage-buffer atlas. |
| `cpu` | CPU reference / WebGPU present | The deterministic software SVO rasterizer writes RGBA8; WebGPU uploads that framebuffer to a texture and presents it. |

All four choices require WebGPU for presentation. Adapter or device creation
failure is reported in the loading HUD; the engine does not silently switch
rendering algorithms.

## Build & run

Prerequisites: Rust (with the `wasm32-unknown-unknown` target) and
[`wasm-pack`](https://rustwasm.github.io/wasm-pack/).
The page imports wasm-bindgen's native ES-module output directly; there is no
webpack build stage in this repository.

```sh
# 1. Build the wasm front end into static/pkg/
npm run build:wasm

# 2. Run the dev server
cargo run

# 3. Open http://localhost:3000
```

## Usage

Click to capture the mouse; WASD to move, ESC to release.

- `http://localhost:3000/` — wasm engine (low-spec profile: 3×3 chunk
  streaming, 0.2 u voxels, adaptive resolution)
- `http://localhost:3000/?spec=high` — high-spec profile (5×5 chunks,
  0.1 u voxels)

The former `/legacy` Three.js client and route have been retired. The native
server still exposes `/maze` and `/octree` as diagnostic data endpoints; the
wasm client does not depend on them.

Worlds are shareable by URL — every query parameter below is optional and
combinable:

| Parameter | Values | Effect |
|---|---|---|
| `spec` | `low`, `high` | Shared generation/rendering quality decision (case-insensitive). Default is low. |
| `voxel_size` | `0.4`, `0.2`, `0.1`, `0.05` | Overrides scene voxel size after seam, progressive-LOD, SVO-depth, and dense-memory validation. `0.4` is high-profile-only; `0.05` is standard-profile-only. |
| `seed` | number or text | World seed. Non-numeric text is hashed (FNV-1a) so words work too. |
| `pillars` | `0`–`4` | Structural column density multiplier (`0` = none). Default `1`. |
| `walls` | `0`–`4` | Office wall density multiplier (`0` = open plan). Default `1`. |
| `atria` | `0`–`4` | How much of the world vaults into tall atria. Default `1`. |
| `lights` | `0`–`4` | Ceiling light panel density. Default `1`. |
| `renderer` | `surface`, `splat`, `raymarch`, `cpu` | Selects one of the four WebGPU strategies (see above). |
| `cpu_preset` | `performance`, `balanced`, `quality`, `maximum`, `custom` | Typed CPU workload profile. Presets render at 12.5%–50% backing resolution per axis and present the complete image over the full page; `balanced` is the default. |
| `cpu_scale` | `0.25`–`2` | Custom CPU backing scale relative to the 25%-viewport baseline; CSS presentation always remains full-page. |
| `cpu_lod_px` | `0.25`–`2` | Custom projected-radius MIP cutoff; lower preserves finer distant geometry. |
| `cpu_splat_radius_px` | `1`–`16` | Custom maximum splat radius; lower values virtually split large leaves for finer surfaces. |
| `cpu_virtual_depth` | `3`–`8` | Custom cap on virtual solid-leaf subdivision. |
| `cpu_mip_occupancy` | `0`–`0.5` | Custom sparse-LOD rejection threshold; lower retains sparser distant detail. |
| `cpu_range` | `16`–`256` | CPU draw distance in world units. |
| `cpu_shadows` | `off`, `hero`, `full` | CPU fixture visibility: unshadowed, one cached hero-fixture approximation, or the expensive reference that traces every point/Gauss-sample endpoint. |
| `workers` | `auto`, `0`, `1`–`4` | Chunk-generation/streaming concurrency. `auto` reserves the main thread and uses at most four workers; explicit counts clamp to reported hardware. `0` uses the synchronous main-thread fallback. This does not parallelize rendering. |
| `level` | `0`, `34` | Debug: boots straight into a level (34 = the grassland) instead of waiting on a noclip roll. |
| `force_anomaly` | `pillars`, `blackout`, `pits`, `archway`, `redroom` | Debug: guarantees one anomaly of that family near spawn, bypassing organic frequency and the spawn keep-out. |
| `rt_hiz` | `0`, `1` | CPU hierarchical-Z rejection. |
| `rt_f2b` | `0`, `1` | CPU traversal plus splat/raymarch near-to-far chunk selection. Off preserves each strategy's deterministic reference order. |
| `rt_skip` | `0`, `1` | Raymarch empty-SVO-leaf skipping; `0` selects finest-cell diagnostic DDA. |
| `rt_mips` | `0`, `1` | CPU projected-size MIP/LOD collapse. |
| `rt_beam_occlusion` | `0`, `1` | CPU cone-light occlusion rays. |
| `rt_deferred` | `0`, `1` | CPU exact fine-depth rejection before hidden-splat shading. Off shades first and depth-tests later; pixels are identical. |
| `rt_ao` | `0`, `1` | CPU sibling-count ambient-occlusion approximation. Off selects unit ambient visibility for the reference path. |
| `rt_shadows` | `0`, `1` | Direct-light visibility. Surface/Splat use a hero-fixture depth map (128² hard comparison on low, 256² four-tap PCF on high); raymarch traces finite SVO segments to each point-light endpoint and one/four rectangle samples. |
| `rt_cells` | `0`, `1` | CPU-side splat cell-range rejection before direct draw submission. |
| `rt_budget` | `0`, `1` | Splat far-chunk face budget. |
| `rt_cull` | `0`, `1` | CPU and all WebGPU strategies reject chunks beyond the configured range; raster strategies also apply conservative camera visibility. Raymarch filters complete chunk bounds before its resident-set cap. |
| `rt_dither` | `0`, `1` | Shared WebGPU output dither. |
| `rt_bake` | `0`, `1` | Optional quantized diffuse fill for CPU, surface, and splat; raymarch remains analytic. |

The top-right HUD names the section you are in (level, zone, region), and
**F3** (or Settings → Debug) toggles the anomaly debug overlay: reality
epochs, resident traversal gates and pit hazards, and streaming counters.

The in-page settings menu (gear icon) also exposes mouse sensitivity, invert
Y, render scale, FOV, control scheme, typed CPU quality presets/custom detail,
and every renderer optimization switch. CPU quality scalars do not silently
change optimization switches: hierarchical Z, MIP collapse, traversal order,
chunk culling, hidden-splat rejection, ambient occlusion, and flashlight scene-visibility rays remain independently
diagnosable. Older `cpuScale`, `cpuSplatPx`, `cpuRange`, and `cpuShadows` links
are still accepted; the formerly ambiguous `cpuSplatPx` is interpreted as a
radius.
Live preferences are saved to `localStorage`; Apply & Reload writes disabled
optimizations into the URL so a regression setup can be shared exactly.

## GitHub Pages

`.github/workflows/pages.yml` builds `wasm_frontend` in release mode and
deploys `static/` on every push to `master` (or via manual dispatch). The
main engine (`/`, low- and high-spec profiles) generates chunks entirely in
the browser, so it needs nothing but static files and runs unmodified from a
Pages project URL (`https://<user>.github.io/<repo>/`) — the worker pool
resolves `worker.js` relative to the page, not from the domain root.

The optional `/maze` and `/octree` diagnostics belong to `src/main.rs` and
are absent on Pages; the wasm engine needs neither endpoint.

One-time setup: in the repo's Settings → Pages, set Source to "GitHub
Actions".

## Tests

```sh
cargo test --workspace
cargo check -p wasm_frontend --target wasm32-unknown-unknown
npm run check:wasm

# Linux: deterministic headless GPU contracts through Vulkan/Lavapipe.
VK_DRIVER_FILES="$(find /usr/share/vulkan/icd.d -name 'lvp_icd*.json' -print -quit)" \
  WGPU_VALIDATION=1 \
  cargo test -p wasm_frontend --features vulkan-tests \
    --test vulkan_renderers -- --test-threads=1
```

The application layers are browser-free by construction, so player physics,
collision, streaming, atlas assembly and input mapping are all covered by
native unit tests. The Vulkan suite uses a surface-free render target, forces
the Vulkan backend, and fails if no Vulkan adapter is available; CI pins it to
Mesa's deterministic Lavapipe software driver and enables Vulkan validation
layers. The exact Lavapipe ICD filename varies by distribution; CI discovers
it under `/usr/share/vulkan/icd.d/` before running the suite.

`npm run build:wasm` also writes a source fingerprint into `static/pkg/`.
`npm run check:wasm` verifies that the checked-in wasm-bindgen glue matches
the current Rust and worker ABI, catching stale worker-call signatures without
bringing a browser automation dependency back into the project.

## Workspace layout

| Path | What it is |
|---|---|
| `src/` | `vackrooms` core: entities, use cases, adapters + native dev server |
| `wasm_frontend/` | Portable application/adapters/WebGPU pipelines plus the wasm browser shell |
| `static/` | Thin HTML shell and wasm bundle output (`pkg/`) |
| `wasm_raycaster/` | Legacy CPU raycaster experiment (reference only) |

## Modifying the engine

Ports (traits) sit at the boundary of each layer; implement the trait rather
than reaching into a concrete type:

- **New procedural level** — implement `LevelGenerator`
  (`src/use_cases/level_generator.rs`) alongside `backrooms_level.rs` /
  `grassland_level.rs`, register a level id, and give it a native unit test
  (generation must be deterministic and seamless at chunk borders — see the
  contract documented on the trait).
- **New renderer strategy** — add a focused pipeline under
  `wasm_frontend/src/drivers/webgpu/pipelines/`, give it typed configuration
  and artifact requirements in `webgpu/config.rs`, then compose it inside the
  single `WebGpuRenderer` port adapter. Browser surface ownership stays out of
  the pipeline.
- **New chunk source** (e.g. a different streaming/caching strategy) —
  implement `ChunkSourcePort` (`wasm_frontend/src/application/ports.rs`); see
  `LocalChunkSource` and `WorkerChunkSource` for the synchronous and
  worker-pool implementations.
- **Player/physics tuning** — `wasm_frontend/src/application/player.rs` and
  `engine.rs`; both are plain Rust, unit-tested natively without a browser.

After changing anything under `src/` or `wasm_frontend/src/`, run
`cargo test --workspace` (native, browser-free) before rebuilding the wasm
bundle with the command in [Build & run](#build--run). See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full layer map and data flow.
