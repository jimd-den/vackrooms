# Vackrooms

A voxel-only rendering engine in Rust, structured with documented Clean
Architecture and compiled to WebAssembly. Procedural "backrooms" world
generation, BFS lighting, sparse voxel octree (SVO) construction, chunk
streaming and sliding player collision all run inside one ~410 KB wasm
module — built to hold up on low-spec hardware. Rendering is pluggable:
a greedy-meshed WebGL2 rasterizer by default, with an instanced face-splat
path, a retained SVO raymarcher, and a CPU-only software fallback all
selectable at runtime (see [Renderer backends](#renderer-backends)).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design (layer map, ports,
data flow, texel encodings, streaming and rendering pipeline).

## How it works

Each chunk goes through the same pipeline, on the main thread or in a
generation worker:

1. **Procedural generation** (`src/use_cases/generate_chunk.rs` +
   `*_level.rs`) — a `LevelGenerator` fills a `VoxelGrid` deterministically
   from `(chunk position, seed, config)`. Level 0 is the office backrooms;
   level 34 is an open grassland. Both tile seamlessly because geometry is
   derived from world-space coordinates, never chunk-local ones.
2. **BFS lighting** — light sources placed by the generator flood-fill
   through open voxels to produce per-voxel RGB light levels.
3. **SVO build + serialize** — the lit grid is packed into a sparse voxel
   octree and serialized into the GPU texel encoding the renderer expects
   (`src/adapters/octree_gpu_serializer.rs`).
4. **Streaming** (`wasm_frontend/src/application/streaming.rs`) — chunks
   around the player are requested, LOD'd by distance, and evicted as the
   player moves; a coarse-first pass keeps distant chunks cheap.
5. **Collision** — player movement is axis-separated (X and Z tested and
   moved independently against a `CollisionWorld`), so hitting a wall
   diagonally slides along it instead of stopping dead.
6. **Render** — one of four interchangeable `RendererPort` implementations
   draws the resident chunks (see below).

A standing gameplay quirk: leaning into a wall for a few seconds has a
chance to "noclip" the player between level 0 (backrooms) and level 34
(grassland) — see `update_noclip` in
`wasm_frontend/src/application/engine.rs`.

### Renderer backends

Chosen with `?renderer=` (see [Usage](#usage)):

| Value | Backend | Notes |
|---|---|---|
| *(default)* | `SurfaceRenderer` | Indexed greedy meshes, WebGL2 fixed-function depth. The SVO is kept for collision/debug only — no fragment shader walks it. |
| `splat` | `SplatRenderer` | Instanced face-splat "microvoxel" path: one instance per visible surface rectangle, lit once in the vertex shader. |
| `raymarch` | `WebGl2Renderer` | The retained hybrid SVO raymarcher: per-fragment octree walk against an RGBA32UI node-atlas texture. |
| `cpu` | `CpuCanvasRenderer` | Software rasterizer blitted via `ImageData`; no WebGL context at all. |

`SurfaceRenderer` and `SplatRenderer` fall back automatically (to CPU, or to
surfaces) if WebGL2 or the requested backend is unavailable.

## Build & run

Prerequisites: Rust (with the `wasm32-unknown-unknown` target) and
[`wasm-pack`](https://rustwasm.github.io/wasm-pack/).

```sh
# 1. Build the wasm front end into static/pkg/
wasm-pack build wasm_frontend --target web --release --no-typescript --out-dir ../static/pkg

# 2. Run the dev server
cargo run

# 3. Open http://localhost:8080
```

## Usage

Click to capture the mouse; WASD to move, ESC to release.

- `http://localhost:8080/` — wasm engine (low-spec profile: 3×3 chunk
  streaming, 0.2 u voxels, adaptive resolution)
- `http://localhost:8080/?spec=high` — high-spec profile (5×5 chunks,
  0.1 u voxels)
- `http://localhost:8080/legacy` — the preserved original Three.js/JS client,
  which fetches chunks from the server's `/maze` and `/octree` endpoints
  instead of generating them in the browser

Worlds are shareable by URL — every query parameter below is optional and
combinable:

| Parameter | Values | Effect |
|---|---|---|
| `spec` | `high` | Switches to the high-spec profile (finer voxels, wider streaming radius). Default is low-spec. |
| `seed` | number or text | World seed. Non-numeric text is hashed (FNV-1a) so words work too. |
| `pillars` | `0`–`4` | Structural column density multiplier (`0` = none). Default `1`. |
| `walls` | `0`–`4` | Office wall density multiplier (`0` = open plan). Default `1`. |
| `atria` | `0`–`4` | How much of the world vaults into tall atria. Default `1`. |
| `lights` | `0`–`4` | Ceiling light panel density. Default `1`. |
| `renderer` | `splat`, `raymarch`, `cpu` | Picks a non-default renderer backend (see above). |
| `workers` | `0` | Disables the Web Worker generation pool and generates chunks synchronously on the main thread. |
| `level` | `0`, `34` | Debug: boots straight into a level (34 = the grassland) instead of waiting on a noclip roll. |
| `force_anomaly` | `pillars`, `blackout`, `pits`, `archway` | Debug: guarantees one anomaly of that family on the spawn's macro cell, bypassing the spawn keep-out. |

The top-right HUD names the section you are in (level, zone, region), and
**F3** (or Settings → Debug) toggles the anomaly debug overlay: reality
epochs, resident traversal gates and pit hazards, and streaming counters.

The in-page settings menu (gear icon) also exposes mouse sensitivity, invert
Y, render scale, FOV and control scheme; those are saved to `localStorage`
per device rather than the URL.

## GitHub Pages

`.github/workflows/pages.yml` builds `wasm_frontend` in release mode and
deploys `static/` on every push to `master` (or via manual dispatch). The
main engine (`/`, low- and high-spec profiles) generates chunks entirely in
the browser, so it needs nothing but static files and runs unmodified from a
Pages project URL (`https://<user>.github.io/<repo>/`) — the worker pool
resolves `worker.js` relative to the page, not from the domain root.

The `/legacy` client and the `/maze`/`/octree` endpoints depend on
`src/main.rs`'s native HTTP server and cannot run on Pages; only the wasm
engine is served there.

One-time setup: in the repo's Settings → Pages, set Source to "GitHub
Actions".

## Tests

```sh
cargo test --workspace
```

The application layers are browser-free by construction, so player physics,
collision, streaming, atlas assembly and input mapping are all covered by
native unit tests (116 tests at the time of writing).

## Workspace layout

| Path | What it is |
|---|---|
| `src/` | `vackrooms` core: entities, use cases, adapters + native dev server |
| `wasm_frontend/` | Browser client: application / adapters / drivers (wasm-only) |
| `static/` | Thin HTML shell, wasm bundle output (`pkg/`), legacy client |
| `wasm_raycaster/` | Legacy CPU raycaster experiment (reference only) |

## Modifying the engine

Ports (traits) sit at the boundary of each layer; implement the trait rather
than reaching into a concrete type:

- **New procedural level** — implement `LevelGenerator`
  (`src/use_cases/level_generator.rs`) alongside `backrooms_level.rs` /
  `grassland_level.rs`, register a level id, and give it a native unit test
  (generation must be deterministic and seamless at chunk borders — see the
  contract documented on the trait).
- **New renderer backend** — implement `RendererPort`
  (`wasm_frontend/src/application/ports.rs`) under `wasm_frontend/src/drivers/`
  and wire a `?renderer=` value into `create_renderer` in
  `wasm_frontend/src/drivers/browser.rs`.
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
