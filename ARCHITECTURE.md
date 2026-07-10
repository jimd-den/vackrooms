# Vackrooms — Architecture

A voxel-only rendering engine built on **Clean Architecture**, targeting
low-spec hardware. The whole engine — procedural generation, lighting, sparse
voxel octree (SVO) construction, chunk streaming, player physics, and WebGL2
raymarching — compiles to a single ~140 KB WebAssembly module. The browser
runs it; a zero-dependency native HTTP server merely serves the files.

## Rendering family

Of the three classical voxel pipeline families (mesh/rasterized,
SVO ray casting, hybrid), this engine implements a **hybrid SVO raymarcher**:

1. **Chunk-level coarse pass** — each resident chunk's AABB is slab-tested
   per fragment and the hits are insertion-sorted by entry distance, so rays
   march the nearest chunk first and exit on the first solid hit.
2. **Fragment SVO traversal** — stack-based descent through the octree with
   **empty-space skipping**: an air leaf or masked-out octant is exited in a
   single step via its AABB exit plane, never voxel-by-voxel.
3. **Shading** — packed 24-bit leaf color × BFS-propagated light level,
   face normal from the hit box, exponential distance fog.

Low-spec strategy: no depth buffer, no MSAA, nearest-filtered integer
textures, and an **adaptive internal-resolution governor** (0.5×–1.0× backing
store, hysteresis + cooldown) instead of temporal checkerboarding — simpler,
and it degrades smoothly on weak GPUs.

## The dependency rule

Source dependencies point inward only. No inner layer names a browser, GPU,
socket, or clock.

```
┌────────────────────────────────────────────────────────────────────┐
│ FRAMEWORKS & DRIVERS                                               │
│   wasm_frontend/drivers   WebGL2, DOM events, rAF, HUD, console    │
│   src/main.rs             native HTTP server (std::net only)      │
│   src/frameworks_drivers  SimpleNoiseProvider, StdTelemetry        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ INTERFACE ADAPTERS                                           │  │
│  │   src/adapters            OctreeGpuSerializer, VoxelMapper,  │  │
│  │                           JsonPresenter                      │  │
│  │   src/interface_adapters  WebRendererAdapter (presenters)    │  │
│  │   wasm_frontend/adapters  InputCollector, LocalChunkSource   │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │ USE CASES (application business rules)                 │  │  │
│  │  │   src/use_cases           GenerateChunk + ports        │  │  │
│  │  │   src/domain/use_cases    BuildOctree, lighting, maze  │  │  │
│  │  │   wasm_frontend/application  Engine (frame loop),      │  │  │
│  │  │      Player, CollisionWorld, StreamingPolicy, atlas    │  │  │
│  │  │      + ports (RendererPort, ChunkSourcePort)           │  │  │
│  │  │  ┌──────────────────────────────────────────────────┐  │  │  │
│  │  │  │ ENTITIES (enterprise business rules)             │  │  │  │
│  │  │  │   VoxelGrid, SparseVoxelOctree, Position, Grid   │  │  │  │
│  │  │  └──────────────────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Crates

| Crate | Role | Targets |
|---|---|---|
| `vackrooms` (root) | Entities + use cases + adapters; native dev-server binary | native **and** wasm32 |
| `wasm_frontend` | Browser client: application/adapters (portable) + drivers (wasm-only) | wasm32 (pure layers test natively) |
| `wasm_raycaster` | Legacy CPU raycaster experiment | kept for reference |

The split inside `wasm_frontend` is enforced mechanically: `web-sys`,
`js-sys` and `wasm-bindgen` are `[target.'cfg(target_arch = "wasm32")']`
dependencies, and `drivers/` is `#[cfg(target_arch = "wasm32")]`. A native
`cargo test` therefore cannot even *see* browser types — the compiler is the
architecture cop.

### Ports (dependency inversion boundaries)

| Port | Defined in (inner) | Implemented by (outer) |
|---|---|---|
| `NoiseProvider` | `src/use_cases/ports.rs` | `SimpleNoiseProvider` |
| `TelemetryPort` | `src/use_cases/ports.rs` | `StdTelemetry` (native stdout), `ConsoleTelemetry` (browser console), `NullTelemetry` (tests) |
| `RendererPort` | `wasm_frontend/application/ports.rs` | `WebGl2Renderer` (browser), recording fakes (tests) |
| `ChunkSourcePort` | `wasm_frontend/application/ports.rs` | `LocalChunkSource` (in-wasm generation); an HTTP-fetching implementation would slot in without touching the engine |

## Data flow: from noise to pixel

```
SimpleNoiseProvider (driver)
      │ NoiseProvider port
      ▼
GenerateChunkArchitectureUseCase ──► VoxelGrid (dense u8 grid + light grid)
      │                                   │
      │                          calculate_voxel_lighting (BFS flood fill)
      ▼                                   ▼
BuildOctreeUseCase ──────────────► SparseVoxelOctree
      │                              (arena Vec<SvoNode>, uniform-collapsed)
      ▼
OctreeGpuSerializer ─────────────► RGBA32UI texel stream, rows of 1024,
      │                              row-padded per chunk
      ▼
application::atlas::build_atlas ─► merged atlas + per-chunk rebased roots
      │ RendererPort port
      ▼
WebGl2Renderer ──────────────────► RGBA32UI texture + uniform chunk table
      ▼
fragment shader (drivers/shaders.rs) — fullscreen quad, sorted chunk AABBs,
stack-based SVO march, empty-space skipping ──► pixels
```

Collision geometry is derived **from the same SVO** (solid WALL/RED_WALL
leaves → world-space AABBs in `local_chunk_source::extract_collision_boxes`),
so physics and visuals can never drift apart. Uniform subtrees collapsed by
the SVO become single large collision boxes for free.

### SVO node texel encoding (RGBA32UI, one node per texel)

| Channel | Internal node (`R == 0`) | Leaf node (`R == 1`) |
|---|---|---|
| R | 0 | 1 |
| G | `child_base_index` (children contiguous at +0..+7) | `voxel_type` |
| B | `child_mask` (bit *i* set = child *i* non-empty) | 24-bit `0xRRGGBB` color |
| A | 0 | light level 0–15 |

The same layout is documented at its source of truth,
`src/adapters/octree_gpu_serializer.rs`, and decoded in
`wasm_frontend/src/drivers/shaders.rs` (`decodeNode`).

## Level 0: architecture first

Level 0 no longer decorates a random maze — it *plans* buildings and then
voxelizes them (`use_cases/region_plan.rs` + `use_cases/backrooms_level.rs`):

1. **Region plans.** The world tiles into fixed 80 u regions. A pure function
   of `(seed, region)` derives 1–3 `ArchitectGenome`s (circulation style,
   structural system, proportions, threshold/ceiling/lighting languages,
   renovation history), routes one dominant 4.8–7.2 u primary spine between
   shared edge portals (so it chains across regions forever), then adds at
   most two 3.6–5.0 u interior branches. It attaches incomplete, 12–24 u
   `AssemblyInstance` masses beside that circulation rather than tiling the
   region with rooms. Thresholds are selected per assembly: narrow lintelled
   doors are rare; most fronts use a broad or unframed portal.
2. **Corruption.** A final pass makes the sane plan Backrooms: suites repeat
   with misaligned copies, one assembly becomes an unlit
   `AbandonedExpansion` shell, renovations overlay contradictory column
   grids.
3. **Voxelization.** `BackroomsLevel` samples the plan per voxel column with
   priority *corridor → assembly → fabric*: corridors carve open under
   3.4–4.2 u (primary) or 3.0–3.6 u (secondary) ceilings; assemblies retain
   long perimeter runs and sparse partitions; everything between is a sparse
   18 u wall fabric that dissolves into open space. A broad ceiling field
   holds the visual baseline at 3.2–3.6 u, with 3.8–4.4 u expanses, 4.5–5.4 u
   vaults, and rare 2.5–2.8 u compression zones. This keeps cheap office
   finishes at an implausible scale instead of producing a low office maze.

All plan geometry snaps to a 0.4 u lattice (one coarse voxel) so every LOD
of a chunk voxelizes the same architecture. Debug hook:
`cargo run --example dump_plan` prints region plans as ASCII;
`debug_region_ascii` renders any plan.

## Chunk streaming

Single-threaded wasm has no background meshing thread, so streaming is
**time-sliced** and **progressive**: `StreamingPolicy` produces the desired
resident set (nearest-first) each frame, and the `Engine` spends a per-tick
cost budget on it in two phases:

1. **Availability** — every missing chunk first loads at a *coarse LOD*
   (voxels 2× the size, ~1/8 the generation/lighting/SVO cost, ~1 budget
   unit), so the whole footprint renders after roughly one tick instead of
   trickling in chunk by chunk.
2. **Refinement** — leftover budget reloads the nearest coarse chunk *within
   `fine_distance`* at full resolution, replacing it in place (same atlas
   slot, partial row upload). Chunks beyond `fine_distance` stay coarse: at
   that range a coarse voxel projects to about the same pixels as a fine one
   — a screen-space-error LOD. Refinement is monotonic while resident, so
   the LOD choice never flaps.

Every LOD of a chunk covers the same world cube (`GeneratorConfig::at_lod`
halves the SVO depth as it doubles `voxel_scale`), and cell borders/door
positions derive from world space, so a coarse chunk is a faithful low-res
proxy of its fine version and payloads are interchangeable to the renderer.
The atlas texture and collision world are rebuilt only on a resident-set
change — steady-state frames upload nothing (asserted by
`steady_state_does_not_reupload_atlas`).

Profiles (selected by URL, `?spec=high`):

| | chunk size | radius | voxels (fine/coarse) | SVO depth | fine ring |
|---|---|---|---|---|---|
| low (default) | 10 u | 1 (3×3) | 0.2 / 0.4 u | 6 / 5 | 15 u (all 9) |
| high | 20 u | 2 (5×5) | 0.1 / 0.2 u | 8 / 7 | 25 u (~9 of 25) |

The shader's chunk table is fixed at 25 entries — exactly the 5×5 high-spec
worst case (`application::atlas::MAX_CHUNKS`).

## Player simulation

`application::player` ports the original client's tuning 1:1: exponential
friction (8 s⁻¹), acceleration 4 u/s² (terminal 0.5 u/s), mouse sensitivity
0.002 rad/px, pitch clamped short of ±90°. Collision is **axis-separated
sliding**: X and Z are moved and tested independently against the
`CollisionWorld`, so hitting a wall diagonally slides along it. `dt` is
clamped to 100 ms so a background-tab hitch cannot teleport the player
through a wall.

## Testing strategy

Ports make the interesting logic natively testable — no browser, no GPU:

- `cargo test --workspace` runs 40 tests: entities, generation, lighting,
  octree build/serialize, plus the front end's player physics, sliding
  collision, streaming policy/eviction, atlas rebasing, input mapping, and
  the resolution governor.
- Renderer/chunk-source **test doubles** verify the engine's contract with
  its ports (upload counts, draw-table sizes) rather than pixels.
- The drivers layer is deliberately thin: translation only, no decisions.

## Native dev server

`src/main.rs` is a zero-dependency HTTP server used for local development:
static files with correct MIME types (`application/wasm` is mandatory for
streaming instantiation), plus the legacy JSON/binary chunk endpoints
(`/maze`, `/octree`) that the preserved Three.js client (`/legacy`) still
consumes. The wasm client needs no data endpoints at all.

## Roadmap (documented non-goals of this iteration)

- Temporal reprojection / checkerboarding to complement adaptive resolution.
- Per-chunk AABB *rasterization* (BackSide boxes + `gl_FragDepth` writeback)
  to replace the fullscreen quad once chunk counts grow beyond 25.
- Web Worker chunk generation (wasm threads) to remove the 1-chunk/tick cap.
- LOD: shallower SVO mip levels for distant chunks.
