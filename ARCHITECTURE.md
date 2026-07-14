# Vackrooms — Architecture

A voxel-only rendering engine built on **Clean Architecture**, targeting
low-spec hardware. The whole engine — procedural generation, lighting, sparse
voxel octree (SVO) construction, chunk streaming, player physics, and
rendering — compiles to a single WebAssembly module. The browser
runs it; a zero-dependency native HTTP server merely serves the files (or, on
GitHub Pages, static files alone — the wasm engine generates chunks entirely
client-side and needs no server).

## Rendering family

`RendererPort` (`wasm_frontend/src/application/ports.rs`) has four
implementations, selected at runtime by `create_renderer`
(`wasm_frontend/src/drivers/browser.rs`) from the `?renderer=` query param:

| `?renderer=` | Driver | Approach |
|---|---|---|
| *(default)* | `surface_webgl::SurfaceRenderer` | Indexed greedy meshes, WebGL2 fixed-function depth. The SVO is retained per chunk for collision/debug only — no fragment shader walks it. |
| `splat` | `splat_webgl::SplatRenderer` | Instanced face-splat "microvoxel" path: one GPU instance per visible surface rectangle, lit once in the vertex shader; the retained mesh is drawn only into the shadow map. |
| `raymarch` | `webgl::WebGl2Renderer` | The hybrid SVO raymarcher described below — a per-fragment octree walk. |
| `cpu` | `cpu_canvas::CpuCanvasRenderer` | Software rasterizer (`adapters::cpu_splatter`) blitted via `ImageData`; no WebGL context at all. |

`SurfaceRenderer` and `SplatRenderer` fall back (to CPU, or to surfaces)
if WebGL2 or the requested backend fails to initialize.

Renderer code is organized by lifetime and responsibility rather than by one
backend-sized class:

| Module | Responsibility |
|---|---|
| `drivers/surface_webgl/{mod,resources,draw}.rs` | surface composition, chunk GPU resources, staged frame pipeline |
| `drivers/splat_webgl/{mod,resources,draw}.rs` | splat composition, instance/shadow resources, staged frame pipeline |
| `drivers/webgl/{mod,atlas,draw}.rs` | raymarch composition, SVO texture lifecycle, fullscreen submission |
| `adapters/cpu_splatter/` | atlas decode, camera, cone light, ray queries, shading, raster traversal, tests |
| `drivers/gl/` | context/program setup, matrices, light selection, shadow targets, timers, visibility |
| `drivers/shaders/` | one literate Rust module per GLSL program plus shared chunks |

Optional shortcuts are plain data in `application::render_settings::RenderToggles`.
Each driver snapshots that switchboard once per frame; inner algorithms never
read browser globals. The same switches are available live in Settings →
Optimize and as shareable `?rt_<name>=0|1` query parameters.

Of the three classical voxel pipeline families (mesh/rasterized,
SVO ray casting, hybrid), the `raymarch` backend implements a
**hybrid SVO raymarcher**:

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
| `RendererPort` | `wasm_frontend/src/application/ports.rs` | `SurfaceRenderer` (default), `SplatRenderer`, `WebGl2Renderer` (raymarch), `CpuCanvasRenderer`, recording fakes (tests) |
| `ChunkSourcePort` | `wasm_frontend/src/application/ports.rs` | `LocalChunkSource` (synchronous, in-wasm generation), `WorkerChunkSource` (pooled Web Worker generation, default); an HTTP-fetching implementation would slot in without touching the engine |

## Data flow: from noise to pixel

The SVO atlas path below feeds the `raymarch` backend directly; the default
`SurfaceRenderer` and `SplatRenderer` consume the same lit `VoxelGrid` but
greedy-mesh or face-splat it instead of walking the octree per fragment (the
SVO they receive is used for collision only).

```
SimpleNoiseProvider (driver)
      │ NoiseProvider port
      ▼
GenerateChunkArchitectureUseCase ──► VoxelGrid (dense u8 grid + light grid)
      │                                   │
      │                          bake_voxel_lighting (world-unit air paths)
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
fragment shader (drivers/shaders/raymarch.rs) — fullscreen quad, sorted chunk AABBs,
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
`wasm_frontend/src/drivers/shaders/raymarch.rs` (`decodeNode`).

## Level 0: architecture first

Level 0 is a **world-planning system**, not a random room generator. Math
selects architectural *intentions*; it never substitutes for them. Planning
descends a strict scale hierarchy, each level constraining the next:

```
world seed
  -> MacroFields         multi-octave parameter fields (world_topology)
    -> MacroCell graph   160 u cells: nodes, portals, red-room events,
                         vertical links (use_cases/world_topology.rs)
      -> RegionPlan      80 u architectural plan (use_cases/region_plan.rs)
        -> ColumnPlan    one voxel column (use_cases/backrooms_level.rs)
          -> VoxelGrid   resolution-dependent output (level_zero/voxelize.rs)
```

**Fractal math's one job.** `world_topology::sample_fields` sums three
noise octaves per parameter (`F(x,z) = Σ aᵢ·N(x/sᵢ, z/sᵢ)`) into `[0, 1]`
fields — `openness`, `vertical_pressure`, `institution_age`,
`anomaly_pressure`, `redroom_pressure`, `style_blend`. Fields only bias
probabilities and style choices (where the building's rules change); a
planner still decides every corridor, room, stair, and threshold. This is
what makes billions of areas *differ* without ever placing a wall by noise.

**The macro graph.** `plan_macro_cell(seed, cell, red_room_scale, noise)`
is a pure function returning one 160 u `MacroCell`: a `WorldNode` per
region (classified `Warren` / `OpenPlate` / `Atrium` / `Stairwell` /
`RedRoomEncounter`), shared-edge `Portal`s (both neighbors of an edge
derive the identical crossing — the infinite-corridor contract),
`VerticalLink` stair reservations, and at most one `RedRoomEvent`. A 5×5
graph snapshot test pins the exact topology of seed 42: an unintentional
world change fails the build.

**Red rooms are graph events, not a biome.** Events are planned on the
macro lattice by a deterministic local tournament: a candidate cell fires
only if no candidate within the separation radius beats its score, so any
two encounters keep a cooldown distance (≥ 3 regions), clustered where
`redroom_pressure` runs high. The region planner merely *realizes* an event
that targets its region by promoting one occupied, reachable assembly.

**Vertical circulation.** `vertical_link_for_region` reserves stairwells
where `vertical_pressure` is high: `OrdinaryStair` and `EndlessAscent`
links are realized by `use_cases/vertical_circulation.rs` as compact Stair
assemblies beside the main spine — a lit vestibule, a monotonic flight of
0.2 u treads (raised floor voxelizes as solid, so it collides), and either
a landing or an endless climb into an unlit 5.4 u shaft. Elevation is an
integer story index on `WorldNode`; `EndlessDescent` / `ServiceShaft`
links stay graph reservations until the engine can stream below elevation
0 and the player gains vertical physics (see roadmap).

**Archways are topological connectors.** A `Transition` arch room carries
an `ArchBehavior`: most are plain `Anchor`s, but a `CultureSeam`'s far
half changes ceiling regime, floor grammar, and lintel height, and a
`ScaleBreach`'s far half repeats the same grammar larger. The seam is
expressed only through legal architectural vocabulary — never impossible
collision or repainted walls.

Within one region, the planner works as before
(`use_cases/region_plan.rs` + `use_cases/backrooms_level.rs`):

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

### Infinite generation and anomalies

The generation code is organized by the language of Level 0, rather than by
delivery mechanism:

```
world_topology (macro fields, graph events, portals, vertical links)
        │
InfiniteRegionWindow ──► RegionPlan(s) ──► architectural ColumnPlan
                                                │
anomalies/{planning,geometry} ──────────────────┤
red_rooms/{planning,geometry,recursive_level} ──┤
vertical_circulation (stair profiles) ──────────┘
                                                │
                                      ColumnField (1-column halo)
                                                │
                                      voxelize_columns ──► VoxelGrid
```

- `domain/entities/anomaly/` contains only domain language: rectilinear
  geometry, immutable anomaly instances, traversal semantics, and canonical
  reality snapshots. The public `anomaly` namespace remains stable.
  `domain/entities/world_topology.rs` holds the macro-graph language
  (`MacroCell`, `WorldNode`, `Portal`, `VerticalLink`, `MacroFields`).
- `use_cases/anomalies/` plans and samples macro anomaly families on the
  same 160 u lattice as the world graph; per-anchor chance is weighted by
  the `anomaly_pressure` field, so anomalies arrive in loose
  constellations, never a uniform sprinkle.
  `use_cases/red_rooms/` owns the assembly-derived encounter and its recursive
  Level 0 address. The old `anomaly_plan` module is only a compatibility
  facade. The pre-planning `?level=1` generator lives untouched in
  `use_cases/legacy_blueprint.rs`, consulted by nothing in Level 0.
- `InfiniteRegionWindow` is the finite query cache for an infinite region
  lattice. A missing plan is an error instead of silently falling back to an
  unrelated region.
- A committed Red Room derives its branch seed and region-aligned translation
  with integer hashes over the full 64-bit anomaly id. Planning and sampling
  use that same address, so worker order, chunk boundaries, and floating-point
  id precision cannot select different worlds.
- `ColumnField` and `voxelize_columns` are resolution-dependent output stages.
  World planning never reads the requested output extent; generating one 20 u
  area or four 10 u areas produces the same voxels at matching coordinates.

Reality is part of chunk identity. Entering a Red Room changes the active
infinite Level 0 address, so all resident chunks are atomically scheduled for
replacement; partially repainting only the vestibule would splice two worlds
at a streaming boundary.

### Generation invariants (enforced by tests)

Massive procedural worlds only work when randomness is constrained. These
contracts are non-negotiable and each is asserted natively:

- The same seed, coordinates, LOD, and reality always produce identical
  geometry; the 5×5 macro-graph snapshot digest pins seed 42's topology.
- Neighboring regions independently compute the same shared portals, and
  the primary route crosses every region boundary (corridors chain forever).
- Every assembly entrance opens onto a corridor; sealed masses are
  intentional (`AbandonedExpansion`), never accidents.
- Stair flights rise monotonically from a flat entrance, keep ≥ 2.2 u of
  headroom over every tread, and reach their declared landing.
- Red-room events keep a minimum separation radius; a red room never
  produces red masonry — only red light over ordinary architecture.
- World planning never reads the requested output extent: one 20 u query
  and four 10 u queries produce the same voxels at matching coordinates.

## Chunk streaming

Chunk generation runs on a pool of Web Workers (`WorkerChunkSource`,
`static/worker.js` — each a second instance of the same wasm module), sized
to `hardware_concurrency - 1` (clamped 1–4), so crossing a streaming
boundary never stalls the frame loop; `?workers=0` forces the older
synchronous in-thread `LocalChunkSource`, which is also the automatic
fallback if the worker pool fails to spin up. Either way, the resident set
itself is **time-sliced** and **progressive**: `StreamingPolicy` produces the
desired resident set (nearest-first) each frame, and the `Engine` spends a
per-tick cost budget on it in two phases:

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

- `cargo test --workspace` runs 212 tests: entities, generation (including
  the macro-graph snapshot, red-room separation, stair-flight, and arch-seam
  invariants), lighting, octree build/serialize, plus the front end's player
  physics, sliding collision, streaming policy/eviction, atlas rebasing,
  input mapping, and the resolution governor.
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

- **Vertical streaming**: chunks keyed by `(x, z, elevation)` so
  `EndlessDescent` / `ServiceShaft` links can voxelize real destinations;
  player step-up and Y physics so planned flights become climbable. The
  topology contracts (`VerticalLink`, integer `elevation`) already exist —
  the endless variants should re-address the world at each landing, never
  mesh a literal infinite staircase.
- Arch seams as *portal transforms*: a `CultureSeam` crossing re-seeding
  the architect genome on the far side, `LoopBreach` returning near the
  origin under a shifted reality epoch (gate machinery exists; the arch
  kinds need their own `TraversalGate` semantics).
- Red-room progression: foreshadowing (warming/flickering fixtures across
  the one or two assemblies adjacent to a planned event, driven by the
  existing event lookup).
- Temporal reprojection / checkerboarding to complement adaptive resolution.
- Per-chunk AABB *rasterization* (BackSide boxes + `gl_FragDepth` writeback)
  to replace the fullscreen quad once chunk counts grow beyond 25 (`raymarch`
  backend only).
- LOD: shallower SVO mip levels for distant chunks.
