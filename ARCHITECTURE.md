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
| `drivers/surface_webgl/` | `draw_lit_world_surfaces`, `upload_world_surface_chunks`, and renderer composition |
| `drivers/splat_webgl/{mod,resources,draw}.rs` | splat composition, instance/shadow resources, staged frame pipeline |
| `drivers/webgl/` | `draw_voxel_scene`, `upload_voxel_atlas`, and raymarch composition |
| `adapters/cpu_splatter/` | atlas decode, camera, cone light, ray queries, shading, raster traversal, tests |
| `adapters/cpu_splatter/settings/` | typed CPU quality presets, one validated scalar model, and explicit flashlight/fixture visibility policies |
| `drivers/gl/` | context/program setup, matrices, light selection, shadow targets, timers, visibility |
| `drivers/shaders/render_world_surfaces/` | reconstruct surfaces → sample optional diffuse field → shade visible surface |
| `drivers/shaders/trace_voxel_scene/` | decode atlas → intersect voxel scene → shade nearest hit |
| `drivers/shaders/{evaluate_scene_lighting,apply_distance_fog,encode_display_color}.rs` | shared linear-light, Beer–Lambert, and display equations |

Optional shortcuts are plain data in `application::render_settings::RenderToggles`.
Each driver snapshots that switchboard once per frame; inner algorithms never
read browser globals. The same switches are available live in Settings →
Optimize and as shareable `?rt_<name>=0|1` query parameters.

The CPU composition root separately snapshots `CpuRenderSettings`. Its named
presets change only bounded workload/quality scalars (resolution, distant LOD,
splat radius, virtual depth, sparse-MIP threshold, range, and fixture-shadow
policy). They do not mutate `RenderToggles`. The CPU canvas resolution factor
is applied before both the canvas and software framebuffer are resized, so a
reduced render always covers the full CSS viewport instead of filling only a
corner of a larger backing store.

CPU fixture visibility is an explicit three-level policy. `off` is the fast
unoccluded diagnostic, `hero` amortizes one real fixture-center segment, and
`full` is the radiometric reference: every point endpoint and every one of a
rectangle's four Gauss endpoints traces a finite SVO segment. CPU MIPs average
linear albedo and colored bake values; ordinary LOD never collapses a subtree
containing emission, while the hard work-cap fallback retains aggregate
emissive coverage/radiance instead of deleting the panel.

The `raymarch` backend is a correctness-first SVO ray caster with two explicit
stepping policies over the same stateless point lookup and hit record:

1. **Chunk intervals** — robust parallel-safe slabs produce entry/exit
   intervals. Near-to-far traversal may stop only when every remaining AABB
   starts beyond the nearest solid hit; padded chunk cubes may overlap.
2. **Reference traversal (`rt_skip=0`)** — exact finest-cell 3-D DDA. Voxel
   size and SVO depth come from each payload; neither is inferred from its
   power-of-two padded world size.
3. **Empty-leaf traversal (`rt_skip=1`)** — restart from the root for each
   sample, then jump to the exact exit of a masked octant or empty leaf. No
   mutable descent stack survives a jump.
4. **Shading** — sRGB material values are decoded to linear RGB, ceiling
   panels use a downward one-sided rectangular-emitter integral. The
   raymarcher intentionally rejects the bake's face-independent leaf value;
   when `rt_shadows=1`, each of the rectangle integral's four samples traces
   an SVO visibility segment through every resident chunk. Beer–Lambert fog
   is composed in linear space before one shared tone-map/sRGB conversion.

The default surface renderer evaluates the same analytic fixture list. Static
fixtures and runtime flares use separate uniform arrays, so dropping a flare
cannot evict a ceiling panel. `rt_bake` defaults off and never removes analytic
lights; on the surface renderer it enables the deliberately approximate,
quantized diffuse-fill field. Surface visibility is also an explicit raster
approximation: `rt_shadows=1` provides one shadow map for the highest-priority
fixture. The raymarcher is the strict all-fixture visibility reference.

There is no global fixture budget in either rewritten GPU path. The complete,
deduplicated fixture list lives in an RGBA32F texture; a conservative
finite-support intersection builds contiguous per-surface or per-chunk
clusters. Clustering changes only which provably zero-contribution lights are
skipped, not the lighting equation or the set of contributing fixtures.

Low-spec strategy: the raymarcher uses one fullscreen pass, the surface path
uses the hardware depth buffer, neither requests MSAA, SVO data stays in
nearest-filtered integer textures, and an **adaptive internal-resolution
governor** (0.5×–1.0× backing store, hysteresis + cooldown) replaces temporal
checkerboarding.

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
fragment shader (drivers/shaders/trace_voxel_scene/) — fullscreen quad,
robust chunk intervals, DDA or empty-leaf stepping ──► pixels
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
| A | 0 | scalar light 0–15, face-occlusion bits, and RGB light 0–15/channel |

The same layout is documented at its source of truth,
`src/adapters/octree_gpu_serializer.rs`, and decoded in
`wasm_frontend/src/drivers/shaders/trace_voxel_scene/decode_voxel_atlas.rs`.

## Level 0: architecture first

Level 0 is a **world-planning system**, not a random room generator. Math
selects architectural *intentions*; it never substitutes for them. Planning
descends a strict scale hierarchy, each level constraining the next:

```
world seed
  -> MacroFields         multi-octave parameter fields (world_topology)
    -> MacroCell graph   160 u cells: nodes, portals, red-room events,
                         vertical links (use_cases/world_topology.rs)
      -> RegionPlan      80 u architectural plan (use_cases/region_plan/)
        -> ColumnPlan    one voxel column (use_cases/level_zero/)
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

Both stages are split by intent, one module per planning concern:
`use_cases/region_plan/` (genome, circulation, suites, corruption, debug)
and `use_cases/level_zero/` (fabric, circulation_sampler, assembly_sampler,
compose_column, generate, column_field, voxelize). Within one region, the
planner works as follows:

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

**The Peripheral Shift.** "Whenever not directly observed, the layout can
warp, stretch, or rearrange itself." `RealitySnapshot` (RTY v3) carries one
drift epoch per 40 u fabric cell. The epoch re-salts only the fabric's
cosmetic and porosity decisions — wall dropout, doorway
direction/position/framing, dead lights — read at each deciding lattice
cell's own anchor, so walls rebuild whole and the binary-tree doorway rule
(hence global connectivity) holds at any epoch mix. Corridors, portals,
assemblies, anomaly identities, arch-anchor surroundings, and the spawn
opening sequence never drift: navigation survives; hallway memory does not.
The engine advances epochs two ways: territory abandoned beyond the
streaming footprint (plus hysteresis) drifts on departure, and *inside a
blackout* the shift runs in real time — cells wholly behind the player's
facing, beyond any light's reach, and fully inside the blackout advance on
a slow cadence and force-rebuild in place, hidden by the dark.

**Fixture decay and flicker.** The `institution_age` field steers the
fabric's dead-light ratio (young wings ~1.15x survival, ancient wings
~0.6x). A deterministic share of warm panels carries a `flicker_mode`
authored at light collection (tired-ballast shimmer, dying tubes); the
engine pre-flickers intensity CPU-side once per frame, like flares, so all
renderers agree and no shader needs a clock.

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
`static/worker.js` — each a second instance of the same wasm module). The
pure `GenerationWorkerPreference` policy accepts `?workers=auto|0|1..4`:
`auto` reserves one reported hardware thread and caps the pool at four,
explicit counts are clamped to reported hardware and the four-worker product
cap, while `0` selects the synchronous in-thread `LocalChunkSource`. That
local source is also the automatic fallback if worker startup fails. These workers parallelize chunk
generation, lighting, meshes, SVO serialization, and collision extraction;
they do **not** parallelize any renderer, including the CPU splatter. Either
way, the resident set
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

`?voxel_size=` overrides the fine voxel edge independently of the streaming
profile. `GeneratorConfig::try_with_voxel_size` is the single validator used
by the browser and every worker: the value must be finite and positive, tile
the chunk edge at both fine and progressive half-resolution LOD, fit the
renderer-wide depth-eight SVO contract, and remain
inside the measured dense-grid budget including the lateral halo. The UI
offers 0.4, 0.2, 0.1, and 0.05 u. The 10 u profile rejects 0.4 u because its
25-cell axis cannot be halved exactly; the 20 u profile rejects 0.05 u because
it would require depth nine. Every chunk payload transports its exact voxel size
and depth, so neither renderer reverse-engineers them from padded bounds.

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

- `cargo test --workspace` covers entities, generation (including the
  macro-graph snapshot, red-room separation, stair-flight, and arch-seam
  invariants), lighting equations and bake invariants, octree build/serialize,
  plus the front end's player physics, sliding collision, streaming policy,
  atlas rebasing, input mapping, and the resolution governor.
- Renderer/chunk-source **test doubles** verify the engine's contract with
  its ports (upload counts, draw-table sizes) rather than pixels.
- Playwright GPU references require visible panel emission and lit room
  geometry, exercise exact raymarched occlusion, and compare every decoded
  pixel when correctness-preserving traversal/culling optimizations are
  toggled live over the same resident scene.

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
