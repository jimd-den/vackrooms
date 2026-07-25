# Vackrooms — Architecture

A voxel-only rendering engine built on **Clean Architecture**, targeting
low-spec hardware. The whole engine — procedural generation, lighting, sparse
voxel octree (SVO) construction, chunk streaming, player physics, and
rendering — compiles to a single WebAssembly module. The browser
runs it; a zero-dependency native HTTP server merely serves the files (or, on
GitHub Pages, static files alone — the wasm engine generates chunks entirely
client-side and needs no server).

## Rendering family

The engine targets **WebGL2** as its primary production rendering API, prioritizing reliable hardware compatibility, static probe volume lighting, and selective dynamic shadows.

| Strategy | Role & Purpose | Target Profile |
|---|---|---|
| `surface` *(default)* | **Primary High-Quality Renderer** | Packed greedy meshes, per-chunk analytic light ranges, hero-light PCF shadows, dynamic dithered fog, and 3D probe volume sampling (`L_baked`). |
| `splat` | **Low-Spec / Dense Geometry** | Instanced face splats with quad expansion; preserves flat retro voxel aesthetic and reads shared 3D probe volume. |
| `raymarch` | **Diagnostic & Reference** | Fullscreen SVO raymarcher for volumetric inspection, special effects, and reference comparison. |
| `cpu` | **Deterministic Fallback** | Software SVO rasterizer fallback for low-spec environments without rich global illumination. |

Initialization failures are reported to the loading HUD gracefully without silent algorithm fallback.

### Renderer module boundaries

| Module | Responsibility |
|---|---|
| `drivers/browser.rs` | async composition root, HUD/settings projection, and rAF orchestration |
| `drivers/browser/input_bindings.rs` | DOM listener lifetimes and translation into the platform-free input adapter |
| `drivers/webgpu/browser_context.rs` | browser adapter/device/surface acquisition and resize/present lifecycle |
| `drivers/webgpu/renderer.rs` | `RendererPort` adapter and strategy composition |
| `drivers/webgpu/config.rs` | target-independent renderer kinds, quality profiles, budgets, optimization switches, and artifact requirements |
| `drivers/webgpu/frame_resources.rs` | shared frame uniforms and light storage |
| `drivers/webgpu/gpu_types.rs` | explicit, aligned CPU-to-WGSL records |
| `drivers/webgpu/pipelines/` | one focused cache/pass implementation per strategy plus shared visibility and `raster_shadow` resource/math contracts |
| `drivers/webgpu/shaders/` | small WGSL programs selected by `shader.rs` |
| `adapters/cpu_splatter/` | browser-free reference rasterizer, traversal, shading, and typed CPU settings |

This split keeps DOM lifetime, GPU resource lifetime, pass encoding, and
renderer policy out of a single god object. Adding a strategy means adding a
pipeline and typed profile, then composing it behind the existing port.

### Work ownership

Procedural generation is CPU-authoritative. World graphs, reality snapshots,
seeded noise, lighting, collision, and SVO construction must not vary with GPU
model, driver, or scheduling order. Workers may parallelize independent chunk
requests, but the same request always produces the same artifact bytes.

The GPU owns renderer-derived work that cannot change world semantics:

- surface vertices are decoded by vertex pulling, avoiding an expanded CPU
  vertex copy;
- splat faces become quad corners in the vertex shader, avoiding CPU-generated
  presentation vertices;
- the raster hero shadow reuses Surface indices or expands the Splat face
  buffer directly, so visibility does not broaden either artifact request;
- each compact supply-label point becomes a six-vertex, camera-facing
  billboard in WGSL and shares the raster strategies' depth target;
- raymarch fragments traverse the compressed SVO directly, avoiding a dense
  persistent voxel field;
- the CPU reference path still uses the same WebGPU presentation lifecycle.

`BuildOctreeUseCase` also prunes any recursive cube whose minimum is already
outside a source-grid dimension. Such cubes can only collapse to the canonical
air leaf, so this skips work without changing arena indices or serialized
bytes. A non-power-of-two fixture compares the exact pruned and exhaustive
node arenas and every addressable lookup.

### Configurable optimizations

`RendererProfile` separates bounded low/high quality budgets from
`RenderToggles`. The browser snapshots toggles once per frame; inner passes do
not read DOM globals. Each strategy consumes only switches it implements:

| Strategy | Explicit work controls |
|---|---|
| surface | chunk-sphere rejection, optional baked diffuse fill, profile-sized hero shadow map (128²/256² with one/four taps), and adaptive backing resolution |
| splat | chunk/cell rejection, near-to-far submission, optional baked diffuse fill, far-face budget, and the same profile-sized hero map using face-expanded casters |
| raymarch | stable versus near-to-far chunk selection, complete-AABB distance rejection before the resident cap, reference traversal versus empty-leaf skipping, one/four-sample finite-SVO direct-light visibility, maximum trace steps, and resident-chunk budget |
| CPU | hierarchical Z, near-to-far order, projected-size MIPs, deferred hidden-splat shading, ambient occlusion, range, and hard work caps |

Compact storage records, GPU vertex pulling/face expansion, fixed-function
depth, and the fullscreen-triangle passes are architectural choices rather
than hidden toggles. Output dithering remains configurable. GPU timestamp
collection is not implemented and is therefore not advertised as a switch.

## The dependency rule

Source dependencies point inward only. No inner layer names a browser, GPU,
socket, or clock.

```
┌────────────────────────────────────────────────────────────────────┐
│ FRAMEWORKS & DRIVERS                                               │
│   wasm_frontend/drivers   WebGPU, DOM events, rAF, HUD, console    │
│   src/main.rs             native HTTP server (std::net only)      │
│   src/frameworks_drivers  SimpleNoiseProvider, StdTelemetry        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ INTERFACE ADAPTERS                                           │  │
│  │   src/adapters            MaterialPalette, WebRenderer,      │  │
│  │                           OctreeGpuSerializer, VoxelMapper   │  │
│  │   wasm_frontend/adapters  InputCollector, LocalChunkSource   │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │ USE CASES (application business rules)                 │  │  │
│  │  │   src/use_cases           GenerateChunk, BuildOctree   │  │  │
│  │  │   src/domain/use_cases    Domain algorithms and maze   │  │  │
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
| `wasm_frontend` | Portable application/adapters/WebGPU pipelines plus the wasm browser shell | wasm32 and native headless Vulkan tests |
| `wasm_raycaster` | Legacy CPU raycaster experiment | kept for reference |

The platform boundary inside `wasm_frontend` is enforced mechanically. `web-sys`,
`js-sys`, and `wasm-bindgen` are wasm-only dependencies; browser context,
events, and rAF are `cfg(wasm32)`. Target-independent WebGPU configuration,
record layouts, shader assembly, and pipelines compile on the host so native
tests exercise the same code through Vulkan without admitting DOM types.

### Ports (dependency inversion boundaries)

| Port | Defined in (inner) | Implemented by (outer) |
|---|---|---|
| `NoiseProvider` | `src/use_cases/ports.rs` | `SimpleNoiseProvider` |
| `TelemetryPort` | `src/use_cases/ports.rs` | `StdTelemetry` (native stdout), `ConsoleTelemetry` (browser console), `NullTelemetry` (tests) |
| `RendererPort` | `wasm_frontend/src/application/ports.rs` | `WebGpuRenderer` with surface/splat/raymarch/CPU strategies; recording fakes in application tests |
| `ChunkSourcePort` | `wasm_frontend/src/application/ports.rs` | `LocalChunkSource` (synchronous, in-wasm generation), `WorkerChunkSource` (pooled Web Worker generation, default); an HTTP-fetching implementation would slot in without touching the engine |

## Data flow: from noise to pixel

The request and world state are authoritative inputs. `RenderArtifactNeeds`
describes which expensive presentation products a selected strategy consumes;
collision remains an application requirement, not a renderer preference.

```
SimpleNoiseProvider (driver)
      │ NoiseProvider port
      ▼
GenerateChunkArchitectureUseCase ──► haloed VoxelGrid
      │                                  │
      │                         bake_voxel_lighting
      ▼                                  ▼
crop authoritative grid ─────────► BuildOctreeUseCase (outside-grid pruning)
      │                                  │
      ├─ greedy mesh ────────────────────┼─► surface storage + index buffers
      ├─ compact face records ───────────┼─► splat storage buffer
      │                                  ├─► collision AABBs
      │                                  ▼
      │                         OctreeGpuSerializer
      │                                  │ four-u32 nodes, 1024-node rows
      │                                  ▼
      │                         merged/rebased SVO atlas
      │                                  ├─► raymarch storage buffer
      │                                  └─► CPU reference rasterizer
      ▼
analytic light records ────────────────► shared WebGPU frame storage
```

Collision geometry is derived from the same SVO built from the rendered voxel
source. Uniform solid subtrees become larger collision boxes without a second
geometry generator.

### SVO node encoding (four `u32` words per node)

| Lane | Internal node (`x == 0`) | Leaf node (`x == 1`) |
|---|---|---|
| x | 0 | 1 |
| y | `child_base_index` (children contiguous at +0..+7) | `voxel_type` |
| z | `child_mask` (bit *i* set = child *i* non-empty) | 24-bit `0xRRGGBB` color |
| w | 0 | scalar light 0–15, face-occlusion bits, and RGB light 0–15/channel |

Rows remain padded to 1024 nodes so atlas slots and partial row updates are
stable. The source of truth is `src/adapters/octree_gpu_serializer.rs`; the
WebGPU decoder is
`wasm_frontend/src/drivers/webgpu/shaders/raymarch.wgsl`.

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
warp, stretch, or rearrange itself." `RealitySnapshot` (RTY v5) carries one
drift epoch per 40 u fabric cell plus the mismanagement *strain* tier
(delirium 0–3). The epoch re-salts the fabric's decisions — wall dropout,
doorway direction/position/framing, dead lights — and re-deals each
corridor section's edge mouths, read at each deciding lattice cell's own
anchor, so walls rebuild whole. At tier 0 the binary-tree doorway rule
(hence global connectivity) holds at any epoch mix. Under strain the
guarantee erodes *by design*: warren cells brick over their guaranteed
doorway (~10 % per tier) into dead-end pockets, second doorways thin,
corridor mouths narrow and (tier ≥ 2) seal whole sections, supply cells
withhold a 15 % share per tier, and rare 0.8 u secret slips open in
otherwise doorless walls (tier ≥ 2) — the punished labyrinth grows secrets
alongside its dead ends. Corridor interiors, portals, assemblies, anomaly
identities, arch-anchor surroundings, and the spawn opening sequence never
drift or strain: the spine survives; hallway memory — and, mismanaged,
hallway mercy — does not.
The engine advances epochs two ways: territory abandoned beyond the
streaming footprint (plus hysteresis) drifts on departure, and *inside a
blackout* the shift runs in real time — cells wholly behind the player's
facing, beyond any light's reach, and fully inside the blackout advance on
a slow cadence and force-rebuild in place, hidden by the dark. Strain is
fed from both ends of resource mismanagement — dehydration bands
(underconsumption) and the excess meter of wasted supply value, i.e.
drinking or eating past a full reserve (overconsumption) — and the max of
the two also shrinks drift hysteresis and speeds the blackout's rear-shift
cadence.

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
The SVO atlas and collision world are rebuilt only on a resident-set
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
depth-eight SVO payload contract, and remain
inside the measured dense-grid budget including the lateral halo. The UI
offers 0.4, 0.2, 0.1, and 0.05 u. The 10 u profile rejects 0.4 u because its
25-cell axis cannot be halved exactly; the 20 u profile rejects 0.05 u because
it would require depth nine. Every chunk payload transports its exact voxel size
and depth, so no strategy reverse-engineers them from padded bounds.

The raymarch storage table is bounded at 25 chunks — exactly the 5×5 high-spec
worst case (`application::atlas::MAX_CHUNKS`).

## Player simulation

`application::player` ports the original client's tuning 1:1: exponential
friction (8 s⁻¹), acceleration 4 u/s² (terminal 0.5 u/s), mouse sensitivity
0.002 rad/px, pitch clamped short of ±90°. Collision is **axis-separated
sliding**: X and Z are moved and tested independently against the
`CollisionWorld`, so hitting a wall diagonally slides along it. `dt` is
clamped to 100 ms so a background-tab hitch cannot teleport the player
through a wall.

`application::body` is the embodied-telemetry layer on top: a pure,
deterministic model of session steps (stride 0.7 m over collision-resolved
walking only), short-term exertion, long-term fatigue with a rest loop, and
a target-BPM pulse with asymmetric smoothing. Its bounded movement factor
(≥ 0.35 before death) feeds back into `Player::step_with_effort`, scaling
acceleration — and therefore terminal speed — without touching collision.
`application::navigation` supplies pure route/bearing selection over the
resident `LevelExit`/gate/pit records: nearest-door prioritization, 0–359°
relative bearings, and deterministic anomaly focus for the diagnostic
aperture. Neither module invents targets — no resident record, no reading.
The browser driver stays a passive presenter of `HudStats`.

## Testing strategy

Ports make the world and application rules natively testable without a
browser or GPU:

- `cargo test --workspace` covers entities, generation (including the
  macro-graph snapshot, red-room separation, stair-flight, and arch-seam
  invariants), lighting equations and bake invariants, octree build/serialize,
  plus the front end's player physics, sliding collision, streaming policy,
  atlas rebasing, input mapping, and the resolution governor.
- Renderer/chunk-source **test doubles** verify the engine's contract with
  its ports (upload counts, draw-table sizes) rather than pixels.
- The `vulkan_renderers` integration suite creates surface-free targets with
  the production pipeline modules. CI forces wgpu's Vulkan backend through
  Mesa Lavapipe, enables validation layers, and fails rather than silently
  skipping when no Vulkan adapter is present. This covers pipeline creation,
  storage layouts, draw encoding, readback, and deterministic renderer
  contracts without browser screenshots.

## Native dev server

`src/main.rs` is a zero-dependency HTTP server used for local development. It
serves static files with correct MIME types (`application/wasm` is mandatory
for streaming instantiation) and retains `/maze` and `/octree` as diagnostic
JSON/binary endpoints. The wasm client needs no data endpoint. The former
Three.js client, its static asset, and the `/legacy` route are retired.

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
- Per-chunk AABB proxy draws with fragment-depth output to replace the
  fullscreen raymarch pass if resident chunk counts grow beyond 25.
- LOD: shallower SVO mip levels for distant chunks.
