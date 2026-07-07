# Vackrooms

A voxel-only rendering engine in Rust, structured with documented Clean
Architecture and compiled to WebAssembly. Procedural "backrooms" world
generation, BFS lighting, sparse voxel octree construction, chunk streaming,
sliding player collision and a WebGL2 hybrid SVO raymarcher all run inside
one ~140 KB wasm module — built to hold up on low-spec hardware.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design (layer map, ports,
data flow, texel encodings, streaming and rendering pipeline).

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

Click to capture the mouse; WASD to move, ESC to release.

- `http://localhost:8080/` — wasm engine (low-spec profile: 3×3 chunk
  streaming, 0.2 u voxels, adaptive resolution)
- `http://localhost:8080/?spec=high` — high-spec profile (5×5 chunks,
  0.1 u voxels)
- `http://localhost:8080/legacy` — the preserved original Three.js/JS client,
  which fetches chunks from the server's `/maze` and `/octree` endpoints
  instead of generating them in the browser

## Tests

```sh
cargo test --workspace
```

The application layers are browser-free by construction, so player physics,
collision, streaming, atlas assembly and input mapping are all covered by
native unit tests (40 tests at the time of writing).

## Workspace layout

| Path | What it is |
|---|---|
| `src/` | `vackrooms` core: entities, use cases, adapters + native dev server |
| `wasm_frontend/` | Browser client: application / adapters / drivers (wasm-only) |
| `static/` | Thin HTML shell, wasm bundle output (`pkg/`), legacy client |
| `wasm_raycaster/` | Legacy CPU raycaster experiment (reference only) |
