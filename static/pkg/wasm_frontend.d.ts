/* tslint:disable */
/* eslint-disable */

export function get_blueprint_svg(seed: number, rx: number, rz: number, voxel_scale: number, size_world: number): string;

export function get_chunk_blueprint_svg(seed: number, chunk_x: number, chunk_z: number, voxel_scale: number, layer: string): string;

/**
 * Fast single-chunk voxel blueprint with semantic overlays.
 *
 * Why fast? This renders only one 50×50 (or 200×200 for high-spec) voxel
 * grid instead of stitching N×N chunks, so it completes in < 5 ms inside
 * WASM.  The browser can call this on every keypress without delay.
 *
 * Arguments (all passed from JavaScript):
 * * `seed`          – World seed.
 * * `chunk_x`       – Chunk origin X in world units (chunk_index * chunk_size).
 * * `chunk_z`       – Chunk origin Z in world units.
 * * `voxel_scale`   – Metres per voxel (0.2 default, 0.1 high-spec).
 * * `pixels_per_voxel` – SVG pixels per cell (8–12 is comfortable).
 * * `show_grid`     – Draw hairline grid lines.
 * * `show_semantics`– Draw corridor / room / door overlays.
 * * `show_ceiling`  – Include ceiling and light voxels.
 */
export function get_chunk_voxel_blueprint_svg(seed: number, chunk_x: number, chunk_z: number, voxel_scale: number, pixels_per_voxel: number, show_grid: boolean, show_semantics: boolean, show_ceiling: boolean): string;

/**
 * Debug payload for one generated chunk: a compact material slice plus the
 * plan objects that explain why those voxels were placed.
 */
export function get_debug_chunk_json(seed: number, chunk_x: number, chunk_z: number, voxel_scale: number): string;

/**
 * Compact geometry feed for the canvas debug map. This deliberately returns
 * planning primitives rather than SVG so the page can redraw cheaply while
 * panning, zooming, or changing overlays.
 */
export function get_debug_region_json(seed: number, region_x: number, region_z: number): string;

export function get_large_voxel_blueprint_svg(seed: number, rx_val: number, rz_val: number, voxel_scale: number, size_world: number, layer: string): string;

export function set_cpu_lod_cutoff(cutoff: number): void;

export function set_cpu_max_draw_distance(dist: number): void;

export function set_cpu_max_splat_half(half: number): void;

export function set_cpu_scale(scale: number): void;

export function set_cpu_shadows(mode: number): void;

export function set_doom_controls(enabled: boolean): void;

export function set_face_weights(top: number, bottom: number, x: number, z: number): void;

/**
 * Sets the vertical field of view in degrees (clamped to 40–110).
 */
export function set_fov(degrees: number): void;

export function set_invert_y(enabled: boolean): void;

/**
 * Mouse/touch look sensitivity multiplier (clamped to 0.1–5.0).
 */
export function set_mouse_sensitivity(multiplier: number): void;

/**
 * Forces the internal render resolution scale (0.25–1.0), or restores the
 * adaptive governor when `scale` is 0.
 */
export function set_render_scale(scale: number): void;

/**
 * Composition root. Runs automatically when the wasm module is
 * instantiated by `static/index.html`.
 *
 * `static/worker.js` instantiates this same module inside a Web Worker
 * with `#[wasm_bindgen(start)]` skipped (`init` is passed
 * `{ skip_start: true }` is not available for start fns, so instead the
 * worker checks for a missing DOM and bails out here).
 */
export function start(): void;

export function worker_generate(_request_id: number, origin_x: number, origin_z: number, level: number, lod: number, reality_words: Uint32Array): Uint8Array;

/**
 * `default_seed` must match the main thread's `WORLD_SEED`.
 */
export function worker_init(query: string, default_seed: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly get_blueprint_svg: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly get_chunk_blueprint_svg: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly get_chunk_voxel_blueprint_svg: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly get_debug_chunk_json: (a: number, b: number, c: number, d: number) => [number, number];
    readonly get_debug_region_json: (a: number, b: number, c: number) => [number, number];
    readonly get_large_voxel_blueprint_svg: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly set_doom_controls: (a: number) => void;
    readonly set_invert_y: (a: number) => void;
    readonly set_cpu_lod_cutoff: (a: number) => void;
    readonly set_cpu_max_draw_distance: (a: number) => void;
    readonly set_cpu_max_splat_half: (a: number) => void;
    readonly set_cpu_scale: (a: number) => void;
    readonly set_mouse_sensitivity: (a: number) => void;
    readonly set_render_scale: (a: number) => void;
    readonly set_cpu_shadows: (a: number) => void;
    readonly worker_generate: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly worker_init: (a: number, b: number, c: number) => void;
    readonly start: () => void;
    readonly set_face_weights: (a: number, b: number, c: number, d: number) => void;
    readonly set_fov: (a: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h4a98415c8f04dfea: (a: number, b: number, c: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h00d81ef20398ee11: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h00d81ef20398ee11_2: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h00d81ef20398ee11_3: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h00d81ef20398ee11_4: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h00d81ef20398ee11_5: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__ha60c758f5c18d619: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
