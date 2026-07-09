/* tslint:disable */
/* eslint-disable */

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
 */
export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly set_face_weights: (a: number, b: number, c: number, d: number) => void;
    readonly set_fov: (a: number) => void;
    readonly set_doom_controls: (a: number) => void;
    readonly set_invert_y: (a: number) => void;
    readonly set_mouse_sensitivity: (a: number) => void;
    readonly set_render_scale: (a: number) => void;
    readonly start: () => void;
    readonly wasm_bindgen__convert__closures_____invoke__h42bad8c94dc1d8fe: (a: number, b: number, c: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1bbff226e3ed8404: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1bbff226e3ed8404_2: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1bbff226e3ed8404_3: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1bbff226e3ed8404_4: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hc38de1062546049f: (a: number, b: number) => void;
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
