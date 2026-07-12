// Generation worker: a second instance of the same wasm module, running the
// full chunk pipeline off the main thread. The onmessage handler is
// registered synchronously at module scope so no message can be lost while
// the wasm module is still instantiating; work is serialized through a
// promise chain.
//
// Protocol (main -> worker):
//   { type: "init", query, seed }        one-time generator setup
//   { type: "gen", requestId, ox, oz, level, lod, reality } one chunk order;
//                                                        reality is Uint32Array
// (worker -> main):
//   { type: "done", requestId, ox, oz, level, lod, reality, ms, buf }
//                                   buf transferred, encoded by chunk_codec
import init, { worker_init, worker_generate } from "./pkg/wasm_frontend.js";

let wasmReady = null;
function ensureWasm() {
  if (!wasmReady) wasmReady = init();
  return wasmReady;
}

let queue = Promise.resolve();
onmessage = (event) => {
  const m = event.data;
  queue = queue.then(async () => {
    await ensureWasm();
    if (m.type === "init") {
      worker_init(m.query, m.seed >>> 0);
    } else if (m.type === "gen") {
      const t0 = performance.now();
      const reality = new Uint32Array(m.reality);
      const bytes = worker_generate(
        m.requestId >>> 0,
        m.ox,
        m.oz,
        m.level,
        m.lod,
        reality
      );
      const ms = performance.now() - t0;
      console.debug(
        `[WORKER] chunk (${m.ox}, ${m.oz}) lod ${m.lod} generated in ${ms.toFixed(1)} ms`
      );
      postMessage(
        {
          type: "done",
          requestId: m.requestId,
          ox: m.ox,
          oz: m.oz,
          level: m.level,
          lod: m.lod,
          reality,
          ms,
          buf: bytes.buffer,
        },
        [bytes.buffer]
      );
    }
  });
};
