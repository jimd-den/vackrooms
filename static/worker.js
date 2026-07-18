// Generation worker: a second instance of the same wasm module, producing
// renderer-selected chunk artifacts off the main thread. The handler is
// registered synchronously at module scope so no message can be lost while
// the wasm module is still instantiating; work is serialized through a
// promise chain.
//
// Protocol (main -> worker):
//   { type: "init", query, seed }        one-time generator setup
//   { type: "gen", requestId, ox, oz, level, lod, artifacts, reality }
//                   one chunk order; artifacts is a u8 bitfield and reality
//                   is Uint32Array
// (worker -> main):
//   { type: "done", requestId, ox, oz, level, lod, artifacts, reality, ms, buf }
//                                   buf transferred, encoded by chunk_codec
//   { type: "failed", requestId, ox, oz, level, lod, artifacts, reality, message }
//                                   one request failed and may be retried
//   { type: "fatal", message }      worker initialization failed
import init, { worker_init, worker_generate } from "./pkg/wasm_frontend.js";

let wasmReady = null;
function ensureWasm() {
  if (!wasmReady) wasmReady = init();
  return wasmReady;
}

let queue = Promise.resolve();

function errorMessage(error) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function reportFailure(message, error) {
  const description = errorMessage(error);
  console.error("[WORKER] generation failure:", description);
  try {
    if (message?.type === "gen") {
      postMessage({
        type: "failed",
        requestId: message.requestId,
        ox: message.ox,
        oz: message.oz,
        level: message.level,
        lod: message.lod,
        artifacts: message.artifacts,
        reality: new Uint32Array(message.reality),
        message: description,
      });
    } else {
      postMessage({ type: "fatal", message: description });
    }
  } catch (reportError) {
    // The browser's Worker error event is the final recovery path when even
    // the structured failure message cannot be cloned or delivered.
    console.error("[WORKER] could not report generation failure:", reportError);
  }
}

async function processMessage(message) {
  await ensureWasm();
  if (message.type === "init") {
    worker_init(message.query, message.seed >>> 0);
    return;
  }
  if (message.type !== "gen") return;

  const t0 = performance.now();
  const reality = new Uint32Array(message.reality);
  const bytes = worker_generate(
    message.requestId >>> 0,
    message.ox,
    message.oz,
    message.level,
    message.lod,
    message.artifacts >>> 0,
    reality
  );
  const ms = performance.now() - t0;
  console.debug(
    `[WORKER] chunk (${message.ox}, ${message.oz}) lod ${message.lod} generated in ${ms.toFixed(1)} ms`
  );
  postMessage(
    {
      type: "done",
      requestId: message.requestId,
      ox: message.ox,
      oz: message.oz,
      level: message.level,
      lod: message.lod,
      artifacts: message.artifacts,
      reality,
      ms,
      buf: bytes.buffer,
    },
    [bytes.buffer]
  );
}

onmessage = (event) => {
  const message = event.data;
  // Catch each queue item independently. A failed request must not leave the
  // promise chain rejected, which would silently skip every later request.
  queue = queue
    .then(() => processMessage(message))
    .catch((error) => reportFailure(message, error));
};
