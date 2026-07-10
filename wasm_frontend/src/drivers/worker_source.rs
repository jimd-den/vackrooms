//! Web Worker chunk source — the multithreaded [`ChunkSourcePort`].
//!
//! Each worker (`static/worker.js`) is a second instance of this same wasm
//! module running on its own OS thread. The main thread posts small
//! `{gen, ox, oz, level, lod}` messages round-robin; workers run the full
//! chunk pipeline (generation, BFS lighting, greedy mesh + face instances,
//! SVO build/serialize, collision) and transfer one encoded byte buffer
//! back (`adapters::chunk_codec`). No WebGL object ever leaves the main
//! thread, and the frame loop never blocks on generation.
//!
//! Stale-result rejection lives in the engine (`install_completed`), keyed
//! by the echoed request; this driver only moves bytes.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{ArrayBuffer, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, Worker, WorkerOptions, WorkerType};

use crate::adapters::chunk_codec::decode_chunk_payload;
use crate::adapters::local_chunk_source::LocalChunkSource;
use crate::application::ports::{
    ChunkPayload, ChunkRequest, ChunkSourcePort, CompletedChunk,
};
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;

pub struct WorkerChunkSource {
    workers: Vec<Worker>,
    next_worker: usize,
    completed: Rc<RefCell<Vec<CompletedChunk>>>,
    /// Keeps the onmessage closures alive for the workers' lifetime.
    _handlers: Vec<Closure<dyn FnMut(MessageEvent)>>,
    /// Same-thread source for the synchronous [`ChunkSourcePort::load`]
    /// contract (the engine never uses it while `is_async` is true).
    fallback: LocalChunkSource<SimpleNoiseProvider>,
}

impl WorkerChunkSource {
    /// Spawns the pool and hands every worker the URL query, so main thread
    /// and workers derive the identical generator configuration.
    pub fn new(query: &str, default_seed: u32) -> Result<Self, JsValue> {
        let concurrency = web_sys::window()
            .map(|w| w.navigator().hardware_concurrency())
            .unwrap_or(2.0);
        // Leave one core for the render thread; a Pi 4 (4 cores) gets 3.
        let pool_size = ((concurrency as i32) - 1).clamp(1, 4) as usize;

        let completed: Rc<RefCell<Vec<CompletedChunk>>> = Rc::new(RefCell::new(Vec::new()));
        let mut workers = Vec::with_capacity(pool_size);
        let mut handlers = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let options = WorkerOptions::new();
            options.set_type(WorkerType::Module);
            let worker = Worker::new_with_options("/worker.js", &options)?;

            let sink = completed.clone();
            let handler = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                if let Some(done) = parse_done_message(&event) {
                    sink.borrow_mut().push(done);
                }
            });
            worker.set_onmessage(Some(handler.as_ref().unchecked_ref()));

            let init = js_sys::Object::new();
            Reflect::set(&init, &"type".into(), &"init".into())?;
            Reflect::set(&init, &"query".into(), &query.into())?;
            Reflect::set(&init, &"seed".into(), &(default_seed as f64).into())?;
            worker.post_message(&init)?;

            workers.push(worker);
            handlers.push(handler);
        }

        let (seed, config) =
            crate::adapters::query_config::generator_setup_from_query(query, default_seed);
        Ok(Self {
            workers,
            next_worker: 0,
            completed,
            _handlers: handlers,
            fallback: LocalChunkSource::new(SimpleNoiseProvider::new(), seed, config),
        })
    }

    pub fn pool_size(&self) -> usize {
        self.workers.len()
    }
}

impl ChunkSourcePort for WorkerChunkSource {
    fn load(&self, origin_x: f32, origin_z: f32, level: u32, lod: u8) -> ChunkPayload {
        self.fallback.load(origin_x, origin_z, level, lod)
    }

    fn is_async(&self) -> bool {
        true
    }

    fn request(&mut self, request: ChunkRequest) {
        let message = js_sys::Object::new();
        let set = |k: &str, v: JsValue| {
            let _ = Reflect::set(&message, &k.into(), &v);
        };
        set("type", "gen".into());
        set("ox", (request.origin_x as f64).into());
        set("oz", (request.origin_z as f64).into());
        set("level", (request.level as f64).into());
        set("lod", (request.lod as f64).into());
        let worker = &self.workers[self.next_worker % self.workers.len()];
        self.next_worker = self.next_worker.wrapping_add(1);
        let _ = worker.post_message(&message);
    }

    fn poll_completed(&mut self) -> Vec<CompletedChunk> {
        std::mem::take(&mut *self.completed.borrow_mut())
    }
}

/// Decodes one `{type:"done", ...}` worker reply. Any malformed message is
/// dropped; the engine re-requests the chunk on a later tick because its
/// pending entry is only cleared by a matching completion or eviction.
fn parse_done_message(event: &MessageEvent) -> Option<CompletedChunk> {
    let data = event.data();
    let field = |name: &str| Reflect::get(&data, &name.into()).ok();
    if field("type")?.as_string()? != "done" {
        return None;
    }
    let request = ChunkRequest {
        origin_x: field("ox")?.as_f64()? as f32,
        origin_z: field("oz")?.as_f64()? as f32,
        level: field("level")?.as_f64()? as u32,
        lod: field("lod")?.as_f64()? as u8,
    };
    let buffer: ArrayBuffer = field("buf")?.dyn_into().ok()?;
    let bytes = Uint8Array::new(&buffer).to_vec();
    let payload = decode_chunk_payload(&bytes)?;
    Some(CompletedChunk { request, payload })
}
