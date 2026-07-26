//! Web Worker chunk source — the multithreaded [`ChunkSourcePort`].
//!
//! Each worker (`static/worker.js`) is a second instance of this same wasm
//! module running on its own OS thread. The main thread posts small
//! `{gen, requestId, ox, oz, level, lod, artifacts, reality}` messages
//! round-robin. Workers always generate, bake, build the collision SVO, and
//! extract collision; the artifact bitfield requests only the active
//! renderer's mesh, face, or serialized-node products. One encoded byte
//! buffer returns through `adapters::chunk_codec`; GPU resources remain owned
//! by the main-thread render driver. Healthy pools keep generation off the
//! frame loop. If every worker fatally fails, the driver uses its same-thread
//! source as a correctness fallback instead of leaving the world pending.
//!
//! Stale-result rejection lives in the engine (`install_completed`), keyed
//! by the echoed request. This driver owns transport failure detection and
//! reports exact failed requests through the same application port.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{ArrayBuffer, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, Worker, WorkerOptions, WorkerType};

use crate::adapters::chunk_codec::decode_chunk_payload;
use crate::adapters::local_chunk_source::LocalChunkSource;
use crate::application::generation_worker_policy::MAX_GENERATION_WORKERS;
use crate::application::ports::{
    ChunkPayload, ChunkRequest, ChunkSourcePort, CompletedChunk, RenderArtifactNeeds,
};
use crate::drivers::generation_worker_requests::GenerationWorkerRequests;
use vackrooms::domain::entities::anomaly::RealitySnapshot;
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;

pub struct WorkerChunkSource {
    workers: Vec<Worker>,
    next_worker: usize,
    completed: Rc<RefCell<Vec<CompletedChunk>>>,
    failed: Rc<RefCell<Vec<ChunkRequest>>>,
    requests: Rc<RefCell<GenerationWorkerRequests>>,
    /// Keeps browser callbacks alive for the workers' lifetime.
    _message_handlers: Vec<Closure<dyn FnMut(MessageEvent)>>,
    _error_handlers: Vec<Closure<dyn FnMut(web_sys::Event)>>,
    /// Same-thread source for the synchronous [`ChunkSourcePort::load`]
    /// contract and as a last-resort recovery if every worker has failed.
    fallback: LocalChunkSource<SimpleNoiseProvider>,
    reported_emergency_fallback: bool,
}

impl WorkerChunkSource {
    /// Spawns the pool and hands every worker the URL query, so main thread
    /// and workers derive the identical generator configuration.
    pub fn new(query: &str, default_seed: u32, pool_size: usize) -> Result<Self, JsValue> {
        if !(1..=usize::from(MAX_GENERATION_WORKERS)).contains(&pool_size) {
            return Err(JsValue::from_str("generation worker pool size must be 1..=32"));
        }

        let completed: Rc<RefCell<Vec<CompletedChunk>>> = Rc::new(RefCell::new(Vec::new()));
        let failed: Rc<RefCell<Vec<ChunkRequest>>> = Rc::new(RefCell::new(Vec::new()));
        let requests = Rc::new(RefCell::new(GenerationWorkerRequests::new(pool_size)));
        // The single-file bundle (scripts/build-single-file.mjs) has no
        // sibling worker.js on disk; it publishes a blob: URL for the inlined
        // worker script on the global scope instead.
        let script_url = Reflect::get(&js_sys::global(), &"__VACKROOMS_WORKER_URL__".into())
            .ok()
            .and_then(|value| value.as_string())
            .unwrap_or_else(|| "worker.js".to_string());
        let mut workers = Vec::with_capacity(pool_size);
        let mut message_handlers = Vec::with_capacity(pool_size);
        let mut error_handlers = Vec::with_capacity(pool_size);
        for worker_index in 0..pool_size {
            let options = WorkerOptions::new();
            options.set_type(WorkerType::Module);
            // Relative (not `/worker.js`): resolved against the document's
            // URL, so this still finds the sibling file when the site is
            // served from a subpath (e.g. a GitHub Pages project page at
            // `<user>.github.io/<repo>/`) instead of the domain root.
            let worker = Worker::new_with_options(&script_url, &options)?;

            let completed_sink = completed.clone();
            let failed_sink = failed.clone();
            let message_requests = requests.clone();
            let worker_for_fatal_reply = worker.clone();
            let handler = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                match parse_worker_reply(&event) {
                    WorkerReply::Completed(done) => {
                        if message_requests
                            .borrow_mut()
                            .finish_exact(worker_index, &done.request)
                        {
                            completed_sink.borrow_mut().push(done);
                        }
                    }
                    WorkerReply::Failed(echoed) => {
                        if let Some(failed_request) = message_requests
                            .borrow_mut()
                            .fail_exact(worker_index, &echoed)
                        {
                            failed_sink.borrow_mut().push(failed_request);
                        }
                    }
                    WorkerReply::Fatal => {
                        let abandoned = message_requests
                            .borrow_mut()
                            .disable_and_drain(worker_index);
                        failed_sink.borrow_mut().extend(abandoned);
                        worker_for_fatal_reply.terminate();
                    }
                    WorkerReply::Malformed => {
                        if let Some(failed_request) =
                            message_requests.borrow_mut().fail_oldest(worker_index)
                        {
                            failed_sink.borrow_mut().push(failed_request);
                        }
                    }
                }
            });
            worker.set_onmessage(Some(handler.as_ref().unchecked_ref()));

            let failed_sink = failed.clone();
            let error_requests = requests.clone();
            let worker_for_error = worker.clone();
            let error_handler = Closure::<dyn FnMut(web_sys::Event)>::new(move |_event| {
                let abandoned = error_requests.borrow_mut().disable_and_drain(worker_index);
                failed_sink.borrow_mut().extend(abandoned);
                worker_for_error.terminate();
            });
            worker.set_onerror(Some(error_handler.as_ref().unchecked_ref()));

            let init = js_sys::Object::new();
            Reflect::set(&init, &"type".into(), &"init".into())?;
            Reflect::set(&init, &"query".into(), &query.into())?;
            Reflect::set(&init, &"seed".into(), &(default_seed as f64).into())?;
            worker.post_message(&init)?;

            workers.push(worker);
            message_handlers.push(handler);
            error_handlers.push(error_handler);
        }

        let (seed, config) =
            crate::adapters::query_config::generator_setup_from_query(query, default_seed);
        Ok(Self {
            workers,
            next_worker: 0,
            completed,
            failed,
            requests,
            _message_handlers: message_handlers,
            _error_handlers: error_handlers,
            fallback: LocalChunkSource::new(SimpleNoiseProvider::new(), seed, config),
            reported_emergency_fallback: false,
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

    fn load_with_reality(
        &self,
        origin_x: f32,
        origin_z: f32,
        level: u32,
        lod: u8,
        reality: &RealitySnapshot,
    ) -> ChunkPayload {
        self.fallback
            .load_with_reality(origin_x, origin_z, level, lod, reality)
    }

    fn load_with_artifacts(
        &self,
        origin_x: f32,
        origin_z: f32,
        level: u32,
        lod: u8,
        reality: &RealitySnapshot,
        artifacts: RenderArtifactNeeds,
    ) -> ChunkPayload {
        self.fallback
            .load_with_artifacts(origin_x, origin_z, level, lod, reality, artifacts)
    }

    fn is_async(&self) -> bool {
        true
    }

    fn max_concurrent_requests(&self) -> usize {
        self.pool_size()
    }

    fn request(&mut self, request: ChunkRequest) {
        let worker_index = self.requests.borrow().next_accepting(self.next_worker);
        let Some(worker_index) = worker_index else {
            // Fatal worker startup/runtime failures must not strand streaming.
            // This path is deliberately exceptional: the healthy path always
            // generates off-thread, while an exhausted pool favors a complete
            // world over a permanently pending chunk.
            if !self.reported_emergency_fallback {
                web_sys::console::warn_1(
                    &"generation worker pool failed; using same-thread recovery".into(),
                );
                self.reported_emergency_fallback = true;
            }
            let payload = self.fallback.load_with_artifacts(
                request.origin_x,
                request.origin_z,
                request.level,
                request.lod,
                &request.reality,
                request.artifacts,
            );
            self.completed
                .borrow_mut()
                .push(CompletedChunk { request, payload });
            return;
        };

        let message = js_sys::Object::new();
        let set = |k: &str, v: JsValue| {
            let _ = Reflect::set(&message, &k.into(), &v);
        };
        set("type", "gen".into());
        set("requestId", (request.request_id as f64).into());
        set("ox", (request.origin_x as f64).into());
        set("oz", (request.origin_z as f64).into());
        set("level", (request.level as f64).into());
        set("lod", (request.lod as f64).into());
        set("artifacts", (request.artifacts.bits() as f64).into());
        let reality_words = request.reality.to_words();
        let reality = js_sys::Uint32Array::from(reality_words.as_slice());
        set("reality", reality.into());
        let worker = &self.workers[worker_index];
        self.next_worker = (worker_index + 1) % self.workers.len();
        self.requests
            .borrow_mut()
            .assign(worker_index, request.clone());
        if let Err(error) = worker.post_message(&message) {
            // A synchronous send failure means this worker can no longer be
            // trusted with its already queued requests. Retire all of them;
            // the engine will validate their ids and retry on healthy slots.
            let abandoned = self.requests.borrow_mut().disable_and_drain(worker_index);
            self.failed.borrow_mut().extend(abandoned);
            worker.terminate();
            web_sys::console::warn_2(
                &"generation worker postMessage failed; disabling worker:".into(),
                &error,
            );
        }
    }

    fn poll_completed(&mut self) -> Vec<CompletedChunk> {
        std::mem::take(&mut *self.completed.borrow_mut())
    }

    fn poll_failed_requests(&mut self) -> Vec<ChunkRequest> {
        std::mem::take(&mut *self.failed.borrow_mut())
    }
}

enum WorkerReply {
    Completed(CompletedChunk),
    Failed(ChunkRequest),
    Fatal,
    Malformed,
}

/// Decodes the deliberately small worker protocol. Malformed replies are a
/// first-class outcome: the worker queue is serial, so the caller can fail its
/// oldest assigned request instead of leaving that chunk pending forever.
fn parse_worker_reply(event: &MessageEvent) -> WorkerReply {
    let data = event.data();
    let message_type = Reflect::get(&data, &"type".into())
        .ok()
        .and_then(|value| value.as_string());
    match message_type.as_deref() {
        Some("done") => parse_request(&data)
            .and_then(|request| {
                parse_payload(&data).map(|payload| CompletedChunk { request, payload })
            })
            .map(WorkerReply::Completed)
            .unwrap_or(WorkerReply::Malformed),
        Some("failed") => parse_request(&data)
            .map(WorkerReply::Failed)
            .unwrap_or(WorkerReply::Malformed),
        Some("fatal") => WorkerReply::Fatal,
        _ => WorkerReply::Malformed,
    }
}

fn parse_request(data: &JsValue) -> Option<ChunkRequest> {
    let field = |name: &str| Reflect::get(data, &name.into()).ok();
    Some(ChunkRequest {
        request_id: exact_u32(field("requestId")?)?,
        origin_x: finite_f32(field("ox")?)?,
        origin_z: finite_f32(field("oz")?)?,
        level: exact_u32(field("level")?)?,
        lod: u8::try_from(exact_u32(field("lod")?)?).ok()?,
        artifacts: RenderArtifactNeeds::from_bits(
            u8::try_from(exact_u32(field("artifacts")?)?).ok()?,
        )?,
        reality: {
            let encoded = field("reality")?;
            let words = encoded.dyn_into::<js_sys::Uint32Array>().ok()?.to_vec();
            RealitySnapshot::from_words(&words).ok()?
        },
    })
}

fn parse_payload(data: &JsValue) -> Option<ChunkPayload> {
    let buffer: ArrayBuffer = Reflect::get(data, &"buf".into()).ok()?.dyn_into().ok()?;
    let bytes = Uint8Array::new(&buffer).to_vec();
    decode_chunk_payload(&bytes)
}

fn exact_u32(value: JsValue) -> Option<u32> {
    let number = value.as_f64()?;
    (number.is_finite() && number.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&number))
        .then_some(number as u32)
}

fn finite_f32(value: JsValue) -> Option<f32> {
    let number = value.as_f64()?;
    let narrowed = number as f32;
    (number.is_finite() && narrowed.is_finite()).then_some(narrowed)
}
