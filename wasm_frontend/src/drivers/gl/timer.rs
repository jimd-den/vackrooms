//! Asynchronous GPU frame timing via `EXT_disjoint_timer_query_webgl2`.
//!
//! One query is in flight at a time; results are polled (never blocked on)
//! at the top of the next frames, so the HUD's GPU number lags a frame or
//! two but the pipeline never stalls. Toggleable (`RenderToggles::gpu_timer`)
//! because some drivers serialize on timer queries.

use js_sys::Object;
use web_sys::{WebGl2RenderingContext as Gl, WebGlQuery};

/// Extension constants absent from WebGL2 core.
const TIME_ELAPSED_EXT: u32 = 0x88BF;
const GPU_DISJOINT_EXT: u32 = 0x8FBB;

pub struct GpuFrameTimer {
    extension: Option<Object>,
    pending: Option<WebGlQuery>,
    last_ms: Option<f32>,
}

impl GpuFrameTimer {
    /// Probes for the extension; a `None` extension makes every method a
    /// no-op so callers never branch on support.
    pub fn new(gl: &Gl) -> Self {
        Self {
            extension: gl
                .get_extension("EXT_disjoint_timer_query_webgl2")
                .ok()
                .flatten(),
            pending: None,
            last_ms: None,
        }
    }

    pub fn last_ms(&self) -> Option<f32> {
        self.last_ms
    }

    /// Collects a finished query, if any. Disjoint intervals (GPU clock
    /// disturbance) discard the sample.
    pub fn poll(&mut self, gl: &Gl) {
        let Some(query) = self.pending.as_ref() else {
            return;
        };
        let available = gl
            .get_query_parameter(query, Gl::QUERY_RESULT_AVAILABLE)
            .as_bool()
            .unwrap_or(false);
        let disjoint = self.extension.is_some()
            && gl
                .get_parameter(GPU_DISJOINT_EXT)
                .ok()
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
        if !available && !disjoint {
            return;
        }
        let query = self.pending.take().expect("pending timer exists");
        if available && !disjoint {
            self.last_ms = gl
                .get_query_parameter(&query, Gl::QUERY_RESULT)
                .as_f64()
                .map(|nanos| (nanos as f32) * 1.0e-6);
        }
        gl.delete_query(Some(&query));
    }

    /// Starts timing this frame. Returns whether a query was issued (pass
    /// it to [`Self::end`]). `enabled` comes from the toggle switchboard.
    pub fn begin(&mut self, gl: &Gl, enabled: bool) -> bool {
        if !enabled {
            // The HUD must not present an old sample as current after timing
            // has been switched off. An in-flight query is still collected
            // by `poll` and then discarded from view on this frame.
            self.last_ms = None;
            return false;
        }
        if self.extension.is_none() || self.pending.is_some() {
            return false;
        }
        let Some(query) = gl.create_query() else {
            return false;
        };
        gl.begin_query(TIME_ELAPSED_EXT, &query);
        self.pending = Some(query);
        true
    }

    pub fn end(&self, gl: &Gl, started: bool) {
        if started {
            gl.end_query(TIME_ELAPSED_EXT);
        }
    }
}
