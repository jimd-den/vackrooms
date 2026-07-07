//! Browser implementation of the core's `TelemetryPort`:
//! `performance.now()` for durations, `console.log` for output.

use vackrooms::use_cases::ports::TelemetryPort;
use wasm_bindgen::JsValue;

pub struct ConsoleTelemetry;

/// Shared instance handed to use cases as `&'static dyn TelemetryPort`.
pub static CONSOLE_TELEMETRY: ConsoleTelemetry = ConsoleTelemetry;

impl TelemetryPort for ConsoleTelemetry {
    fn now_micros(&self) -> u64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| (p.now() * 1000.0) as u64)
            .unwrap_or(0)
    }

    fn log(&self, message: &str) {
        web_sys::console::log_1(&JsValue::from_str(message));
    }
}
