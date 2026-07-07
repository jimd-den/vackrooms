use std::time::{SystemTime, UNIX_EPOCH};

use crate::use_cases::ports::TelemetryPort;

/// Native (non-wasm) TelemetryPort adapter: stdout + `SystemTime`.
/// This is the Frameworks & Drivers implementation of the port defined by the
/// Use Case layer in `use_cases::ports`. The wasm front end provides its own
/// implementation backed by `performance.now()` and `console.log`.
pub struct StdTelemetry;

impl TelemetryPort for StdTelemetry {
    fn now_micros(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0)
    }

    fn log(&self, message: &str) {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        println!("[{}] {}", secs, message);
    }
}
