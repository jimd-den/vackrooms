//! Inward-facing interfaces required by application use cases.
//!
//! Use cases define only the capability they need. Adapters and drivers own
//! concrete noise, visual-palette, telemetry, and clock implementations.

use crate::domain::entities::position::Position;

/// `NoiseProvider` is a pure interface for deterministic continuous noise.
/// We use this to evaluate the Macro Field (Drift, Density, Motif Selection).
pub trait NoiseProvider {
    /// Returns a noise value, ideally in the range [-1.0, 1.0].
    /// The function must be deterministic given the same seed and position.
    fn evaluate_2d(&self, seed: u32, position: Position) -> f32;
}

/// Visual lookup required by use cases that prepare renderer-facing artifacts.
/// The policy lives in an outer adapter; core material ids remain presentation
/// agnostic and tests can inject a tiny deterministic palette.
pub trait MaterialPalette {
    fn color(&self, material: u8) -> u32;
}

/// `TelemetryPort` abstracts wall-clock access and log emission so use cases
/// never touch `std::time` or stdout directly. Both are frameworks concerns,
/// and `SystemTime::now()` panics on `wasm32-unknown-unknown`.
pub trait TelemetryPort {
    /// Monotonic-ish timestamp in microseconds from an arbitrary epoch.
    /// Only ever used to compute durations, never absolute dates.
    fn now_micros(&self) -> u64;
    /// Emits one log line to whatever sink the outer layer provides.
    fn log(&self, message: &str);
}

/// Silent TelemetryPort used by default and in tests.
pub struct NullTelemetry;

impl TelemetryPort for NullTelemetry {
    fn now_micros(&self) -> u64 {
        0
    }
    fn log(&self, _message: &str) {}
}

/// Shared instance so constructors can hand out `&'static dyn TelemetryPort`
/// without forcing callers to own a telemetry object.
pub static NULL_TELEMETRY: NullTelemetry = NullTelemetry;
