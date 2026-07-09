use crate::entities::models::Position;

/// The Ports layer implements the Dependency Inversion Principle (SOLID).
/// Instead of the Use Cases depending on a concrete external noise library or framework,
/// the Use Cases define the interface (Port) they need, and the external layers
/// (Adapters/Drivers) must implement this interface.

/// NoiseProvider is a pure interface for extracting deterministic continuous noise.
/// We use this to evaluate the Macro Field (Drift, Density, Motif Selection).
pub trait NoiseProvider {
    /// Returns a noise value, ideally in the range [-1.0, 1.0].
    /// The function must be deterministic given the same seed and position.
    fn evaluate_2d(&self, seed: u32, position: Position) -> f32;
}

/// TelemetryPort abstracts wall-clock access and log emission so Use Cases
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
