//! Compatibility facade for Level 0 anomaly planning.
//!
//! New code belongs in the feature-oriented [`super::anomalies`] and
//! [`super::red_rooms`] modules. This re-export keeps the established use-case
//! API stable for region planning, examples, and downstream crates.

pub use super::anomalies::plan_anomalies_for_region;
