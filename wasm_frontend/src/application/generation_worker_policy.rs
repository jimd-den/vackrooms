//! Select concurrency for the browser's chunk-generation worker pool.
//!
//! This policy configures world generation and streaming only. The software
//! splat renderer remains on the browser's main thread. Keeping query parsing
//! and hardware clamping here makes the contract natively testable and keeps
//! browser APIs out of the application layer.

/// Safety ceiling for generation workers — not a target. `Auto` should use
/// every hardware thread the browser reports (minus one reserved for the
/// main/render thread), including on high-core-count machines; this only
/// guards against a browser misreporting an absurd `hardwareConcurrency`
/// value and spawning an unreasonable number of Worker instances (each one
/// duplicates the WASM module and atlas-building scratch memory).
pub const MAX_GENERATION_WORKERS: u8 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationWorkerPreference {
    /// Reserve one reported hardware thread for the browser/render loop.
    Auto,
    /// Generate synchronously through `LocalChunkSource` on the main thread.
    Disabled,
    /// Request an explicit number, still bounded by reported hardware.
    Fixed(u8),
}

impl Default for GenerationWorkerPreference {
    fn default() -> Self {
        Self::Auto
    }
}

impl GenerationWorkerPreference {
    /// Resolve a safe pool size from `Navigator.hardwareConcurrency`.
    /// Unknown/invalid reports use the conservative historical fallback of
    /// two hardware threads; automatic selection therefore uses one worker.
    pub fn resolve(self, hardware_concurrency: f64) -> usize {
        if self == Self::Disabled {
            return 0;
        }

        let reported_threads = if hardware_concurrency.is_finite() && hardware_concurrency >= 1.0 {
            hardware_concurrency.floor().min(u32::MAX as f64) as u32
        } else {
            2
        };
        let auto_workers = reported_threads
            .saturating_sub(1)
            .clamp(1, u32::from(MAX_GENERATION_WORKERS)) as u8;
        let fixed_worker_ceiling =
            reported_threads.clamp(1, u32::from(MAX_GENERATION_WORKERS)) as u8;
        match self {
            Self::Auto => auto_workers,
            Self::Fixed(requested) => requested.clamp(1, fixed_worker_ceiling),
            Self::Disabled => 0,
        }
        .into()
    }
}

/// Parses the last exact `workers` query pair. Canonical values are `auto`,
/// `0`, and positive integers; oversized integers clamp to the product cap.
/// Missing or malformed values safely restore automatic selection.
pub fn parse_generation_worker_preference(query: &str) -> GenerationWorkerPreference {
    let mut preference = GenerationWorkerPreference::Auto;
    for (key, value) in query
        .trim_start_matches('?')
        .split('&')
        .filter_map(|pair| pair.split_once('='))
    {
        if key != "workers" {
            continue;
        }
        preference = match value {
            "auto" => GenerationWorkerPreference::Auto,
            "0" => GenerationWorkerPreference::Disabled,
            _ => value
                .parse::<u32>()
                .ok()
                .filter(|requested| *requested > 0)
                .map(|requested| {
                    GenerationWorkerPreference::Fixed(
                        requested.min(u32::from(MAX_GENERATION_WORKERS)) as u8,
                    )
                })
                .unwrap_or(GenerationWorkerPreference::Auto),
        };
    }
    preference
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_and_malformed_values_restore_auto() {
        assert_eq!(
            parse_generation_worker_preference("?seed=7"),
            GenerationWorkerPreference::Auto
        );
        assert_eq!(
            parse_generation_worker_preference("?workers=banana"),
            GenerationWorkerPreference::Auto
        );
        assert_eq!(
            parse_generation_worker_preference("?not_workers=0"),
            GenerationWorkerPreference::Auto
        );
    }

    #[test]
    fn parses_disabled_auto_and_bounded_fixed_counts() {
        assert_eq!(
            parse_generation_worker_preference("?workers=0"),
            GenerationWorkerPreference::Disabled
        );
        assert_eq!(
            parse_generation_worker_preference("?workers=auto"),
            GenerationWorkerPreference::Auto
        );
        assert_eq!(
            parse_generation_worker_preference("?workers=3"),
            GenerationWorkerPreference::Fixed(3)
        );
        assert_eq!(
            parse_generation_worker_preference("?workers=999"),
            GenerationWorkerPreference::Fixed(MAX_GENERATION_WORKERS)
        );
    }

    #[test]
    fn last_duplicate_pair_wins_deterministically() {
        assert_eq!(
            parse_generation_worker_preference("?workers=4&workers=0"),
            GenerationWorkerPreference::Disabled
        );
    }

    #[test]
    fn resolution_reserves_the_main_thread_and_clamps_to_hardware() {
        assert_eq!(GenerationWorkerPreference::Auto.resolve(8.0), 7);
        assert_eq!(GenerationWorkerPreference::Auto.resolve(4.0), 3);
        assert_eq!(GenerationWorkerPreference::Auto.resolve(2.0), 1);
        assert_eq!(GenerationWorkerPreference::Auto.resolve(1.0), 1);
        assert_eq!(GenerationWorkerPreference::Fixed(4).resolve(2.0), 2);
        assert_eq!(GenerationWorkerPreference::Fixed(2).resolve(8.0), 2);
        assert_eq!(GenerationWorkerPreference::Disabled.resolve(8.0), 0);
    }

    /// A 16-thread machine must not be artificially throttled to a low
    /// worker count: `Auto` should use (almost) every reported thread. This
    /// is the regression test for the historical `MAX_GENERATION_WORKERS =
    /// 4` ceiling, which silently capped every machine above 5 hardware
    /// threads regardless of what was actually available.
    #[test]
    fn auto_scales_to_high_core_counts_instead_of_a_low_fixed_cap() {
        assert_eq!(GenerationWorkerPreference::Auto.resolve(16.0), 15);
        assert_eq!(
            GenerationWorkerPreference::Fixed(16).resolve(16.0),
            16,
            "an explicit request for 16 workers on 16-thread hardware must not be clamped below it"
        );
    }

    #[test]
    fn invalid_hardware_reports_use_a_conservative_fallback() {
        assert_eq!(GenerationWorkerPreference::Auto.resolve(f64::NAN), 1);
        assert_eq!(GenerationWorkerPreference::Auto.resolve(0.0), 1);
        assert_eq!(
            GenerationWorkerPreference::Fixed(4).resolve(f64::INFINITY),
            2
        );
    }
}
