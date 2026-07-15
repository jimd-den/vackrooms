//! Select concurrency for the browser's chunk-generation worker pool.
//!
//! This policy configures world generation and streaming only. The software
//! splat renderer remains on the browser's main thread. Keeping query parsing
//! and hardware clamping here makes the contract natively testable and keeps
//! browser APIs out of the application layer.

/// Hard ceiling for generation workers. More workers duplicate the WASM
/// module and atlas-building scratch memory without improving frame latency.
pub const MAX_GENERATION_WORKERS: u8 = 4;

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
            GenerationWorkerPreference::Fixed(4)
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
        assert_eq!(GenerationWorkerPreference::Auto.resolve(8.0), 4);
        assert_eq!(GenerationWorkerPreference::Auto.resolve(4.0), 3);
        assert_eq!(GenerationWorkerPreference::Auto.resolve(2.0), 1);
        assert_eq!(GenerationWorkerPreference::Auto.resolve(1.0), 1);
        assert_eq!(GenerationWorkerPreference::Fixed(4).resolve(2.0), 2);
        assert_eq!(GenerationWorkerPreference::Fixed(2).resolve(8.0), 2);
        assert_eq!(GenerationWorkerPreference::Disabled.resolve(8.0), 0);
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
