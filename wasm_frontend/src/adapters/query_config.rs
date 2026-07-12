//! URL query → generation settings. Lets players share worlds by URL:
//!
//! `?seed=1234&pillars=0.5&walls=1.2&anomalies=0.8&remap_intensity=1.2`
//!
//! * `seed`    — world seed (u32; any text is hashed so words work too).
//! * `pillars` — structural column density multiplier (0 = none).
//! * `walls`   — office wall density multiplier (0 = open plan).
//! * `atria`   — how much of the world vaults into tall atria.
//! * `lights`  — ceiling light panel density.
//! * `anomalies`, `anomaly_size` and the per-family knobs control Level 0
//!   phenomena without changing ordinary fabric density.
//! * `remap_intensity`, `remap_distance` and `anomaly_safe_radius` control
//!   deterministic traversal-epoch transformations.
//!
//! Multipliers default to 1.0 and are clamped to 0..=4; physical distances
//! use narrower documented ranges. Kept free of
//! web-sys so it is natively unit-tested; the browser driver only hands in
//! `window.location.search`.

use vackrooms::use_cases::generate_chunk::{AnomalyTuning, GeneratorConfig, LevelTuning};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationParams {
    pub seed: u32,
    pub tuning: LevelTuning,
    pub anomalies: AnomalyTuning,
}

/// Parses `window.location.search` (with or without the leading `?`).
/// Unknown keys are ignored; malformed values fall back to defaults.
pub fn parse_generation_params(query: &str, default_seed: u32) -> GenerationParams {
    let mut params = GenerationParams {
        seed: default_seed,
        tuning: LevelTuning::default(),
        anomalies: AnomalyTuning::default(),
    };

    for pair in query.trim_start_matches('?').split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        let knob = |t: &mut f32| {
            if let Ok(v) = value.parse::<f32>() {
                if v.is_finite() {
                    *t = v.clamp(0.0, 4.0);
                }
            }
        };
        let ranged = |t: &mut f32, min: f32, max: f32| {
            if let Ok(v) = value.parse::<f32>() {
                if v.is_finite() {
                    *t = v.clamp(min, max);
                }
            }
        };
        match key {
            "seed" => {
                params.seed = value.parse::<u32>().unwrap_or_else(|_| hash_seed(value));
            }
            "pillars" => knob(&mut params.tuning.pillars),
            "walls" => knob(&mut params.tuning.walls),
            "atria" => knob(&mut params.tuning.atria),
            "lights" => knob(&mut params.tuning.lights),
            "anomalies" => knob(&mut params.anomalies.frequency),
            "anomaly_size" => ranged(&mut params.anomalies.size, 0.5, 2.0),
            "pillar_expanses" => knob(&mut params.anomalies.pillar_expanses),
            "blackouts" => knob(&mut params.anomalies.blackouts),
            "red_rooms" => knob(&mut params.anomalies.red_rooms),
            "pit_lattices" => knob(&mut params.anomalies.pit_lattices),
            "remap_intensity" => knob(&mut params.anomalies.remap_intensity),
            "remap_distance" => ranged(&mut params.anomalies.remap_distance, 4.0, 64.0),
            "anomaly_safe_radius" => ranged(&mut params.anomalies.safe_radius, 4.0, 32.0),
            "red_escape_bias" => ranged(&mut params.anomalies.red_escape_bias, 0.0, 1.0),
            _ => {}
        }
    }
    params
}

/// Seed + generator configuration derived from the URL query. The main
/// thread and every generation worker call this with the same query string,
/// so all of them voxelize the identical world by construction.
pub fn generator_setup_from_query(query: &str, default_seed: u32) -> (u32, GeneratorConfig) {
    let params = parse_generation_params(query, default_seed);
    let base = if query.contains("spec=high") {
        GeneratorConfig::high_spec()
    } else {
        GeneratorConfig::low_spec()
    };
    (
        params.seed,
        base.with_tuning(params.tuning)
            .with_anomalies(params.anomalies),
    )
}

/// Non-numeric seeds ("?seed=kitten") hash to a stable u32 (FNV-1a).
fn hash_seed(text: &str) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for b in text.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_when_query_is_empty() {
        let p = parse_generation_params("", 42);
        assert_eq!(p.seed, 42);
        assert_eq!(p.tuning, LevelTuning::default());
        assert_eq!(p.anomalies, AnomalyTuning::default());
    }

    #[test]
    fn parses_seed_and_knobs() {
        let p = parse_generation_params("?seed=7&pillars=0.5&walls=2&atria=0&lights=1.5", 42);
        assert_eq!(p.seed, 7);
        assert_eq!(p.tuning.pillars, 0.5);
        assert_eq!(p.tuning.walls, 2.0);
        assert_eq!(p.tuning.atria, 0.0);
        assert_eq!(p.tuning.lights, 1.5);
    }

    #[test]
    fn text_seeds_hash_deterministically() {
        let a = parse_generation_params("seed=kitten", 42);
        let b = parse_generation_params("seed=kitten", 42);
        assert_eq!(a.seed, b.seed);
        assert_ne!(a.seed, 42);
    }

    #[test]
    fn junk_values_are_ignored_and_clamped() {
        let p = parse_generation_params("?pillars=banana&walls=99&renderer=cpu&spec=high", 42);
        assert_eq!(p.tuning.pillars, 1.0);
        assert_eq!(p.tuning.walls, 4.0);
    }

    #[test]
    fn parses_namespaced_anomaly_controls() {
        let p = parse_generation_params(
            "?anomalies=0.5&anomaly_size=9&pillar_expanses=2&blackouts=0\
             &red_rooms=1.5&pit_lattices=0.25&remap_intensity=3\
             &remap_distance=2&anomaly_safe_radius=99&red_escape_bias=0.4",
            42,
        );
        assert_eq!(p.anomalies.frequency, 0.5);
        assert_eq!(p.anomalies.size, 2.0);
        assert_eq!(p.anomalies.pillar_expanses, 2.0);
        assert_eq!(p.anomalies.blackouts, 0.0);
        assert_eq!(p.anomalies.red_rooms, 1.5);
        assert_eq!(p.anomalies.pit_lattices, 0.25);
        assert_eq!(p.anomalies.remap_intensity, 3.0);
        assert_eq!(p.anomalies.remap_distance, 4.0);
        assert_eq!(p.anomalies.safe_radius, 32.0);
        assert_eq!(p.anomalies.red_escape_bias, 0.4);
    }
}
