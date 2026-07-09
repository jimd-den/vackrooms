//! URL query → generation settings. Lets players share worlds by URL:
//!
//! `?seed=1234&pillars=0.5&walls=1.2&atria=2&lights=0.8`
//!
//! * `seed`    — world seed (u32; any text is hashed so words work too).
//! * `pillars` — structural column density multiplier (0 = none).
//! * `walls`   — office wall density multiplier (0 = open plan).
//! * `atria`   — how much of the world vaults into tall atria.
//! * `lights`  — ceiling light panel density.
//!
//! All multipliers default to 1.0 and are clamped to 0..=4. Kept free of
//! web-sys so it is natively unit-tested; the browser driver only hands in
//! `window.location.search`.

use vackrooms::use_cases::generate_chunk::LevelTuning;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationParams {
    pub seed: u32,
    pub tuning: LevelTuning,
}

/// Parses `window.location.search` (with or without the leading `?`).
/// Unknown keys are ignored; malformed values fall back to defaults.
pub fn parse_generation_params(query: &str, default_seed: u32) -> GenerationParams {
    let mut params = GenerationParams {
        seed: default_seed,
        tuning: LevelTuning::default(),
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
        match key {
            "seed" => {
                params.seed = value.parse::<u32>().unwrap_or_else(|_| hash_seed(value));
            }
            "pillars" => knob(&mut params.tuning.pillars),
            "walls" => knob(&mut params.tuning.walls),
            "atria" => knob(&mut params.tuning.atria),
            "lights" => knob(&mut params.tuning.lights),
            _ => {}
        }
    }
    params
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
}
