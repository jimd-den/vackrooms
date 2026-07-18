//! URL query → generation settings. Lets players share worlds by URL:
//!
//! `?seed=1234&voxel_size=0.1&pillars=0.5&walls=1.2&anomalies=0.8`
//!
//! * `seed`    — world seed (u32; any text is hashed so words work too).
//! * `spec`    — shared case-insensitive `low`/`high` quality decision.
//! * `pillars` — structural column density multiplier (0 = none).
//! * `walls`   — office wall density multiplier (0 = open plan).
//! * `atria`   — how much of the world vaults into tall atria.
//! * `lights`  — ceiling light panel density.
//! * `voxel_size` — optional scene resolution in world units. The selected
//!   profile remains in force when the value violates generator limits.
//! * `anomalies`, `anomaly_size` and the per-family knobs control Level 0
//!   phenomena without changing ordinary fabric density.
//! * `remap_intensity`, `remap_distance` and `anomaly_safe_radius` control
//!   deterministic traversal-epoch transformations.
//!
//! Multipliers default to 1.0 and are clamped to 0..=4; physical distances
//! use narrower documented ranges. Kept free of
//! web-sys so it is natively unit-tested; the browser driver only hands in
//! `window.location.search`.

use vackrooms::domain::entities::anomaly::AnomalyKind;
use vackrooms::use_cases::generate_chunk::{AnomalyTuning, GeneratorConfig, LevelTuning};

use crate::application::quality::QualityProfile;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationParams {
    pub seed: u32,
    pub voxel_size: Option<f32>,
    pub tuning: LevelTuning,
    pub anomalies: AnomalyTuning,
}

/// Exact-key lookup in a location search string (`?a=1&b=2`, leading `?`
/// optional). Every query consumer must go through this (or an equivalent
/// `split('&')` loop): raw `str::find("key=")` substring probes have caused
/// settings to apply that nobody asked for — `?spawn_yaw=2` silently
/// matching `yaw=`, values containing `capture=1`, and the like.
pub fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .trim_start_matches('?')
        .split('&')
        .find_map(|pair| match pair.split_once('=') {
            Some((k, v)) if k == key => Some(v),
            _ => None,
        })
}

/// True when the query holds exactly `key=1`.
pub fn query_flag(query: &str, key: &str) -> bool {
    query_param(query, key) == Some("1")
}

/// Parses `window.location.search` (with or without the leading `?`).
/// Unknown keys are ignored; malformed values fall back to defaults.
pub fn parse_generation_params(query: &str, default_seed: u32) -> GenerationParams {
    let mut params = GenerationParams {
        seed: default_seed,
        voxel_size: None,
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
            "voxel_size" => {
                params.voxel_size = value.parse::<f32>().ok().filter(|v| v.is_finite());
            }
            "pillars" => knob(&mut params.tuning.pillars),
            "walls" => knob(&mut params.tuning.walls),
            "atria" => knob(&mut params.tuning.atria),
            "lights" => knob(&mut params.tuning.lights),
            // World provisioning: drink, food, and level-door frequency.
            // 0 removes that class of content entirely.
            "almond_water" => knob(&mut params.tuning.almond_water),
            "rations" => knob(&mut params.tuning.rations),
            "level_doors" => knob(&mut params.tuning.level_doors),
            "anomalies" => knob(&mut params.anomalies.frequency),
            "anomaly_size" => ranged(&mut params.anomalies.size, 0.5, 2.0),
            "pillar_expanses" => knob(&mut params.anomalies.pillar_expanses),
            "blackouts" => knob(&mut params.anomalies.blackouts),
            "red_rooms" => knob(&mut params.anomalies.red_rooms),
            "pit_lattices" => knob(&mut params.anomalies.pit_lattices),
            "remap_intensity" => knob(&mut params.anomalies.remap_intensity),
            "remap_distance" => ranged(&mut params.anomalies.remap_distance, 4.0, 64.0),
            "anomaly_safe_radius" => ranged(&mut params.anomalies.safe_radius, 4.0, 32.0),
            // Gameplay tuning: the probability per closed-loop epoch that a
            // red room hashes a single far-side escape breach. 0 seals every
            // loop, 1 guarantees a breach; it is never the entrance.
            "red_escape_bias" => ranged(&mut params.anomalies.red_escape_bias, 0.0, 1.0),
            "archways" => knob(&mut params.anomalies.archways),
            // Deception: the fraction of blackout glimmers placed one
            // segment off the recovery skeleton. The default is high on
            // purpose — most glimmers lie.
            "blackout_decoys" => ranged(&mut params.anomalies.blackout_decoys, 0.0, 0.9),
            // Debug: guarantee one anomaly of this family on the spawn's
            // macro cell (settings menu "Spawn anomaly"). Unknown values
            // leave the world untouched.
            "force_anomaly" => {
                params.anomalies.forced_kind = match value {
                    "pillars" => Some(AnomalyKind::PillarExpanse),
                    "blackout" => Some(AnomalyKind::BlackoutExpanse),
                    "pits" => Some(AnomalyKind::PitLattice),
                    "archway" => Some(AnomalyKind::ArchwayRoom),
                    "redroom" => Some(AnomalyKind::RedRoom),
                    _ => None,
                };
            }
            _ => {}
        }
    }
    params
}

/// Seed + generator configuration derived from the URL query. The main
/// thread and every generation worker call this with the same query string,
/// so all of them voxelize the identical world by construction.
pub fn generator_setup_from_query(query: &str, default_seed: u32) -> (u32, GeneratorConfig) {
    generator_setup_for_quality(query, default_seed, quality_profile_from_query(query))
}

/// Parses the shared `?spec=low|high` decision. Values are case-insensitive,
/// matching renderer selection, while malformed or missing values retain the
/// established low-profile default.
pub fn quality_profile_from_query(query: &str) -> QualityProfile {
    query_param(query, "spec")
        .and_then(QualityProfile::parse)
        .unwrap_or_default()
}

/// Resolves generation from an already parsed quality decision. The browser
/// composition root uses this together with renderer profile construction so
/// both sides are guaranteed to observe the same value.
pub fn generator_setup_for_quality(
    query: &str,
    default_seed: u32,
    quality: QualityProfile,
) -> (u32, GeneratorConfig) {
    let params = parse_generation_params(query, default_seed);
    let base = match quality {
        QualityProfile::Low => GeneratorConfig::low_spec(),
        QualityProfile::High => GeneratorConfig::high_spec(),
    };
    let configured = base
        .with_tuning(params.tuning)
        .with_anomalies(params.anomalies);
    let configured = params
        .voxel_size
        .and_then(|voxel_size| configured.try_with_voxel_size(voxel_size).ok())
        .unwrap_or(configured);
    (params.seed, configured)
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
        assert_eq!(p.voxel_size, None);
        assert_eq!(p.tuning, LevelTuning::default());
        assert_eq!(p.anomalies, AnomalyTuning::default());
    }

    #[test]
    fn query_params_match_exact_keys_only() {
        // The historical bug: substring probes made `?spawn_yaw=2` set the
        // capture yaw, and any value containing `capture=1` froze the game.
        assert_eq!(query_param("?spawn_yaw=2&pitch2=9", "yaw"), None);
        assert_eq!(query_param("?spawn_yaw=2&yaw=0.5", "yaw"), Some("0.5"));
        assert_eq!(query_param("?note=capture=1", "capture"), None);
        assert!(!query_flag("?recapture=1", "capture"));
        assert!(query_flag("capture=1", "capture"));
        assert_eq!(query_param("?renderer=cpux", "renderer"), Some("cpux"));
        assert_eq!(query_param("", "anything"), None);
    }

    #[test]
    fn parses_seed_and_knobs() {
        let p = parse_generation_params(
            "?seed=7&voxel_size=0.1&pillars=0.5&walls=2&atria=0&lights=1.5",
            42,
        );
        assert_eq!(p.seed, 7);
        assert_eq!(p.voxel_size, Some(0.1));
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
        let p = parse_generation_params(
            "?pillars=banana&walls=99&voxel_size=banana&renderer=cpu&spec=high",
            42,
        );
        assert_eq!(p.tuning.pillars, 1.0);
        assert_eq!(p.tuning.walls, 4.0);
        assert_eq!(p.voxel_size, None);
    }

    #[test]
    fn generator_setup_applies_a_valid_voxel_override_to_both_profiles() {
        let (_, low) = generator_setup_from_query("?voxel_size=0.1", 42);
        let (_, high) = generator_setup_from_query("?spec=high&voxel_size=0.2", 42);
        assert_eq!(low.voxel_scale, 0.1);
        assert_eq!(high.voxel_scale, 0.2);
    }

    #[test]
    fn profile_selection_requires_an_exact_query_key() {
        let (_, config) = generator_setup_from_query("?not_spec=high&voxel_size=0.2", 42);
        assert_eq!(config.chunk_size, GeneratorConfig::low_spec().chunk_size);
    }

    #[test]
    fn generation_and_rendering_share_one_case_insensitive_quality_decision() {
        for query in ["?spec=high", "?spec=HIGH", "?spec=High"] {
            let quality = quality_profile_from_query(query);
            let (_, config) = generator_setup_for_quality(query, 42, quality);
            assert_eq!(quality, QualityProfile::High);
            assert_eq!(config.chunk_size, GeneratorConfig::high_spec().chunk_size);
        }

        for query in [
            "",
            "?spec=low",
            "?spec=LOW",
            "?spec=ultra",
            "?not_spec=high",
        ] {
            let quality = quality_profile_from_query(query);
            let (_, config) = generator_setup_for_quality(query, 42, quality);
            assert_eq!(quality, QualityProfile::Low, "query {query:?}");
            assert_eq!(
                config.chunk_size,
                GeneratorConfig::low_spec().chunk_size,
                "query {query:?}"
            );
        }
    }

    #[test]
    fn generator_setup_keeps_profile_default_for_unsupported_voxel_size() {
        let (_, non_tiling) = generator_setup_from_query("?voxel_size=0.3", 42);
        let (_, odd_progressive_lod) = generator_setup_from_query("?voxel_size=0.4", 42);
        let (_, over_budget) = generator_setup_from_query("?spec=high&voxel_size=0.05", 42);
        let (_, non_positive) = generator_setup_from_query("?voxel_size=-0.1", 42);
        assert_eq!(
            non_tiling.voxel_scale,
            GeneratorConfig::low_spec().voxel_scale
        );
        assert_eq!(
            odd_progressive_lod.voxel_scale,
            GeneratorConfig::low_spec().voxel_scale
        );
        assert_eq!(
            over_budget.voxel_scale,
            GeneratorConfig::high_spec().voxel_scale
        );
        assert_eq!(
            non_positive.voxel_scale,
            GeneratorConfig::low_spec().voxel_scale
        );
    }

    #[test]
    fn parses_force_anomaly_values() {
        let p = parse_generation_params("?force_anomaly=pillars", 42);
        assert_eq!(p.anomalies.forced_kind, Some(AnomalyKind::PillarExpanse));
        let p = parse_generation_params("?force_anomaly=blackout", 42);
        assert_eq!(p.anomalies.forced_kind, Some(AnomalyKind::BlackoutExpanse));
        let p = parse_generation_params("?force_anomaly=pits", 42);
        assert_eq!(p.anomalies.forced_kind, Some(AnomalyKind::PitLattice));
        let p = parse_generation_params("?force_anomaly=archway", 42);
        assert_eq!(p.anomalies.forced_kind, Some(AnomalyKind::ArchwayRoom));
        let p = parse_generation_params("?force_anomaly=redroom", 42);
        assert_eq!(p.anomalies.forced_kind, Some(AnomalyKind::RedRoom));
        let p = parse_generation_params("?force_anomaly=banana", 42);
        assert_eq!(p.anomalies.forced_kind, None);
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
