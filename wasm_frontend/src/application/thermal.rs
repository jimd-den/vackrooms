//! Ambient temperature felt by the player, derived from the lit scene.
//!
//! The Backrooms' heat *is* its lighting: fluorescent tubes bake the air,
//! so standing under working fixtures warms the wanderer while darkness
//! (blackouts, dead bays, Level 1's failing grid) lets the air fall back to
//! the level's unlit baseline. Enclosure matters too — a lit closet cooks
//! faster than a lit atrium — approximated here by how much collision mass
//! crowds the player. Everything is pure math over the same deduplicated,
//! flicker-scaled fixture list every renderer sees: a sputtering tube that
//! is dark this instant radiates nothing this instant.

use crate::application::ports::LightSource;

/// Unlit-air baseline per level, °C. Level 0's dark corners still hold the
/// building's stale warmth; Level 1's raw concrete is genuinely cold.
pub fn dark_baseline_c(level: u32) -> f32 {
    match level {
        0 => 24.0,
        1 => 13.0,
        _ => 21.0,
    }
}

/// Maximum heating the fixture field can add above the baseline, °C.
const MAX_LIGHT_HEAT_C: f32 = 13.0;
/// Scales summed fixture irradiance into °C before the cap.
const HEAT_PER_IRRADIANCE: f32 = 9.0;
/// Enclosure raises effective heating by up to this factor.
const ENCLOSURE_GAIN: f32 = 0.6;
/// Radius probed for enclosing collision mass.
pub const ENCLOSURE_RADIUS: f32 = 6.0;
/// Collision-box count that reads as "fully enclosed".
const ENCLOSURE_SATURATION: f32 = 36.0;

/// The instantaneous ambient temperature at the player position.
///
/// `lights` must already carry this frame's flicker gain so failing tubes
/// heat exactly as much as they shine. The result is *unsmoothed*; the
/// engine applies thermal inertia so doorways don't flip the rate per frame.
pub fn ambient_celsius(
    level: u32,
    player: [f32; 3],
    lights: &[LightSource],
    enclosing_collision_boxes: usize,
) -> f32 {
    let mut irradiance = 0.0f32;
    for light in lights {
        let dx = light.position[0] - player[0];
        let dy = light.position[1] - player[1];
        let dz = light.position[2] - player[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let range = light.radius.max(0.01);
        let falloff = (1.0 - distance / range).clamp(0.0, 1.0);
        irradiance += light.intensity * falloff * falloff;
    }

    let enclosure = (enclosing_collision_boxes as f32 / ENCLOSURE_SATURATION).clamp(0.0, 1.0);

    let heat = (irradiance * HEAT_PER_IRRADIANCE).min(MAX_LIGHT_HEAT_C)
        * (1.0 + ENCLOSURE_GAIN * enclosure);
    dark_baseline_c(level) + heat.min(MAX_LIGHT_HEAT_C * (1.0 + ENCLOSURE_GAIN))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::collision::Aabb;
    use crate::application::ports::LightKind;

    fn fixture(position: [f32; 3], intensity: f32) -> LightSource {
        LightSource {
            id: 1,
            position,
            half_size: [0.5, 0.5],
            color: [1.0, 1.0, 1.0],
            radius: 16.0,
            intensity,
            kind: LightKind::CeilingPanel,
            flicker_mode: 0,
            enabled: true,
        }
    }

    fn walls_around(count: usize) -> Vec<Aabb> {
        (0..count)
            .map(|i| {
                let angle = i as f32;
                Aabb::new(
                    [angle.cos() * 2.0, 0.0, angle.sin() * 2.0],
                    [angle.cos() * 2.0 + 0.4, 3.0, angle.sin() * 2.0 + 0.4],
                )
            })
            .collect()
    }

    fn nearby_wall_count(player: [f32; 3], walls: &[Aabb]) -> usize {
        walls
            .iter()
            .filter(|wall| {
                let center_x = (wall.min[0] + wall.max[0]) * 0.5;
                let center_z = (wall.min[2] + wall.max[2]) * 0.5;
                let dx = center_x - player[0];
                let dz = center_z - player[2];
                dx * dx + dz * dz <= ENCLOSURE_RADIUS * ENCLOSURE_RADIUS
            })
            .count()
    }

    #[test]
    fn darkness_rests_at_the_level_baseline() {
        let ambient = ambient_celsius(0, [0.0, 1.7, 0.0], &[], 0);
        assert_eq!(ambient, dark_baseline_c(0));
        assert!(dark_baseline_c(1) < dark_baseline_c(0), "Level 1 runs cold");
    }

    #[test]
    fn standing_under_a_fixture_heats_the_air() {
        let lit = ambient_celsius(0, [0.0, 1.7, 0.0], &[fixture([0.0, 3.0, 0.0], 1.5)], 0);
        let far = ambient_celsius(0, [14.0, 1.7, 0.0], &[fixture([0.0, 3.0, 0.0], 1.5)], 0);
        assert!(
            lit > dark_baseline_c(0) + 2.0,
            "lit air must run hot: {lit}"
        );
        assert!(far < lit, "heat falls off with distance");
    }

    #[test]
    fn enclosure_amplifies_lit_spaces_but_not_dark_ones() {
        let light = [fixture([0.0, 3.0, 0.0], 1.5)];
        let player = [0.0, 1.7, 0.0];
        let walls = walls_around(40);
        let open = ambient_celsius(0, player, &light, 0);
        let tight = ambient_celsius(0, player, &light, nearby_wall_count(player, &walls));
        assert!(tight > open, "small lit rooms cook: {tight} vs {open}");

        let dark_open = ambient_celsius(0, player, &[], 0);
        let dark_tight = ambient_celsius(0, player, &[], nearby_wall_count(player, &walls));
        assert_eq!(dark_open, dark_tight, "enclosure without light is inert");
    }

    #[test]
    fn a_flickered_out_tube_radiates_nothing() {
        // The engine multiplies flicker gain into intensity before calling;
        // a dropout frame arrives here as near-zero intensity.
        let out = ambient_celsius(0, [0.0, 1.7, 0.0], &[fixture([0.0, 3.0, 0.0], 0.02)], 0);
        assert!(out - dark_baseline_c(0) < 0.5);
    }
}
