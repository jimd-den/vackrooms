//! Survival vitals: the wanderer's hydration, satiety, and condition.
//!
//! Pure domain math — no clocks, no frames, no renderer knowledge. The
//! application advances a value copy once per simulation step with the
//! ambient temperature it measured, and reads back plain numbers for the
//! HUD. All quantities are normalized 0..=1 so presenters never re-scale.

/// Neutral indoor temperature, °C. Depletion multipliers are centered here.
const COMFORT_TEMPERATURE_C: f32 = 21.0;

/// Seconds for full hydration to deplete at comfort temperature.
const HYDRATION_SECONDS: f32 = 18.0 * 60.0;
/// Seconds for full satiety to deplete (hunger is the slower pressure).
const SATIETY_SECONDS: f32 = 40.0 * 60.0;
/// Extra thirst per °C above comfort: at Level 0's ~33 °C the multiplier
/// is 1 + 12 * 0.07 ≈ 1.8x, turning the mono-yellow heat into real urgency.
const HEAT_THIRST_PER_DEGREE: f32 = 0.07;
/// Seconds for full condition to collapse while a vital sits at zero.
/// Both vitals empty drain twice as fast.
const CONDITION_SECONDS: f32 = 90.0;
/// Condition slowly mends while both vitals are comfortable (> 0.3).
const CONDITION_RECOVERY_SECONDS: f32 = 120.0;

/// One almond water restores this much hydration (canon: the safest liquid
/// in the Backrooms) and calms a fraction of lost condition. Public so the
/// application can price the *waste* of drinking past a full reserve.
pub const ALMOND_WATER_HYDRATION: f32 = 0.65;
const ALMOND_WATER_CONDITION: f32 = 0.15;
/// One ration restores this much satiety. Public for the same waste math.
pub const RATION_SATIETY: f32 = 0.55;

/// Normalized survival state. `1.0` is fully provisioned; `0.0` is empty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerVitals {
    /// Thirst reserve. Depletes constantly; ambient heat accelerates it.
    pub hydration: f32,
    /// Hunger reserve. Depletes constantly, slower than hydration.
    pub satiety: f32,
    /// Physical condition. Collapses only while a reserve is empty;
    /// reaching zero is death (the application decides what death does).
    pub condition: f32,
}

impl Default for PlayerVitals {
    fn default() -> Self {
        Self {
            hydration: 1.0,
            satiety: 1.0,
            condition: 1.0,
        }
    }
}

impl PlayerVitals {
    /// Advances the vitals by `dt_seconds` under `ambient_c` degrees Celsius.
    /// Pure: same inputs, same outputs; the application owns the clock.
    #[must_use]
    pub fn advanced(mut self, dt_seconds: f32, ambient_c: f32) -> Self {
        let dt = dt_seconds.max(0.0);
        let heat = 1.0 + (ambient_c - COMFORT_TEMPERATURE_C).max(0.0) * HEAT_THIRST_PER_DEGREE;
        self.hydration = (self.hydration - dt * heat / HYDRATION_SECONDS).clamp(0.0, 1.0);
        self.satiety = (self.satiety - dt / SATIETY_SECONDS).clamp(0.0, 1.0);

        let starved = u32::from(self.hydration <= 0.0) + u32::from(self.satiety <= 0.0);
        self.condition = if starved > 0 {
            (self.condition - dt * starved as f32 / CONDITION_SECONDS).clamp(0.0, 1.0)
        } else if self.hydration > 0.3 && self.satiety > 0.3 {
            (self.condition + dt / CONDITION_RECOVERY_SECONDS).clamp(0.0, 1.0)
        } else {
            self.condition
        };
        self
    }

    /// Drinks one almond water bottle.
    #[must_use]
    pub fn drank_almond_water(mut self) -> Self {
        self.hydration = (self.hydration + ALMOND_WATER_HYDRATION).clamp(0.0, 1.0);
        self.condition = (self.condition + ALMOND_WATER_CONDITION).clamp(0.0, 1.0);
        self
    }

    /// Eats one ration.
    #[must_use]
    pub fn ate_ration(mut self) -> Self {
        self.satiety = (self.satiety + RATION_SATIETY).clamp(0.0, 1.0);
        self
    }

    /// True once condition has fully collapsed.
    pub fn is_dead(&self) -> bool {
        self.condition <= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_accelerates_thirst_but_not_hunger() {
        let cool = PlayerVitals::default().advanced(600.0, 21.0);
        let hot = PlayerVitals::default().advanced(600.0, 33.0);
        assert!(hot.hydration < cool.hydration, "heat must cost extra water");
        assert!((hot.satiety - cool.satiety).abs() < 1e-6);
    }

    #[test]
    fn empty_reserves_collapse_condition_and_zero_condition_is_death() {
        let mut vitals = PlayerVitals {
            hydration: 0.0,
            satiety: 0.0,
            condition: 1.0,
        };
        assert!(!vitals.is_dead());
        vitals = vitals.advanced(30.0, 21.0);
        assert!(vitals.condition < 0.4, "double starvation drains fast");
        vitals = vitals.advanced(60.0, 21.0);
        assert!(vitals.is_dead());
    }

    #[test]
    fn condition_recovers_only_while_provisioned() {
        let hurt = PlayerVitals {
            hydration: 0.8,
            satiety: 0.8,
            condition: 0.5,
        };
        assert!(hurt.advanced(60.0, 21.0).condition > 0.5);

        let parched = PlayerVitals {
            hydration: 0.1,
            satiety: 0.8,
            condition: 0.5,
        };
        assert!(parched.advanced(60.0, 21.0).condition <= 0.5);
    }

    #[test]
    fn supplies_restore_their_vital() {
        let weary = PlayerVitals {
            hydration: 0.2,
            satiety: 0.2,
            condition: 0.4,
        };
        let refreshed = weary.drank_almond_water();
        assert!((refreshed.hydration - 0.85).abs() < 1e-6);
        assert!(refreshed.condition > 0.4);
        let fed = weary.ate_ration();
        assert!((fed.satiety - 0.75).abs() < 1e-6);
    }

    #[test]
    fn vitals_never_leave_the_unit_interval() {
        let v = PlayerVitals::default()
            .drank_almond_water()
            .ate_ration()
            .advanced(1_000_000.0, 50.0);
        assert_eq!(v.hydration, 0.0);
        assert_eq!(v.satiety, 0.0);
        assert!(v.condition >= 0.0 && v.condition <= 1.0);
    }
}
