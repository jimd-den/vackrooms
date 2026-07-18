//! Survival state and carried-supply policy for one wanderer.
//!
//! The frame orchestrator measures the world (notably its instantaneous
//! temperature) and passes those observations here. This object owns the
//! resulting felt temperature, vitals, inventory, consumption decisions,
//! overconsumption strain, and attrition bookkeeping. Keeping those rules
//! together prevents presentation, streaming, and player relocation concerns
//! from becoming another way to mutate survival state.

use vackrooms::domain::entities::player_vitals::{
    ALMOND_WATER_HYDRATION, PlayerVitals, RATION_SATIETY,
};
use vackrooms::domain::entities::supplies::SupplyKind;

/// Carried supplies beyond immediate need, per kind.
const CARRY_CAP: u32 = 3;
/// Assisted consumption acts once the matching reserve falls below this.
const AUTO_CONSUME_AT: f32 = 0.5;
/// Share of restoration beyond a full reserve charged to excess strain.
const EXCESS_PER_WASTE: f32 = 0.8;
/// Frugal time drains the excess meter to zero in five minutes.
const EXCESS_DECAY_PER_S: f32 = 1.0 / 300.0;
/// Felt air closes roughly 63% of the gap to measured air in this many
/// seconds. The bounded frame step makes the linear blend stable.
const THERMAL_TAU_S: f32 = 8.0;

/// Edge-triggered choices made during one simulation step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ConsumptionIntent {
    pub drink: bool,
    pub eat: bool,
}

/// Immutable presenter/body view of the survival aggregate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SurvivalReadout {
    pub vitals: PlayerVitals,
    pub excess: f32,
    pub almond_bottles: u32,
    pub rations: u32,
    pub assisted_consumption: bool,
    pub ambient_c: f32,
    pub deaths: u32,
}

/// Cohesive survival aggregate. World mutation after a pickup or death stays
/// with the engine; all survival-state mutation stays here.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SurvivalInventory {
    vitals: PlayerVitals,
    excess: f32,
    almond_bottles: u32,
    rations: u32,
    assisted_consumption: bool,
    ambient_c: f32,
    deaths: u32,
}

impl SurvivalInventory {
    pub fn new(initial_ambient_c: f32) -> Self {
        Self {
            vitals: PlayerVitals::default(),
            excess: 0.0,
            almond_bottles: 0,
            rations: 0,
            assisted_consumption: false,
            ambient_c: initial_ambient_c,
            deaths: 0,
        }
    }

    pub fn readout(&self) -> SurvivalReadout {
        SurvivalReadout {
            vitals: self.vitals,
            excess: self.excess,
            almond_bottles: self.almond_bottles,
            rations: self.rations,
            assisted_consumption: self.assisted_consumption,
            ambient_c: self.ambient_c,
            deaths: self.deaths,
        }
    }

    pub fn vitals(&self) -> PlayerVitals {
        self.vitals
    }

    pub fn set_assisted_consumption(&mut self, enabled: bool) {
        self.assisted_consumption = enabled;
    }

    /// Smooths the measured air temperature, then advances physiological
    /// depletion under that felt temperature.
    pub fn advance_vitals(&mut self, dt: f32, instantaneous_c: f32) {
        let blend = (dt / THERMAL_TAU_S).clamp(0.0, 1.0);
        self.ambient_c += (instantaneous_c - self.ambient_c) * blend;
        self.vitals = self.vitals.advanced(dt, self.ambient_c);
    }

    /// Applies deliberate choices first, then the optional accessibility
    /// assist. This order ensures a deliberate drink that restores the reserve
    /// does not immediately trigger a second automatic drink.
    pub fn consume(&mut self, intent: ConsumptionIntent) {
        if intent.drink {
            self.drink_carried();
        }
        if intent.eat {
            self.eat_carried();
        }
        if self.assisted_consumption {
            self.auto_consume_supplies();
        }
    }

    /// Attempts to stock one world pickup. `false` means the matching pocket
    /// is full, so the caller must leave the pickup and reality state intact.
    pub fn try_pickup(&mut self, kind: SupplyKind) -> bool {
        let carried = match kind {
            SupplyKind::AlmondWater => &mut self.almond_bottles,
            SupplyKind::Ration => &mut self.rations,
        };
        if *carried >= CARRY_CAP {
            return false;
        }
        *carried += 1;
        true
    }

    /// Applies attrition exactly once. The caller owns relocation and other
    /// world-facing consequences of the death.
    pub fn succumb_if_dead(&mut self) -> bool {
        if !self.vitals.is_dead() {
            return false;
        }
        self.deaths += 1;
        self.vitals = PlayerVitals::default();
        self.almond_bottles = 0;
        self.rations = 0;
        true
    }

    /// Advances overconsumption recovery and returns the reality delirium
    /// tier induced by either deprivation or excess.
    pub fn advance_strain(&mut self, dt: f32) -> u8 {
        self.excess = (self.excess - dt * EXCESS_DECAY_PER_S).max(0.0);
        let thirst_tier = match self.vitals.hydration {
            h if h > 0.6 => 0,
            h if h > 0.35 => 1,
            h if h > 0.15 => 2,
            _ => 3,
        };
        let excess_tier = match self.excess {
            e if e < 0.25 => 0,
            e if e < 0.5 => 1,
            e if e < 0.75 => 2,
            _ => 3,
        };
        thirst_tier.max(excess_tier)
    }

    /// World's aggression dial: deprivation and waste are symmetric forms of
    /// mismanagement.
    pub fn mismanagement(&self) -> f32 {
        let dehydration = (1.0 - self.vitals.hydration).clamp(0.0, 1.0);
        dehydration.max(self.excess)
    }

    /// Test seam for long-running attrition/strain integration tests.
    #[cfg(test)]
    pub(crate) fn set_vitals_for_test(&mut self, vitals: PlayerVitals) {
        self.vitals = vitals;
    }

    /// Test seam used to prove that Engine forwards this aggregate's tier to
    /// the reality snapshot without simulating a particular pickup layout.
    #[cfg(test)]
    pub(crate) fn register_waste_for_test(&mut self, reserve_before: f32, restores: f32) {
        self.register_waste(reserve_before, restores);
    }

    fn drink_carried(&mut self) {
        if self.almond_bottles == 0 {
            return;
        }
        self.almond_bottles -= 1;
        self.register_waste(self.vitals.hydration, ALMOND_WATER_HYDRATION);
        self.vitals = self.vitals.drank_almond_water();
    }

    fn eat_carried(&mut self) {
        if self.rations == 0 {
            return;
        }
        self.rations -= 1;
        self.register_waste(self.vitals.satiety, RATION_SATIETY);
        self.vitals = self.vitals.ate_ration();
    }

    fn auto_consume_supplies(&mut self) {
        if self.vitals.hydration < AUTO_CONSUME_AT && self.almond_bottles > 0 {
            self.drink_carried();
        }
        if self.vitals.satiety < AUTO_CONSUME_AT && self.rations > 0 {
            self.eat_carried();
        }
    }

    fn register_waste(&mut self, reserve_before: f32, restores: f32) {
        let waste = (reserve_before + restores - 1.0).max(0.0);
        self.excess = (self.excess + waste * EXCESS_PER_WASTE).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pickup_policy_caps_each_supply_kind_independently() {
        let mut survival = SurvivalInventory::new(21.0);
        for _ in 0..CARRY_CAP {
            assert!(survival.try_pickup(SupplyKind::AlmondWater));
            assert!(survival.try_pickup(SupplyKind::Ration));
        }
        assert!(!survival.try_pickup(SupplyKind::AlmondWater));
        assert!(!survival.try_pickup(SupplyKind::Ration));
        let readout = survival.readout();
        assert_eq!(readout.almond_bottles, CARRY_CAP);
        assert_eq!(readout.rations, CARRY_CAP);
    }

    #[test]
    fn consumption_is_deliberate_unless_assistance_is_enabled() {
        let mut survival = SurvivalInventory::new(21.0);
        survival.try_pickup(SupplyKind::AlmondWater);
        survival.set_vitals_for_test(PlayerVitals {
            hydration: 0.2,
            ..PlayerVitals::default()
        });

        survival.consume(ConsumptionIntent::default());
        assert_eq!(survival.readout().almond_bottles, 1);

        survival.set_assisted_consumption(true);
        survival.consume(ConsumptionIntent::default());
        assert_eq!(survival.readout().almond_bottles, 0);
        assert!(survival.vitals().hydration > 0.5);
    }

    #[test]
    fn felt_temperature_has_inertia_and_drives_vitals() {
        let mut survival = SurvivalInventory::new(20.0);
        survival.advance_vitals(0.1, 36.0);
        let readout = survival.readout();
        assert!(readout.ambient_c > 20.0 && readout.ambient_c < 36.0);
        assert!(readout.vitals.hydration < 1.0);
    }

    #[test]
    fn attrition_resets_body_supplies_but_preserves_excess_history() {
        let mut survival = SurvivalInventory::new(21.0);
        survival.try_pickup(SupplyKind::AlmondWater);
        survival.try_pickup(SupplyKind::Ration);
        survival.register_waste_for_test(1.0, 0.65);
        let excess = survival.readout().excess;
        survival.set_vitals_for_test(PlayerVitals {
            condition: 0.0,
            ..PlayerVitals::default()
        });

        assert!(survival.succumb_if_dead());
        assert!(!survival.succumb_if_dead(), "one death is counted once");
        let readout = survival.readout();
        assert_eq!(readout.deaths, 1);
        assert_eq!(readout.almond_bottles, 0);
        assert_eq!(readout.rations, 0);
        assert_eq!(readout.vitals, PlayerVitals::default());
        assert_eq!(readout.excess, excess);
    }

    #[test]
    fn deprivation_and_excess_choose_the_more_severe_tier() {
        let mut survival = SurvivalInventory::new(21.0);
        survival.set_vitals_for_test(PlayerVitals {
            hydration: 0.5,
            ..PlayerVitals::default()
        });
        assert_eq!(survival.advance_strain(0.0), 1);

        survival.register_waste_for_test(1.0, 1.0);
        assert_eq!(survival.advance_strain(0.0), 3);
        assert_eq!(survival.mismanagement(), 0.8);
    }
}
