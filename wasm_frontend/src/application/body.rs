//! The wanderer's embodied telemetry: pedometer, exertion, fatigue, and the
//! heart signal the interface listens to.
//!
//! Pure application math — no clocks, no DOM, no renderer knowledge. The
//! engine feeds it the one collision-resolved horizontal movement delta per
//! tick (relocations, teleports, and level transitions already excluded)
//! plus the survival state it owns, and reads back plain presentation-safe
//! numbers. Everything is finite, clamped, and deterministic: same inputs,
//! same outputs, on every machine.
//!
//! Units are explicit throughout: seconds, meters, meters/second, beats per
//! minute, and dimensionless normalized factors in `0..=1`.

/// World meters of resolved walking that count as one step. The pedometer
/// keeps the raw distance in `f64`; only whole steps are presented.
pub const STRIDE_LENGTH_M: f32 = 0.7;

/// Resting pulse of a healthy wanderer, beats per minute.
const BASELINE_BPM: f32 = 58.0;
/// Hard physiological ceiling; the target is clamped below this.
const MAX_BPM: f32 = 185.0;
/// Hard floor (bradycardia is not simulated).
const MIN_BPM: f32 = 45.0;
/// Seconds for the pulse to close ~63% of the gap while rising.
const BPM_RISE_TAU_S: f32 = 4.0;
/// Seconds for the pulse to recover while falling; dehydration and fatigue
/// stretch this, which is what "delayed recovery" feels like.
const BPM_FALL_TAU_S: f32 = 9.0;

/// Movement at terminal walk speed reaches this much of full intensity.
/// (Terminal speed is `player::ACCELERATION / player::FRICTION` = 2.5 u/s;
/// the model is tuned against it but only sees measured meters.)
const FULL_INTENSITY_SPEED_MPS: f32 = 2.5;

/// Equilibrium exertion of a healthy wanderer walking flat out. Baseline
/// exploration must stay cheap: 0.4 costs almost no capacity (see
/// `exertion_capacity_multiplier`).
const HEALTHY_WALK_EXERTION: f32 = 0.4;
/// Seconds for exertion to close ~63% of the gap while rising.
const EXERTION_RISE_TAU_S: f32 = 22.0;
/// Seconds for exertion to recover at rest with full recovery quality.
const EXERTION_FALL_TAU_S: f32 = 16.0;

/// Fatigue accrues only above this exertion level, so ordinary walking
/// never wears the wanderer down permanently.
const FATIGUE_EXERTION_THRESHOLD: f32 = 0.6;
/// Fatigue per second at maximal exertion (~80 s of redline to exhaustion).
const FATIGUE_RISE_PER_S: f32 = 0.0125;
/// Seconds of deliberate stillness before fatigue starts to mend.
const FATIGUE_REST_DELAY_S: f32 = 3.0;
/// Fatigue recovered per second of rest (~2 min of standing still from
/// fully exhausted). Rest is the loop; there is no sleep system yet — the
/// extension point is `resting` below.
const FATIGUE_RECOVERY_PER_S: f32 = 1.0 / 120.0;

/// Ambient °C above which heat begins to stress the body.
const HEAT_STRESS_FLOOR_C: f32 = 27.0;
/// °C span from first stress to full heat stress.
const HEAT_STRESS_SPAN_C: f32 = 12.0;

/// The lowest combined movement multiplier before death: severe depletion
/// is alarming and restrictive, never immobilizing.
const MOVEMENT_FLOOR: f32 = 0.35;

/// Presentation-safe classification of the heart signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulseSignal {
    /// Calm body: the HUD shows only a faint beat.
    Quiet,
    /// Working body: elevated pulse worth glancing at.
    Active,
    /// Something is wrong — heat, thirst, exhaustion, or injury.
    Strained,
    /// Imminent physiological danger.
    Critical,
}

/// Survival context the engine already owns, passed by value each tick.
#[derive(Debug, Clone, Copy)]
pub struct BodyContext {
    /// Hydration reserve, 0..=1.
    pub hydration: f32,
    /// Satiety reserve, 0..=1.
    pub satiety: f32,
    /// Physical condition, 0..=1 (0 is death).
    pub condition: f32,
    /// Felt ambient temperature, °C.
    pub ambient_c: f32,
}

/// One tick's presentation-safe reading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyReadout {
    /// Whole session steps.
    pub steps: u64,
    /// Current smoothed pulse, BPM.
    pub bpm: f32,
    /// Where the pulse is heading, BPM (debug only; never shown normally).
    pub target_bpm: f32,
    /// Normalized beat prominence 0..=1 (amplitude of the visual pulse).
    pub beat_intensity: f32,
    /// Signal classification for sparse color/reveal decisions.
    pub signal: PulseSignal,
    /// Short-term exertion, 0 fresh ..= 1 spent.
    pub exertion: f32,
    /// Long-term fatigue, 0 rested ..= 1 exhausted.
    pub fatigue: f32,
    /// Combined bounded movement multiplier, `MOVEMENT_FLOOR..=1`.
    pub movement_factor: f32,
}

/// Debug-only decomposition of the movement multiplier and pulse target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyDebug {
    pub exertion_capacity: f32,
    pub hydration_factor: f32,
    pub satiety_factor: f32,
    pub condition_factor: f32,
    pub fatigue_factor: f32,
    pub heat_stress: f32,
    pub recovery_quality: f32,
}

/// The wanderer's body. Advanced once per engine tick.
#[derive(Debug, Clone)]
pub struct Body {
    /// Precise resolved walking distance, meters. Steps derive from this.
    walked_m: f64,
    exertion: f32,
    fatigue: f32,
    bpm: f32,
    target_bpm: f32,
    /// Beat phase 0..1; wraps once per beat.
    beat_phase: f32,
    /// True on the tick the phase wrapped (a beat event).
    beat_event: bool,
    /// Seconds of near-stillness, for the fatigue rest gate.
    still_seconds: f32,
    context: BodyContext,
}

impl Default for Body {
    fn default() -> Self {
        Self {
            walked_m: 0.0,
            exertion: 0.0,
            fatigue: 0.0,
            bpm: BASELINE_BPM,
            target_bpm: BASELINE_BPM,
            beat_phase: 0.0,
            beat_event: false,
            still_seconds: 0.0,
            context: BodyContext {
                hydration: 1.0,
                satiety: 1.0,
                condition: 1.0,
                ambient_c: 21.0,
            },
        }
    }
}

fn finite01(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Smoothstep over `edge0..edge1`, safe for any finite input.
fn smooth(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Body {
    /// Death/respawn: strain resets with the new body, but the session
    /// pedometer keeps counting — the record of ground covered survives.
    pub fn reset_strain(&mut self) {
        self.exertion = 0.0;
        self.fatigue = 0.0;
        self.bpm = BASELINE_BPM;
        self.target_bpm = BASELINE_BPM;
        self.still_seconds = 0.0;
    }

    /// Advances one tick.
    ///
    /// `moved_m` must be the collision-resolved horizontal movement of this
    /// tick only — the caller excludes relocations, teleports, pit recovery,
    /// respawns, and level transitions before calling. Non-finite or absurd
    /// inputs are clamped; dt is bounded so a background-tab hitch cannot
    /// spike the model.
    pub fn advance(&mut self, dt_seconds: f32, moved_m: f32, context: BodyContext) {
        let dt = if dt_seconds.is_finite() {
            dt_seconds.clamp(0.0, 0.25)
        } else {
            0.0
        };
        let moved = if moved_m.is_finite() {
            // One tick of walking can never legitimately exceed a few meters.
            moved_m.clamp(0.0, 2.0)
        } else {
            0.0
        };
        self.context = BodyContext {
            hydration: finite01(context.hydration),
            satiety: finite01(context.satiety),
            condition: finite01(context.condition),
            ambient_c: if context.ambient_c.is_finite() {
                context.ambient_c.clamp(-50.0, 80.0)
            } else {
                21.0
            },
        };
        if dt <= 0.0 {
            self.beat_event = false;
            return;
        }

        self.walked_m += moved as f64;

        // -- movement intensity ------------------------------------------------
        let speed = moved / dt;
        let intensity = (speed / FULL_INTENSITY_SPEED_MPS).clamp(0.0, 1.5).powf(1.3);

        // -- stress inputs -----------------------------------------------------
        let c = self.context;
        let heat_stress = smooth(
            HEAT_STRESS_FLOOR_C,
            HEAT_STRESS_FLOOR_C + HEAT_STRESS_SPAN_C,
            c.ambient_c,
        );
        let dehydration = smooth(0.5, 0.0, c.hydration); // 0 fine .. 1 parched
        let hunger = smooth(0.35, 0.0, c.satiety); // 0 fine .. 1 starving
        let injury = smooth(0.6, 0.0, c.condition); // 0 fine .. 1 broken

        // Exertion cost multiplier: adverse conditions make the same walk
        // more expensive. Bounded to keep equilibria inside 0..1.
        let cost = (1.0 + 0.9 * heat_stress + 0.9 * dehydration + 0.5 * hunger + 0.6 * injury)
            .clamp(1.0, 3.0);
        // Recovery quality: how well rest actually restores.
        let recovery_quality = (1.0
            - 0.45 * dehydration
            - 0.30 * hunger
            - 0.25 * self.fatigue
            - 0.25 * heat_stress)
            .clamp(0.25, 1.0);

        // -- exertion: approach a cost-scaled target ---------------------------
        let target_exertion = (intensity * HEALTHY_WALK_EXERTION * cost).clamp(0.0, 1.0);
        if target_exertion > self.exertion {
            let blend = (dt / EXERTION_RISE_TAU_S).clamp(0.0, 1.0);
            self.exertion += (target_exertion - self.exertion) * blend;
        } else {
            let blend = (dt * recovery_quality / EXERTION_FALL_TAU_S).clamp(0.0, 1.0);
            self.exertion += (target_exertion - self.exertion) * blend;
        }
        self.exertion = finite01(self.exertion);

        // -- fatigue: sustained redline wears; deliberate rest mends -----------
        let resting = intensity < 0.05;
        self.still_seconds = if resting {
            (self.still_seconds + dt).min(3600.0)
        } else {
            0.0
        };
        if self.exertion > FATIGUE_EXERTION_THRESHOLD {
            let over = (self.exertion - FATIGUE_EXERTION_THRESHOLD)
                / (1.0 - FATIGUE_EXERTION_THRESHOLD);
            self.fatigue += over * FATIGUE_RISE_PER_S * dt;
        } else if resting && self.still_seconds >= FATIGUE_REST_DELAY_S {
            self.fatigue -= FATIGUE_RECOVERY_PER_S * recovery_quality * dt;
        }
        self.fatigue = finite01(self.fatigue);

        // -- pulse: target model with asymmetric smoothing ---------------------
        let deficit_stress = 0.5 * hunger + 0.5 * dehydration;
        let mut target = BASELINE_BPM
            + 78.0 * self.exertion
            + 26.0 * heat_stress.max(dehydration) // heat and thirst share a pathway
            + 10.0 * deficit_stress
            + 14.0 * self.fatigue
            + 24.0 * injury;
        target = target.clamp(MIN_BPM, MAX_BPM);
        self.target_bpm = target;
        let tau = if target > self.bpm {
            BPM_RISE_TAU_S
        } else {
            // Recovery slows when the body is depleted: stopping helps, but
            // the pulse settles reluctantly.
            BPM_FALL_TAU_S / recovery_quality
        };
        let blend = (dt / tau).clamp(0.0, 1.0);
        self.bpm += (target - self.bpm) * blend;
        self.bpm = self.bpm.clamp(MIN_BPM, MAX_BPM);

        // -- beat phase --------------------------------------------------------
        let next_phase = self.beat_phase + self.bpm / 60.0 * dt;
        self.beat_event = next_phase >= 1.0;
        self.beat_phase = next_phase.fract();
    }

    /// Whole steps walked this session.
    pub fn steps(&self) -> u64 {
        (self.walked_m / STRIDE_LENGTH_M as f64) as u64
    }

    /// Precise resolved walking distance, meters (debug/diagnostics).
    pub fn walked_m(&self) -> f64 {
        self.walked_m
    }

    /// True exactly on ticks where the beat phase wrapped.
    pub fn beat_event(&self) -> bool {
        self.beat_event
    }

    /// Beat phase, 0..1.
    pub fn beat_phase(&self) -> f32 {
        self.beat_phase
    }

    fn heat_stress(&self) -> f32 {
        smooth(
            HEAT_STRESS_FLOOR_C,
            HEAT_STRESS_FLOOR_C + HEAT_STRESS_SPAN_C,
            self.context.ambient_c,
        )
    }

    /// Debug decomposition of every factor feeding movement and pulse.
    pub fn debug(&self) -> BodyDebug {
        let c = self.context;
        let dehydration = smooth(0.5, 0.0, c.hydration);
        let hunger = smooth(0.35, 0.0, c.satiety);
        BodyDebug {
            exertion_capacity: self.exertion_capacity_multiplier(),
            hydration_factor: self.hydration_multiplier(),
            satiety_factor: self.satiety_multiplier(),
            condition_factor: self.condition_multiplier(),
            fatigue_factor: self.fatigue_multiplier(),
            heat_stress: self.heat_stress(),
            recovery_quality: (1.0
                - 0.45 * dehydration
                - 0.30 * hunger
                - 0.25 * self.fatigue
                - 0.25 * self.heat_stress())
            .clamp(0.25, 1.0),
        }
    }

    /// High exertion trims sustainable pace; quadratic so the healthy walk
    /// equilibrium (~0.4) costs almost nothing.
    fn exertion_capacity_multiplier(&self) -> f32 {
        1.0 - 0.35 * self.exertion * self.exertion
    }

    fn hydration_multiplier(&self) -> f32 {
        0.75 + 0.25 * smooth(0.0, 0.5, self.context.hydration)
    }

    fn satiety_multiplier(&self) -> f32 {
        0.85 + 0.15 * smooth(0.0, 0.35, self.context.satiety)
    }

    fn condition_multiplier(&self) -> f32 {
        0.6 + 0.4 * self.context.condition
    }

    fn fatigue_multiplier(&self) -> f32 {
        1.0 - 0.25 * self.fatigue
    }

    /// The single bounded multiplier the movement integrator applies:
    ///
    /// `effective = base * exertion_capacity * hydration * satiety *
    /// condition * fatigue`, floored at `MOVEMENT_FLOOR` so every nonfatal
    /// state keeps meaningful movement.
    pub fn movement_factor(&self) -> f32 {
        let product = self.exertion_capacity_multiplier()
            * self.hydration_multiplier()
            * self.satiety_multiplier()
            * self.condition_multiplier()
            * self.fatigue_multiplier();
        product.clamp(MOVEMENT_FLOOR, 1.0)
    }

    fn signal(&self) -> PulseSignal {
        let c = self.context;
        if c.condition < 0.25 || self.bpm > 160.0 {
            PulseSignal::Critical
        } else if self.bpm > 118.0 || c.hydration < 0.2 || self.fatigue > 0.7 || c.condition < 0.6
        {
            PulseSignal::Strained
        } else if self.bpm > 76.0 {
            PulseSignal::Active
        } else {
            PulseSignal::Quiet
        }
    }

    /// The presentation snapshot the HUD consumes.
    pub fn readout(&self) -> BodyReadout {
        let signal = self.signal();
        // Beat prominence follows how far the pulse sits above baseline.
        let beat_intensity = smooth(BASELINE_BPM + 4.0, 150.0, self.bpm).max(match signal {
            PulseSignal::Critical => 0.9,
            PulseSignal::Strained => 0.45,
            _ => 0.12,
        });
        BodyReadout {
            steps: self.steps(),
            bpm: self.bpm,
            target_bpm: self.target_bpm,
            beat_intensity,
            signal,
            exertion: self.exertion,
            fatigue: self.fatigue,
            movement_factor: self.movement_factor(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn healthy() -> BodyContext {
        BodyContext {
            hydration: 1.0,
            satiety: 1.0,
            condition: 1.0,
            ambient_c: 21.0,
        }
    }

    fn walk(body: &mut Body, seconds: f32, speed_mps: f32, context: BodyContext) {
        let ticks = (seconds / DT) as usize;
        for _ in 0..ticks {
            body.advance(DT, speed_mps * DT, context);
        }
    }

    #[test]
    fn steps_derive_from_resolved_distance_only() {
        let mut body = Body::default();
        walk(&mut body, 10.0, 2.5, healthy());
        let expected = (25.0 / STRIDE_LENGTH_M) as u64;
        assert!(
            body.steps().abs_diff(expected) <= 1,
            "steps {} expected ~{expected}",
            body.steps()
        );
    }

    #[test]
    fn non_walking_input_is_rejected() {
        let mut body = Body::default();
        // A teleport-sized delta in one tick is clamped, never counted whole.
        body.advance(DT, 500.0, healthy());
        assert!(body.walked_m() <= 2.0 + 1e-6);
        // Non-finite input counts nothing and poisons nothing.
        body.advance(DT, f32::NAN, healthy());
        body.advance(f32::INFINITY, 1.0, healthy());
        assert!(body.walked_m().is_finite());
        assert!(body.readout().bpm.is_finite());
    }

    #[test]
    fn exertion_rises_while_moving_and_recovers_at_rest() {
        let mut body = Body::default();
        walk(&mut body, 60.0, 2.5, healthy());
        let worked = body.readout().exertion;
        assert!(worked > 0.2, "exertion after a minute's walk: {worked}");
        walk(&mut body, 60.0, 0.0, healthy());
        assert!(
            body.readout().exertion < worked * 0.5,
            "rest must recover exertion: {} -> {}",
            worked,
            body.readout().exertion
        );
    }

    #[test]
    fn exertion_stays_bounded_under_hostile_dt() {
        let mut body = Body::default();
        for _ in 0..100 {
            body.advance(1e9, 2.0, healthy());
            body.advance(-5.0, 2.0, healthy());
            body.advance(f32::NAN, 2.0, healthy());
        }
        let r = body.readout();
        assert!((0.0..=1.0).contains(&r.exertion));
        assert!((0.0..=1.0).contains(&r.fatigue));
        assert!(r.bpm.is_finite() && r.bpm <= 185.0 && r.bpm >= 45.0);
    }

    #[test]
    fn dehydration_and_heat_worsen_cost_and_recovery() {
        let parched = BodyContext {
            hydration: 0.05,
            ambient_c: 36.0,
            ..healthy()
        };
        let mut fresh = Body::default();
        let mut thirsty = Body::default();
        walk(&mut fresh, 120.0, 2.5, healthy());
        walk(&mut thirsty, 120.0, 2.5, parched);
        assert!(
            thirsty.readout().exertion > fresh.readout().exertion + 0.1,
            "adverse conditions must cost more: {} vs {}",
            thirsty.readout().exertion,
            fresh.readout().exertion
        );

        // Same strain, then rest under each condition: recovery is slower
        // while parched and hot.
        let strained = thirsty.clone();
        let mut rest_fresh = strained.clone();
        let mut rest_thirsty = strained;
        walk(&mut rest_fresh, 30.0, 0.0, healthy());
        walk(&mut rest_thirsty, 30.0, 0.0, parched);
        assert!(
            rest_thirsty.readout().exertion > rest_fresh.readout().exertion,
            "parched recovery must lag: {} vs {}",
            rest_thirsty.readout().exertion,
            rest_fresh.readout().exertion
        );
        assert!(
            rest_thirsty.readout().bpm > rest_fresh.readout().bpm,
            "parched pulse recovery must lag"
        );
    }

    #[test]
    fn hunger_trims_sustainable_effort_without_instant_tachycardia() {
        let hungry = BodyContext {
            satiety: 0.0,
            ..healthy()
        };
        let mut body = Body::default();
        // A few seconds of hunger at rest must not spike the pulse.
        walk(&mut body, 5.0, 0.0, hungry);
        assert!(
            body.readout().bpm < 85.0,
            "ordinary hunger is not an arrhythmia: {}",
            body.readout().bpm
        );
        // But sustainable movement is reduced.
        assert!(body.movement_factor() < 1.0);
        let full = Body::default();
        assert!(body.movement_factor() < full.movement_factor());
    }

    #[test]
    fn fatigue_has_a_recoverable_rest_loop() {
        let mut body = Body::default();
        // Redline: hot, parched, hard sustained movement.
        let brutal = BodyContext {
            hydration: 0.0,
            satiety: 0.1,
            condition: 0.8,
            ambient_c: 38.0,
        };
        walk(&mut body, 300.0, 3.0, brutal);
        let tired = body.readout().fatigue;
        assert!(tired > 0.1, "sustained redline must accrue fatigue: {tired}");
        assert!(body.movement_factor() < 0.9);

        // Deliberate rest under decent conditions mends it.
        walk(&mut body, 300.0, 0.0, healthy());
        assert!(
            body.readout().fatigue < tired * 0.5,
            "rest must mend fatigue: {tired} -> {}",
            body.readout().fatigue
        );
    }

    #[test]
    fn ordinary_walking_never_accrues_fatigue() {
        let mut body = Body::default();
        walk(&mut body, 600.0, 2.5, healthy());
        assert_eq!(
            body.readout().fatigue,
            0.0,
            "healthy walking sits below the fatigue threshold"
        );
    }

    #[test]
    fn movement_stays_meaningful_before_death() {
        let mut body = Body::default();
        let dying = BodyContext {
            hydration: 0.0,
            satiety: 0.0,
            condition: 0.01,
            ambient_c: 40.0,
        };
        walk(&mut body, 600.0, 3.0, dying);
        let factor = body.movement_factor();
        assert!(
            (0.35..0.6).contains(&factor),
            "severe depletion restricts but never immobilizes: {factor}"
        );
    }

    #[test]
    fn bpm_rises_under_exertion_and_decays_smoothly_at_rest() {
        let mut body = Body::default();
        let resting = body.readout().bpm;
        walk(&mut body, 90.0, 2.5, healthy());
        let working = body.readout().bpm;
        assert!(working > resting + 15.0, "work raises pulse: {working}");

        // One tick of rest must not snap back — recovery is gradual.
        body.advance(DT, 0.0, healthy());
        assert!(body.readout().bpm > working - 1.0, "no instant reset");
        walk(&mut body, 120.0, 0.0, healthy());
        assert!(
            body.readout().bpm < resting + 8.0,
            "long rest settles toward baseline: {}",
            body.readout().bpm
        );
    }

    #[test]
    fn signal_escalates_under_critical_strain() {
        let mut calm = Body::default();
        calm.advance(DT, 0.0, healthy());
        assert_eq!(calm.readout().signal, PulseSignal::Quiet);

        let mut broken = Body::default();
        let critical = BodyContext {
            hydration: 0.0,
            satiety: 0.0,
            condition: 0.1,
            ambient_c: 38.0,
        };
        walk(&mut broken, 120.0, 2.5, critical);
        let r = broken.readout();
        assert_eq!(r.signal, PulseSignal::Critical);
        assert!(r.beat_intensity > 0.8, "critical beat is prominent");
        assert!(
            r.beat_intensity > calm.readout().beat_intensity,
            "strain is more prominent than calm"
        );
    }

    #[test]
    fn beat_phase_wraps_and_fires_events() {
        let mut body = Body::default();
        let mut beats = 0;
        for _ in 0..(60.0 / DT) as usize {
            body.advance(DT, 0.0, healthy());
            if body.beat_event() {
                beats += 1;
            }
        }
        // A minute at ~58 BPM: allow smoothing slack.
        assert!((54..=62).contains(&beats), "beats in a minute: {beats}");
    }

    #[test]
    fn strain_reset_keeps_the_pedometer() {
        let mut body = Body::default();
        walk(&mut body, 30.0, 2.5, healthy());
        let steps = body.steps();
        body.reset_strain();
        assert_eq!(body.steps(), steps);
        assert_eq!(body.readout().exertion, 0.0);
    }
}
