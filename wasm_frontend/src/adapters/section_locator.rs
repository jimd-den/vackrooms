//! Names the section of the world the player is standing in, for the HUD.
//!
//! The locator re-derives the deterministic region plan for the player's
//! current region (cached per region key, so the plan is recomputed only on
//! a region crossing) and classifies the position against, in priority
//! order: anomaly footprints, assembly footprints, and circulation spines.
//! Platform-agnostic — no browser types — so the classification is natively
//! unit-testable.

use vackrooms::domain::entities::anomaly::AnomalyKind;
use vackrooms::domain::entities::architecture::{RegionPlan, SpaceProgram};
use vackrooms::domain::entities::position::Position;
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::{REGION_SIZE, generate_region_plan};

/// Backrooms level ids, mirroring the engine's.
const LEVEL_HABITABLE: u32 = 1;
const LEVEL_GRASSLAND: u32 = 34;

pub struct SectionLocator {
    seed: u32,
    config: GeneratorConfig,
    noise: SimpleNoiseProvider,
    cached_region: Option<(i64, i64)>,
    plan: Option<RegionPlan>,
}

impl SectionLocator {
    pub fn new(seed: u32, config: GeneratorConfig) -> Self {
        Self {
            seed,
            config,
            noise: SimpleNoiseProvider::new(),
            cached_region: None,
            plan: None,
        }
    }

    /// Human-readable "where am I" line for the HUD, e.g.
    /// `LEVEL 0 · PILLAR EXPANSE · REGION (2, -1)`.
    pub fn describe(&mut self, level: u32, x: f32, z: f32) -> String {
        if level == LEVEL_GRASSLAND {
            return "LEVEL 34 · THE GRASSLAND".to_string();
        }
        if level == LEVEL_HABITABLE {
            // Level 1's sectors are a pure world-lattice function; no
            // region plan exists (or is needed) on the Habitable Zone.
            let sector =
                vackrooms::use_cases::level_one::Sector::at(self.seed, x, z).name();
            return format!("LEVEL 1 · HABITABLE ZONE · {sector}");
        }
        let rx = (x / REGION_SIZE).floor() as i64;
        let rz = (z / REGION_SIZE).floor() as i64;
        let zone = self.zone_name(rx, rz, x, z);
        format!("LEVEL {level} · {zone} · REGION ({rx}, {rz})")
    }

    fn zone_name(&mut self, rx: i64, rz: i64, x: f32, z: f32) -> &'static str {
        if self.cached_region != Some((rx, rz)) {
            self.plan = Some(generate_region_plan(
                self.seed,
                Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
                REGION_SIZE,
                &self.config,
                &self.noise,
            ));
            self.cached_region = Some((rx, rz));
        }
        let plan = self.plan.as_ref().expect("plan cached above");

        // Anomalies dominate: standing inside one matters more than which
        // office footprint it swallowed.
        for anomaly in &plan.anomalies {
            if anomaly.footprint.contains(x, z) {
                return match anomaly.kind {
                    AnomalyKind::PillarExpanse => "PILLAR EXPANSE",
                    AnomalyKind::BlackoutExpanse => "BLACKOUT EXPANSE",
                    AnomalyKind::PitLattice => "PIT LATTICE",
                    AnomalyKind::RedRoom => "RED LOOP",
                    AnomalyKind::ArchwayRoom => "ARCHWAY ANCHOR",
                };
            }
        }
        for assembly in &plan.assemblies {
            if assembly.footprint.contains(x, z) {
                return program_name(assembly.program);
            }
        }
        for corridor in &plan.corridors {
            if corridor.distance(x, z) <= corridor.width * 0.5 {
                return program_name(corridor.spine_kind);
            }
        }
        "OPEN FABRIC"
    }
}

fn program_name(program: SpaceProgram) -> &'static str {
    match program {
        SpaceProgram::Arrival => "ARRIVAL",
        SpaceProgram::Reception => "RECEPTION",
        SpaceProgram::MainCorridor => "MAIN CORRIDOR",
        SpaceProgram::SecondaryHall => "SECONDARY HALL",
        SpaceProgram::OpenOffice => "OPEN OFFICE",
        SpaceProgram::PrivateOffice => "PRIVATE OFFICE",
        SpaceProgram::ConferenceRoom => "CONFERENCE ROOM",
        SpaceProgram::WaitingArea => "WAITING AREA",
        SpaceProgram::BreakRoom => "BREAK ROOM",
        SpaceProgram::Storage => "STORAGE",
        SpaceProgram::ServerRoom => "SERVER ROOM",
        SpaceProgram::RestroomCore => "RESTROOM CORE",
        SpaceProgram::Stair => "STAIRWELL",
        SpaceProgram::Mechanical => "MECHANICAL",
        SpaceProgram::Atrium => "ATRIUM",
        SpaceProgram::AbandonedExpansion => "ABANDONED EXPANSION",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vackrooms::use_cases::region_plan::spawn_point;

    #[test]
    fn grassland_never_computes_a_plan() {
        let mut locator = SectionLocator::new(42, GeneratorConfig::low_spec());
        assert_eq!(
            locator.describe(34, 12345.0, -9876.0),
            "LEVEL 34 · THE GRASSLAND"
        );
        assert!(locator.plan.is_none(), "no region plan needed outdoors");
    }

    #[test]
    fn spawn_is_on_the_main_corridor() {
        let mut locator = SectionLocator::new(42, GeneratorConfig::low_spec());
        let at = spawn_point(42);
        let text = locator.describe(0, at.x, at.z);
        assert!(
            text.contains("MAIN CORRIDOR") || text.contains("ARRIVAL"),
            "spawn should sit on readable circulation, got: {text}"
        );
        assert!(text.starts_with("LEVEL 0 · "), "{text}");
        assert!(text.contains("REGION (0, 0)"), "{text}");
    }

    #[test]
    fn region_plan_is_cached_until_a_region_crossing() {
        let mut locator = SectionLocator::new(42, GeneratorConfig::low_spec());
        locator.describe(0, 5.0, 5.0);
        let first = locator.cached_region;
        locator.describe(0, 6.0, 6.0);
        assert_eq!(locator.cached_region, first, "same region: no recompute");
        locator.describe(0, 5.0 + REGION_SIZE, 5.0);
        assert_ne!(locator.cached_region, first, "crossing must recompute");
    }

    #[test]
    fn forced_anomaly_is_reported_at_its_own_center() {
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.forced_kind =
            Some(vackrooms::domain::entities::anomaly::AnomalyKind::PillarExpanse);
        let mut locator = SectionLocator::new(42, config);
        // Find the forced instance's center via the same plan the locator
        // uses; standing on its center must classify as the anomaly.
        let plan = generate_region_plan(
            42,
            Position::new(0.0, 0.0),
            REGION_SIZE,
            &config,
            &SimpleNoiseProvider::new(),
        );
        let forced = plan
            .anomalies
            .iter()
            .find(|a| a.kind == vackrooms::domain::entities::anomaly::AnomalyKind::PillarExpanse);
        if let Some(instance) = forced {
            let c = instance.footprint.center;
            let text = locator.describe(0, c.x, c.z);
            assert!(text.contains("PILLAR EXPANSE"), "{text}");
        }
    }
}
