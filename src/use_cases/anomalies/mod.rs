//! Level 0 anomaly planning use case.
//!
//! This module composes two independent sources of immutable plans:
//! world-lattice macro anomalies and Red Rooms promoted from architectural
//! assemblies. Voxelization and encounter-state transitions live elsewhere.

pub mod config;
pub(crate) mod determinism;
pub(crate) mod geometry;
mod macro_planner;

use crate::domain::entities::anomaly::{AnomalyInstance, WorldBounds};
use crate::domain::entities::architecture::AssemblyInstance;
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::red_rooms::planning::plan_red_rooms;

/// Returns every immutable anomaly instance whose footprint overlaps a region.
///
/// Identity is independent of the query partition: callers may ask through
/// adjacent regions or different voxel sizes and receive the same instance.
pub fn plan_anomalies_for_region(
    seed: u32,
    region_origin: Position,
    region_size: f32,
    assemblies: &[AssemblyInstance],
    protected_point: Position,
    config: &GeneratorConfig,
) -> Vec<AnomalyInstance> {
    let query = WorldBounds::new(
        region_origin.x,
        region_origin.z,
        region_origin.x + region_size,
        region_origin.z + region_size,
    );
    let mut instances = macro_planner::plan_macro_anomalies(seed, query, protected_point, config);
    instances.extend(plan_red_rooms(seed, region_origin, assemblies, config));
    instances.sort_by_key(|instance| instance.id);
    instances.dedup_by_key(|instance| instance.id);
    instances
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::anomaly::AnomalyKind;
    use crate::domain::entities::architecture::{
        CorruptionProfile, Opening, Polygon2, SpaceProgram, StructuralSystem,
        StructuralSystemInstance,
    };

    fn red_assembly() -> AssemblyInstance {
        AssemblyInstance {
            id: 7,
            program: SpaceProgram::PrivateOffice,
            footprint: Polygon2::rect(0.0, 0.0, 12.0, 8.0),
            entrances: vec![Opening {
                center: Position::new(6.0, 0.0),
                width: 1.2,
                through_x_wall: true,
                lintel_units: Some(2.2),
            }],
            spaces: Vec::new(),
            structure: StructuralSystemInstance {
                system: StructuralSystem::RegularGrid,
                bay_x: 4.0,
                bay_z: 4.0,
                phase: (0.0, 0.0),
                column_side: 0.4,
            },
            ceiling_zones: Vec::new(),
            fixtures: Vec::new(),
            service_voids: Vec::new(),
            corruption: CorruptionProfile {
                red_room: true,
                ..CorruptionProfile::default()
            },
        }
    }

    #[test]
    fn disabled_anomalies_produce_no_instances() {
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.frequency = 0.0;
        assert!(
            plan_anomalies_for_region(
                42,
                Position::new(800.0, 800.0),
                80.0,
                &[],
                Position::new(0.0, 0.0),
                &config,
            )
            .is_empty()
        );
    }

    #[test]
    fn forced_macro_kind_survives_zero_frequency() {
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.forced_kind = Some(AnomalyKind::PillarExpanse);
        config.anomalies.frequency = 0.0;
        let plans = plan_anomalies_for_region(
            42,
            Position::new(0.0, 0.0),
            80.0,
            &[],
            Position::new(6.0, 40.0),
            &config,
        );
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].kind, AnomalyKind::PillarExpanse);
    }

    #[test]
    fn compatibility_api_returns_forced_red_room_at_zero_frequency() {
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.frequency = 0.0;
        config.anomalies.red_rooms = 0.0;
        config.anomalies.forced_kind = Some(AnomalyKind::RedRoom);
        let plans = crate::use_cases::anomaly_plan::plan_anomalies_for_region(
            42,
            Position::new(0.0, 0.0),
            80.0,
            &[red_assembly()],
            Position::new(6.0, 0.0),
            &config,
        );
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].kind, AnomalyKind::RedRoom);
        assert_eq!(plans[0].gates.len(), 2);
    }

    #[test]
    fn region_partition_keeps_macro_instance_identity() {
        let mut config = GeneratorConfig::low_spec();
        config.anomalies.frequency = 4.0;
        let protected = Position::new(0.0, 0.0);
        let mut shared = false;
        for region_z in 4..20 {
            for region_x in 4..20 {
                let left = plan_anomalies_for_region(
                    42,
                    Position::new(region_x as f32 * 80.0, region_z as f32 * 80.0),
                    80.0,
                    &[],
                    protected,
                    &config,
                );
                let right = plan_anomalies_for_region(
                    42,
                    Position::new((region_x + 1) as f32 * 80.0, region_z as f32 * 80.0),
                    80.0,
                    &[],
                    protected,
                    &config,
                );
                if let Some(one) = left
                    .iter()
                    .find(|one| right.iter().any(|two| two.id == one.id))
                {
                    assert_eq!(one, right.iter().find(|two| two.id == one.id).unwrap());
                    shared = true;
                    break;
                }
            }
            if shared {
                break;
            }
        }
        assert!(shared, "density scan found no cross-region instance");
    }
}
