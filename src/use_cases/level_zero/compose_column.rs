//! Column composition: the one priority order in which Level 0's systems
//! claim a world column. Circulation beats anomalies beats assemblies beats
//! fabric — and archway anchors freeze every mutable family around them.

use crate::domain::entities::anomaly::{AnomalyKind, RealitySnapshot};
use crate::domain::entities::architecture::{
    RegionPlan, SpaceProgram, StructuralSystemInstance,
};
use crate::use_cases::anomalies::geometry::sample_anomaly;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::red_rooms::geometry::sample_red_room;
use crate::use_cases::region_plan::{PLAN_WALL_T, region_index};

use super::{BackroomsLevel, ColumnPlan};

#[cfg(test)]
use crate::use_cases::generate_chunk::LevelTuning;

impl BackroomsLevel {

    /// The architectural column plan: corridor beats assembly beats fabric.
    #[cfg(test)]
    pub(crate) fn plan_column(
        plan: &RegionPlan,
        noise: &dyn NoiseProvider,
        seed: u32,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        Self::plan_column_in_reality(
            plan,
            noise,
            seed,
            &GeneratorConfig::low_spec().with_tuning(*tuning),
            &RealitySnapshot::empty(),
            wx,
            wz,
        )
    }

    pub(crate) fn plan_column_in_reality(
        plan: &RegionPlan,
        noise: &dyn NoiseProvider,
        seed: u32,
        config: &GeneratorConfig,
        reality: &RealitySnapshot,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        let tuning = &config.tuning;
        // -- circulation ----------------------------------------------------
        let mut in_corridor = false;
        let mut corridor_ceiling = 0.0f32;
        let mut corridor_light = false;
        let mut corridor_wall = false;
        let mut corridor_gap = false;
        for s in &plan.corridors {
            let (d, along, is_horizontal) = s.nearest(wx, wz);
            let half = s.width * 0.5;
            if d <= half {
                in_corridor = true;
                corridor_ceiling =
                    corridor_ceiling.max(Self::corridor_ceiling(s, noise, seed, wx, wz));
                // Light strip modules follow the corridor in world space.
                if d < 0.45 && along.rem_euclid(4.0) < 1.0 {
                    corridor_light = true;
                }
            } else if d <= half + PLAN_WALL_T {
                corridor_wall = true;
                corridor_ceiling =
                    corridor_ceiling.max(Self::corridor_ceiling(s, noise, seed, wx, wz));
                if Self::corridor_edge_opens(s, noise, seed, along, is_horizontal, wx, wz) {
                    corridor_gap = true;
                }
            }
        }
        if in_corridor {
            return ColumnPlan {
                light: corridor_light && tuning.lights > 0.0,
                ..ColumnPlan::open(corridor_ceiling)
            };
        }

        // Archway anchors take precedence over every hostile family, and any
        // hostile column within an anchor's margin generates as if no epoch
        // had ever advanced: arch rooms and their surroundings are immune to
        // non-Euclidean transformation by construction, not by policy.
        if let Some(anchor) = plan
            .anomalies
            .iter()
            .find(|a| a.kind == AnomalyKind::ArchwayRoom && a.contains(wx, wz))
        {
            return sample_anomaly(anchor, noise, seed, config, reality, wx, wz);
        }
        let near_anchor = plan.anomalies.iter().any(|a| {
            a.kind == AnomalyKind::ArchwayRoom
                && a.footprint.bounds().expanded(3.2).contains(wx, wz)
        });

        if let Some(instance) = plan
            .anomalies
            .iter()
            .filter(|a| {
                a.kind != AnomalyKind::RedRoom
                    && a.kind != AnomalyKind::ArchwayRoom
                    && a.contains(wx, wz)
            })
            .max_by(|a, b| {
                a.normalized_depth(wx, wz)
                    .total_cmp(&b.normalized_depth(wx, wz))
            })
        {
            let frozen = RealitySnapshot::empty();
            let effective_reality = if near_anchor { &frozen } else { reality };
            return sample_anomaly(instance, noise, seed, config, effective_reality, wx, wz);
        }

        // -- assemblies -------------------------------------------------------
        {
            let renovator_structure = plan.architects.get(1).map(|g| StructuralSystemInstance {
                system: g.structural_system,
                bay_x: 3.6,
                bay_z: 4.4,
                phase: (1.6, 2.4),
                column_side: 0.4,
            });
            for a in &plan.assemblies {
                let b = a.footprint.bounds();
                let t = PLAN_WALL_T;
                if wx < b.0 - t || wx > b.2 + t || wz < b.1 - t || wz > b.3 + t {
                    continue;
                }
                let inside = a.footprint.contains(wx, wz);
                let mut base =
                    Self::assembly_column(a, inside, renovator_structure.as_ref(), tuning, wx, wz);
                // A stair assembly shapes its interior as a flight: the
                // vertical link that reserved it decides whether the flight
                // lands or climbs endlessly. Stairs are architecture, so a
                // walls=0 debug world flattens them along with everything.
                if a.program == SpaceProgram::Stair && inside && !base.solid && tuning.walls > 0.0 {
                    let rx = region_index(plan.origin_world.x + 0.1);
                    let rz = region_index(plan.origin_world.z + 0.1);
                    let kind = crate::use_cases::world_topology::vertical_link_for_region(
                        seed, noise, rx, rz,
                    )
                    .map(|link| link.kind)
                    .unwrap_or(
                        crate::domain::entities::world_topology::VerticalLinkKind::OrdinaryStair,
                    );
                    crate::use_cases::vertical_circulation::apply_stair_profile(
                        a, kind, &mut base, wx, wz,
                    );
                }
                if a.corruption.red_room
                    && let Some(red) = plan.anomalies.iter().find(|r| {
                        r.kind == AnomalyKind::RedRoom
                            && r.footprint
                                .bounds()
                                .expanded(PLAN_WALL_T + 0.05)
                                .contains(wx, wz)
                    })
                {
                    return sample_red_room(red, base, config, reality, wx, wz);
                }
                return base;
            }
        }

        // -- corridor edge walls through fabric -------------------------------
        if corridor_wall && tuning.walls > 0.0 && !corridor_gap {
            return ColumnPlan {
                solid: true,
                ..ColumnPlan::open(corridor_ceiling.max(3.2))
            };
        }

        // -- the endless unplanned office fabric -------------------------------
        Self::column_plan(noise, seed, tuning, wx, wz)
    }
}
