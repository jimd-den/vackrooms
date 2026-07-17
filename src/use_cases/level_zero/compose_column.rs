//! Column composition: the one priority order in which Level 0's systems
//! claim a world column. Circulation beats anomalies beats assemblies beats
//! fabric — and archway anchors freeze every mutable family around them.

use crate::domain::entities::anomaly::{AnomalyKind, RealitySnapshot};
use crate::domain::entities::architecture::{
    CirculationSpine, RegionPlan, SpaceProgram, StructuralSystemInstance,
};
use crate::use_cases::anomalies::geometry::sample_anomaly;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::red_rooms::geometry::sample_red_room;
use crate::use_cases::region_plan::{PLAN_WALL_T, region_index, spawn_point};

use super::circulation_sampler::SPAWN_READABLE_RADIUS;
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
        // Edge-gap decisions are deferred: they read the Peripheral Shift,
        // and whether this column's fabric is frozen (spawn radius, arch
        // anchors) is only known further down.
        let mut gap_probes: Vec<(&CirculationSpine, f32, bool)> = Vec::new();
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
                gap_probes.push((s, along, is_horizontal));
            }
        }
        if in_corridor {
            let mut column = ColumnPlan {
                light: corridor_light && tuning.lights > 0.0,
                ..ColumnPlan::open(corridor_ceiling)
            };
            // Geometry priority and presentation priority are different
            // things: the corridor owns passability — its route through a
            // blackout stays carved open — but a blackout owns the *dark*.
            // Past the approach band the corridor's light strips die like
            // every other fixture, thinning through the same gradient, so
            // circulation never reads as a lit tunnel through the anomaly.
            if let Some(blackout) = plan
                .anomalies
                .iter()
                .find(|a| a.kind == AnomalyKind::BlackoutExpanse && a.contains(wx, wz))
            {
                let depth = blackout.normalized_depth(wx, wz);
                if depth > 0.22 {
                    column.light = false;
                } else if depth > 0.0 && column.light {
                    let module = (wx.floor() as i64) ^ ((wz.floor() as i64) << 17);
                    let hash = {
                        let mut h = (blackout.id ^ module as u64)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        h ^= h >> 33;
                        (h >> 40) as f32 / (1u64 << 24) as f32
                    };
                    column.light = hash < 1.0 - depth / 0.22;
                }
            }
            return column;
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

        // -- the endless unplanned office fabric -------------------------------
        // Ordinary fabric is the Peripheral Shift's territory — with two
        // sanctuaries. Arch-anchor surroundings are immune by construction
        // (the same guarantee their hostile-family freeze gives above), and
        // the spawn opening sequence never rearranges: the first hallways a
        // wanderer learns are the last ones the level may take away.
        let sp = spawn_point(seed);
        let near_spawn = (wx - sp.x) * (wx - sp.x) + (wz - sp.z) * (wz - sp.z)
            < SPAWN_READABLE_RADIUS * SPAWN_READABLE_RADIUS;
        let still_fabric = RealitySnapshot::empty();
        let fabric_reality = if near_anchor || near_spawn {
            &still_fabric
        } else {
            reality
        };

        // -- corridor edge walls through fabric -------------------------------
        // The mouths onto circulation drift and seal with the same reality
        // as the fabric they open into (frozen in the sanctuaries above).
        if corridor_wall && tuning.walls > 0.0 {
            let corridor_gap = gap_probes.iter().any(|&(s, along, is_horizontal)| {
                Self::corridor_edge_opens(
                    s,
                    noise,
                    seed,
                    fabric_reality,
                    along,
                    is_horizontal,
                    wx,
                    wz,
                )
            });
            if !corridor_gap {
                return ColumnPlan {
                    solid: true,
                    ..ColumnPlan::open(corridor_ceiling.max(3.2))
                };
            }
        }
        Self::column_plan_in_reality(noise, seed, tuning, fabric_reality, wx, wz)
    }
}
