//! Sampling one planned assembly: perimeter walls pierced by entrances,
//! sparse interior partitions, structural bays (plus a renovator's
//! contradictory grid), and fixtures on the room's ceiling modules.

use crate::domain::entities::architecture::{
    AssemblyInstance, CeilingLanguage, CeilingZone, StructuralSystem, StructuralSystemInstance,
};
use crate::use_cases::generate_chunk::LevelTuning;

use super::{BackroomsLevel, ColumnPlan};

fn distance_to_zone(zone: &CeilingZone, wx: f32, wz: f32) -> f32 {
    let (x0, z0, x1, z1) = zone.area.bounds();
    let dx = if wx < x0 {
        x0 - wx
    } else if wx > x1 {
        wx - x1
    } else {
        0.0
    };
    let dz = if wz < z0 {
        z0 - wz
    } else if wz > z1 {
        wz - z1
    } else {
        0.0
    };
    dx * dx + dz * dz
}

/// A ceiling zone owns the columns inside its polygon. Wall bands can sit a
/// fraction outside both adjacent polygons, so they inherit the nearest zone
/// rather than falling back to whichever zone happened to be stored first.
fn ceiling_zone_at(assembly: &AssemblyInstance, wx: f32, wz: f32) -> Option<&CeilingZone> {
    assembly
        .ceiling_zones
        .iter()
        .find(|zone| zone.area.contains(wx, wz))
        .or_else(|| {
            assembly
                .ceiling_zones
                .iter()
                .min_by(|a, b| distance_to_zone(a, wx, wz).total_cmp(&distance_to_zone(b, wx, wz)))
        })
}

impl BackroomsLevel {
    /// Is (wx, wz) on a structural column of this system?
    pub(super) fn on_column(st: &StructuralSystemInstance, wx: f32, wz: f32) -> bool {
        if st.system == StructuralSystem::CoreAndShell {
            // Core-and-shell designers hide columns in walls; none inside.
            return false;
        }
        let mut mx = (wx - st.phase.0).rem_euclid(st.bay_x);
        let mz = (wz - st.phase.1).rem_euclid(st.bay_z);
        if st.system == StructuralSystem::OffsetGrid {
            let row = ((wz - st.phase.1) / st.bay_z).floor() as i64;
            if row.rem_euclid(2) == 1 {
                mx = (wx - st.phase.0 + st.bay_x * 0.5).rem_euclid(st.bay_x);
            }
        }
        mx < st.column_side && mz < st.column_side
    }

    /// Column plan for a point inside an assembly footprint or its
    /// surrounding wall band (`inside == false`).
    pub(super) fn assembly_column(
        a: &AssemblyInstance,
        renovator: Option<&StructuralSystemInstance>,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        let walls_on = tuning.walls > 0.0;
        let zone = ceiling_zone_at(a, wx, wz);
        let mut ceiling_units = zone.map_or(3.4, |c| c.height_units);
        // Coffered reads as a shading pattern now (see the splat shader),
        // not stepped geometry.
        if let Some(CeilingLanguage::ExposedSoffit) = zone.map(|c| c.language) {
            ceiling_units -= 0.2;
        }
        let mut plan = ColumnPlan::open(ceiling_units);

        // Hosts own wall geometry; openings cut only their referenced host.
        // This remains stable when a grammar moves, removes, or duplicates a
        // wall because the relationship is identity-based, not rediscovered
        // from coincident room bounds at every sample.
        let mut covering_hosts = 0usize;
        let mut cut_hosts = 0usize;
        let mut lintel_units = 0.0f32;
        for host in a.hosts.iter().filter(|host| host.contains_plan(wx, wz)) {
            covering_hosts += 1;
            ceiling_units = ceiling_units.max(host.top_units);
            if let Some(opening) = a.openings.iter().find(|opening| {
                opening.host == host.id && opening.contains_plan(host, wx, wz)
            }) {
                cut_hosts += 1;
                lintel_units = lintel_units.max(opening.lintel_units.unwrap_or(0.0));
            }
        }
        if covering_hosts > 0 {
            plan.ceiling_units = ceiling_units;
            if cut_hosts == covering_hosts {
                plan.lintel_from_units = walls_on
                    .then_some(lintel_units)
                    .filter(|height| *height > 0.0);
            } else {
                plan.solid = walls_on;
            }
            return plan;
        }

        // Structure: the original grid, plus the renovator's contradictory
        // grid where a renovation overlays the assembly.
        if tuning.pillars > 0.0 && !plan.solid {
            if Self::on_column(&a.structure, wx, wz) {
                plan.solid = true;
            }
            if let Some(r) = renovator
                && a.corruption.renovation_overlay
                && Self::on_column(r, wx, wz)
            {
                // Renovation columns intrude on the original bay rhythm;
                // the contradiction itself is the corruption. They stay
                // ordinary wall material — Level 0 has no red masonry.
                plan.solid = true;
            }
        }

        // Fixtures, tied to the assembly's ceiling modules. In a red room
        // every fixture burns red: the anomaly is the room's light, applied
        // after all geometry decisions and on the same fixture spacing.
        if !plan.solid && tuning.lights > 0.0 {
            for f in &a.fixtures {
                if f.lit && (wx - f.at.x).abs() <= f.half_x && (wz - f.at.z).abs() <= f.half_z {
                    plan.light = true;
                    plan.red_light = a.corruption.red_room;
                    break;
                }
            }
        }
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::architecture::{
        AssemblyInstance, CeilingZone, CorruptionProfile, HostSegment, Polygon2, SpaceProgram,
        StructuralSystem,
    };

    fn zoned_assembly() -> AssemblyInstance {
        let footprint = Polygon2::rect(0.0, 0.0, 10.0, 4.0);
        AssemblyInstance {
            id: 1,
            program: SpaceProgram::OpenOffice,
            footprint: footprint.clone(),
            hosts: HostSegment::rectangular_shell(&footprint, 0.4, 5.0),
            openings: Vec::new(),
            spaces: Vec::new(),
            structure: StructuralSystemInstance {
                system: StructuralSystem::CoreAndShell,
                bay_x: 4.0,
                bay_z: 4.0,
                phase: (0.0, 0.0),
                column_side: 0.4,
            },
            ceiling_zones: vec![
                CeilingZone {
                    area: Polygon2::rect(0.0, 0.0, 5.0, 4.0),
                    language: CeilingLanguage::FlatTiles,
                    height_units: 3.2,
                },
                CeilingZone {
                    area: Polygon2::rect(5.0, 0.0, 5.0, 4.0),
                    language: CeilingLanguage::FlatTiles,
                    height_units: 4.8,
                },
            ],
            fixtures: Vec::new(),
            service_voids: Vec::new(),
            corruption: CorruptionProfile::default(),
        }
    }

    #[test]
    fn assembly_columns_use_the_spatially_containing_ceiling_zone() {
        let assembly = zoned_assembly();
        let tuning = LevelTuning::default();
        let low = BackroomsLevel::assembly_column(&assembly, None, &tuning, 2.0, 2.0);
        let high = BackroomsLevel::assembly_column(&assembly, None, &tuning, 8.0, 2.0);
        assert_eq!(low.ceiling_units, 3.2);
        assert_eq!(high.ceiling_units, 4.8);
    }
}
