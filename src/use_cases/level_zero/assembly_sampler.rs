//! Sampling one planned assembly: perimeter walls pierced by entrances,
//! sparse interior partitions, structural bays (plus a renovator's
//! contradictory grid), and fixtures on the room's ceiling modules.

use crate::domain::entities::architecture::{
    AssemblyInstance, CeilingLanguage, StructuralSystem, StructuralSystemInstance,
};
use crate::use_cases::generate_chunk::LevelTuning;
use crate::use_cases::region_plan::PLAN_WALL_T;

use super::{BackroomsLevel, ColumnPlan, DOOR_HEIGHT};

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
        inside: bool,
        renovator: Option<&StructuralSystemInstance>,
        tuning: &LevelTuning,
        wx: f32,
        wz: f32,
    ) -> ColumnPlan {
        let walls_on = tuning.walls > 0.0;
        let zone = a.ceiling_zones.first();
        let mut ceiling_units = zone.map_or(3.4, |c| c.height_units);
        match zone.map(|c| c.language) {
            // Coffered reads as a shading pattern now (see the splat
            // shader), not stepped geometry.
            Some(CeilingLanguage::ExposedSoffit) => ceiling_units -= 0.2,
            _ => {}
        }
        let mut plan = ColumnPlan::open(ceiling_units);

        // Entrances pierce the wall band (and win over everything solid).
        for e in &a.entrances {
            let (da, db) = if e.through_x_wall {
                ((wx - e.center.x).abs(), (wz - e.center.z).abs())
            } else {
                ((wz - e.center.z).abs(), (wx - e.center.x).abs())
            };
            if da < e.width * 0.5 && db <= PLAN_WALL_T + 0.05 {
                plan.lintel_from_units = walls_on
                    .then_some(e.lintel_units.unwrap_or(0.0))
                    .filter(|u| *u > 0.0);
                return plan;
            }
        }

        if !inside {
            // Perimeter wall band.
            plan.solid = walls_on;
            return plan;
        }

        // Interior partitions: walls on space boundaries that are not the
        // footprint perimeter, each with a centered doorway.
        let fb = a.footprint.bounds();
        if walls_on {
            for s in &a.spaces {
                let sb = s.footprint.bounds();
                if wz >= sb.1 && wz <= sb.3 {
                    for plane in [sb.0, sb.2] {
                        if (plane - fb.0).abs() > 0.1
                            && (plane - fb.2).abs() > 0.1
                            && (wx - plane).abs() < PLAN_WALL_T * 0.5
                        {
                            let door_c = (sb.1 + sb.3) * 0.5;
                            if (wz - door_c).abs() < 0.6 {
                                plan.lintel_from_units = Some(DOOR_HEIGHT);
                            } else {
                                plan.solid = true;
                            }
                        }
                    }
                }
                if wx >= sb.0 && wx <= sb.2 {
                    for plane in [sb.1, sb.3] {
                        if (plane - fb.1).abs() > 0.1
                            && (plane - fb.3).abs() > 0.1
                            && (wz - plane).abs() < PLAN_WALL_T * 0.5
                        {
                            let door_c = (sb.0 + sb.2) * 0.5;
                            if (wx - door_c).abs() < 0.6 {
                                plan.lintel_from_units = Some(DOOR_HEIGHT);
                            } else {
                                plan.solid = true;
                            }
                        }
                    }
                }
            }
        }

        // Structure: the original grid, plus the renovator's contradictory
        // grid where a renovation overlays the assembly.
        if tuning.pillars > 0.0 && !plan.solid {
            if Self::on_column(&a.structure, wx, wz) {
                plan.solid = true;
            }
            if let Some(r) = renovator {
                if a.corruption.renovation_overlay && Self::on_column(r, wx, wz) {
                    // Renovation columns intrude on the original bay rhythm;
                    // the contradiction itself is the corruption. They stay
                    // ordinary wall material — Level 0 has no red masonry.
                    plan.solid = true;
                }
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
