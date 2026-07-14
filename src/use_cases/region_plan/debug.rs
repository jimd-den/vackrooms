//! Debug hook: ASCII plan view of one region.

use crate::domain::entities::anomaly::AnomalyKind;
use crate::domain::entities::architecture::*;

/// ASCII plan view at `step` world units per character. Corridors `=`, macro
/// anomalies `P/B/H` (pillar/blackout/pits), red loops `R`, assembly interiors
/// by program initial, entrances `+`, empty fabric `.`.
pub fn debug_region_ascii(plan: &RegionPlan, step: f32) -> String {
    let n = (plan.size_world / step) as usize;
    let mut out = String::with_capacity((n + 1) * n);
    for iz in 0..n {
        for ix in 0..n {
            let x = plan.origin_world.x + (ix as f32 + 0.5) * step;
            let z = plan.origin_world.z + (iz as f32 + 0.5) * step;
            let mut ch = '.';
            for a in &plan.assemblies {
                if a.footprint.contains(x, z) {
                    ch = match a.program {
                        SpaceProgram::OpenOffice => 'o',
                        SpaceProgram::PrivateOffice => 'p',
                        SpaceProgram::ConferenceRoom => 'c',
                        SpaceProgram::BreakRoom => 'b',
                        SpaceProgram::Storage => 's',
                        SpaceProgram::ServerRoom => 'v',
                        SpaceProgram::WaitingArea => 'w',
                        SpaceProgram::AbandonedExpansion => 'x',
                        SpaceProgram::Atrium => 'A',
                        SpaceProgram::Stair => 'S',
                        _ => 'r',
                    };
                }
            }
            for anomaly in &plan.anomalies {
                if anomaly.contains(x, z) {
                    ch = match anomaly.kind {
                        AnomalyKind::PillarExpanse => 'P',
                        AnomalyKind::BlackoutExpanse => 'B',
                        AnomalyKind::PitLattice => 'H',
                        AnomalyKind::RedRoom => 'R',
                        AnomalyKind::ArchwayRoom => 'M',
                    };
                }
            }
            for s in &plan.corridors {
                if s.distance(x, z) <= s.width * 0.5 {
                    ch = '=';
                }
            }
            for a in &plan.assemblies {
                for e in &a.entrances {
                    if (x - e.center.x).abs() < e.width * 0.5 && (z - e.center.z).abs() < step {
                        ch = '+';
                    }
                }
            }
            out.push(ch);
        }
        out.push('\n');
    }
    out
}
