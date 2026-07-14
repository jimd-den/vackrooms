//! Program masses beside the dominant route: suite placement, structure,
//! ceilings, fixtures, and sparse interior partitioning.

use crate::domain::entities::architecture::*;
use crate::entities::models::Position;

use super::{EDGE_MARGIN, snap};

/// The suite program palette placed beside main corridors, roughly weighted.
/// Large, unfinished open-office masses dominate. Small private rooms remain
/// present only as occasional evidence that this once had an office program.
pub(super) const SUITE_PROGRAMS: [SpaceProgram; 12] = [
    SpaceProgram::OpenOffice,
    SpaceProgram::OpenOffice,
    SpaceProgram::OpenOffice,
    SpaceProgram::OpenOffice,
    SpaceProgram::OpenOffice,
    SpaceProgram::ConferenceRoom,
    SpaceProgram::ConferenceRoom,
    SpaceProgram::BreakRoom,
    SpaceProgram::Storage,
    SpaceProgram::ServerRoom,
    SpaceProgram::WaitingArea,
    SpaceProgram::PrivateOffice,
];

fn ceiling_height_for(program: SpaceProgram, aseed: f32) -> f32 {
    match program {
        SpaceProgram::Atrium => 4.8 + 0.6 * aseed,
        // Compression is a deliberate contrast, never the default ceiling.
        SpaceProgram::ServerRoom | SpaceProgram::Mechanical => 2.6 + 0.2 * aseed,
        SpaceProgram::Storage | SpaceProgram::RestroomCore => 3.0 + 0.3 * aseed,
        SpaceProgram::OpenOffice | SpaceProgram::ConferenceRoom => 3.5 + 0.7 * aseed,
        _ => 3.2 + 0.5 * aseed,
    }
}

fn structure_for(genome: &ArchitectGenome, aseed: f32) -> StructuralSystemInstance {
    let bay = snap(4.4 + 1.6 * aseed);
    let (bay_x, bay_z) = match genome.structural_system {
        StructuralSystem::DeepSpansWithBeams => (bay * 1.5, bay),
        _ => (bay, bay),
    };
    StructuralSystemInstance {
        system: genome.structural_system,
        bay_x,
        bay_z,
        phase: (snap(aseed * 4.0), snap(aseed * 8.0 % 4.0)),
        column_side: 0.4,
    }
}

/// Fixtures for a rectangular footprint in the genome's lighting language.
fn fixtures_for(
    genome: &ArchitectGenome,
    footprint: &Polygon2,
    lit: bool,
    aseed: f32,
) -> Vec<Fixture> {
    let (x0, z0, x1, z1) = footprint.bounds();
    let mut out = Vec::new();
    match genome.lighting_language {
        LightingLanguage::GridPanels => {
            let step = 2.4;
            let mut z = z0 + 1.2;
            while z < z1 - 0.8 {
                let mut x = x0 + 1.2;
                while x < x1 - 0.8 {
                    out.push(Fixture {
                        at: Position::new(snap(x), snap(z)),
                        half_x: 0.4,
                        half_z: 0.4,
                        lit,
                    });
                    x += step;
                }
                z += step;
            }
        }
        LightingLanguage::StripsAlongCirculation => {
            // Strips run down the room's long axis.
            let long_x = (x1 - x0) >= (z1 - z0);
            let (cx, cz) = ((x0 + x1) * 0.5, (z0 + z1) * 0.5);
            let step = 3.2;
            if long_x {
                let mut x = x0 + 1.6;
                while x < x1 - 1.2 {
                    out.push(Fixture {
                        at: Position::new(snap(x), snap(cz)),
                        half_x: 1.0,
                        half_z: 0.25,
                        lit,
                    });
                    x += step;
                }
            } else {
                let mut z = z0 + 1.6;
                while z < z1 - 1.2 {
                    out.push(Fixture {
                        at: Position::new(snap(cx), snap(z)),
                        half_x: 0.25,
                        half_z: 1.0,
                        lit,
                    });
                    z += step;
                }
            }
        }
        LightingLanguage::SparsePendants => {
            let n = 1 + (aseed * 2.0) as i32;
            for k in 0..n {
                let fx = x0 + (x1 - x0) * (0.3 + 0.4 * ((k as f32 * 0.618 + aseed) % 1.0));
                let fz = z0 + (z1 - z0) * (0.3 + 0.4 * ((k as f32 * 0.382 + aseed * 2.0) % 1.0));
                out.push(Fixture {
                    at: Position::new(snap(fx), snap(fz)),
                    half_x: 0.3,
                    half_z: 0.3,
                    lit,
                });
            }
        }
    }
    out
}

/// Interior partitioning is deliberately sparse. A private-office suite may
/// split once, producing one long interruption rather than a cell grid.
fn spaces_for(
    program: SpaceProgram,
    footprint: &Polygon2,
    genome: &ArchitectGenome,
    aseed: f32,
) -> Vec<Space> {
    if program != SpaceProgram::PrivateOffice {
        return Vec::new();
    }
    let (x0, z0, x1, z1) = footprint.bounds();
    let long_x = (x1 - x0) >= (z1 - z0);
    let span = if long_x { x1 - x0 } else { z1 - z0 };
    let want = genome.room_proportions.min_side.max(12.0);
    let n = ((span / want) as usize).clamp(1, 2);
    let n = if aseed > 0.62 { n } else { 1 };
    if n <= 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for k in 0..n {
        let (a, b) = (
            snap(k as f32 / n as f32 * span),
            snap((k + 1) as f32 / n as f32 * span),
        );
        let rect = if long_x {
            Polygon2::rect(x0 + a, z0, b - a, z1 - z0)
        } else {
            Polygon2::rect(x0, z0 + a, x1 - x0, b - a)
        };
        out.push(Space {
            program: SpaceProgram::PrivateOffice,
            footprint: rect,
        });
    }
    out
}

pub(super) fn aabb_overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32), gap: f32) -> bool {
    a.0 < b.2 + gap && b.0 < a.2 + gap && a.1 < b.3 + gap && b.1 < a.3 + gap
}

/// Places one suite beside a horizontal corridor segment. Returns `None` if
/// the footprint would leave the region, collide with a prior assembly, or
/// cross another corridor.
#[allow(clippy::too_many_arguments)]
pub(super) fn place_suite(
    id: u32,
    program: SpaceProgram,
    genome: &ArchitectGenome,
    aseed: f32,
    threshold_seed: f32,
    cursor_x: f32,
    corridor_z: f32,
    corridor_half: f32,
    side: f32, // +1 = suite on +Z side of corridor, -1 = -Z side
    origin: Position,
    size: f32,
    taken: &[(f32, f32, f32, f32)],
    spines: &[CirculationSpine],
) -> Option<AssemblyInstance> {
    let p = &genome.room_proportions;
    let w = snap((p.min_side + (p.max_side - p.min_side) * aseed).clamp(12.0, 24.0));
    let d = snap((w * (0.72 - 0.28 * p.elongation)).clamp(8.0, 18.0));

    let x0 = snap(cursor_x);
    let z_front = snap(if side > 0.0 {
        corridor_z + corridor_half
    } else {
        corridor_z - corridor_half - d
    });
    let footprint = Polygon2::rect(x0, z_front, w, d);
    let b = footprint.bounds();

    // Stay inside the region with margin.
    if b.0 < origin.x + EDGE_MARGIN
        || b.1 < origin.z + EDGE_MARGIN
        || b.2 > origin.x + size - EDGE_MARGIN
        || b.3 > origin.z + size - EDGE_MARGIN
    {
        return None;
    }
    if taken.iter().any(|t| aabb_overlap(*t, b, 0.8)) {
        return None;
    }
    // Don't let a suite swallow a *different* corridor (its front corridor
    // touching the footprint edge is fine and expected).
    let center = ((b.0 + b.2) * 0.5, (b.1 + b.3) * 0.5);
    for s in spines {
        if s.distance(center.0, center.1) < s.width * 0.5 + d.min(w) * 0.25 {
            return None;
        }
    }

    // Entrance on the corridor-facing wall. The genome biases the language,
    // but a narrow framed door is only about 8--14% of thresholds globally.
    // Most fronts dissolve into a room through an unframed or broad portal.
    let threshold = match genome.threshold_language {
        ThresholdLanguage::DoorWithLintel if threshold_seed < 0.14 => {
            ThresholdLanguage::DoorWithLintel
        }
        ThresholdLanguage::OpenPortal if threshold_seed < 0.08 => ThresholdLanguage::DoorWithLintel,
        ThresholdLanguage::WidePortal if threshold_seed < 0.10 => ThresholdLanguage::DoorWithLintel,
        ThresholdLanguage::OpenPortal if threshold_seed < 0.62 => ThresholdLanguage::OpenPortal,
        ThresholdLanguage::WidePortal if threshold_seed < 0.50 => ThresholdLanguage::OpenPortal,
        ThresholdLanguage::DoorWithLintel if threshold_seed < 0.50 => ThresholdLanguage::OpenPortal,
        _ => ThresholdLanguage::WidePortal,
    };
    let front_z = if side > 0.0 { b.1 } else { b.3 };
    let (width, lintel) = match threshold {
        ThresholdLanguage::DoorWithLintel => (1.2, Some(2.2)),
        ThresholdLanguage::OpenPortal => (4.8, None),
        ThresholdLanguage::WidePortal => (3.6, Some(3.0)),
    };
    let edge = width * 0.5 + 0.8;
    let door_x = snap((b.0 + w * (0.25 + 0.5 * threshold_seed)).clamp(b.0 + edge, b.2 - edge));
    let entrances = vec![Opening {
        center: Position::new(door_x, front_z),
        width,
        through_x_wall: true,
        lintel_units: lintel,
    }];

    let ceiling = CeilingZone {
        area: footprint.clone(),
        language: genome.ceiling_language,
        height_units: ceiling_height_for(program, aseed),
    };
    Some(AssemblyInstance {
        id,
        program,
        spaces: spaces_for(program, &footprint, genome, aseed),
        structure: structure_for(genome, aseed),
        ceiling_zones: vec![ceiling],
        fixtures: fixtures_for(genome, &footprint, true, aseed),
        service_voids: Vec::new(),
        corruption: CorruptionProfile::default(),
        entrances,
        footprint,
    })
}
