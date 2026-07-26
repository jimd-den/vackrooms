//! Program masses beside the dominant route: suite placement, structure,
//! ceilings, fixtures, and sparse interior partitioning.

use crate::domain::entities::architecture::*;
use crate::domain::entities::position::Position;

use super::{EDGE_MARGIN, PLAN_WALL_T, snap};

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
    // A designer with no tolerance for asymmetry (1) centers the column
    // grid on the footprint instead of letting it fall wherever an offset
    // phase lands; one with none of that tolerance (0) keeps today's
    // freely offset grid.
    let tol = genome.tolerance_for_symmetry.clamp(0.0, 1.0);
    let phase = (
        snap(aseed * 4.0 * (1.0 - tol)),
        snap(aseed * 8.0 % 4.0 * (1.0 - tol)),
    );
    StructuralSystemInstance {
        system: genome.structural_system,
        bay_x,
        bay_z,
        phase,
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
            // A designer with high tolerance for symmetry mirrors and
            // centers pendants along the room's long axis instead of
            // scattering them by golden-ratio jitter; `tol` blends between
            // the two rather than switching sharply between them.
            let tol = genome.tolerance_for_symmetry.clamp(0.0, 1.0);
            let (cx, cz) = ((x0 + x1) * 0.5, (z0 + z1) * 0.5);
            let long_x = (x1 - x0) >= (z1 - z0);
            for k in 0..n {
                let jx = x0 + (x1 - x0) * (0.3 + 0.4 * ((k as f32 * 0.618 + aseed) % 1.0));
                let jz = z0 + (z1 - z0) * (0.3 + 0.4 * ((k as f32 * 0.382 + aseed * 2.0) % 1.0));
                let centered = if n > 1 {
                    k as f32 / (n - 1) as f32 - 0.5
                } else {
                    0.0
                };
                let (sx, sz) = if long_x {
                    (cx + centered * (x1 - x0) * 0.6, cz)
                } else {
                    (cx, cz + centered * (z1 - z0) * 0.6)
                };
                let fx = jx + (sx - jx) * tol;
                let fz = jz + (sz - jz) * tol;
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

/// Recursively divide a program envelope along its dominant axis. This is a
/// small shape grammar, not a random room packer: every child remains inside
/// its parent, split planes snap to the plan lattice, and stable path keys
/// make the result independent of traversal, chunks, and voxel resolution.
#[derive(Default)]
struct SpaceLayout {
    spaces: Vec<Space>,
    partitions: Vec<(Position, Position)>,
}

#[cfg(test)]
fn spaces_for(
    program: SpaceProgram,
    footprint: &Polygon2,
    genome: &ArchitectGenome,
    aseed: f32,
) -> Vec<Space> {
    space_layout_for(program, footprint, genome, aseed).spaces
}

fn space_layout_for(
    program: SpaceProgram,
    footprint: &Polygon2,
    genome: &ArchitectGenome,
    aseed: f32,
) -> SpaceLayout {
    if !matches!(
        program,
        SpaceProgram::PrivateOffice | SpaceProgram::OpenOffice | SpaceProgram::ConferenceRoom
    ) {
        return SpaceLayout::default();
    }
    let split_threshold = (1.0 - genome.furnishing_density).clamp(0.05, 0.95);
    if aseed <= split_threshold {
        return SpaceLayout::default();
    }

    let rule = SubdivisionRule {
        target_side: (genome.room_proportions.min_side * 0.42).clamp(4.0, 7.2),
        max_depth: 1 + (genome.furnishing_density.clamp(0.0, 1.0) * 2.5) as u8,
        seed: aseed,
    };
    let mut leaves = Vec::new();
    let mut partitions = Vec::new();
    subdivide_space(footprint.bounds(), 0, 1, &rule, &mut leaves, &mut partitions);
    if leaves.len() <= 1 {
        return SpaceLayout::default();
    }
    SpaceLayout {
        spaces: leaves
            .into_iter()
            .map(|(x0, z0, x1, z1)| Space {
                program,
                footprint: Polygon2::rect(x0, z0, x1 - x0, z1 - z0),
            })
            .collect(),
        partitions,
    }
}

struct SubdivisionRule {
    target_side: f32,
    max_depth: u8,
    seed: f32,
}

fn subdivide_space(
    bounds: (f32, f32, f32, f32),
    depth: u8,
    path: u32,
    rule: &SubdivisionRule,
    leaves: &mut Vec<(f32, f32, f32, f32)>,
    partitions: &mut Vec<(Position, Position)>,
) {
    let width = bounds.2 - bounds.0;
    let depth_units = bounds.3 - bounds.1;
    let split_x = width >= depth_units;
    let span = if split_x { width } else { depth_units };
    if depth >= rule.max_depth || span < rule.target_side * 2.05 {
        leaves.push(bounds);
        return;
    }

    let jitter = keyed_unit(rule.seed, path) - 0.5;
    let split = (span * (0.5 + jitter * 0.16) / 0.4).round() * 0.4;
    if split < rule.target_side || span - split < rule.target_side {
        leaves.push(bounds);
        return;
    }
    let (a, b) = if split_x {
        let plane = bounds.0 + split;
        partitions.push((
            Position::new(plane, bounds.1),
            Position::new(plane, bounds.3),
        ));
        (
            (bounds.0, bounds.1, plane, bounds.3),
            (plane, bounds.1, bounds.2, bounds.3),
        )
    } else {
        let plane = bounds.1 + split;
        partitions.push((
            Position::new(bounds.0, plane),
            Position::new(bounds.2, plane),
        ));
        (
            (bounds.0, bounds.1, bounds.2, plane),
            (bounds.0, plane, bounds.2, bounds.3),
        )
    };
    subdivide_space(a, depth + 1, path << 1, rule, leaves, partitions);
    subdivide_space(b, depth + 1, path << 1 | 1, rule, leaves, partitions);
}

fn keyed_unit(seed: f32, key: u32) -> f32 {
    let mut value = seed.to_bits() ^ key.wrapping_mul(0x9E37_79B9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x85EB_CA6B);
    value ^= value >> 13;
    (value >> 8) as f32 / (1u32 << 24) as f32
}

/// Lower a suite's room subdivision into explicit wall hosts and hosted
/// openings. The recursive/automata layer can rewrite these elements later;
/// voxel sampling no longer has to rediscover partitions from rectangles.
fn hosts_and_openings_for(
    footprint: &Polygon2,
    partition_segments: &[(Position, Position)],
    ceiling_units: f32,
    entrance: Opening,
    partition_density: f32,
    grammar_seed: f32,
) -> (Vec<HostSegment>, Vec<Opening>) {
    let mut hosts = HostSegment::rectangular_shell(footprint, PLAN_WALL_T, ceiling_units);
    let mut openings = vec![entrance];

    // A short totalistic cellular automaton acts as a renovation/demolition
    // pass over the partition graph. It can merge neighboring program cells,
    // but never touches the shell and never creates an unpierced retained
    // partition, so topology stays legible and walkable.
    let retained = evolve_partitions(partition_segments, partition_density, grammar_seed);
    for (&(start, end), retain) in partition_segments.iter().zip(retained) {
        if retain {
            let host_id = HostId(hosts.len() as u32);
            let horizontal = (start.z - end.z).abs() <= 1e-4;
            let center = Position::new((start.x + end.x) * 0.5, (start.z + end.z) * 0.5);
            hosts.push(HostSegment {
                id: host_id,
                start,
                end,
                thickness: PLAN_WALL_T,
                base_units: 0.0,
                top_units: ceiling_units,
                role: HostRole::Partition,
            });
            openings.push(Opening {
                id: OpeningId(openings.len() as u32),
                host: host_id,
                role: OpeningRole::Interior,
                center,
                width: 1.2,
                through_x_wall: horizontal,
                lintel_units: Some(2.2),
            });
        }
    }
    (hosts, openings)
}

fn point_on_segment(point: Position, segment: (Position, Position)) -> bool {
    const EPSILON: f32 = 1e-4;
    if (segment.0.z - segment.1.z).abs() <= EPSILON {
        (point.z - segment.0.z).abs() <= EPSILON
            && point.x >= segment.0.x.min(segment.1.x) - EPSILON
            && point.x <= segment.0.x.max(segment.1.x) + EPSILON
    } else {
        (point.x - segment.0.x).abs() <= EPSILON
            && point.z >= segment.0.z.min(segment.1.z) - EPSILON
            && point.z <= segment.0.z.max(segment.1.z) + EPSILON
    }
}

fn segments_touch(a: (Position, Position), b: (Position, Position)) -> bool {
    point_on_segment(a.0, b)
        || point_on_segment(a.1, b)
        || point_on_segment(b.0, a)
        || point_on_segment(b.1, a)
}

fn evolve_partitions(segments: &[(Position, Position)], density: f32, seed: f32) -> Vec<bool> {
    if segments.is_empty() {
        return Vec::new();
    }
    let survival = (0.25 + density.clamp(0.0, 1.0) * 0.75).clamp(0.0, 1.0);
    let mut state: Vec<bool> = segments
        .iter()
        .enumerate()
        .map(|(index, _)| keyed_unit(seed, index as u32 ^ 0xA170) < survival)
        .collect();
    for _ in 0..2 {
        state = segments
            .iter()
            .enumerate()
            .map(|(index, segment)| {
                let neighbours = segments
                    .iter()
                    .enumerate()
                    .filter(|(other_index, other)| {
                        *other_index != index && segments_touch(*segment, **other)
                    })
                    .filter(|(other_index, _)| state[*other_index])
                    .count();
                if state[index] {
                    neighbours <= 3
                } else {
                    neighbours == 2
                }
            })
            .collect();
    }
    if !state.iter().any(|retained| *retained) {
        let fallback = (keyed_unit(seed, 0xFA11) * segments.len() as f32) as usize;
        state[fallback.min(segments.len() - 1)] = true;
    }
    state
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
    // Wall centerlines may occupy the half-step of the 0.4-unit plan lattice:
    // coarse voxel centers live there too. Snapping this midpoint back to a
    // full step can move the shell outside the corridor wall it is hosting.
    let z_front = if side > 0.0 {
        corridor_z + corridor_half
    } else {
        corridor_z - corridor_half - d
    };
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
    let ceiling = CeilingZone {
        area: footprint.clone(),
        language: genome.ceiling_language,
        height_units: ceiling_height_for(program, aseed),
    };
    let layout = space_layout_for(program, &footprint, genome, aseed);
    let entrance = Opening {
        id: OpeningId(0),
        host: if side > 0.0 { HostId(0) } else { HostId(2) },
        role: OpeningRole::Entrance,
        center: Position::new(door_x, front_z),
        width,
        through_x_wall: true,
        lintel_units: lintel,
    };
    let door_leaves = if threshold == ThresholdLanguage::DoorWithLintel {
        vec![DoorLeaf {
            opening: entrance.id,
        }]
    } else {
        Vec::new()
    };
    let (hosts, openings) = hosts_and_openings_for(
        &footprint,
        &layout.partitions,
        ceiling.height_units,
        entrance,
        genome.furnishing_density,
        aseed,
    );
    Some(AssemblyInstance {
        id,
        program,
        spaces: layout.spaces,
        structure: structure_for(genome, aseed),
        ceiling_zones: vec![ceiling],
        fixtures: fixtures_for(genome, &footprint, true, aseed),
        service_voids: Vec::new(),
        corruption: CorruptionProfile::default(),
        hosts,
        openings,
        door_leaves,
        footprint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn genome_with(furnishing_density: f32, tolerance_for_symmetry: f32) -> ArchitectGenome {
        ArchitectGenome {
            circulation: CirculationStyle::StraightSpine,
            structural_system: StructuralSystem::RegularGrid,
            room_proportions: ProportionRules {
                min_side: 12.0,
                max_side: 18.0,
                elongation: 0.5,
            },
            threshold_language: ThresholdLanguage::OpenPortal,
            ceiling_language: CeilingLanguage::FlatTiles,
            lighting_language: LightingLanguage::SparsePendants,
            furnishing_density,
            renovation_history: RenovationStyle::Untouched,
            tolerance_for_symmetry,
        }
    }

    #[test]
    fn furnishing_density_scales_how_readily_a_suite_partitions() {
        // Wide enough (30u span, 12u min_side) that a split candidate exists
        // at all (n = 2 before the density gate).
        let footprint = Polygon2::rect(0.0, 0.0, 30.0, 12.0);
        let bare_shell = genome_with(0.0, 0.5);
        let dense = genome_with(1.0, 0.5);
        assert!(
            spaces_for(SpaceProgram::OpenOffice, &footprint, &bare_shell, 0.5).is_empty(),
            "furnishing_density 0 should almost never partition"
        );
        assert!(
            spaces_for(SpaceProgram::OpenOffice, &footprint, &dense, 0.5).len() >= 2,
            "furnishing_density 1 should recursively partition"
        );
    }

    #[test]
    fn interior_partitions_apply_beyond_private_office() {
        let footprint = Polygon2::rect(0.0, 0.0, 30.0, 12.0);
        let dense = genome_with(1.0, 0.5);
        for program in [
            SpaceProgram::PrivateOffice,
            SpaceProgram::OpenOffice,
            SpaceProgram::ConferenceRoom,
        ] {
            assert!(
                !spaces_for(program, &footprint, &dense, 0.5).is_empty(),
                "{program:?} should be eligible for interior partitioning"
            );
        }
        assert!(
            spaces_for(SpaceProgram::Storage, &footprint, &dense, 0.5).is_empty(),
            "programs outside the eligible set must stay bare shells"
        );
    }

    #[test]
    fn tolerance_for_symmetry_centers_the_structural_phase() {
        let aseed = 0.37;
        let free = structure_for(&genome_with(0.5, 0.0), aseed);
        assert_eq!(
            free.phase,
            (snap(aseed * 4.0), snap(aseed * 8.0 % 4.0)),
            "tolerance 0 must reproduce the original freely offset phase"
        );
        let centered = structure_for(&genome_with(0.5, 1.0), aseed);
        assert_eq!(
            centered.phase,
            (0.0, 0.0),
            "tolerance 1 must center the bay grid on the footprint"
        );
    }

    #[test]
    fn tolerance_for_symmetry_mirrors_and_centers_sparse_pendants() {
        let footprint = Polygon2::rect(0.0, 0.0, 20.0, 10.0);
        let aseed = 0.6;
        let jittered = fixtures_for(&genome_with(0.5, 0.0), &footprint, true, aseed);
        let centered = fixtures_for(&genome_with(0.5, 1.0), &footprint, true, aseed);
        assert_eq!(jittered.len(), centered.len());
        assert!(
            jittered
                .iter()
                .zip(&centered)
                .any(|(j, c)| { (j.at.x - c.at.x).abs() > 1e-3 || (j.at.z - c.at.z).abs() > 1e-3 }),
            "full symmetry tolerance must move at least one pendant off its jittered spot"
        );
        // At full tolerance, an odd fixture count's middle pendant must land
        // exactly on the room's centerline (the mirrored layout's fixed
        // point), once snapped to the same lattice every position here uses.
        let (x0, z0, x1, z1) = footprint.bounds();
        let (cx, cz) = (snap((x0 + x1) * 0.5), snap((z0 + z1) * 0.5));
        if centered.len() % 2 == 1 {
            let mid = &centered[centered.len() / 2];
            assert!((mid.at.x - cx).abs() < 1e-4 && (mid.at.z - cz).abs() < 1e-4);
        }
    }

    #[test]
    fn recursive_subdivision_is_snapped_bounded_and_replayable() {
        let footprint = Polygon2::rect(-4.0, 8.0, 30.0, 18.0);
        let genome = genome_with(1.0, 0.5);
        let first = spaces_for(SpaceProgram::OpenOffice, &footprint, &genome, 0.67);
        let second = spaces_for(SpaceProgram::OpenOffice, &footprint, &genome, 0.67);
        assert!(first.len() >= 4, "dense grammar did not recurse");
        assert_eq!(first.len(), second.len());
        let outer = footprint.bounds();
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.footprint.bounds(), b.footprint.bounds());
            let bounds = a.footprint.bounds();
            assert!(bounds.0 >= outer.0 && bounds.1 >= outer.1);
            assert!(bounds.2 <= outer.2 && bounds.3 <= outer.3);
            for coordinate in [bounds.0, bounds.1, bounds.2, bounds.3] {
                assert!(
                    (coordinate / 0.4 - (coordinate / 0.4).round()).abs() < 1e-4,
                    "subdivision left the 0.4-unit plan lattice at {coordinate}"
                );
            }
        }
    }

    #[test]
    fn partition_automaton_preserves_shell_and_pierces_every_retained_wall() {
        let footprint = Polygon2::rect(0.0, 0.0, 30.0, 18.0);
        let genome = genome_with(1.0, 0.5);
        let layout = space_layout_for(SpaceProgram::OpenOffice, &footprint, &genome, 0.67);
        let build = || {
            hosts_and_openings_for(
                &footprint,
                &layout.partitions,
                3.6,
                Opening {
                    id: OpeningId(0),
                    host: HostId(0),
                    role: OpeningRole::Entrance,
                    center: Position::new(15.0, 0.0),
                    width: 3.6,
                    through_x_wall: true,
                    lintel_units: Some(3.0),
                },
                0.72,
                0.67,
            )
        };
        let (hosts, openings) = build();
        let (replayed_hosts, replayed_openings) = build();
        assert_eq!(hosts.len(), replayed_hosts.len());
        assert_eq!(openings.len(), replayed_openings.len());
        assert_eq!(
            hosts.iter().map(|host| host.id).collect::<Vec<_>>(),
            replayed_hosts
                .iter()
                .map(|host| host.id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            hosts
                .iter()
                .filter(|host| host.role == HostRole::Shell)
                .count(),
            4
        );
        assert!(hosts.iter().any(|host| host.role == HostRole::Partition));
        for host in hosts.iter().filter(|host| host.role == HostRole::Partition) {
            assert!(
                openings.iter().any(|opening| opening.host == host.id),
                "retained partition {:?} has no traversable opening",
                host.id
            );
        }
    }
}
