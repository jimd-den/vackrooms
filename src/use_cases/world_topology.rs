//! The pure macro-scale world planner: an infinite graph, one cell at a time.
//!
//! [`plan_macro_cell`] is a pure function of `(seed, cell id, red-room
//! scale)` — the top of the planning hierarchy documented in
//! [`crate::domain::entities::world_topology`]. It owns three jobs:
//!
//! 1. **Parameter fields.** [`sample_fields`] turns the noise port into
//!    multi-octave `[0, 1]` fields (openness, vertical pressure, anomaly
//!    pressure, ...). This is the *only* legitimate role of fractal math in
//!    the world: it chooses where the building's rules change; it never
//!    substitutes for an architectural decision.
//! 2. **Graph events.** Red rooms are planned here as separated events on
//!    the macro lattice (a deterministic tournament gives every event a
//!    cooldown radius), and vertical links reserve stairwell footprints
//!    where vertical pressure is high.
//! 3. **Continuity contracts.** Primary-route portals are derived from the
//!    *edge's* lattice coordinate, so both neighbors of an edge compute the
//!    identical crossing — the invariant that lets circulation chain across
//!    regions forever.
//!
//! Everything downstream (region planner, anomaly planner, voxelizer)
//! consumes these obligations; nothing here writes a voxel or knows a chunk
//! size. Determinism is testable: see the 5×5 snapshot test at the bottom.

use crate::domain::entities::world_topology::{
    MACRO_CELL_SIZE, MacroCell, MacroCellId, MacroFields, PlaceKind, Portal, REGIONS_PER_MACRO_CELL,
    RedRoomEvent, VerticalLink, VerticalLinkKind, WorldNode,
};
use crate::entities::models::Position;
use crate::use_cases::anomalies::determinism::{hash, unit};
use crate::use_cases::ports::NoiseProvider;

/// Side of a region, world units. Kept in lockstep with
/// `region_plan::REGION_SIZE` by a compile-time-adjacent test below.
const REGION_SIZE: f32 = MACRO_CELL_SIZE / REGIONS_PER_MACRO_CELL as f32;

// ---------------------------------------------------------------------------
// Shared 32-bit deterministic hashing.
//
// All architectural "randomness" in region planning flows through these two
// functions. They live here — at the top of the planning hierarchy — and
// `region_plan` imports them, so the macro graph and the regional plans can
// never disagree about a shared derivation (portals, most importantly).
// ---------------------------------------------------------------------------

pub(crate) fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}

/// Hash of a seed plus any number of integer keys, uniform in [0, 1).
pub(crate) fn hash01(seed: u32, keys: &[i64]) -> f32 {
    let mut h = mix(seed ^ 0x9E37_79B9);
    for &k in keys {
        h = mix(h ^ (k as u32)).wrapping_add(mix((k >> 32) as u32));
    }
    (mix(h) >> 8) as f32 / (1u32 << 24) as f32
}

// ---------------------------------------------------------------------------
// Portals: the cross-region continuity contract.
// ---------------------------------------------------------------------------

/// World Z of the primary-route portal on the vertical edge
/// `x = edge_x * REGION_SIZE` of region row `row_z`. Both regions sharing the
/// edge derive the same value — this function is the single source of truth
/// (the region planner routes its main spine between two of these).
pub fn primary_portal_z(seed: u32, edge_x: i64, row_z: i64) -> f32 {
    let f = 0.30 + 0.40 * hash01(seed, &[0x0E1, edge_x, row_z]);
    snap((row_z as f32 + f) * REGION_SIZE)
}

/// Snap to the 0.4 u plan lattice (one coarse voxel).
fn snap(v: f32) -> f32 {
    (v / 0.4).round() * 0.4
}

fn portal(seed: u32, edge_x: i64, row_z: i64) -> Portal {
    Portal {
        id: hash(seed, 0x7047_A100, edge_x, row_z),
        edge_x,
        row_z,
        at: Position::new(
            edge_x as f32 * REGION_SIZE,
            primary_portal_z(seed, edge_x, row_z),
        ),
    }
}

// ---------------------------------------------------------------------------
// Parameter fields.
// ---------------------------------------------------------------------------

/// One multi-octave field in [0, 1]:
/// `F(x,z) = Σ aᵢ · N(x/sᵢ, z/sᵢ)`, rescaled and clamped. The provider's
/// base wavelength is ~20 u, so `20 / wavelength` stretches each octave.
fn field(
    noise: &dyn NoiseProvider,
    seed: u32,
    salt: u32,
    wx: f32,
    wz: f32,
    wavelengths: [f32; 3],
    weights: [f32; 3],
) -> f32 {
    let mut sum = 0.0;
    let mut norm = 0.0;
    for (octave, (&wavelength, &weight)) in wavelengths.iter().zip(weights.iter()).enumerate() {
        let s = 20.0 / wavelength;
        sum += weight
            * noise.evaluate_2d(
                seed ^ salt.wrapping_add((octave as u32).wrapping_mul(0x9E37_79B9)),
                Position::new(wx * s, wz * s),
            );
        norm += weight;
    }
    // Octave sums cluster toward zero; stretch before biasing to [0, 1] so
    // thresholds see a usable spread instead of everything at 0.5.
    ((sum / norm) * 0.8 + 0.5).clamp(0.0, 1.0)
}

/// Sample every macro parameter field at one world point. Pure and
/// world-space: two callers asking about the same point always agree,
/// regardless of which cell, region, or chunk asked.
pub fn sample_fields(noise: &dyn NoiseProvider, seed: u32, wx: f32, wz: f32) -> MacroFields {
    MacroFields {
        openness: field(noise, seed, 0xF1E1_0001, wx, wz, [350.0, 130.0, 45.0], [1.0, 0.5, 0.2]),
        vertical_pressure: field(
            noise, seed, 0xF1E1_0002, wx, wz, [520.0, 170.0, 60.0], [1.0, 0.4, 0.15],
        ),
        institution_age: field(
            noise, seed, 0xF1E1_0003, wx, wz, [260.0, 90.0, 30.0], [1.0, 0.5, 0.25],
        ),
        anomaly_pressure: field(
            noise, seed, 0xF1E1_0004, wx, wz, [620.0, 210.0, 70.0], [1.0, 0.35, 0.1],
        ),
        redroom_pressure: field(
            noise, seed, 0xF1E1_0005, wx, wz, [700.0, 230.0, 80.0], [1.0, 0.3, 0.1],
        ),
        style_blend: field(
            noise, seed, 0xF1E1_0006, wx, wz, [420.0, 140.0, 50.0], [1.0, 0.45, 0.2],
        ),
    }
}

// ---------------------------------------------------------------------------
// Red rooms as separated graph events.
// ---------------------------------------------------------------------------

/// Cooldown radius in macro cells (Chebyshev). Two committed events are
/// always at least one full macro cell (160 u) apart, so red rooms read as
/// rare landmarks instead of a recurring biome.
const RED_ROOM_SEPARATION_CELLS: i64 = 1;

/// Base per-cell candidate chance at `red_room_scale == 1.0`.
const RED_ROOM_BASE_CHANCE: f32 = 0.32;

/// The candidate score of one macro cell: `Some(score)` if the cell wants a
/// red room this epoch, lower score = higher priority in the tournament.
fn red_room_candidate(
    seed: u32,
    noise: &dyn NoiseProvider,
    cell: MacroCellId,
    red_room_scale: f32,
) -> Option<f32> {
    if red_room_scale <= 0.0 {
        return None;
    }
    let center = cell.center();
    let pressure = sample_fields(noise, seed, center.x, center.z).redroom_pressure;
    let score = unit(hash(seed, 0x8ED8_0011, cell.x, cell.z));
    // Pressure clusters events without ever making them uniform or certain.
    let chance = (RED_ROOM_BASE_CHANCE * red_room_scale).min(0.85) * (0.35 + 0.65 * pressure);
    (score < chance).then_some(score)
}

/// The red-room event a macro cell *wins*, if any. A candidate cell fires
/// only when no candidate within the separation radius beats its score
/// (ties broken by lattice coordinate), so any two firing cells are more
/// than the separation radius apart — a deterministic Matérn-style
/// exclusion, computed locally without global state.
pub fn red_room_event_for_cell(
    seed: u32,
    noise: &dyn NoiseProvider,
    cell: MacroCellId,
    red_room_scale: f32,
) -> Option<RedRoomEvent> {
    let mine = red_room_candidate(seed, noise, cell, red_room_scale)?;
    for dz in -RED_ROOM_SEPARATION_CELLS..=RED_ROOM_SEPARATION_CELLS {
        for dx in -RED_ROOM_SEPARATION_CELLS..=RED_ROOM_SEPARATION_CELLS {
            if dx == 0 && dz == 0 {
                continue;
            }
            let rival_cell = MacroCellId::new(cell.x + dx, cell.z + dz);
            if let Some(rival) = red_room_candidate(seed, noise, rival_cell, red_room_scale) {
                let rival_wins = (rival, rival_cell.x, rival_cell.z) < (mine, cell.x, cell.z);
                if rival_wins {
                    return None;
                }
            }
        }
    }
    // The event targets exactly one of the cell's 2×2 regions.
    let pick = hash(seed, 0x8ED8_0022, cell.x, cell.z);
    let region = (
        cell.x * REGIONS_PER_MACRO_CELL + (pick & 1) as i64,
        cell.z * REGIONS_PER_MACRO_CELL + ((pick >> 1) & 1) as i64,
    );
    Some(RedRoomEvent {
        id: hash(seed, 0x8ED8_0033, cell.x, cell.z),
        region,
        score: mine,
    })
}

/// Does a red-room event target this specific region? This is the query the
/// region planner asks; the answer is independent of which chunk, region, or
/// LOD triggered the plan.
pub fn red_room_event_for_region(
    seed: u32,
    noise: &dyn NoiseProvider,
    region_x: i64,
    region_z: i64,
    red_room_scale: f32,
) -> Option<RedRoomEvent> {
    red_room_event_for_cell(
        seed,
        noise,
        MacroCellId::of_region(region_x, region_z),
        red_room_scale,
    )
    .filter(|event| event.region == (region_x, region_z))
}

// ---------------------------------------------------------------------------
// Vertical links.
// ---------------------------------------------------------------------------

/// The vertical link reserved inside one region, if any. Stairwells are
/// planned per region so the region planner can honor the reservation while
/// laying out its assemblies; the *kind* mix leans ordinary, with the
/// endless variants appearing only under high vertical pressure.
pub fn vertical_link_for_region(
    seed: u32,
    noise: &dyn NoiseProvider,
    region_x: i64,
    region_z: i64,
) -> Option<VerticalLink> {
    let center_x = (region_x as f32 + 0.5) * REGION_SIZE;
    let center_z = (region_z as f32 + 0.5) * REGION_SIZE;
    let pressure = sample_fields(noise, seed, center_x, center_z).vertical_pressure;
    let gate = unit(hash(seed, 0x57A1_0001, region_x, region_z));
    if gate >= 0.06 + 0.30 * pressure * pressure {
        return None;
    }

    let kind_roll = unit(hash(seed, 0x57A1_0002, region_x, region_z));
    let (kind, to_elevation) = if kind_roll < 0.55 {
        (VerticalLinkKind::OrdinaryStair, 1)
    } else if kind_roll < 0.75 && pressure > 0.55 {
        (VerticalLinkKind::EndlessAscent, 1)
    } else if kind_roll < 0.90 && pressure > 0.55 {
        (VerticalLinkKind::EndlessDescent, -1)
    } else if kind_roll < 0.97 {
        (VerticalLinkKind::OrdinaryStair, 1)
    } else {
        (VerticalLinkKind::ServiceShaft, -1)
    };

    // Anchor well inside the region so the stair mass never has to fight the
    // edge margin. The region planner snaps the actual footprint to its
    // corridor network; the anchor only says "near here".
    let ax = unit(hash(seed, 0x57A1_0003, region_x, region_z));
    let az = unit(hash(seed, 0x57A1_0004, region_x, region_z));
    Some(VerticalLink {
        id: hash(seed, 0x57A1_0005, region_x, region_z),
        kind,
        region: (region_x, region_z),
        anchor: Position::new(
            snap((region_x as f32 + 0.3 + 0.4 * ax) * REGION_SIZE),
            snap((region_z as f32 + 0.3 + 0.4 * az) * REGION_SIZE),
        ),
        from_elevation: 0,
        to_elevation,
    })
}

// ---------------------------------------------------------------------------
// The cell planner.
// ---------------------------------------------------------------------------

fn classify_region(
    seed: u32,
    noise: &dyn NoiseProvider,
    region: (i64, i64),
    red_room_event: Option<&RedRoomEvent>,
    vertical: Option<&VerticalLink>,
) -> PlaceKind {
    if red_room_event.is_some_and(|event| event.region == region) {
        return PlaceKind::RedRoomEncounter;
    }
    if vertical.is_some() {
        return PlaceKind::Stairwell;
    }
    let center_x = (region.0 as f32 + 0.5) * REGION_SIZE;
    let center_z = (region.1 as f32 + 0.5) * REGION_SIZE;
    let fields = sample_fields(noise, seed, center_x, center_z);
    if fields.openness > 0.66 {
        if fields.vertical_pressure > 0.60 {
            PlaceKind::Atrium
        } else {
            PlaceKind::OpenPlate
        }
    } else {
        PlaceKind::Warren
    }
}

/// Plan one macro cell of the infinite world graph.
///
/// Pure: the same `(seed, cell, red_room_scale)` always yields the same
/// graph, and shared contracts (portals, red-room tournaments) agree with
/// every neighboring cell's own computation. `red_room_scale` is the
/// world-wide `anomalies.frequency * anomalies.red_rooms` product; passing
/// different scales for different cells of one world is a caller bug.
pub fn plan_macro_cell(
    seed: u32,
    cell: MacroCellId,
    red_room_scale: f32,
    noise: &dyn NoiseProvider,
) -> MacroCell {
    let center = cell.center();
    let fields = sample_fields(noise, seed, center.x, center.z);
    let red_room_event = red_room_event_for_cell(seed, noise, cell, red_room_scale);

    let region_x0 = cell.x * REGIONS_PER_MACRO_CELL;
    let region_z0 = cell.z * REGIONS_PER_MACRO_CELL;

    let mut nodes = Vec::with_capacity((REGIONS_PER_MACRO_CELL * REGIONS_PER_MACRO_CELL) as usize);
    let mut vertical_links = Vec::new();
    for dz in 0..REGIONS_PER_MACRO_CELL {
        for dx in 0..REGIONS_PER_MACRO_CELL {
            let region = (region_x0 + dx, region_z0 + dz);
            let vertical = vertical_link_for_region(seed, noise, region.0, region.1);
            nodes.push(WorldNode {
                id: hash(seed, 0x40DE_0001, region.0, region.1),
                region,
                kind: classify_region(
                    seed,
                    noise,
                    region,
                    red_room_event.as_ref(),
                    vertical.as_ref(),
                ),
                elevation: 0,
            });
            vertical_links.extend(vertical);
        }
    }

    // Every vertical edge the cell's regions touch: interior, west boundary,
    // and east boundary. Boundary portals are recomputed identically by the
    // neighboring cell (asserted by test below).
    let mut portals = Vec::new();
    for dz in 0..REGIONS_PER_MACRO_CELL {
        for edge in 0..=REGIONS_PER_MACRO_CELL {
            portals.push(portal(seed, region_x0 + edge, region_z0 + dz));
        }
    }

    MacroCell {
        id: cell,
        fields,
        nodes,
        portals,
        vertical_links,
        red_room_event,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frameworks_drivers::simple_noise::SimpleNoiseProvider;
    use crate::use_cases::anomalies::determinism::mix64;
    use std::fmt::Write as _;

    const SEED: u32 = 42;

    fn noise() -> SimpleNoiseProvider {
        SimpleNoiseProvider::new()
    }

    /// Compact, stable text form of a cell for snapshot comparison.
    fn describe(cell: &MacroCell) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "cell ({}, {}) open={:.3} vert={:.3} age={:.3} anom={:.3} red={:.3} blend={:.3}",
            cell.id.x,
            cell.id.z,
            cell.fields.openness,
            cell.fields.vertical_pressure,
            cell.fields.institution_age,
            cell.fields.anomaly_pressure,
            cell.fields.redroom_pressure,
            cell.fields.style_blend,
        )
        .unwrap();
        for node in &cell.nodes {
            writeln!(
                out,
                "  node {:016x} region ({}, {}) {:?} elev {}",
                node.id, node.region.0, node.region.1, node.kind, node.elevation
            )
            .unwrap();
        }
        for p in &cell.portals {
            writeln!(
                out,
                "  portal {:016x} edge {} row {} at ({:.1}, {:.1})",
                p.id, p.edge_x, p.row_z, p.at.x, p.at.z
            )
            .unwrap();
        }
        for v in &cell.vertical_links {
            writeln!(
                out,
                "  vlink {:016x} {:?} region ({}, {}) at ({:.1}, {:.1}) {} -> {}",
                v.id,
                v.kind,
                v.region.0,
                v.region.1,
                v.anchor.x,
                v.anchor.z,
                v.from_elevation,
                v.to_elevation
            )
            .unwrap();
        }
        if let Some(event) = &cell.red_room_event {
            writeln!(
                out,
                "  redroom {:016x} region ({}, {}) score {:.4}",
                event.id, event.region.0, event.region.1, event.score
            )
            .unwrap();
        }
        out
    }

    fn snapshot_5x5() -> String {
        let noise = noise();
        let mut out = String::new();
        for cz in -2i64..=2 {
            for cx in -2i64..=2 {
                out.push_str(&describe(&plan_macro_cell(
                    SEED,
                    MacroCellId::new(cx, cz),
                    1.0,
                    &noise,
                )));
            }
        }
        out
    }

    fn digest(text: &str) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for byte in text.bytes() {
            h = mix64(h ^ byte as u64);
        }
        h
    }

    /// The same seed, coordinates and scale must always produce the same
    /// graph — byte for byte. If this digest changes, the world changed for
    /// every player: update the constant only for an *intentional*
    /// generation change, and say so in the commit message.
    #[test]
    fn five_by_five_topology_snapshot_is_stable() {
        let text = snapshot_5x5();
        assert_eq!(text, snapshot_5x5(), "topology is not replayable");
        assert_eq!(
            digest(&text),
            SNAPSHOT_DIGEST,
            "macro topology for seed {SEED} changed; snapshot follows:\n{text}"
        );
    }

    const SNAPSHOT_DIGEST: u64 = 11827557378851379719;

    /// Both neighbors of a shared edge must derive the identical portal.
    #[test]
    fn shared_edge_portals_agree_between_neighbor_cells() {
        let noise = noise();
        for cz in -2i64..=2 {
            for cx in -2i64..=2 {
                let west = plan_macro_cell(SEED, MacroCellId::new(cx, cz), 1.0, &noise);
                let east = plan_macro_cell(SEED, MacroCellId::new(cx + 1, cz), 1.0, &noise);
                for p in &west.portals {
                    for q in &east.portals {
                        if p.edge_x == q.edge_x && p.row_z == q.row_z {
                            assert_eq!(p, q, "portal mismatch on edge {}", p.edge_x);
                        }
                    }
                }
            }
        }
    }

    /// Any two committed red-room events keep the cooldown distance: no two
    /// events within the separation radius over a large scan.
    #[test]
    fn red_room_events_respect_their_separation_radius() {
        let noise = noise();
        let mut events = Vec::new();
        for cz in -12i64..=12 {
            for cx in -12i64..=12 {
                if let Some(event) =
                    red_room_event_for_cell(SEED, &noise, MacroCellId::new(cx, cz), 1.0)
                {
                    events.push(((cx, cz), event));
                }
            }
        }
        assert!(
            events.len() >= 5,
            "red rooms too rare to test separation ({} events in 625 cells)",
            events.len()
        );
        for (i, ((ax, az), _)) in events.iter().enumerate() {
            for ((bx, bz), _) in events.iter().skip(i + 1) {
                let chebyshev = (ax - bx).abs().max((az - bz).abs());
                assert!(
                    chebyshev > RED_ROOM_SEPARATION_CELLS,
                    "events at ({ax},{az}) and ({bx},{bz}) violate the cooldown"
                );
            }
        }
    }

    /// The per-region query and the per-cell tournament are one system.
    #[test]
    fn region_query_matches_cell_tournament() {
        let noise = noise();
        for cz in -6i64..=6 {
            for cx in -6i64..=6 {
                let cell = MacroCellId::new(cx, cz);
                let event = red_room_event_for_cell(SEED, &noise, cell, 1.0);
                for dz in 0..REGIONS_PER_MACRO_CELL {
                    for dx in 0..REGIONS_PER_MACRO_CELL {
                        let region = (
                            cx * REGIONS_PER_MACRO_CELL + dx,
                            cz * REGIONS_PER_MACRO_CELL + dz,
                        );
                        let hit =
                            red_room_event_for_region(SEED, &noise, region.0, region.1, 1.0);
                        match (&event, hit) {
                            (Some(e), Some(h)) => {
                                assert_eq!(e.region, region);
                                assert_eq!(*e, h);
                            }
                            (Some(e), None) => assert_ne!(e.region, region),
                            (None, Some(_)) => panic!("region fired without its cell"),
                            (None, None) => {}
                        }
                    }
                }
            }
        }
    }

    /// Zero scale silences the event system entirely.
    #[test]
    fn zero_red_room_scale_produces_no_events() {
        let noise = noise();
        for cz in -6i64..=6 {
            for cx in -6i64..=6 {
                assert!(
                    red_room_event_for_cell(SEED, &noise, MacroCellId::new(cx, cz), 0.0).is_none()
                );
            }
        }
    }

    /// Fields are world-space pure and bounded.
    #[test]
    fn fields_are_bounded_and_replayable() {
        let noise = noise();
        let mut spread = (f32::MAX, f32::MIN);
        for z in (-2000..=2000).step_by(160) {
            for x in (-2000..=2000).step_by(160) {
                let a = sample_fields(&noise, SEED, x as f32, z as f32);
                let b = sample_fields(&noise, SEED, x as f32, z as f32);
                assert_eq!(a, b);
                for v in [
                    a.openness,
                    a.vertical_pressure,
                    a.institution_age,
                    a.anomaly_pressure,
                    a.redroom_pressure,
                    a.style_blend,
                ] {
                    assert!((0.0..=1.0).contains(&v), "field out of range: {v}");
                }
                spread.0 = spread.0.min(a.openness);
                spread.1 = spread.1.max(a.openness);
            }
        }
        assert!(
            spread.1 - spread.0 > 0.4,
            "openness field is too flat to steer anything: {spread:?}"
        );
    }

    /// Vertical links appear at a believable rate and always change story.
    #[test]
    fn vertical_links_are_present_but_not_everywhere() {
        let noise = noise();
        let mut links = 0usize;
        let mut regions = 0usize;
        for rz in -20i64..=20 {
            for rx in -20i64..=20 {
                regions += 1;
                if let Some(link) = vertical_link_for_region(SEED, &noise, rx, rz) {
                    links += 1;
                    assert_ne!(link.from_elevation, link.to_elevation);
                    assert_eq!(link.region, (rx, rz));
                    // The anchor stays well inside its own region.
                    let x0 = rx as f32 * REGION_SIZE;
                    let z0 = rz as f32 * REGION_SIZE;
                    assert!(link.anchor.x > x0 + 8.0 && link.anchor.x < x0 + REGION_SIZE - 8.0);
                    assert!(link.anchor.z > z0 + 8.0 && link.anchor.z < z0 + REGION_SIZE - 8.0);
                }
            }
        }
        let ratio = links as f32 / regions as f32;
        assert!(
            (0.05..=0.35).contains(&ratio),
            "vertical links in {:.0}% of regions",
            ratio * 100.0
        );
    }
}
