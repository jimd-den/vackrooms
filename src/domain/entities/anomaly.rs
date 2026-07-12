//! Immutable anomaly plans and deterministic encounter-state snapshots.
//!
//! The planner owns instance placement; this module owns only plain data and
//! coordinate/transport helpers. Stateful geometry is still pure: a chunk is
//! sampled from an [`AnomalyInstance`] plus the matching stamp in a
//! [`RealitySnapshot`].

use std::collections::BTreeMap;

use crate::entities::models::Position;

pub type AnomalyId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum AnomalyKind {
    PillarExpanse = 0,
    BlackoutExpanse = 1,
    PitLattice = 2,
    RedRoom = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Axis2 {
    X = 0,
    Z = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum AxisDirection {
    Negative = 0,
    Positive = 1,
}

impl AxisDirection {
    pub fn sign(self) -> f32 {
        match self {
            Self::Negative => -1.0,
            Self::Positive => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QuarterTurn {
    Zero = 0,
    Clockwise = 1,
}

/// Axis-aligned bounds in world X/Z. All anomaly footprints are rectangles
/// rotated in 90-degree increments so construction remains rectilinear.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldBounds {
    pub min_x: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_z: f32,
}

impl WorldBounds {
    pub fn new(min_x: f32, min_z: f32, max_x: f32, max_z: f32) -> Self {
        Self {
            min_x: min_x.min(max_x),
            min_z: min_z.min(max_z),
            max_x: min_x.max(max_x),
            max_z: min_z.max(max_z),
        }
    }

    pub fn contains(&self, x: f32, z: f32) -> bool {
        x >= self.min_x && x <= self.max_x && z >= self.min_z && z <= self.max_z
    }

    pub fn intersects(&self, other: Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_z <= other.max_z
            && self.max_z >= other.min_z
    }

    pub fn expanded(self, margin: f32) -> Self {
        Self::new(
            self.min_x - margin,
            self.min_z - margin,
            self.max_x + margin,
            self.max_z + margin,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthoBasis {
    pub turn: QuarterTurn,
}

impl OrthoBasis {
    pub fn to_local(self, dx: f32, dz: f32) -> (f32, f32) {
        match self.turn {
            QuarterTurn::Zero => (dx, dz),
            QuarterTurn::Clockwise => (dz, -dx),
        }
    }

    pub fn to_world(self, lx: f32, lz: f32) -> (f32, f32) {
        match self.turn {
            QuarterTurn::Zero => (lx, lz),
            QuarterTurn::Clockwise => (-lz, lx),
        }
    }

    pub fn local_axis_in_world(self, axis: Axis2) -> Axis2 {
        match (self.turn, axis) {
            (QuarterTurn::Zero, a) => a,
            (QuarterTurn::Clockwise, Axis2::X) => Axis2::Z,
            (QuarterTurn::Clockwise, Axis2::Z) => Axis2::X,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrientedFootprint {
    pub center: Position,
    /// Half extents in the instance's local coordinates.
    pub half_x: f32,
    pub half_z: f32,
    pub basis: OrthoBasis,
}

impl OrientedFootprint {
    pub fn local_coords(&self, wx: f32, wz: f32) -> (f32, f32) {
        self.basis.to_local(wx - self.center.x, wz - self.center.z)
    }

    pub fn world_coords(&self, lx: f32, lz: f32) -> Position {
        let (dx, dz) = self.basis.to_world(lx, lz);
        Position::new(self.center.x + dx, self.center.z + dz)
    }

    pub fn contains(&self, wx: f32, wz: f32) -> bool {
        let (x, z) = self.local_coords(wx, wz);
        x.abs() <= self.half_x && z.abs() <= self.half_z
    }

    pub fn boundary_distance(&self, wx: f32, wz: f32) -> f32 {
        let (x, z) = self.local_coords(wx, wz);
        (self.half_x - x.abs()).min(self.half_z - z.abs())
    }

    pub fn normalized_depth(&self, wx: f32, wz: f32) -> f32 {
        (self.boundary_distance(wx, wz) / self.half_x.min(self.half_z).max(0.001)).clamp(0.0, 1.0)
    }

    pub fn bounds(&self) -> WorldBounds {
        let (hx, hz) = match self.basis.turn {
            QuarterTurn::Zero => (self.half_x, self.half_z),
            QuarterTurn::Clockwise => (self.half_z, self.half_x),
        };
        WorldBounds::new(
            self.center.x - hx,
            self.center.z - hz,
            self.center.x + hx,
            self.center.z + hz,
        )
    }
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PillarLattice {
    pub bay_x: f32,
    pub bay_z: f32,
    pub phase_x: f32,
    pub phase_z: f32,
    pub min_side: f32,
    pub max_side: f32,
    pub variation_seed: u64,
}

impl PillarLattice {
    /// Variable wallpapered piers, always snapped to the shared 0.4u plan
    /// lattice. Entry regularity comes from placement, not a fixed size.
    pub fn pillar_size(&self, cell_x: i64, cell_z: i64) -> f32 {
        let h = mix64(
            self.variation_seed
                ^ (cell_x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (cell_z as u64).rotate_left(29),
        );
        let steps = (((self.max_side - self.min_side) / 0.4).round() as u64).max(1);
        let step = (h % (steps + 1)) as f32;
        (self.min_side + 0.4 * step).clamp(self.min_side, self.max_side)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitLattice {
    pub spacing_x: f32,
    pub spacing_z: f32,
    pub phase_x: f32,
    pub phase_z: f32,
    pub side: f32,
    pub depth: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TraversalGateKind {
    Remap = 0,
    RedThreshold = 1,
}

/// A semantic plane crossing. `span_*` lies on the perpendicular world axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraversalGate {
    pub id: u64,
    pub instance_id: AnomalyId,
    pub anomaly_kind: AnomalyKind,
    pub kind: TraversalGateKind,
    pub axis: Axis2,
    pub plane: f32,
    pub span_min: f32,
    pub span_max: f32,
    /// For directional thresholds (red rooms), the authored inward direction.
    pub forward: AxisDirection,
    pub affected_bounds: WorldBounds,
}

impl TraversalGate {
    /// Returns traversal direction when a movement segment crosses the plane
    /// inside its span. Merely touching or moving parallel does not fire.
    pub fn crossing(&self, old: [f32; 2], new: [f32; 2]) -> Option<AxisDirection> {
        let (a0, a1, b0, b1) = match self.axis {
            Axis2::X => (old[0], new[0], old[1], new[1]),
            Axis2::Z => (old[1], new[1], old[0], new[0]),
        };
        let delta = a1 - a0;
        if delta.abs() < 1e-6 {
            return None;
        }
        let t = (self.plane - a0) / delta;
        if !(0.0 < t && t <= 1.0) {
            return None;
        }
        let across = b0 + (b1 - b0) * t;
        if across < self.span_min || across > self.span_max {
            return None;
        }
        Some(if delta > 0.0 {
            AxisDirection::Positive
        } else {
            AxisDirection::Negative
        })
    }

    pub const WORDS: usize = 15;

    pub fn to_words(self) -> [u32; Self::WORDS] {
        [
            self.id as u32,
            (self.id >> 32) as u32,
            self.instance_id as u32,
            (self.instance_id >> 32) as u32,
            self.anomaly_kind as u32,
            self.kind as u32,
            self.axis as u32,
            self.plane.to_bits(),
            self.span_min.to_bits(),
            self.span_max.to_bits(),
            self.forward as u32,
            self.affected_bounds.min_x.to_bits(),
            self.affected_bounds.min_z.to_bits(),
            self.affected_bounds.max_x.to_bits(),
            self.affected_bounds.max_z.to_bits(),
        ]
    }

    pub fn from_words(w: &[u32]) -> Option<Self> {
        if w.len() != Self::WORDS {
            return None;
        }
        Some(Self {
            id: w[0] as u64 | ((w[1] as u64) << 32),
            instance_id: w[2] as u64 | ((w[3] as u64) << 32),
            anomaly_kind: decode_kind(w[4])?,
            kind: match w[5] {
                0 => TraversalGateKind::Remap,
                1 => TraversalGateKind::RedThreshold,
                _ => return None,
            },
            axis: decode_axis(w[6])?,
            plane: f32::from_bits(w[7]),
            span_min: f32::from_bits(w[8]),
            span_max: f32::from_bits(w[9]),
            forward: decode_direction(w[10])?,
            affected_bounds: WorldBounds::new(
                f32::from_bits(w[11]),
                f32::from_bits(w[12]),
                f32::from_bits(w[13]),
                f32::from_bits(w[14]),
            ),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitHazard {
    pub id: u64,
    pub instance_id: AnomalyId,
    pub center: Position,
    pub half_side: f32,
    pub depth: f32,
    pub recovery: Position,
}

impl PitHazard {
    pub fn contains(&self, x: f32, z: f32) -> bool {
        (x - self.center.x).abs() < self.half_side && (z - self.center.z).abs() < self.half_side
    }

    pub const TRANSPORT_WORDS: usize = 10;

    pub fn write_words(self, out: &mut Vec<u32>) {
        out.extend_from_slice(&[
            self.id as u32,
            (self.id >> 32) as u32,
            self.instance_id as u32,
            (self.instance_id >> 32) as u32,
            self.center.x.to_bits(),
            self.center.z.to_bits(),
            self.half_side.to_bits(),
            self.depth.to_bits(),
            self.recovery.x.to_bits(),
            self.recovery.z.to_bits(),
        ]);
    }

    pub fn from_transport_words(w: &[u32]) -> Option<Self> {
        if w.len() != Self::TRANSPORT_WORDS {
            return None;
        }
        Some(Self {
            id: w[0] as u64 | ((w[1] as u64) << 32),
            instance_id: w[2] as u64 | ((w[3] as u64) << 32),
            center: Position::new(f32::from_bits(w[4]), f32::from_bits(w[5])),
            half_side: f32::from_bits(w[6]),
            depth: f32::from_bits(w[7]),
            recovery: Position::new(f32::from_bits(w[8]), f32::from_bits(w[9])),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnomalyInstance {
    pub id: AnomalyId,
    pub kind: AnomalyKind,
    pub footprint: OrientedFootprint,
    pub basis: OrthoBasis,
    pub macro_anchor: (i64, i64),
    pub pillar_lattice: Option<PillarLattice>,
    pub pit_lattice: Option<PitLattice>,
    pub gates: Vec<TraversalGate>,
    /// Epoch-invariant internal route half-width.
    pub skeleton_half_width: f32,
    /// Boundary depth kept regular before corruption/remapping begins.
    pub entry_band: f32,
}

impl AnomalyInstance {
    pub fn contains(&self, x: f32, z: f32) -> bool {
        self.footprint.contains(x, z)
    }
    pub fn local_coords(&self, x: f32, z: f32) -> (f32, f32) {
        self.footprint.local_coords(x, z)
    }
    pub fn world_coords(&self, x: f32, z: f32) -> Position {
        self.footprint.world_coords(x, z)
    }
    pub fn boundary_distance(&self, x: f32, z: f32) -> f32 {
        self.footprint.boundary_distance(x, z)
    }
    pub fn normalized_depth(&self, x: f32, z: f32) -> f32 {
        self.footprint.normalized_depth(x, z)
    }
    pub fn state<'a>(&self, reality: &'a RealitySnapshot) -> Option<&'a AnomalyStateStamp> {
        reality.lookup(self.id)
    }
    pub fn traversal_gates(&self) -> &[TraversalGate] {
        &self.gates
    }
    pub fn pillar_size(&self, cx: i64, cz: i64) -> Option<f32> {
        self.pillar_lattice.map(|p| p.pillar_size(cx, cz))
    }

    pub fn pit_hazards_for_bounds(&self, bounds: WorldBounds) -> Vec<PitHazard> {
        let Some(lattice) = self.pit_lattice else {
            return Vec::new();
        };
        let corners = [
            self.local_coords(bounds.min_x, bounds.min_z),
            self.local_coords(bounds.min_x, bounds.max_z),
            self.local_coords(bounds.max_x, bounds.min_z),
            self.local_coords(bounds.max_x, bounds.max_z),
        ];
        let min_lx = corners.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let max_lx = corners
            .iter()
            .map(|p| p.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_lz = corners.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let max_lz = corners
            .iter()
            .map(|p| p.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let ix0 = ((min_lx - lattice.phase_x) / lattice.spacing_x).floor() as i64 - 1;
        let ix1 = ((max_lx - lattice.phase_x) / lattice.spacing_x).ceil() as i64 + 1;
        let iz0 = ((min_lz - lattice.phase_z) / lattice.spacing_z).floor() as i64 - 1;
        let iz1 = ((max_lz - lattice.phase_z) / lattice.spacing_z).ceil() as i64 + 1;
        let mut out = Vec::new();
        for iz in iz0..=iz1 {
            for ix in ix0..=ix1 {
                let lx = lattice.phase_x + ix as f32 * lattice.spacing_x;
                let lz = lattice.phase_z + iz as f32 * lattice.spacing_z;
                let center = self.world_coords(lx, lz);
                if !self.contains(center.x, center.z)
                    || self.boundary_distance(center.x, center.z) < lattice.side
                    || !bounds.expanded(lattice.side).contains(center.x, center.z)
                {
                    continue;
                }
                // The immutable center lane remains solid and navigable.
                if lz.abs() < self.skeleton_half_width + lattice.side {
                    continue;
                }
                // Cell center between four pits: deterministically clear of
                // this lattice even when pit side/spacing vary by instance.
                let recovery =
                    self.world_coords(lx + lattice.spacing_x * 0.5, lz + lattice.spacing_z * 0.5);
                let id = mix64(self.id ^ (ix as u64).rotate_left(17) ^ (iz as u64).rotate_left(43));
                out.push(PitHazard {
                    id,
                    instance_id: self.id,
                    center,
                    half_side: lattice.side * 0.5,
                    depth: lattice.depth,
                    recovery,
                });
            }
        }
        out.sort_by_key(|h| h.id);
        out.dedup_by_key(|h| h.id);
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnomalyStateStamp {
    pub instance_id: AnomalyId,
    pub epoch: u32,
    pub gate_plane_milli: i32,
    pub gate_axis: Axis2,
    pub travel_direction: AxisDirection,
}

impl AnomalyStateStamp {
    pub fn new(
        instance_id: AnomalyId,
        epoch: u32,
        gate_plane: f32,
        gate_axis: Axis2,
        travel_direction: AxisDirection,
    ) -> Self {
        Self {
            instance_id,
            epoch,
            gate_plane_milli: (gate_plane * 1000.0).round() as i32,
            gate_axis,
            travel_direction,
        }
    }

    pub fn gate_plane(&self) -> f32 {
        self.gate_plane_milli as f32 / 1000.0
    }

    pub fn point_is_in_wake(&self, x: f32, z: f32, min_distance: f32) -> bool {
        let coordinate = match self.gate_axis {
            Axis2::X => x,
            Axis2::Z => z,
        };
        match self.travel_direction {
            AxisDirection::Positive => coordinate < self.gate_plane() - min_distance,
            AxisDirection::Negative => coordinate > self.gate_plane() + min_distance,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RealitySnapshot {
    stamps: Vec<AnomalyStateStamp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealitySnapshotDecodeError {
    Header,
    Length,
    Value,
    Fingerprint,
}

impl RealitySnapshot {
    const MAGIC: u32 = 0x5254_5901; // "RTY" v1
    const STAMP_WORDS: usize = 5;

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn new(stamps: Vec<AnomalyStateStamp>) -> Self {
        let mut by_id: BTreeMap<AnomalyId, AnomalyStateStamp> = BTreeMap::new();
        for stamp in stamps {
            by_id
                .entry(stamp.instance_id)
                .and_modify(|old| {
                    if stamp > *old {
                        *old = stamp;
                    }
                })
                .or_insert(stamp);
        }
        Self {
            stamps: by_id.into_values().collect(),
        }
    }

    pub fn stamps(&self) -> &[AnomalyStateStamp] {
        &self.stamps
    }

    pub fn lookup(&self, id: AnomalyId) -> Option<&AnomalyStateStamp> {
        self.stamps
            .binary_search_by_key(&id, |s| s.instance_id)
            .ok()
            .map(|i| &self.stamps[i])
    }

    pub fn with_advanced_gate(&self, gate: &TraversalGate, direction: AxisDirection) -> Self {
        let next = self
            .lookup(gate.instance_id)
            .map_or(1, |s| s.epoch.wrapping_add(1));
        let mut stamps = self.stamps.clone();
        stamps.retain(|s| s.instance_id != gate.instance_id);
        stamps.push(AnomalyStateStamp::new(
            gate.instance_id,
            next,
            gate.plane,
            gate.axis,
            direction,
        ));
        Self::new(stamps)
    }

    fn hash_words(words: &[u32]) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for &word in words {
            h ^= word as u64;
            h = h.wrapping_mul(0x100_0000_01B3);
        }
        mix64(h)
    }

    pub fn fingerprint(&self) -> u64 {
        let words = self.words_without_fingerprint();
        Self::hash_words(&words)
    }

    fn words_without_fingerprint(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(2 + self.stamps.len() * Self::STAMP_WORDS);
        out.push(Self::MAGIC);
        out.push(self.stamps.len() as u32);
        for s in &self.stamps {
            out.extend_from_slice(&[
                s.instance_id as u32,
                (s.instance_id >> 32) as u32,
                s.epoch,
                s.gate_plane_milli as u32,
                (s.gate_axis as u32) | ((s.travel_direction as u32) << 8),
            ]);
        }
        out
    }

    pub fn to_words(&self) -> Vec<u32> {
        let mut out = self.words_without_fingerprint();
        let fp = Self::hash_words(&out);
        out.push(fp as u32);
        out.push((fp >> 32) as u32);
        out
    }

    pub fn from_words(words: &[u32]) -> Result<Self, RealitySnapshotDecodeError> {
        if words.len() < 4 || words[0] != Self::MAGIC {
            return Err(RealitySnapshotDecodeError::Header);
        }
        let count = words[1] as usize;
        let body_len = 2 + count.saturating_mul(Self::STAMP_WORDS);
        if words.len() != body_len + 2 {
            return Err(RealitySnapshotDecodeError::Length);
        }
        let expected = words[body_len] as u64 | ((words[body_len + 1] as u64) << 32);
        if Self::hash_words(&words[..body_len]) != expected {
            return Err(RealitySnapshotDecodeError::Fingerprint);
        }
        let mut stamps = Vec::with_capacity(count);
        for c in words[2..body_len].chunks_exact(Self::STAMP_WORDS) {
            stamps.push(AnomalyStateStamp {
                instance_id: c[0] as u64 | ((c[1] as u64) << 32),
                epoch: c[2],
                gate_plane_milli: c[3] as i32,
                gate_axis: decode_axis(c[4] & 0xff).ok_or(RealitySnapshotDecodeError::Value)?,
                travel_direction: decode_direction((c[4] >> 8) & 0xff)
                    .ok_or(RealitySnapshotDecodeError::Value)?,
            });
        }
        let decoded = Self::new(stamps);
        if decoded.to_words() != words {
            return Err(RealitySnapshotDecodeError::Value);
        }
        Ok(decoded)
    }
}

fn decode_axis(v: u32) -> Option<Axis2> {
    match v {
        0 => Some(Axis2::X),
        1 => Some(Axis2::Z),
        _ => None,
    }
}
fn decode_direction(v: u32) -> Option<AxisDirection> {
    match v {
        0 => Some(AxisDirection::Negative),
        1 => Some(AxisDirection::Positive),
        _ => None,
    }
}
fn decode_kind(v: u32) -> Option<AnomalyKind> {
    match v {
        0 => Some(AnomalyKind::PillarExpanse),
        1 => Some(AnomalyKind::BlackoutExpanse),
        2 => Some(AnomalyKind::PitLattice),
        3 => Some(AnomalyKind::RedRoom),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reality_transport_is_canonical_and_lossless() {
        let a = AnomalyStateStamp::new(9, 2, -12.4, Axis2::Z, AxisDirection::Negative);
        let b = AnomalyStateStamp::new(3, 7, 8.0, Axis2::X, AxisDirection::Positive);
        let one = RealitySnapshot::new(vec![a, b]);
        let two = RealitySnapshot::new(vec![b, a]);
        assert_eq!(one, two);
        assert_eq!(one.fingerprint(), two.fingerprint());
        assert_eq!(RealitySnapshot::from_words(&one.to_words()), Ok(one));
    }

    #[test]
    fn pillar_sizes_vary_but_stay_snapped() {
        let lattice = PillarLattice {
            bay_x: 4.4,
            bay_z: 4.0,
            phase_x: 0.0,
            phase_z: 0.0,
            min_side: 1.2,
            max_side: 1.6,
            variation_seed: 42,
        };
        let mut seen = Vec::new();
        for z in 0..12 {
            for x in 0..12 {
                let s = lattice.pillar_size(x, z);
                assert!((1.2..=1.6).contains(&s));
                assert!(((s / 0.4).round() - s / 0.4).abs() < 1e-4);
                if !seen.contains(&s.to_bits()) {
                    seen.push(s.to_bits());
                }
            }
        }
        assert!(seen.len() >= 2);
    }

    #[test]
    fn gate_crossing_is_directional_and_span_bounded() {
        let gate = TraversalGate {
            id: 1,
            instance_id: 2,
            anomaly_kind: AnomalyKind::RedRoom,
            kind: TraversalGateKind::RedThreshold,
            axis: Axis2::X,
            plane: 5.0,
            span_min: -1.0,
            span_max: 1.0,
            forward: AxisDirection::Positive,
            affected_bounds: WorldBounds::new(0.0, -5.0, 10.0, 5.0),
        };
        assert_eq!(
            gate.crossing([[4.0, 0.0][0], [4.0, 0.0][1]], [6.0, 0.0]),
            Some(AxisDirection::Positive)
        );
        assert_eq!(gate.crossing([4.0, 2.0], [6.0, 2.0]), None);
        assert_eq!(gate.crossing([5.0, 0.0], [6.0, 0.0]), None);
    }
}
