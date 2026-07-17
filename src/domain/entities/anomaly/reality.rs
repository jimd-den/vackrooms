//! Immutable encounter state and its deterministic transition reducer.
//!
//! A snapshot is request identity: workers receive it by value and generate a
//! complete chunk from that exact reality. Gate interpretation is concentrated
//! here so geometry sampling never mutates ambient state.

use std::collections::BTreeMap;

use super::geometry::{Axis2, AxisDirection, decode_axis, decode_direction};
use super::mix64;
use super::phenomena::AnomalyId;
use super::traversal::{TraversalGate, TraversalGateKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum RedRoomPhase {
    Outside = 0,
    Sealed = 1,
    EscapeOpen = 2,
}

impl Default for RedRoomPhase {
    fn default() -> Self {
        Self::Outside
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnomalyStateStamp {
    pub instance_id: AnomalyId,
    pub epoch: u32,
    pub gate_plane_milli: i32,
    pub gate_axis: Axis2,
    pub travel_direction: AxisDirection,
    pub phase: RedRoomPhase,
    pub loop_count: u8,
    pub last_gate_id: u64,
}

impl AnomalyStateStamp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instance_id: AnomalyId,
        epoch: u32,
        gate_plane: f32,
        gate_axis: Axis2,
        travel_direction: AxisDirection,
        phase: RedRoomPhase,
        loop_count: u8,
        last_gate_id: u64,
    ) -> Self {
        Self {
            instance_id,
            epoch,
            gate_plane_milli: (gate_plane * 1000.0).round() as i32,
            gate_axis,
            travel_direction,
            phase,
            loop_count,
            last_gate_id,
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

/// Side of one fabric-drift cell, world units. Drift is the Peripheral
/// Shift of the *ordinary* fabric: the unit of territory whose cosmetic and
/// porosity salts re-derive together when the level rearranges unobserved.
/// Coarser than a chunk (so a shift reads as a neighborhood changing, not a
/// tile) and finer than a region (so corridors and assemblies — which never
/// drift — anchor navigation through any number of shifts).
pub const FABRIC_DRIFT_CELL: f32 = 40.0;

/// Drift-lattice index of a world coordinate.
pub fn fabric_drift_cell_of(w: f32) -> i64 {
    (w / FABRIC_DRIFT_CELL).floor() as i64
}

/// How many times one drift cell's fabric has rearranged while unobserved.
/// Only cells that have actually drifted are recorded; epoch 0 is implicit
/// and means the fabric still matches its first observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FabricDriftStamp {
    pub cell_x: i64,
    pub cell_z: i64,
    pub epoch: u32,
}

/// Highest delirium tier. Tiers derive from resource mismanagement in
/// either direction — hydration bands (want) and the engine's excess meter
/// of wasted supplies (waste): a parched or glutted wanderer perceives (and
/// therefore *gets*) a more anomalous Backrooms — sealed doorways, shut
/// corridor mouths, thinner provisions, secret slips.
pub const MAX_DELIRIUM: u8 = 3;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RealitySnapshot {
    stamps: Vec<AnomalyStateStamp>,
    /// Peripheral Shift state, canonical-sorted by (cell_x, cell_z).
    drifts: Vec<FabricDriftStamp>,
    /// Supply items the wanderer has consumed, canonical-sorted by id.
    /// Generation omits a consumed item and its marker deterministically.
    consumed_supplies: Vec<u64>,
    /// Mismanagement-driven perception tier (0 provisioned .. 3 ruined,
    /// from deep thirst or from squandered supplies). Part of request
    /// identity: chunks generated while delirious carry more anomalous
    /// infill, more deceptive glimmers, sealed doorways, and less mercy.
    delirium: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealitySnapshotDecodeError {
    Header,
    Length,
    Value,
    Fingerprint,
}

/// The mutable values of one stamp while a single gate event is reduced.
/// Location/direction metadata comes from the event and is written afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TransitionState {
    epoch: u32,
    phase: RedRoomPhase,
    loop_count: u8,
}

impl TransitionState {
    fn before(existing: Option<&AnomalyStateStamp>, first_epoch: u32) -> Self {
        existing.map_or(
            Self {
                epoch: first_epoch,
                phase: RedRoomPhase::Outside,
                loop_count: 0,
            },
            |stamp| Self {
                epoch: stamp.epoch,
                phase: stamp.phase,
                loop_count: stamp.loop_count,
            },
        )
    }
}

/// Remap gates always commit a new geometry epoch while retaining any
/// red-room phase information already associated with the instance.
fn advance_remap(mut state: TransitionState, next_epoch: u32) -> TransitionState {
    state.epoch = next_epoch;
    state
}

/// The inner red threshold seals only in its authored forward direction.
fn advance_red_threshold(
    mut state: TransitionState,
    next_epoch: u32,
    direction: AxisDirection,
    forward: AxisDirection,
) -> TransitionState {
    if direction == forward {
        state.phase = RedRoomPhase::Sealed;
        state.epoch = next_epoch;
    }
    state
}

/// A loop crossing matters only after commitment. The escape is almost
/// never offered: seven accepted crossings expose it, and the very next
/// crossing seals the room again and resets the count — a wanderer who
/// doesn't find the breach before touching the checkpoint loses it. Each
/// accepted crossing also advances the geometry epoch, so the breach never
/// reopens on the same wall twice in a row.
const LOOPS_TO_OPEN_ESCAPE: u8 = 7;

fn advance_red_loop(mut state: TransitionState, next_epoch: u32) -> TransitionState {
    match state.phase {
        RedRoomPhase::Sealed => {
            state.loop_count = state.loop_count.saturating_add(1);
            if state.loop_count >= LOOPS_TO_OPEN_ESCAPE {
                state.phase = RedRoomPhase::EscapeOpen;
            }
            state.epoch = next_epoch;
        }
        RedRoomPhase::EscapeOpen => {
            state.phase = RedRoomPhase::Sealed;
            state.loop_count = 0;
            state.epoch = next_epoch;
        }
        RedRoomPhase::Outside => {}
    }
    state
}

/// RedEscape is currently observational: crossing it records event metadata
/// without changing phase or epoch. Keeping the no-op named makes that part of
/// the v2 behavior explicit until the encounter design assigns an exit effect.
fn advance_red_escape(state: TransitionState) -> TransitionState {
    state
}

fn transition_for(
    gate: &TraversalGate,
    direction: AxisDirection,
    existing: Option<&AnomalyStateStamp>,
    next_epoch: u32,
) -> TransitionState {
    let state = TransitionState::before(existing, next_epoch);
    match gate.kind {
        TraversalGateKind::Remap => advance_remap(state, next_epoch),
        TraversalGateKind::RedThreshold => {
            advance_red_threshold(state, next_epoch, direction, gate.forward)
        }
        TraversalGateKind::RedLoop => advance_red_loop(state, next_epoch),
        TraversalGateKind::RedEscape => advance_red_escape(state),
    }
}

impl RealitySnapshot {
    const MAGIC: u32 = 0x5254_5905; // "RTY" v5: v4 plus the delirium tier
    const STAMP_WORDS: usize = 7;
    const DRIFT_WORDS: usize = 5;

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn new(stamps: Vec<AnomalyStateStamp>) -> Self {
        Self::with_drifts(stamps, Vec::new())
    }

    pub fn with_drifts(stamps: Vec<AnomalyStateStamp>, drifts: Vec<FabricDriftStamp>) -> Self {
        Self::with_drifts_and_supplies(stamps, drifts, Vec::new())
    }

    pub fn with_drifts_and_supplies(
        stamps: Vec<AnomalyStateStamp>,
        drifts: Vec<FabricDriftStamp>,
        consumed_supplies: Vec<u64>,
    ) -> Self {
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
        let mut by_cell: BTreeMap<(i64, i64), u32> = BTreeMap::new();
        for drift in drifts {
            let epoch = by_cell.entry((drift.cell_x, drift.cell_z)).or_insert(0);
            *epoch = (*epoch).max(drift.epoch);
        }
        let mut consumed = consumed_supplies;
        consumed.sort_unstable();
        consumed.dedup();
        Self {
            stamps: by_id.into_values().collect(),
            drifts: by_cell
                .into_iter()
                .filter(|(_, epoch)| *epoch > 0)
                .map(|((cell_x, cell_z), epoch)| FabricDriftStamp {
                    cell_x,
                    cell_z,
                    epoch,
                })
                .collect(),
            consumed_supplies: consumed,
            delirium: 0,
        }
    }

    /// The dehydration-driven perception tier (0..=MAX_DELIRIUM).
    pub fn delirium(&self) -> u8 {
        self.delirium
    }

    /// The same reality perceived at a different delirium tier.
    pub fn with_delirium(&self, tier: u8) -> Self {
        Self {
            delirium: tier.min(MAX_DELIRIUM),
            ..self.clone()
        }
    }

    /// True when the wanderer has already consumed this supply item.
    pub fn supply_consumed(&self, id: u64) -> bool {
        self.consumed_supplies.binary_search(&id).is_ok()
    }

    /// Records one consumed supply. Everything else is untouched.
    pub fn with_supply_consumed(&self, id: u64) -> Self {
        let mut consumed = self.consumed_supplies.clone();
        if let Err(index) = consumed.binary_search(&id) {
            consumed.insert(index, id);
        }
        Self {
            stamps: self.stamps.clone(),
            drifts: self.drifts.clone(),
            consumed_supplies: consumed,
            delirium: self.delirium,
        }
    }

    pub fn stamps(&self) -> &[AnomalyStateStamp] {
        &self.stamps
    }

    pub fn fabric_drifts(&self) -> &[FabricDriftStamp] {
        &self.drifts
    }

    /// The Peripheral Shift epoch of the drift cell containing a world point.
    /// Epoch 0 — the overwhelmingly common answer — means "never rearranged"
    /// and reproduces the fabric exactly as first observed.
    pub fn fabric_drift_epoch(&self, wx: f32, wz: f32) -> u32 {
        let key = (fabric_drift_cell_of(wx), fabric_drift_cell_of(wz));
        self.drifts
            .binary_search_by_key(&key, |drift| (drift.cell_x, drift.cell_z))
            .map(|index| self.drifts[index].epoch)
            .unwrap_or(0)
    }

    /// One more unobserved rearrangement of a drift cell. Everything else in
    /// the snapshot — encounter stamps and other cells — is untouched.
    pub fn with_fabric_drift_advanced(&self, cell_x: i64, cell_z: i64) -> Self {
        let mut drifts = self.drifts.clone();
        match drifts.binary_search_by_key(&(cell_x, cell_z), |drift| (drift.cell_x, drift.cell_z))
        {
            Ok(index) => drifts[index].epoch = drifts[index].epoch.saturating_add(1),
            Err(index) => drifts.insert(
                index,
                FabricDriftStamp {
                    cell_x,
                    cell_z,
                    epoch: 1,
                },
            ),
        }
        Self {
            stamps: self.stamps.clone(),
            drifts,
            consumed_supplies: self.consumed_supplies.clone(),
            delirium: self.delirium,
        }
    }

    pub fn lookup(&self, id: AnomalyId) -> Option<&AnomalyStateStamp> {
        self.stamps
            .binary_search_by_key(&id, |stamp| stamp.instance_id)
            .ok()
            .map(|index| &self.stamps[index])
    }

    pub fn with_advanced_gate(&self, gate: &TraversalGate, direction: AxisDirection) -> Self {
        let existing = self.lookup(gate.instance_id);
        // Streaming can report the same crossing more than once, so an
        // identical gate+direction event is idempotent.  The opposite
        // direction is a real traversal, however: a Red Room loop advances
        // by walking back and forth through its one authored checkpoint.
        if existing.is_some_and(|stamp| {
            gate.id == stamp.last_gate_id && direction == stamp.travel_direction
        }) {
            return self.clone();
        }

        // Epoch is the snapshot-wide encounter revision, not a per-instance
        // counter. That makes the most recently entered recursive Red Room
        // unambiguous even when a child encounter has a different stable id.
        let next_epoch = self
            .stamps
            .iter()
            .map(|stamp| stamp.epoch)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let next = transition_for(gate, direction, existing, next_epoch);
        let mut stamps = self.stamps.clone();
        stamps.retain(|stamp| stamp.instance_id != gate.instance_id);
        stamps.push(AnomalyStateStamp::new(
            gate.instance_id,
            next.epoch,
            gate.plane,
            gate.axis,
            direction,
            next.phase,
            next.loop_count,
            gate.id,
        ));
        Self {
            delirium: self.delirium,
            ..Self::with_drifts_and_supplies(
                stamps,
                self.drifts.clone(),
                self.consumed_supplies.clone(),
            )
        }
    }

    fn hash_words(words: &[u32]) -> u64 {
        let mut hash = 0xCBF2_9CE4_8422_2325u64;
        for &word in words {
            hash ^= word as u64;
            hash = hash.wrapping_mul(0x100_0000_01B3);
        }
        mix64(hash)
    }

    pub fn fingerprint(&self) -> u64 {
        let words = self.words_without_fingerprint();
        Self::hash_words(&words)
    }

    fn words_without_fingerprint(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(
            4 + self.stamps.len() * Self::STAMP_WORDS
                + self.drifts.len() * Self::DRIFT_WORDS
                + self.consumed_supplies.len() * 2,
        );
        out.push(Self::MAGIC);
        out.push(self.stamps.len() as u32);
        for stamp in &self.stamps {
            out.extend_from_slice(&[
                stamp.instance_id as u32,
                (stamp.instance_id >> 32) as u32,
                stamp.epoch,
                stamp.gate_plane_milli as u32,
                (stamp.gate_axis as u32)
                    | ((stamp.travel_direction as u32) << 8)
                    | ((stamp.phase as u32) << 16)
                    | ((stamp.loop_count as u32) << 24),
                stamp.last_gate_id as u32,
                (stamp.last_gate_id >> 32) as u32,
            ]);
        }
        out.push(self.drifts.len() as u32);
        for drift in &self.drifts {
            out.extend_from_slice(&[
                drift.cell_x as u32,
                (drift.cell_x >> 32) as u32,
                drift.cell_z as u32,
                (drift.cell_z >> 32) as u32,
                drift.epoch,
            ]);
        }
        out.push(self.consumed_supplies.len() as u32);
        for &id in &self.consumed_supplies {
            out.push(id as u32);
            out.push((id >> 32) as u32);
        }
        out.push(self.delirium as u32);
        out
    }

    pub fn to_words(&self) -> Vec<u32> {
        let mut out = self.words_without_fingerprint();
        let fingerprint = Self::hash_words(&out);
        out.push(fingerprint as u32);
        out.push((fingerprint >> 32) as u32);
        out
    }

    pub fn from_words(words: &[u32]) -> Result<Self, RealitySnapshotDecodeError> {
        if words.len() < 5 || words[0] != Self::MAGIC {
            return Err(RealitySnapshotDecodeError::Header);
        }
        let count = words[1] as usize;
        let drift_header = 2 + count.saturating_mul(Self::STAMP_WORDS);
        if words.len() < drift_header + 4 {
            return Err(RealitySnapshotDecodeError::Length);
        }
        let drift_count = words[drift_header] as usize;
        let supply_header = drift_header + 1 + drift_count.saturating_mul(Self::DRIFT_WORDS);
        if words.len() < supply_header + 4 {
            return Err(RealitySnapshotDecodeError::Length);
        }
        let supply_count = words[supply_header] as usize;
        let delirium_at = supply_header + 1 + supply_count.saturating_mul(2);
        let body_len = delirium_at + 1;
        if words.len() != body_len + 2 {
            return Err(RealitySnapshotDecodeError::Length);
        }
        if words[delirium_at] > MAX_DELIRIUM as u32 {
            return Err(RealitySnapshotDecodeError::Value);
        }
        let expected = words[body_len] as u64 | ((words[body_len + 1] as u64) << 32);
        if Self::hash_words(&words[..body_len]) != expected {
            return Err(RealitySnapshotDecodeError::Fingerprint);
        }

        let mut stamps = Vec::with_capacity(count);
        for chunk in words[2..drift_header].chunks_exact(Self::STAMP_WORDS) {
            stamps.push(AnomalyStateStamp {
                instance_id: chunk[0] as u64 | ((chunk[1] as u64) << 32),
                epoch: chunk[2],
                gate_plane_milli: chunk[3] as i32,
                gate_axis: decode_axis(chunk[4] & 0xff).ok_or(RealitySnapshotDecodeError::Value)?,
                travel_direction: decode_direction((chunk[4] >> 8) & 0xff)
                    .ok_or(RealitySnapshotDecodeError::Value)?,
                phase: decode_phase((chunk[4] >> 16) & 0xff)
                    .ok_or(RealitySnapshotDecodeError::Value)?,
                loop_count: ((chunk[4] >> 24) & 0xff) as u8,
                last_gate_id: chunk[5] as u64 | ((chunk[6] as u64) << 32),
            });
        }
        let mut drifts = Vec::with_capacity(drift_count);
        for chunk in words[drift_header + 1..supply_header].chunks_exact(Self::DRIFT_WORDS) {
            drifts.push(FabricDriftStamp {
                cell_x: chunk[0] as u64 as i64 | (((chunk[1] as u64) << 32) as i64),
                cell_z: chunk[2] as u64 as i64 | (((chunk[3] as u64) << 32) as i64),
                epoch: chunk[4],
            });
        }
        let mut consumed = Vec::with_capacity(supply_count);
        for chunk in words[supply_header + 1..delirium_at].chunks_exact(2) {
            consumed.push(chunk[0] as u64 | ((chunk[1] as u64) << 32));
        }
        let decoded = Self::with_drifts_and_supplies(stamps, drifts, consumed)
            .with_delirium(words[delirium_at] as u8);
        if decoded.to_words() != words {
            return Err(RealitySnapshotDecodeError::Value);
        }
        Ok(decoded)
    }
}

fn decode_phase(value: u32) -> Option<RedRoomPhase> {
    match value {
        0 => Some(RedRoomPhase::Outside),
        1 => Some(RedRoomPhase::Sealed),
        2 => Some(RedRoomPhase::EscapeOpen),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::anomaly::{AnomalyKind, WorldBounds};

    fn gate(id: u64, kind: TraversalGateKind) -> TraversalGate {
        TraversalGate {
            id,
            instance_id: 77,
            anomaly_kind: AnomalyKind::RedRoom,
            kind,
            axis: Axis2::X,
            plane: 5.0,
            span_min: -1.0,
            span_max: 1.0,
            forward: AxisDirection::Positive,
            affected_bounds: WorldBounds::new(0.0, -5.0, 10.0, 5.0),
        }
    }

    #[test]
    fn forward_red_threshold_seals_the_room() {
        let next = RealitySnapshot::empty().with_advanced_gate(
            &gate(1, TraversalGateKind::RedThreshold),
            AxisDirection::Positive,
        );
        let stamp = next.lookup(77).unwrap();
        assert_eq!(stamp.epoch, 1);
        assert_eq!(stamp.phase, RedRoomPhase::Sealed);
        assert_eq!(stamp.loop_count, 0);
        assert_eq!(stamp.last_gate_id, 1);
    }

    #[test]
    fn reverse_red_threshold_preserves_outside_phase() {
        let next = RealitySnapshot::empty().with_advanced_gate(
            &gate(1, TraversalGateKind::RedThreshold),
            AxisDirection::Negative,
        );
        let stamp = next.lookup(77).unwrap();
        assert_eq!(stamp.epoch, 1);
        assert_eq!(stamp.phase, RedRoomPhase::Outside);
    }

    #[test]
    fn seven_accepted_loop_events_open_the_escape_and_one_more_seals_it() {
        let mut reality = RealitySnapshot::empty().with_advanced_gate(
            &gate(1, TraversalGateKind::RedThreshold),
            AxisDirection::Positive,
        );
        let loop_gate = gate(2, TraversalGateKind::RedLoop);
        let mut direction = AxisDirection::Negative;
        for _ in 0..7 {
            reality = reality.with_advanced_gate(&loop_gate, direction);
            direction = match direction {
                AxisDirection::Positive => AxisDirection::Negative,
                AxisDirection::Negative => AxisDirection::Positive,
            };
        }
        let stamp = reality.lookup(77).unwrap();
        assert_eq!(stamp.epoch, 8);
        assert_eq!(stamp.loop_count, 7);
        assert_eq!(stamp.phase, RedRoomPhase::EscapeOpen);

        // Touching the checkpoint again withdraws the offer entirely.
        let resealed = reality.with_advanced_gate(&loop_gate, direction);
        let stamp = resealed.lookup(77).unwrap();
        assert_eq!(stamp.phase, RedRoomPhase::Sealed);
        assert_eq!(stamp.loop_count, 0);
    }

    #[test]
    fn duplicate_gate_event_is_idempotent() {
        let loop_gate = gate(2, TraversalGateKind::RedLoop);
        let sealed = RealitySnapshot::empty().with_advanced_gate(
            &gate(1, TraversalGateKind::RedThreshold),
            AxisDirection::Positive,
        );
        let once = sealed.with_advanced_gate(&loop_gate, AxisDirection::Positive);
        let twice = once.with_advanced_gate(&loop_gate, AxisDirection::Positive);
        assert_eq!(twice, once);
    }

    #[test]
    fn remap_advances_epoch_without_losing_red_state() {
        let sealed = RealitySnapshot::empty().with_advanced_gate(
            &gate(1, TraversalGateKind::RedThreshold),
            AxisDirection::Positive,
        );
        let remapped =
            sealed.with_advanced_gate(&gate(2, TraversalGateKind::Remap), AxisDirection::Negative);
        let stamp = remapped.lookup(77).unwrap();
        assert_eq!(stamp.epoch, 2);
        assert_eq!(stamp.phase, RedRoomPhase::Sealed);
        assert_eq!(stamp.loop_count, 0);
    }

    #[test]
    fn epoch_orders_encounters_across_different_instances() {
        let first_gate = gate(1, TraversalGateKind::RedThreshold);
        let mut second_gate = gate(2, TraversalGateKind::RedThreshold);
        second_gate.instance_id = 88;
        let first =
            RealitySnapshot::empty().with_advanced_gate(&first_gate, AxisDirection::Positive);
        let second = first.with_advanced_gate(&second_gate, AxisDirection::Positive);

        assert_eq!(second.lookup(77).unwrap().epoch, 1);
        assert_eq!(second.lookup(88).unwrap().epoch, 2);
    }

    #[test]
    fn red_escape_records_crossing_without_changing_state() {
        let sealed = RealitySnapshot::empty().with_advanced_gate(
            &gate(1, TraversalGateKind::RedThreshold),
            AxisDirection::Positive,
        );
        let escaped = sealed.with_advanced_gate(
            &gate(2, TraversalGateKind::RedEscape),
            AxisDirection::Positive,
        );
        let before = sealed.lookup(77).unwrap();
        let after = escaped.lookup(77).unwrap();
        assert_eq!(after.epoch, before.epoch);
        assert_eq!(after.phase, before.phase);
        assert_eq!(after.loop_count, before.loop_count);
        assert_eq!(after.last_gate_id, 2);
    }

    #[test]
    fn fabric_drift_advances_per_cell_and_survives_gate_transitions() {
        let reality = RealitySnapshot::empty()
            .with_fabric_drift_advanced(2, -3)
            .with_fabric_drift_advanced(0, 0)
            .with_fabric_drift_advanced(2, -3);
        assert_eq!(reality.fabric_drift_epoch(2.5 * 40.0, -2.5 * 40.0), 2);
        assert_eq!(reality.fabric_drift_epoch(1.0, 1.0), 1);
        assert_eq!(reality.fabric_drift_epoch(400.0, 400.0), 0, "unvisited");

        // Encounter transitions carry drift along untouched.
        let sealed = reality.with_advanced_gate(
            &gate(1, TraversalGateKind::RedThreshold),
            AxisDirection::Positive,
        );
        assert_eq!(sealed.fabric_drift_epoch(2.5 * 40.0, -2.5 * 40.0), 2);
        assert_eq!(sealed.lookup(77).unwrap().phase, RedRoomPhase::Sealed);

        // And drift is part of transport identity.
        assert_eq!(
            RealitySnapshot::from_words(&sealed.to_words()),
            Ok(sealed.clone())
        );
        assert_ne!(reality.fingerprint(), RealitySnapshot::empty().fingerprint());
    }

    #[test]
    fn consumed_supplies_are_canonical_reality_and_survive_transport() {
        let reality = RealitySnapshot::empty()
            .with_supply_consumed(42)
            .with_supply_consumed(7)
            .with_supply_consumed(42);
        assert!(reality.supply_consumed(7));
        assert!(reality.supply_consumed(42));
        assert!(!reality.supply_consumed(9));

        // Order of consumption does not change identity.
        let other = RealitySnapshot::empty()
            .with_supply_consumed(7)
            .with_supply_consumed(42);
        assert_eq!(reality, other);
        assert_eq!(reality.fingerprint(), other.fingerprint());

        // Consumption is part of transport identity and survives both
        // drift advances and gate transitions.
        let round = RealitySnapshot::from_words(&reality.to_words()).unwrap();
        assert_eq!(round, reality);
        let drifted = reality.with_fabric_drift_advanced(1, 2);
        assert!(drifted.supply_consumed(42));
        let sealed = drifted.with_advanced_gate(
            &gate(1, TraversalGateKind::RedThreshold),
            AxisDirection::Positive,
        );
        assert!(sealed.supply_consumed(42));
        assert_ne!(
            reality.fingerprint(),
            RealitySnapshot::empty().fingerprint()
        );
    }

    #[test]
    fn delirium_is_identity_survives_transitions_and_transport() {
        let calm = RealitySnapshot::empty();
        let parched = calm.with_delirium(3);
        assert_ne!(calm, parched);
        assert_ne!(calm.fingerprint(), parched.fingerprint());
        assert_eq!(parched.with_delirium(9).delirium(), MAX_DELIRIUM, "clamped");

        // Every transition preserves the tier.
        let after = parched
            .with_supply_consumed(5)
            .with_fabric_drift_advanced(1, 1)
            .with_advanced_gate(
                &gate(1, TraversalGateKind::RedThreshold),
                AxisDirection::Positive,
            );
        assert_eq!(after.delirium(), 3);

        // And it is part of transport identity.
        let round = RealitySnapshot::from_words(&after.to_words()).unwrap();
        assert_eq!(round, after);
    }

    #[test]
    fn reality_transport_is_canonical_and_lossless() {
        let a = AnomalyStateStamp::new(
            9,
            2,
            -12.4,
            Axis2::Z,
            AxisDirection::Negative,
            RedRoomPhase::Outside,
            0,
            0,
        );
        let b = AnomalyStateStamp::new(
            3,
            7,
            8.0,
            Axis2::X,
            AxisDirection::Positive,
            RedRoomPhase::Sealed,
            4,
            12345,
        );
        let one = RealitySnapshot::new(vec![a, b]);
        let two = RealitySnapshot::new(vec![b, a]);
        assert_eq!(one, two);
        assert_eq!(one.fingerprint(), two.fingerprint());
        assert_eq!(RealitySnapshot::from_words(&one.to_words()), Ok(one));
    }
}
