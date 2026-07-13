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

/// A loop crossing matters only after commitment. Three accepted crossings
/// expose the escape while each accepted crossing advances the geometry epoch.
fn advance_red_loop(mut state: TransitionState, next_epoch: u32) -> TransitionState {
    if state.phase == RedRoomPhase::Sealed {
        state.loop_count = state.loop_count.saturating_add(1);
        if state.loop_count >= 3 {
            state.phase = RedRoomPhase::EscapeOpen;
        }
        state.epoch = next_epoch;
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
    const MAGIC: u32 = 0x5254_5902; // "RTY" v2
    const STAMP_WORDS: usize = 7;

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
        Self::new(stamps)
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
        let mut out = Vec::with_capacity(2 + self.stamps.len() * Self::STAMP_WORDS);
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
        for chunk in words[2..body_len].chunks_exact(Self::STAMP_WORDS) {
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
        let decoded = Self::new(stamps);
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
    fn three_accepted_loop_events_open_the_escape() {
        let mut reality = RealitySnapshot::empty().with_advanced_gate(
            &gate(1, TraversalGateKind::RedThreshold),
            AxisDirection::Positive,
        );
        let loop_gate = gate(2, TraversalGateKind::RedLoop);
        for direction in [
            AxisDirection::Positive,
            AxisDirection::Negative,
            AxisDirection::Positive,
        ] {
            reality = reality.with_advanced_gate(&loop_gate, direction);
        }
        let stamp = reality.lookup(77).unwrap();
        assert_eq!(stamp.epoch, 4);
        assert_eq!(stamp.loop_count, 3);
        assert_eq!(stamp.phase, RedRoomPhase::EscapeOpen);
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
