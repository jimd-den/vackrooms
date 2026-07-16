//! Semantic crossings and hazards exported beside generated voxel geometry.
//!
//! These records describe interaction surfaces; they do not decide how a
//! reality changes. The application detects a crossing, then the reality
//! reducer interprets it as a deterministic state transition.

use crate::entities::models::Position;

use super::geometry::{Axis2, AxisDirection, WorldBounds, decode_axis, decode_direction};
use super::phenomena::{AnomalyId, AnomalyKind, decode_kind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TraversalGateKind {
    Remap = 0,
    RedThreshold = 1,
    RedLoop = 2,
    RedEscape = 3,
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

    pub fn from_words(words: &[u32]) -> Option<Self> {
        if words.len() != Self::WORDS {
            return None;
        }
        Some(Self {
            id: words[0] as u64 | ((words[1] as u64) << 32),
            instance_id: words[2] as u64 | ((words[3] as u64) << 32),
            anomaly_kind: decode_kind(words[4])?,
            kind: match words[5] {
                0 => TraversalGateKind::Remap,
                1 => TraversalGateKind::RedThreshold,
                2 => TraversalGateKind::RedLoop,
                3 => TraversalGateKind::RedEscape,
                _ => return None,
            },
            axis: decode_axis(words[6])?,
            plane: f32::from_bits(words[7]),
            span_min: f32::from_bits(words[8]),
            span_max: f32::from_bits(words[9]),
            forward: decode_direction(words[10])?,
            affected_bounds: WorldBounds::new(
                f32::from_bits(words[11]),
                f32::from_bits(words[12]),
                f32::from_bits(words[13]),
                f32::from_bits(words[14]),
            ),
        })
    }
}

/// A physical doorway between Backrooms levels. Unlike a [`TraversalGate`],
/// crossing one does not advance encounter state — it swaps which level
/// generator the application streams from, arriving at `arrival`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelExit {
    pub id: u64,
    /// The `GeneratorConfig::level` id on the far side of the door.
    pub target_level: u32,
    /// Trigger-volume center on the walking plane.
    pub center: Position,
    /// Trigger half-extent (square) around `center`.
    pub half_extent: f32,
    /// Arrival point in the *target* level's coordinate space.
    pub arrival: Position,
}

impl LevelExit {
    pub const WORDS: usize = 9;

    pub fn contains(&self, x: f32, z: f32) -> bool {
        (x - self.center.x).abs() <= self.half_extent
            && (z - self.center.z).abs() <= self.half_extent
    }

    pub fn to_words(self) -> [u32; Self::WORDS] {
        [
            self.id as u32,
            (self.id >> 32) as u32,
            self.target_level,
            self.center.x.to_bits(),
            self.center.z.to_bits(),
            self.half_extent.to_bits(),
            self.arrival.x.to_bits(),
            self.arrival.z.to_bits(),
            0, // reserved (orientation, future arrival yaw)
        ]
    }

    pub fn from_words(words: &[u32]) -> Option<Self> {
        if words.len() != Self::WORDS {
            return None;
        }
        Some(Self {
            id: words[0] as u64 | ((words[1] as u64) << 32),
            target_level: words[2],
            center: Position::new(f32::from_bits(words[3]), f32::from_bits(words[4])),
            half_extent: f32::from_bits(words[5]),
            arrival: Position::new(f32::from_bits(words[6]), f32::from_bits(words[7])),
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

    pub fn from_transport_words(words: &[u32]) -> Option<Self> {
        if words.len() != Self::TRANSPORT_WORDS {
            return None;
        }
        Some(Self {
            id: words[0] as u64 | ((words[1] as u64) << 32),
            instance_id: words[2] as u64 | ((words[3] as u64) << 32),
            center: Position::new(f32::from_bits(words[4]), f32::from_bits(words[5])),
            half_side: f32::from_bits(words[6]),
            depth: f32::from_bits(words[7]),
            recovery: Position::new(f32::from_bits(words[8]), f32::from_bits(words[9])),
        })
    }
}
