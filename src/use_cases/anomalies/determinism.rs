//! Shared deterministic primitives for anomaly planners.
//!
//! These helpers are intentionally integer-first. A candidate's identity must
//! depend on the world seed and its stable lattice anchor, never on which
//! region or voxel chunk happened to request it.

use crate::domain::entities::anomaly::AnomalyKind;

pub(crate) const SNAP: f32 = 0.4;

pub(crate) fn snap(value: f32) -> f32 {
    (value / SNAP).round() * SNAP
}

pub(crate) fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

pub(crate) fn hash(seed: u32, salt: u64, x: i64, z: i64) -> u64 {
    mix64(
        seed as u64
            ^ salt
            ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (z as u64).rotate_left(31),
    )
}

pub(crate) fn unit(hash: u64) -> f32 {
    (hash >> 40) as f32 / (1u64 << 24) as f32
}

pub(crate) fn stable_id(seed: u32, kind: AnomalyKind, anchor_x: i64, anchor_z: i64) -> u64 {
    hash(
        seed,
        0xA110_4A1E_0000_0000 | kind as u64,
        anchor_x,
        anchor_z,
    )
}
