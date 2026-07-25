//! Consumable supply items exported beside generated voxel geometry.
//!
//! A supply item is a semantic pickup point: generation places it (and a
//! small marker of matching voxels), the application detects proximity and
//! consumes it. Whether an item still exists is *reality* state — consumed
//! ids live in the `RealitySnapshot`, so a regenerated chunk deterministically
//! omits what the wanderer already drank.

use crate::domain::entities::position::Position;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SupplyKind {
    /// Canon Backrooms hydration: the safest liquid on Level 0 and Level 1.
    AlmondWater = 0,
    /// Scavenged food (Level 1 supply crates, rare Level 0 finds).
    Ration = 1,
}

pub(crate) fn decode_supply_kind(value: u32) -> Option<SupplyKind> {
    match value {
        0 => Some(SupplyKind::AlmondWater),
        1 => Some(SupplyKind::Ration),
        _ => None,
    }
}

/// One placed consumable. `position` is the marker's floor point; `rest_y`
/// is the marker's base height in world units (bottles sit on crates too).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupplyItem {
    pub id: u64,
    pub kind: SupplyKind,
    pub position: Position,
    pub rest_y: f32,
}

impl SupplyItem {
    pub const WORDS: usize = 6;

    pub fn within_reach(&self, x: f32, z: f32, reach: f32) -> bool {
        let dx = x - self.position.x;
        let dz = z - self.position.z;
        dx * dx + dz * dz <= reach * reach
    }

    pub fn to_words(self) -> [u32; Self::WORDS] {
        [
            self.id as u32,
            (self.id >> 32) as u32,
            self.kind as u32,
            self.position.x.to_bits(),
            self.position.z.to_bits(),
            self.rest_y.to_bits(),
        ]
    }

    pub fn from_words(words: &[u32]) -> Option<Self> {
        if words.len() != Self::WORDS {
            return None;
        }
        Some(Self {
            id: words[0] as u64 | ((words[1] as u64) << 32),
            kind: decode_supply_kind(words[2])?,
            position: Position::new(f32::from_bits(words[3]), f32::from_bits(words[4])),
            rest_y: f32::from_bits(words[5]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_words_round_trip() {
        let item = SupplyItem {
            id: 0xDEAD_BEEF_0042_1177,
            kind: SupplyKind::Ration,
            position: Position::new(-12.25, 830.5),
            rest_y: 0.8,
        };
        assert_eq!(SupplyItem::from_words(&item.to_words()), Some(item));
        assert_eq!(SupplyItem::from_words(&[0; 3]), None);
    }

    #[test]
    fn reach_is_a_horizontal_disc() {
        let item = SupplyItem {
            id: 1,
            kind: SupplyKind::AlmondWater,
            position: Position::new(10.0, 10.0),
            rest_y: 0.0,
        };
        assert!(item.within_reach(10.5, 10.0, 0.6));
        assert!(!item.within_reach(11.0, 11.0, 0.6));
    }
}
