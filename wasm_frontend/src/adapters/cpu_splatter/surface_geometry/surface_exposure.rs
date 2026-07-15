//! Compact directional surface-area data.

use vackrooms::domain::entities::voxel_grid::{
    FACE_OCCLUDED_NEGATIVE_X, FACE_OCCLUDED_NEGATIVE_Y, FACE_OCCLUDED_NEGATIVE_Z,
    FACE_OCCLUDED_POSITIVE_X, FACE_OCCLUDED_POSITIVE_Y, FACE_OCCLUDED_POSITIVE_Z,
};

/// Stable direction order shared by traversal and MIP preparation.
///
/// The discriminants index [`SurfaceExposure`]'s six bytes. Keeping the
/// normal here makes it impossible for propagation and shading to disagree
/// about bit/axis conventions.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceFace {
    NegativeX = 0,
    PositiveX = 1,
    NegativeY = 2,
    PositiveY = 3,
    NegativeZ = 4,
    PositiveZ = 5,
}

impl SurfaceFace {
    pub const ALL: [Self; 6] = [
        Self::NegativeX,
        Self::PositiveX,
        Self::NegativeY,
        Self::PositiveY,
        Self::NegativeZ,
        Self::PositiveZ,
    ];

    pub const fn normal(self) -> [f32; 3] {
        match self {
            Self::NegativeX => [-1.0, 0.0, 0.0],
            Self::PositiveX => [1.0, 0.0, 0.0],
            Self::NegativeY => [0.0, -1.0, 0.0],
            Self::PositiveY => [0.0, 1.0, 0.0],
            Self::NegativeZ => [0.0, 0.0, -1.0],
            Self::PositiveZ => [0.0, 0.0, 1.0],
        }
    }

    pub const fn opposite(self) -> Self {
        match self {
            Self::NegativeX => Self::PositiveX,
            Self::PositiveX => Self::NegativeX,
            Self::NegativeY => Self::PositiveY,
            Self::PositiveY => Self::NegativeY,
            Self::NegativeZ => Self::PositiveZ,
            Self::PositiveZ => Self::NegativeZ,
        }
    }

    pub(super) const fn axis_bit(self) -> u32 {
        match self {
            Self::NegativeX | Self::PositiveX => 1,
            Self::NegativeY | Self::PositiveY => 2,
            Self::NegativeZ | Self::PositiveZ => 4,
        }
    }

    pub(super) const fn is_positive(self) -> bool {
        matches!(self, Self::PositiveX | Self::PositiveY | Self::PositiveZ)
    }
}

/// Exposed area in each axis direction, normalized to one node face.
///
/// `0` means no known surface in that direction and `255` means one complete
/// node face. MIP preparation may use intermediate values or saturate above
/// one node-face worth of internal surface. Six bytes per node are enough to
/// preserve orientation without putting floats in the atlas-side cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SurfaceExposure {
    area: [u8; 6],
}

impl SurfaceExposure {
    pub const NONE: Self = Self { area: [0; 6] };
    pub const ALL_FACES: Self = Self { area: [255; 6] };

    pub const fn from_area(area: [u8; 6]) -> Self {
        Self { area }
    }

    /// Decodes the authoritative six neighbor bits carried by a real atlas
    /// leaf. Clear occupancy means air, hence a fully exposed face.
    pub const fn from_neighbor_occlusion(mask: u8) -> Self {
        Self::from_area([
            exposed_area(mask, FACE_OCCLUDED_NEGATIVE_X),
            exposed_area(mask, FACE_OCCLUDED_POSITIVE_X),
            exposed_area(mask, FACE_OCCLUDED_NEGATIVE_Y),
            exposed_area(mask, FACE_OCCLUDED_POSITIVE_Y),
            exposed_area(mask, FACE_OCCLUDED_NEGATIVE_Z),
            exposed_area(mask, FACE_OCCLUDED_POSITIVE_Z),
        ])
    }

    pub fn area(self, face: SurfaceFace) -> u8 {
        self.area[face as usize]
    }

    pub fn contains(self, face: SurfaceFace) -> bool {
        self.area(face) != 0
    }

    pub(super) fn set_area(&mut self, face: SurfaceFace, area: u8) {
        self.area[face as usize] = area;
    }

    pub(crate) fn add_saturating(&mut self, other: Self) {
        for face in SurfaceFace::ALL {
            let index = face as usize;
            self.area[index] = self.area[index].saturating_add(other.area[index]);
        }
    }

    pub(super) fn is_empty(self) -> bool {
        self.area == [0; 6]
    }

    /// Directional area of a child subtree expressed in parent-face units.
    /// One octant face has one quarter of its parent's face area.
    pub(crate) fn scaled_to_parent(self) -> Self {
        Self::from_area(self.area.map(|area| ((u16::from(area) + 2) / 4) as u8))
    }
}

const fn exposed_area(mask: u8, neighbor_bit: u8) -> u8 {
    if mask & neighbor_bit == 0 { 255 } else { 0 }
}
