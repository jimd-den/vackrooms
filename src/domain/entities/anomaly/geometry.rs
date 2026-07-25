//! Rectilinear world geometry shared by every anomaly family.
//!
//! Anomalies are authored on an orthogonal lattice. Keeping the transforms in
//! one small module makes that invariant visible: planners choose a footprint,
//! while samplers only translate between world and footprint-local space.

use crate::domain::entities::position::Position;

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

pub(super) fn decode_axis(value: u32) -> Option<Axis2> {
    match value {
        0 => Some(Axis2::X),
        1 => Some(Axis2::Z),
        _ => None,
    }
}

pub(super) fn decode_direction(value: u32) -> Option<AxisDirection> {
    match value {
        0 => Some(AxisDirection::Negative),
        1 => Some(AxisDirection::Positive),
        _ => None,
    }
}
