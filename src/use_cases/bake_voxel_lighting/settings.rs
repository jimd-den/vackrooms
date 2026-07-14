use std::error::Error;
use std::fmt::{Display, Formatter};

/// Default finite reach of baked fixture light, measured in world units.
///
/// Six units keeps ordinary 3.2--5.4 unit rooms illuminated while still
/// reaching zero before distant fixtures can flatten a whole streamed scene.
pub const DEFAULT_MAX_LIGHT_RANGE_WORLD_UNITS: f32 = 6.0;

/// Physical sampling settings for one voxel-light bake.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelLightingSettings {
    voxel_size_world_units: f32,
    max_range_world_units: f32,
}

impl VoxelLightingSettings {
    /// Creates validated settings. Both values are lengths in the same world
    /// coordinate system used by generated architecture and runtime lights.
    pub fn new(
        voxel_size_world_units: f32,
        max_range_world_units: f32,
    ) -> Result<Self, InvalidVoxelLightingSettings> {
        if !voxel_size_world_units.is_finite() || voxel_size_world_units <= 0.0 {
            return Err(InvalidVoxelLightingSettings::VoxelSize);
        }
        if !max_range_world_units.is_finite() || max_range_world_units <= 0.0 {
            return Err(InvalidVoxelLightingSettings::MaxRange);
        }
        Ok(Self {
            voxel_size_world_units,
            max_range_world_units,
        })
    }

    /// Uses the engine's ordinary finite fixture range with an exact scene
    /// voxel size supplied by the generation configuration.
    pub fn with_default_range(
        voxel_size_world_units: f32,
    ) -> Result<Self, InvalidVoxelLightingSettings> {
        Self::new(voxel_size_world_units, DEFAULT_MAX_LIGHT_RANGE_WORLD_UNITS)
    }

    pub fn voxel_size_world_units(self) -> f32 {
        self.voxel_size_world_units
    }

    pub fn max_range_world_units(self) -> f32 {
        self.max_range_world_units
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidVoxelLightingSettings {
    VoxelSize,
    MaxRange,
}

impl Display for InvalidVoxelLightingSettings {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VoxelSize => {
                formatter.write_str("voxel size must be finite and greater than zero")
            }
            Self::MaxRange => {
                formatter.write_str("maximum light range must be finite and greater than zero")
            }
        }
    }
}

impl Error for InvalidVoxelLightingSettings {}
