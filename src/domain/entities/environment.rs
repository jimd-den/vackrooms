//! Shared material/environment semantics for Level 0.
//!
//! Generation decides *meaning* here — wall treatment, carpet depth and
//! condition, floor state, ceiling character, fixture policy, ambience, and
//! stability class — and every consumer (voxelization, renderers via the
//! material palette, engine audio hooks, debug tooling) derives from the same
//! profile instead of re-inventing ad hoc zone checks.

use crate::domain::entities::voxel_grid::{
    VOXEL_DAMAGED_WALL, VOXEL_DEEP_CARPET, VOXEL_DRY_CARPET, VOXEL_FLOOR, VOXEL_FLUID,
    VOXEL_GLIMMER, VOXEL_LIGHT, VOXEL_PALE_WALL, VOXEL_RED_LIGHT, VOXEL_RED_WALL,
    VOXEL_STICKY_CARPET, VOXEL_WALL,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallTreatment {
    /// Ordinary Level 0 yellow wallpaper.
    YellowWallpaper,
    /// Pale bone-tinted arch-room masonry.
    PaleArch,
    /// Rough, damaged blackout-expanse surface.
    RoughBlackout,
    /// Peeled wallpaper with crimson underneath (red-room contamination).
    CrimsonPeeled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarpetDepth {
    Shallow,
    Normal,
    Deep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarpetCondition {
    Dry,
    Damp,
    /// Wet, sticky, coarse — the red-room approach and interior.
    WetSticky,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloorState {
    Level,
    /// Recessed basin pooling dark ankle-deep fluid (visual-only; collision
    /// stays flat so no random unavoidable deaths).
    RecessedFluid,
    /// A genuine opening (pit lattice). The player is relocated on entry.
    Open,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeilingCharacter {
    Compressed,
    Dropped,
    Open,
    Vaulted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixturePolicy {
    /// Dense office fluorescent grid.
    OfficeGrid,
    /// Wider, sparser rhythm (expanses).
    SparseWide,
    /// Only the cool recovery-skeleton glimmers.
    GlimmerSkeleton,
    /// Every fixture burns red (red rooms).
    RedPressure,
    /// Sparse warm pendants (arch rooms).
    WarmPendant,
    /// No fixtures at all.
    Unlit,
}

/// Ambience the space asks for. Audio playback does not exist yet; this is
/// the shared cue contract a future audio driver consumes, and what debug
/// tooling reports today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioAmbience {
    FluorescentBuzz,
    FadingBuzz,
    BlackoutSilence,
    /// Directional glimmer/buzz cue toward the recovery skeleton.
    GlimmerCue,
    RedPressure,
}

/// How a space participates in the spatial-reality contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityClass {
    /// Never accepts epoch input by type (arch rooms). The fixed landmark
    /// the mutable world is measured against.
    ImmutableAnchor,
    /// Ordinary static Level 0 fabric — deterministic and epoch-independent.
    StaticSpace,
    /// May re-hash in the committed wake behind the player (pillar and
    /// blackout expanses).
    WakeRemappable,
    /// Regenerates as a closed loop through explicit threshold events
    /// (red rooms).
    ThresholdLoop,
}

/// One compact semantic profile for a column/space of Level 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvironmentProfile {
    pub wall: WallTreatment,
    pub carpet_depth: CarpetDepth,
    pub carpet_condition: CarpetCondition,
    pub floor: FloorState,
    pub ceiling: CeilingCharacter,
    pub fixtures: FixturePolicy,
    pub audio: AudioAmbience,
    pub stability: StabilityClass,
}

impl EnvironmentProfile {
    pub const fn level0_fabric() -> Self {
        Self {
            wall: WallTreatment::YellowWallpaper,
            carpet_depth: CarpetDepth::Normal,
            carpet_condition: CarpetCondition::Damp,
            floor: FloorState::Level,
            ceiling: CeilingCharacter::Dropped,
            fixtures: FixturePolicy::OfficeGrid,
            audio: AudioAmbience::FluorescentBuzz,
            stability: StabilityClass::StaticSpace,
        }
    }

    /// Calmer, drier, paler than ordinary Level 0.
    pub const fn pillar_expanse() -> Self {
        Self {
            wall: WallTreatment::YellowWallpaper,
            carpet_depth: CarpetDepth::Shallow,
            carpet_condition: CarpetCondition::Dry,
            floor: FloorState::Level,
            ceiling: CeilingCharacter::Open,
            fixtures: FixturePolicy::SparseWide,
            audio: AudioAmbience::FluorescentBuzz,
            stability: StabilityClass::WakeRemappable,
        }
    }

    /// `depth` is the normalized distance into the footprint (0 boundary,
    /// 1 core): the approach earns its darkness gradually.
    pub fn blackout(depth: f32) -> Self {
        Self {
            wall: if depth > 0.15 {
                WallTreatment::RoughBlackout
            } else {
                WallTreatment::YellowWallpaper
            },
            carpet_depth: CarpetDepth::Normal,
            carpet_condition: CarpetCondition::Damp,
            floor: if depth > 0.35 {
                FloorState::RecessedFluid
            } else {
                FloorState::Level
            },
            ceiling: if depth > 0.68 {
                CeilingCharacter::Compressed
            } else {
                CeilingCharacter::Dropped
            },
            fixtures: if depth > 0.22 {
                FixturePolicy::GlimmerSkeleton
            } else {
                FixturePolicy::OfficeGrid
            },
            audio: if depth > 0.55 {
                AudioAmbience::BlackoutSilence
            } else if depth > 0.22 {
                AudioAmbience::GlimmerCue
            } else {
                AudioAmbience::FadingBuzz
            },
            stability: StabilityClass::WakeRemappable,
        }
    }

    /// `contamination` is 0 outside the warning band, 1 at the threshold.
    pub fn red_room(contamination: f32) -> Self {
        Self {
            wall: if contamination > 0.33 {
                WallTreatment::CrimsonPeeled
            } else {
                WallTreatment::YellowWallpaper
            },
            carpet_depth: CarpetDepth::Deep,
            carpet_condition: CarpetCondition::WetSticky,
            floor: FloorState::Level,
            ceiling: CeilingCharacter::Compressed,
            fixtures: FixturePolicy::RedPressure,
            audio: AudioAmbience::RedPressure,
            stability: StabilityClass::ThresholdLoop,
        }
    }

    pub const fn archway_room() -> Self {
        Self {
            wall: WallTreatment::PaleArch,
            carpet_depth: CarpetDepth::Deep,
            carpet_condition: CarpetCondition::WetSticky,
            floor: FloorState::Level,
            ceiling: CeilingCharacter::Dropped,
            fixtures: FixturePolicy::WarmPendant,
            audio: AudioAmbience::FluorescentBuzz,
            stability: StabilityClass::ImmutableAnchor,
        }
    }

    /// The wall material this profile voxelizes as.
    pub fn wall_voxel(&self) -> u8 {
        match self.wall {
            WallTreatment::YellowWallpaper => VOXEL_WALL,
            WallTreatment::PaleArch => VOXEL_PALE_WALL,
            WallTreatment::RoughBlackout => VOXEL_DAMAGED_WALL,
            WallTreatment::CrimsonPeeled => VOXEL_RED_WALL,
        }
    }

    /// The floor material this profile voxelizes as.
    pub fn floor_voxel(&self) -> u8 {
        if self.floor == FloorState::RecessedFluid {
            return VOXEL_FLUID;
        }
        match (self.carpet_depth, self.carpet_condition) {
            (CarpetDepth::Shallow, _) | (_, CarpetCondition::Dry) => VOXEL_DRY_CARPET,
            (CarpetDepth::Deep, CarpetCondition::WetSticky) => match self.wall {
                WallTreatment::CrimsonPeeled => VOXEL_STICKY_CARPET,
                _ => VOXEL_DEEP_CARPET,
            },
            (CarpetDepth::Deep, _) => VOXEL_DEEP_CARPET,
            _ => VOXEL_FLOOR,
        }
    }

    /// The fixture material this profile voxelizes as.
    pub fn light_voxel(&self) -> u8 {
        match self.fixtures {
            FixturePolicy::GlimmerSkeleton => VOXEL_GLIMMER,
            FixturePolicy::RedPressure => VOXEL_RED_LIGHT,
            _ => VOXEL_LIGHT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_map_to_distinct_materials() {
        let fabric = EnvironmentProfile::level0_fabric();
        let pillar = EnvironmentProfile::pillar_expanse();
        let blackout = EnvironmentProfile::blackout(0.9);
        let red = EnvironmentProfile::red_room(1.0);
        let arch = EnvironmentProfile::archway_room();

        assert_eq!(fabric.wall_voxel(), VOXEL_WALL);
        assert_eq!(fabric.floor_voxel(), VOXEL_FLOOR);
        assert_eq!(pillar.floor_voxel(), VOXEL_DRY_CARPET);
        assert_eq!(blackout.wall_voxel(), VOXEL_DAMAGED_WALL);
        assert_eq!(blackout.floor_voxel(), VOXEL_FLUID);
        assert_eq!(blackout.light_voxel(), VOXEL_GLIMMER);
        assert_eq!(red.wall_voxel(), VOXEL_RED_WALL);
        assert_eq!(red.floor_voxel(), VOXEL_STICKY_CARPET);
        assert_eq!(red.light_voxel(), VOXEL_RED_LIGHT);
        assert_eq!(arch.wall_voxel(), VOXEL_PALE_WALL);
        assert_eq!(arch.floor_voxel(), VOXEL_DEEP_CARPET);
    }

    #[test]
    fn blackout_gradient_earns_its_darkness() {
        let approach = EnvironmentProfile::blackout(0.1);
        assert_eq!(approach.wall, WallTreatment::YellowWallpaper);
        assert_eq!(approach.fixtures, FixturePolicy::OfficeGrid);
        let core = EnvironmentProfile::blackout(0.9);
        assert_eq!(core.ceiling, CeilingCharacter::Compressed);
        assert_eq!(core.audio, AudioAmbience::BlackoutSilence);
    }

    #[test]
    fn arch_rooms_are_immutable_anchors() {
        assert_eq!(
            EnvironmentProfile::archway_room().stability,
            StabilityClass::ImmutableAnchor
        );
    }
}
