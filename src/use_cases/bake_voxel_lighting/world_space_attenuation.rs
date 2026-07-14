use super::exposed_emissive_faces::EmissionProfile;
use super::settings::VoxelLightingSettings;

/// Adds one emission class after finite-range linear attenuation.
///
/// `cell_offsets` accounts for the half-cell from an emitting face to its
/// first air center. Air samples use 0.5. A solid receiver adds the second
/// half-cell from its neighboring air center to the receiving face and uses
/// 1.0. Those two halves make equal physical gaps agree at 0.1 and 0.2 unit
/// voxel sizes instead of differing by one resolution-dependent cell center.
pub(crate) fn accumulate_attenuated(
    accumulated: &mut [f32; 3],
    profile: EmissionProfile,
    distance_steps: u32,
    cell_offsets: f32,
    settings: VoxelLightingSettings,
) {
    let distance_world = (distance_steps as f32 + cell_offsets) * settings.voxel_size_world_units();
    let remaining = (1.0 - distance_world / settings.max_range_world_units()).clamp(0.0, 1.0);
    for channel in 0..3 {
        accumulated[channel] += profile.rgb[channel] * remaining;
    }
}

pub(crate) fn pack_rgb(accumulated: [f32; 3]) -> [u8; 3] {
    accumulated.map(|channel| channel.round().clamp(0.0, 15.0) as u8)
}

pub(crate) fn pack_emission(rgb: [f32; 3]) -> [u8; 3] {
    pack_rgb(rgb)
}
