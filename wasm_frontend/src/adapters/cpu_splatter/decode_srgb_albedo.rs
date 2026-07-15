//! Decode authored 8-bit sRGB albedo at data-preparation boundaries.
//!
//! Production splats carry this linear value from the per-upload atlas/MIP
//! pass, so the hot shading loop never evaluates three transfer-function
//! powers. Reference callers that begin with an encoded color use the same
//! functions and therefore retain exactly the same radiometric contract.

use crate::application::rendering::LinearRgb;

pub(crate) fn srgb_channel_to_linear(byte: f32) -> f32 {
    let encoded = (byte / 255.0).clamp(0.0, 1.0);
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn decode_albedo(bytes: [f32; 3]) -> LinearRgb {
    bytes.map(srgb_channel_to_linear)
}

pub(crate) fn decode_packed_albedo(packed: u32) -> LinearRgb {
    decode_albedo([
        ((packed >> 16) & 0xFF) as f32,
        ((packed >> 8) & 0xFF) as f32,
        (packed & 0xFF) as f32,
    ])
}

#[cfg(test)]
mod tests {
    use super::{decode_albedo, decode_packed_albedo, srgb_channel_to_linear};

    #[test]
    fn packed_and_unpacked_authored_colors_share_one_decode() {
        assert_eq!(
            decode_packed_albedo(0x80_FF_00),
            decode_albedo([128.0, 255.0, 0.0])
        );
        assert!((srgb_channel_to_linear(128.0) - 0.215_86).abs() < 1.0e-4);
    }
}
