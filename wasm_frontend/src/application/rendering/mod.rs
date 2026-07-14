//! Renderer-independent reference equations.
//!
//! This module defines the small, platform-free mathematical contract that
//! presentation drivers can implement. Values are linear (never gamma
//! encoded), distances are in world units, and all cosine terms are
//! dimensionless.

/// Linear RGB radiance or radiance-like intensity.
pub type LinearRgb = [f32; 3];

/// Beer-Lambert transmittance through a homogeneous medium.
///
/// ```text
/// T(d) = exp(-sigma_t * d)
/// ```
///
/// `distance` is a path length in world units and `extinction` is the total
/// extinction coefficient `sigma_t` in inverse world units. Negative lengths
/// and coefficients are treated as zero. A `NaN` input produces full
/// transmittance rather than contaminating the frame with `NaN` values.
pub fn beer_lambert_transmittance(distance: f32, extinction: f32) -> f32 {
    let Some(distance) = nonnegative(distance) else {
        return 1.0;
    };
    let Some(extinction) = nonnegative(extinction) else {
        return 1.0;
    };

    if distance == 0.0 || extinction == 0.0 {
        return 1.0;
    }

    (-extinction * distance).exp()
}

/// Beer-Lambert transmittance with a clear near field.
///
/// Fog occupies only the portion of the view ray beyond `fog_start`:
///
/// ```text
/// d_fog = max(view_distance - fog_start, 0)
/// T     = exp(-sigma_t * d_fog)
/// ```
///
/// All distances use world units. Negative distances and starts are clamped
/// to zero. Invalid (`NaN`) values fail open with full transmittance.
pub fn fog_transmittance(view_distance: f32, fog_start: f32, extinction: f32) -> f32 {
    let (Some(view_distance), Some(fog_start)) =
        (nonnegative(view_distance), nonnegative(fog_start))
    else {
        return 1.0;
    };

    let fog_distance = if view_distance <= fog_start {
        0.0
    } else {
        view_distance - fog_start
    };
    beer_lambert_transmittance(fog_distance, extinction)
}

/// Composes surface radiance with homogeneous fog in linear RGB.
///
/// ```text
/// L = T * L_surface + (1 - T) * L_fog
/// ```
///
/// Both RGB inputs must use the same linear radiometric units. Negative and
/// `NaN` channels are treated as zero. `transmittance` is clamped to `[0, 1]`;
/// `NaN` fails open (`T = 1`). Endpoint branches avoid `0 * infinity`.
pub fn compose_fog(
    surface_radiance: LinearRgb,
    fog_radiance: LinearRgb,
    transmittance: f32,
) -> LinearRgb {
    let surface = surface_radiance.map(nonnegative_radiance);
    let fog = fog_radiance.map(nonnegative_radiance);
    let transmittance = unit_interval_or_one(transmittance);

    if transmittance == 1.0 {
        return surface;
    }
    if transmittance == 0.0 {
        return fog;
    }

    let scattered = 1.0 - transmittance;
    [
        surface[0] * transmittance + fog[0] * scattered,
        surface[1] * transmittance + fog[1] * scattered,
        surface[2] * transmittance + fog[2] * scattered,
    ]
}

/// Smooth compact-support window for a finite-range light.
///
/// For distance `d` and range `R` (both world lengths), the window is
///
/// ```text
/// w(d, R) = (1 - (d / R)^4)^2,  0 <= d < R
///           0,                    d >= R
/// ```
///
/// The value and first derivative are zero at `R`, avoiding a hard lighting
/// seam. A non-positive, `NaN`, or otherwise invalid range disables the light.
pub fn compact_support_window(distance: f32, range: f32) -> f32 {
    let Some(distance) = nonnegative(distance) else {
        return 0.0;
    };
    let Some(range) = positive(range) else {
        return 0.0;
    };

    if distance >= range || distance == f32::INFINITY {
        return 0.0;
    }
    if range == f32::INFINITY {
        return 1.0;
    }

    let ratio_squared = (distance / range).powi(2);
    let one_minus_fourth = 1.0 - ratio_squared * ratio_squared;
    one_minus_fourth * one_minus_fourth
}

/// Finite-range inverse-square attenuation.
///
/// ```text
/// A(d) = w(d, R) / max(d, d_min)^2
/// ```
///
/// `distance`, `range`, and `minimum_distance` are world lengths, so the
/// result has units of inverse world-units squared. `minimum_distance` is the
/// finite emitter-size clamp that removes the point-light singularity. A
/// non-positive or `NaN` range/clamp disables the term; `NaN` distance also
/// returns zero.
pub fn finite_range_inverse_square_attenuation(
    distance: f32,
    range: f32,
    minimum_distance: f32,
) -> f32 {
    let Some(distance) = nonnegative(distance) else {
        return 0.0;
    };
    let Some(minimum_distance) = positive(minimum_distance) else {
        return 0.0;
    };

    let window = compact_support_window(distance, range);
    if window == 0.0 {
        return 0.0;
    }

    let denominator = distance.max(minimum_distance).powi(2);
    if denominator == f32::INFINITY {
        0.0
    } else {
        window / denominator
    }
}

/// Lambertian receiver cosine.
///
/// ```text
/// cos_receiver = max(dot(n_receiver, omega_to_light), 0)
/// ```
///
/// Vectors are normalized internally, so callers may use any finite non-zero
/// magnitude. Zero-length or non-finite vectors return zero.
pub fn lambertian_receiver_cosine(receiver_normal: [f32; 3], direction_to_light: [f32; 3]) -> f32 {
    positive_cosine(receiver_normal, direction_to_light)
}

/// One-sided emission cosine for a horizontal rectangular ceiling panel.
///
/// The panel's emitting normal is `(0, -1, 0)`. For a direction from a point
/// on the emitter toward the receiver,
///
/// ```text
/// cos_emitter = max(dot((0, -1, 0), omega_to_receiver), 0)
/// ```
///
/// Rectangle area belongs to the later irradiance integral; this function is
/// only its dimensionless directional factor. Zero-length or non-finite
/// directions return zero.
pub fn downward_rectangular_emitter_cosine(direction_to_receiver: [f32; 3]) -> f32 {
    positive_cosine([0.0, -1.0, 0.0], direction_to_receiver)
}

fn positive_cosine(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (Some(a), Some(b)) = (normalize(a), normalize(b)) else {
        return 0.0;
    };
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(0.0, 1.0)
}

/// Normalizes without overflowing when a finite component is near `f32::MAX`.
fn normalize(vector: [f32; 3]) -> Option<[f32; 3]> {
    if vector.iter().any(|component| !component.is_finite()) {
        return None;
    }

    let scale = vector
        .iter()
        .fold(0.0_f32, |largest, component| largest.max(component.abs()));
    if scale == 0.0 {
        return None;
    }

    let scaled = vector.map(|component| component / scale);
    let length = (scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2]).sqrt();
    Some(scaled.map(|component| component / length))
}

fn nonnegative(value: f32) -> Option<f32> {
    if value.is_nan() {
        None
    } else {
        Some(value.max(0.0))
    }
}

fn positive(value: f32) -> Option<f32> {
    if value.is_nan() || value <= 0.0 {
        None
    } else {
        Some(value)
    }
}

fn nonnegative_radiance(value: f32) -> f32 {
    if value.is_nan() || value < 0.0 {
        0.0
    } else {
        value
    }
}

fn unit_interval_or_one(value: f32) -> f32 {
    if value.is_nan() {
        1.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-6;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }

    fn assert_rgb_close(actual: LinearRgb, expected: LinearRgb, tolerance: f32) {
        for channel in 0..3 {
            assert_close(actual[channel], expected[channel], tolerance);
        }
    }

    #[test]
    fn beer_lambert_at_zero_distance_is_one() {
        for extinction in [0.0, 0.01, 0.5, 10.0, f32::INFINITY] {
            assert_eq!(beer_lambert_transmittance(0.0, extinction), 1.0);
        }
    }

    #[test]
    fn fog_is_clear_through_fog_start_and_monotone_after_it() {
        let start = 7.5;
        for distance in [0.0, 1.0, start, -4.0] {
            assert_eq!(fog_transmittance(distance, start, 0.2), 1.0);
        }

        let mut previous = 1.0;
        for step in 0..=200 {
            let distance = start + step as f32 * 0.25;
            let current = fog_transmittance(distance, start, 0.2);
            assert!(current <= previous + EPSILON);
            assert!((0.0..=1.0).contains(&current));
            previous = current;
        }
    }

    #[test]
    fn beer_lambert_obeys_the_multiplicative_distance_law() {
        for extinction in [0.01, 0.1, 0.7] {
            for a in [0.0, 0.25, 2.0, 9.0] {
                for b in [0.0, 0.5, 3.0, 11.0] {
                    let together = beer_lambert_transmittance(a + b, extinction);
                    let segmented = beer_lambert_transmittance(a, extinction)
                        * beer_lambert_transmittance(b, extinction);
                    assert_close(together, segmented, 2.0e-6);
                }
            }
        }
    }

    #[test]
    fn fog_start_only_removes_the_clear_part_of_the_ray() {
        let start = 5.0;
        let extinction = 0.17;
        for active_distance in [0.0, 0.25, 3.0, 20.0] {
            assert_close(
                fog_transmittance(start + active_distance, start, extinction),
                beer_lambert_transmittance(active_distance, extinction),
                EPSILON,
            );
        }
    }

    #[test]
    fn fog_composition_has_correct_endpoints_and_converges() {
        let surface = [4.0, 2.0, 1.0];
        let fog = [0.1, 0.2, 0.3];
        assert_eq!(compose_fog(surface, fog, 1.0), surface);
        assert_eq!(compose_fog(surface, fog, 0.0), fog);

        let transmittance = beer_lambert_transmittance(10_000.0, 0.01);
        assert_rgb_close(compose_fog(surface, fog, transmittance), fog, EPSILON);
    }

    #[test]
    fn composing_two_medium_segments_equals_one_combined_segment() {
        let surface = [3.0, 1.0, 0.5];
        let fog = [0.2, 0.3, 0.4];
        for (a, b) in [(0.1, 0.7), (1.0, 2.0), (5.0, 13.0)] {
            let ta = beer_lambert_transmittance(a, 0.3);
            let tb = beer_lambert_transmittance(b, 0.3);
            let segmented = compose_fog(compose_fog(surface, fog, ta), fog, tb);
            let combined = compose_fog(surface, fog, ta * tb);
            assert_rgb_close(segmented, combined, 2.0e-6);
        }
    }

    #[test]
    fn atmospheric_math_rejects_non_physical_inputs_without_nan() {
        assert_eq!(beer_lambert_transmittance(100.0, 0.0), 1.0);
        assert_eq!(beer_lambert_transmittance(-2.0, 0.4), 1.0);
        assert_eq!(beer_lambert_transmittance(2.0, -0.4), 1.0);
        assert_eq!(beer_lambert_transmittance(f32::NAN, 0.4), 1.0);
        assert_eq!(fog_transmittance(f32::NAN, 1.0, 0.4), 1.0);
        assert_eq!(fog_transmittance(5.0, f32::NAN, 0.4), 1.0);

        let composed = compose_fog([f32::NAN, -1.0, 2.0], [1.0, 1.0, 1.0], 0.5);
        assert_eq!(composed, [0.5, 0.5, 1.5]);
        assert_eq!(compose_fog([1.0; 3], [2.0; 3], f32::NAN), [1.0; 3]);
    }

    #[test]
    fn compact_window_is_bounded_monotone_and_zero_at_range() {
        let range = 12.0;
        let mut previous = 1.0;
        for step in 0..=120 {
            let value = compact_support_window(step as f32 * 0.1, range);
            assert!((0.0..=1.0).contains(&value));
            assert!(value <= previous + EPSILON);
            previous = value;
        }
        assert_eq!(compact_support_window(range, range), 0.0);
        assert_eq!(compact_support_window(range + 1.0, range), 0.0);
    }

    #[test]
    fn attenuation_has_inverse_square_ratio_away_from_cutoff() {
        let near = finite_range_inverse_square_attenuation(2.0, 100.0, 0.1);
        let far = finite_range_inverse_square_attenuation(4.0, 100.0, 0.1);
        assert_close(near / far, 4.0, 0.001);
    }

    #[test]
    fn attenuation_is_finite_at_the_source_and_zero_outside_range() {
        assert_eq!(finite_range_inverse_square_attenuation(0.0, 10.0, 0.5), 4.0);
        assert_eq!(
            finite_range_inverse_square_attenuation(10.0, 10.0, 0.5),
            0.0
        );
        assert_eq!(
            finite_range_inverse_square_attenuation(20.0, 10.0, 0.5),
            0.0
        );
    }

    #[test]
    fn attenuation_handles_invalid_lengths_without_nan_or_phantom_light() {
        assert_eq!(
            finite_range_inverse_square_attenuation(f32::NAN, 10.0, 0.5),
            0.0
        );
        assert_eq!(
            finite_range_inverse_square_attenuation(2.0, f32::NAN, 0.5),
            0.0
        );
        assert_eq!(finite_range_inverse_square_attenuation(2.0, 0.0, 0.5), 0.0);
        assert_eq!(finite_range_inverse_square_attenuation(2.0, 10.0, 0.0), 0.0);
        assert_eq!(
            finite_range_inverse_square_attenuation(2.0, -10.0, 0.5),
            0.0
        );
        assert_eq!(
            finite_range_inverse_square_attenuation(-1.0, 10.0, 0.5),
            finite_range_inverse_square_attenuation(0.0, 10.0, 0.5)
        );
    }

    #[test]
    fn lambertian_receiver_uses_normalized_positive_cosine() {
        assert_eq!(
            lambertian_receiver_cosine([0.0, 2.0, 0.0], [0.0, 5.0, 0.0]),
            1.0
        );
        assert_eq!(
            lambertian_receiver_cosine([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]),
            0.0
        );
        assert_close(
            lambertian_receiver_cosine([0.0, 1.0, 0.0], [1.0, 1.0, 0.0]),
            std::f32::consts::FRAC_1_SQRT_2,
            EPSILON,
        );
    }

    #[test]
    fn rectangular_panel_emits_downward_but_not_upward() {
        assert_eq!(downward_rectangular_emitter_cosine([0.0, -1.0, 0.0]), 1.0);
        assert_eq!(downward_rectangular_emitter_cosine([0.0, 1.0, 0.0]), 0.0);
        assert_eq!(downward_rectangular_emitter_cosine([1.0, 0.0, 0.0]), 0.0);
        assert_close(
            downward_rectangular_emitter_cosine([0.0, -2.0, 2.0]),
            std::f32::consts::FRAC_1_SQRT_2,
            EPSILON,
        );
    }

    #[test]
    fn cosine_terms_return_zero_for_zero_or_non_finite_vectors() {
        assert_eq!(lambertian_receiver_cosine([0.0; 3], [0.0, 1.0, 0.0]), 0.0);
        assert_eq!(
            lambertian_receiver_cosine([0.0, 1.0, 0.0], [f32::NAN, 0.0, 0.0]),
            0.0
        );
        assert_eq!(
            downward_rectangular_emitter_cosine([0.0, f32::INFINITY, 0.0]),
            0.0
        );
    }

    #[test]
    fn cosine_normalization_handles_large_finite_vectors() {
        let large = f32::MAX / 2.0;
        assert_close(
            lambertian_receiver_cosine([large, 0.0, 0.0], [large, large, 0.0]),
            std::f32::consts::FRAC_1_SQRT_2,
            EPSILON,
        );
    }
}
