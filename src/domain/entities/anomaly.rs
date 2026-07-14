//! Immutable anomaly plans and deterministic encounter-state snapshots.
//!
//! The public surface remains a single `anomaly` namespace for callers, while
//! its internals follow the domain's actual vocabulary:
//!
//! - [`geometry`] owns rectilinear bounds and coordinate transforms;
//! - [`phenomena`] owns immutable anomaly instances and family parameters;
//! - [`traversal`] owns semantic gates and hazards;
//! - [`reality`] owns snapshots, transitions, and RTY v2 transport.
//!
//! The planner owns instance placement. Stateful geometry remains pure: a
//! chunk is sampled from an [`AnomalyInstance`] plus the matching stamp in a
//! [`RealitySnapshot`].

mod geometry;
mod phenomena;
mod reality;
mod traversal;

pub use geometry::*;
pub use phenomena::*;
pub use reality::*;
pub use traversal::*;

/// SplitMix64 finalizer shared by stable anomaly identities and fingerprints.
/// It stays private to the anomaly aggregate; callers consume only the stable
/// domain values produced from it.
fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pillar_sizes_vary_but_stay_snapped() {
        let lattice = PillarLattice {
            bay_x: 4.4,
            bay_z: 4.0,
            phase_x: 0.0,
            phase_z: 0.0,
            min_side: 1.2,
            max_side: 1.6,
            variation_seed: 42,
        };
        let mut seen = Vec::new();
        for z in 0..12 {
            for x in 0..12 {
                let side = lattice.pillar_size(x, z);
                assert!((1.2..=1.6).contains(&side));
                assert!(((side / 0.4).round() - side / 0.4).abs() < 1e-4);
                if !seen.contains(&side.to_bits()) {
                    seen.push(side.to_bits());
                }
            }
        }
        assert!(seen.len() >= 2);
    }

    #[test]
    fn gate_crossing_is_directional_and_span_bounded() {
        let gate = TraversalGate {
            id: 1,
            instance_id: 2,
            anomaly_kind: AnomalyKind::RedRoom,
            kind: TraversalGateKind::RedThreshold,
            axis: Axis2::X,
            plane: 5.0,
            span_min: -1.0,
            span_max: 1.0,
            forward: AxisDirection::Positive,
            affected_bounds: WorldBounds::new(0.0, -5.0, 10.0, 5.0),
        };
        assert_eq!(
            gate.crossing([4.0, 0.0], [6.0, 0.0]),
            Some(AxisDirection::Positive)
        );
        assert_eq!(gate.crossing([4.0, 2.0], [6.0, 2.0]), None);
        assert_eq!(gate.crossing([5.0, 0.0], [6.0, 0.0]), None);
    }
}
