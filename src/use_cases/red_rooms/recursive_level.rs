//! Addressing for the Level 0 world reached through a committed Red Room.
//!
//! A Red Room does not invent a second procedural grammar.  It addresses a
//! distant, deterministic copy of Level 0 and changes its presentation.  The
//! address must therefore be stable across chunk order and worker boundaries,
//! while retaining all 64 bits of anomaly identity until hashing is complete.
//! Converting an `AnomalyId` directly to `f32` loses most of those bits and can
//! make distinct rooms share a world; this module derives integer region
//! coordinates first and converts only the small final translation.

use crate::domain::entities::anomaly::{
    AnomalyId, AnomalyStateStamp, Axis2, PitHazard, RealitySnapshot, RedRoomPhase, TraversalGate,
    WorldBounds,
};
use crate::domain::entities::architecture::RegionPlan;
use crate::entities::models::Position;
use crate::use_cases::generate_chunk::GeneratorConfig;
use crate::use_cases::infinite_level::InfiniteRegionWindow;
use crate::use_cases::level_generator::LEVEL_BACKROOMS;
use crate::use_cases::ports::NoiseProvider;
use crate::use_cases::region_plan::REGION_SIZE;

const ADDRESS_DOMAIN: u64 = 0x5245_4452_4F4F_4D00; // "REDROOM\0"
// Keep translated coordinates near enough to the origin that f32 retains
// sub-centimetre detail for future fine voxel requests. Branch uniqueness is
// carried by the derived seed, so the translation itself need not be huge.
const MIN_TRANSLATION_REGIONS: i64 = 64;
const TRANSLATION_REGION_SPAN: u64 = 192;

/// Stable procedural address of the recursive Level 0 associated with one
/// Red Room instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RecursiveLevelAddress {
    instance_id: AnomalyId,
    derived_seed: u32,
    translation_regions: (i64, i64),
}

/// The finite planning window needed to render one request into a recursive
/// Level 0 branch.  The branch remains infinite; this value is only its local
/// query cache, just like an ordinary chunk's region window.
pub(crate) struct RecursiveLevelWindow {
    address: RecursiveLevelAddress,
    seed: u32,
    config: GeneratorConfig,
    regions: InfiniteRegionWindow,
}

impl RecursiveLevelWindow {
    pub(crate) fn around_chunk(
        source_origin: Position,
        chunk_size: f32,
        halo: f32,
        world_seed: u32,
        base_config: GeneratorConfig,
        noise: &dyn NoiseProvider,
        reality: &RealitySnapshot,
    ) -> Option<Self> {
        let (address, _) = RecursiveLevelAddress::from_reality(world_seed, reality)?;
        let seed = address.derived_seed();
        let config = address.recursive_config(base_config);
        let recursive_origin = address.to_recursive_world(source_origin);
        let regions = InfiniteRegionWindow::around_chunk(
            recursive_origin,
            chunk_size,
            halo,
            seed,
            &config,
            noise,
        );
        Some(Self {
            address,
            seed,
            config,
            regions,
        })
    }

    pub(crate) fn seed(&self) -> u32 {
        self.seed
    }

    pub(crate) fn instance_id(&self) -> AnomalyId {
        self.address.instance_id
    }

    pub(crate) fn config(&self) -> &GeneratorConfig {
        &self.config
    }

    /// Returns both the recursive coordinate and the plan that owns it.
    /// Keeping them paired prevents the old failure where a recursive plan
    /// was sampled with the base world's seed and coordinates.
    pub(crate) fn plan_at(&self, source: Position) -> Option<(Position, &RegionPlan)> {
        let recursive = self.address.to_recursive_world(source);
        self.regions
            .plan_at(recursive)
            .map(|plan| (recursive, plan))
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = ((i64, i64), &RegionPlan)> {
        self.regions.iter()
    }

    /// Project a semantic gate authored in the recursive branch back into
    /// the player's visible coordinate frame.
    pub(crate) fn project_gate(&self, mut gate: TraversalGate) -> TraversalGate {
        let offset = self.address.translation_world();
        match gate.axis {
            Axis2::X => {
                gate.plane -= offset.x;
                gate.span_min -= offset.z;
                gate.span_max -= offset.z;
            }
            Axis2::Z => {
                gate.plane -= offset.z;
                gate.span_min -= offset.x;
                gate.span_max -= offset.x;
            }
        }
        gate.affected_bounds = self.project_bounds(gate.affected_bounds);
        gate
    }

    pub(crate) fn project_hazard(&self, mut hazard: PitHazard) -> PitHazard {
        hazard.center = self.address.to_source_world(hazard.center);
        hazard.recovery = self.address.to_source_world(hazard.recovery);
        hazard
    }

    pub(crate) fn project_position(&self, recursive: Position) -> Position {
        self.address.to_source_world(recursive)
    }

    pub(crate) fn to_recursive_bounds(&self, bounds: WorldBounds) -> WorldBounds {
        let min = self
            .address
            .to_recursive_world(Position::new(bounds.min_x, bounds.min_z));
        let max = self
            .address
            .to_recursive_world(Position::new(bounds.max_x, bounds.max_z));
        WorldBounds::new(min.x, min.z, max.x, max.z)
    }

    fn project_bounds(&self, bounds: WorldBounds) -> WorldBounds {
        let min = self
            .address
            .to_source_world(Position::new(bounds.min_x, bounds.min_z));
        let max = self
            .address
            .to_source_world(Position::new(bounds.max_x, bounds.max_z));
        WorldBounds::new(min.x, min.z, max.x, max.z)
    }
}

impl RecursiveLevelAddress {
    /// Derives an address using integer avalanche hashes only.  Both halves of
    /// the 64-bit instance id participate in the seed and in each axis.
    pub(crate) fn derive(world_seed: u32, instance_id: AnomalyId) -> Self {
        let root = mix64(
            ADDRESS_DOMAIN ^ instance_id ^ (world_seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        let seed_hash = mix64(root ^ 0x5EED_5EED_A11C_0DE5);
        let derived_seed = (seed_hash as u32) ^ ((seed_hash >> 32) as u32);
        let translation_regions = (
            translation_component(root, 0x5841_5849_5300_0001),
            translation_component(root, 0x5A41_5849_5300_0002),
        );

        Self {
            instance_id,
            derived_seed,
            translation_regions,
        }
    }

    /// Selects the committed Red Room represented by a canonical reality
    /// snapshot, then derives its recursive address.
    pub(crate) fn from_reality(
        world_seed: u32,
        reality: &RealitySnapshot,
    ) -> Option<(Self, &AnomalyStateStamp)> {
        let state = active_red_room(reality)?;
        Some((Self::derive(world_seed, state.instance_id), state))
    }

    pub(crate) fn derived_seed(self) -> u32 {
        self.derived_seed
    }

    pub(crate) fn translation_regions(self) -> (i64, i64) {
        self.translation_regions
    }

    /// Exact transform for region keys.  Planning should prefer this form so
    /// large-world floating-point precision never participates in identity.
    pub(crate) fn to_recursive_region(self, source_region: (i64, i64)) -> (i64, i64) {
        (
            source_region.0.saturating_add(self.translation_regions.0),
            source_region.1.saturating_add(self.translation_regions.1),
        )
    }

    pub(crate) fn to_source_region(self, recursive_region: (i64, i64)) -> (i64, i64) {
        (
            recursive_region
                .0
                .saturating_sub(self.translation_regions.0),
            recursive_region
                .1
                .saturating_sub(self.translation_regions.1),
        )
    }

    /// World-space form used by column sampling.  The translation is an exact
    /// multiple of the 80u region lattice, preserving portals and 0.4u plan
    /// geometry across every chunk and LOD.
    pub(crate) fn to_recursive_world(self, source: Position) -> Position {
        let (region, local) = split_world_position(source);
        let recursive = self.to_recursive_region(region);
        join_world_position(recursive, local)
    }

    pub(crate) fn to_source_world(self, recursive: Position) -> Position {
        let (region, local) = split_world_position(recursive);
        let source = self.to_source_region(region);
        join_world_position(source, local)
    }

    pub(crate) fn translation_world(self) -> Position {
        let translation = self.translation_regions();
        Position::new(
            translation.0 as f32 * REGION_SIZE,
            translation.1 as f32 * REGION_SIZE,
        )
    }

    /// Produces a Level 0 config for the addressed world.  Organic anomaly
    /// tuning is preserved, allowing recursive Level 0 to remain the complete
    /// generator; only the one-shot debug override is cleared so it cannot be
    /// cloned into every recursive address.
    pub(crate) fn recursive_config(self, mut base: GeneratorConfig) -> GeneratorConfig {
        base.level = LEVEL_BACKROOMS;
        base.anomalies.forced_kind = None;
        base
    }
}

/// Chooses one committed Red Room without depending on snapshot insertion
/// order.  Under the current state machine, only red thresholds can produce a
/// non-`Outside` phase.  Progress is compared explicitly because `epoch` is
/// the best available recency signal; stable identity is the final tie-break.
pub(crate) fn active_red_room(reality: &RealitySnapshot) -> Option<&AnomalyStateStamp> {
    reality
        .stamps()
        .iter()
        .filter(|state| state.phase != RedRoomPhase::Outside)
        .max_by_key(|state| {
            (
                state.epoch,
                state.phase,
                state.loop_count,
                state.last_gate_id,
                state.instance_id,
            )
        })
}

fn translation_component(root: u64, axis_domain: u64) -> i64 {
    let hash = mix64(root ^ axis_domain);
    let magnitude = MIN_TRANSLATION_REGIONS + (hash % TRANSLATION_REGION_SPAN) as i64;
    if hash & (1 << 63) == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn split_world_position(position: Position) -> ((i64, i64), (f32, f32)) {
    let region = (
        (position.x / REGION_SIZE).floor() as i64,
        (position.z / REGION_SIZE).floor() as i64,
    );
    let local = (
        position.x - region.0 as f32 * REGION_SIZE,
        position.z - region.1 as f32 * REGION_SIZE,
    );
    (region, local)
}

fn join_world_position(region: (i64, i64), local: (f32, f32)) -> Position {
    Position::new(
        region.0 as f32 * REGION_SIZE + local.0,
        region.1 as f32 * REGION_SIZE + local.1,
    )
}

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
    use crate::domain::entities::anomaly::AnomalyKind;
    use crate::domain::entities::anomaly::{Axis2, AxisDirection};

    fn state(instance_id: AnomalyId, epoch: u32, phase: RedRoomPhase) -> AnomalyStateStamp {
        AnomalyStateStamp::new(
            instance_id,
            epoch,
            12.4,
            Axis2::Z,
            AxisDirection::Positive,
            phase,
            epoch as u8,
            instance_id.rotate_left(7),
        )
    }

    #[test]
    fn full_instance_identity_changes_the_recursive_address() {
        let low = RecursiveLevelAddress::derive(42, 0x0000_0001_1234_5678);
        let high = RecursiveLevelAddress::derive(42, 0x8000_0001_1234_5678);
        assert_ne!(low, high);
        assert_ne!(low.derived_seed(), high.derived_seed());
    }

    #[test]
    fn translation_is_region_aligned_and_round_trips_region_keys() {
        let address = RecursiveLevelAddress::derive(42, 0xDEAD_BEEF_1234_5678);
        let source = (-91, 207);
        let recursive = address.to_recursive_region(source);
        assert_eq!(address.to_source_region(recursive), source);

        let offset = address.translation_world();
        assert_eq!(
            offset.x / REGION_SIZE,
            address.translation_regions().0 as f32
        );
        assert_eq!(
            offset.z / REGION_SIZE,
            address.translation_regions().1 as f32
        );

        let world = Position::new(12.345, -98.765);
        let round_trip = address.to_source_world(address.to_recursive_world(world));
        assert!((round_trip.x - world.x).abs() < 0.005);
        assert!((round_trip.z - world.z).abs() < 0.005);
    }

    #[test]
    fn active_selection_is_insertion_order_independent() {
        let outside = state(3, 99, RedRoomPhase::Outside);
        let older = state(8, 2, RedRoomPhase::Sealed);
        let active = state(5, 4, RedRoomPhase::EscapeOpen);
        let a = RealitySnapshot::new(vec![outside, older, active]);
        let b = RealitySnapshot::new(vec![active, outside, older]);

        assert_eq!(active_red_room(&a), Some(&active));
        assert_eq!(active_red_room(&b), Some(&active));
        assert_eq!(
            RecursiveLevelAddress::from_reality(42, &a).map(|pair| pair.0),
            RecursiveLevelAddress::from_reality(42, &b).map(|pair| pair.0),
        );
    }

    #[test]
    fn recursive_config_keeps_organic_tuning_but_clears_debug_forcing() {
        let address = RecursiveLevelAddress::derive(42, 7);
        let mut config = GeneratorConfig::low_spec().with_level(34);
        config.anomalies.frequency = 2.75;
        config.anomalies.forced_kind = Some(AnomalyKind::RedRoom);

        let recursive = address.recursive_config(config);
        assert_eq!(recursive.level, LEVEL_BACKROOMS);
        assert_eq!(recursive.anomalies.frequency, 2.75);
        assert_eq!(recursive.anomalies.forced_kind, None);
    }
}
