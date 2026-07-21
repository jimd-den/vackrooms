//! HUD/diagnostic presentation over engine state: everything the debug
//! overlay and the diagnostic aperture read, and the `HudStats` snapshot the
//! ordinary HUD renders from. Pure formatting/aggregation — no simulation
//! logic lives here, only reading and describing what `Engine::tick` already
//! decided. A child module of `engine` (not a sibling): its private fields
//! stay private to outside callers, but Rust visibility cascades to
//! descendant modules, so these methods reach `self.player`, `self.store`,
//! etc. exactly as if they were still defined in `engine.rs`.

use std::fmt::Write;

use crate::application::streaming::chunk_key;

use super::{AWARENESS_RANGE_M, Engine, HudStats};
use crate::application::navigation::{relative_bearing_deg, select_anomaly_focus};

impl Engine {
    /// Multi-line plain-text report of the live anomaly machinery, for the
    /// debug overlay: reality-snapshot epochs, resident traversal gates and
    /// pit hazards (nearest first), and the streaming pipeline counters.
    /// Pure formatting over engine state — no browser types, natively
    /// testable.
    pub fn anomaly_debug_text(&self) -> String {
        let p = self.player.position;
        let mut out = String::with_capacity(1024);
        let _ = writeln!(
            out,
            "pos ({:.1}, {:.1}, {:.1})  level {}  yaw {:.2}",
            p[0], p[1], p[2], self.level, self.player.yaw
        );
        let _ = writeln!(
            out,
            "chunks {} resident ({} fine) | pending {} | backlog {} | forced reloads {}",
            self.store.len(),
            self.store.iter_ordered().filter(|c| c.lod == 0).count(),
            self.pending.len(),
            self.completed_backlog.len(),
            self.forced_reloads.len(),
        );
        let _ = writeln!(
            out,
            "flares {} | push {:.2}s",
            self.flares.len(),
            self.push_seconds
        );

        let stamps = self.reality.stamps();
        let _ = writeln!(out, "reality: {} stamp(s)", stamps.len());
        let _ = writeln!(
            out,
            "peripheral shift: {} drifted cell(s), {} watched",
            self.reality.fabric_drifts().len(),
            self.drift_near.len()
        );
        let (inside, eligible, total) = self.blackout_shift_debug;
        let _ = writeln!(
            out,
            "blackout shift: inside={inside} eligible={eligible} \
             shifted_total={total} cooldown={:.1}s",
            self.blackout_shift_cooldown
        );
        for stamp in stamps {
            let _ = writeln!(
                out,
                "  {:016x} epoch {}  plane {:.1} {:?} {:?}",
                stamp.instance_id,
                stamp.epoch,
                stamp.gate_plane(),
                stamp.gate_axis,
                stamp.travel_direction,
            );
        }

        let mut gates: Vec<_> = self.store.all_traversal_gates().copied().collect();
        gates.sort_by_key(|g| g.id);
        gates.dedup_by_key(|g| g.id);
        gates.sort_by(|a, b| {
            let d = |g: &vackrooms::domain::entities::anomaly::TraversalGate| match g.axis {
                vackrooms::domain::entities::anomaly::Axis2::X => (g.plane - p[0]).abs(),
                vackrooms::domain::entities::anomaly::Axis2::Z => (g.plane - p[2]).abs(),
            };
            d(a).total_cmp(&d(b))
        });
        let _ = writeln!(out, "gates resident: {}", gates.len());
        for gate in gates.iter().take(6) {
            let dist = match gate.axis {
                vackrooms::domain::entities::anomaly::Axis2::X => (gate.plane - p[0]).abs(),
                vackrooms::domain::entities::anomaly::Axis2::Z => (gate.plane - p[2]).abs(),
            };
            let epoch = self.reality.lookup(gate.instance_id).map_or(0, |s| s.epoch);
            let _ = writeln!(
                out,
                "  {:?}/{:?} of {:?} {:08x}  plane {:.1} on {:?}  d={:.1}  epoch {}",
                gate.kind,
                gate.forward,
                gate.anomaly_kind,
                (gate.instance_id & 0xFFFF_FFFF) as u32,
                gate.plane,
                gate.axis,
                dist,
                epoch,
            );
        }

        let mut pits: Vec<_> = self.store.all_pit_hazards().copied().collect();
        pits.sort_by_key(|h| h.id);
        pits.dedup_by_key(|h| h.id);
        pits.sort_by(|a, b| {
            let d = |h: &vackrooms::domain::entities::anomaly::PitHazard| {
                let dx = h.center.x - p[0];
                let dz = h.center.z - p[2];
                dx * dx + dz * dz
            };
            d(a).total_cmp(&d(b))
        });
        let _ = writeln!(out, "pit hazards resident: {}", pits.len());
        for pit in pits.iter().take(4) {
            let dx = pit.center.x - p[0];
            let dz = pit.center.z - p[2];
            let _ = writeln!(
                out,
                "  {:08x} at ({:.1}, {:.1})  half {:.1}  d={:.1}  recovery ({:.1}, {:.1})",
                (pit.id & 0xFFFF_FFFF) as u32,
                pit.center.x,
                pit.center.z,
                pit.half_side,
                (dx * dx + dz * dz).sqrt(),
                pit.recovery.x,
                pit.recovery.z,
            );
        }
        out
    }

    pub fn stats(&self) -> HudStats {
        let spawn_key = chunk_key(
            (self.config.spawn[0] / self.config.chunk_size).floor() * self.config.chunk_size,
            (self.config.spawn[2] / self.config.chunk_size).floor() * self.config.chunk_size,
        );
        let survival = self.survival.readout();
        HudStats {
            resident_chunks: self.store.len(),
            fine_chunks: self.store.iter_ordered().filter(|c| c.lod == 0).count(),
            atlas_nodes: self.atlas_nodes,
            collision_boxes: self.world.len(),
            resolution_scale: self.governor.scale(),
            ready: self.store.contains(spawn_key),
            distance_m: self.distance_m as f32,
            hydration: survival.vitals.hydration,
            satiety: survival.vitals.satiety,
            condition: survival.vitals.condition,
            almond_bottles: survival.almond_bottles,
            rations: survival.rations,
            ambient_c: survival.ambient_c,
            deaths: survival.deaths,
            level: self.level,
            body: self.body.readout(),
            route: self.route_anchor(),
        }
    }

    /// The diagnostic aperture's full report: information-dense, truthful
    /// about unavailable data, and organized into `§`-headed sections the
    /// presenter renders as panels. Pure formatting over engine state —
    /// natively testable, and it deliberately reveals everything the normal
    /// HUD hides (raw vitals, multipliers, stress contributions, targets).
    pub fn diagnostic_text(&self) -> String {
        let mut out = String::with_capacity(2048);
        let readout = self.body.readout();
        let debug = self.body.debug();
        let survival = self.survival.readout();

        let _ = writeln!(out, "§ SUBJECT");
        let _ = writeln!(
            out,
            "steps {}  walked {:.1} m  (raw session distance {:.1} m)",
            readout.steps,
            self.body.walked_m(),
            self.distance_m,
        );
        let _ = writeln!(
            out,
            "pulse {:.0} bpm -> target {:.0}  signal {:?}  beat x{:.2}",
            readout.bpm, readout.target_bpm, readout.signal, readout.beat_intensity,
        );
        let _ = writeln!(
            out,
            "exertion {:.2}  fatigue {:.2}  movement x{:.2}",
            readout.exertion, readout.fatigue, readout.movement_factor,
        );
        let _ = writeln!(
            out,
            "factors: capacity x{:.2}  hydr x{:.2}  sati x{:.2}  cond x{:.2}  fatig x{:.2}",
            debug.exertion_capacity,
            debug.hydration_factor,
            debug.satiety_factor,
            debug.condition_factor,
            debug.fatigue_factor,
        );
        let _ = writeln!(
            out,
            "recovery quality {:.2}  heat stress {:.2}  ambient {:.1} C",
            debug.recovery_quality, debug.heat_stress, survival.ambient_c,
        );
        let _ = writeln!(
            out,
            "hydration {:.2}  satiety {:.2}  condition {:.2}  excess {:.2}",
            survival.vitals.hydration,
            survival.vitals.satiety,
            survival.vitals.condition,
            survival.excess,
        );
        let _ = writeln!(
            out,
            "carried: water {}  rations {}  | assisted consumption {}  deaths {}",
            survival.almond_bottles,
            survival.rations,
            if survival.assisted_consumption {
                "on"
            } else {
                "off"
            },
            survival.deaths,
        );

        let _ = writeln!(out, "§ ROUTE");
        match (self.route_target, self.route_anchor()) {
            (Some(target), Some(anchor)) => {
                let _ = writeln!(
                    out,
                    "door {:08x} -> level {}  at ({:.1}, {:.1})",
                    (target.exit_id & 0xFFFF_FFFF) as u32,
                    target.target_level,
                    target.position[0],
                    target.position[1],
                );
                let _ = writeln!(
                    out,
                    "range {:.1} m  bearing {:03}  [straight-line; no route graph exists]",
                    anchor.range_m, anchor.bearing_deg,
                );
            }
            _ => {
                let _ = writeln!(out, "unresolved: no door inside the streamed field");
            }
        }

        let _ = writeln!(out, "§ AWARENESS");
        let player = [self.player.position[0], self.player.position[2]];
        let focus = select_anomaly_focus(
            self.store.all_traversal_gates(),
            self.store.all_pit_hazards(),
            player,
            AWARENESS_RANGE_M,
        );
        match focus {
            Some(focus) => {
                let _ = writeln!(
                    out,
                    "{:?} {:08x}  range {:.1} m  bearing {:03}  priority {}",
                    focus.kind,
                    (focus.instance_id & 0xFFFF_FFFF) as u32,
                    focus.range_m,
                    relative_bearing_deg(self.player.yaw, player, focus.position),
                    focus.priority,
                );
                let _ = writeln!(
                    out,
                    "(debug-only: no player-facing sensing mechanic exists yet)"
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "no anomaly signature resident within {AWARENESS_RANGE_M:.0} m"
                );
            }
        }

        let _ = writeln!(out, "§ FIELD");
        out.push_str(&self.anomaly_debug_text());
        out
    }
}
