//! The navigation-objective mechanic: which resident level exit the HUD's
//! route needle points at, reselected on a bounded cadence and invalidated
//! on level/reality transitions. Thin — `navigation.rs` owns the actual
//! selection/bearing math; this module is only the engine-state glue around
//! it. A child module of `engine`; see `presenter.rs` for why that gives it
//! access to `Engine`'s private fields.

use crate::application::navigation::{range_m, relative_bearing_deg, select_route_target};

use super::{Engine, ROUTE_RECOMPUTE_PERIOD_S, RouteAnchor};

impl Engine {
    /// Drops the cached navigation objective; the next tick reselects from
    /// whatever the *current* level/reality has resident. Called on level
    /// switches, reality transitions, and death so telemetry never crosses
    /// a reality address.
    pub(super) fn invalidate_route(&mut self) {
        self.route_target = None;
        self.route_cooldown = 0.0;
    }

    /// Reselects the route objective on a bounded cadence from the resident
    /// level exits — authoritative generator output, never invented. No
    /// resident door means no route, truthfully.
    pub(super) fn update_route(&mut self, dt: f32) {
        self.route_cooldown -= dt;
        if self.route_cooldown > 0.0 {
            return;
        }
        self.route_cooldown = ROUTE_RECOMPUTE_PERIOD_S;
        let player = [self.player.position[0], self.player.position[2]];
        let selected = select_route_target(self.store.all_level_exits(), player);
        self.route_target = selected;
    }

    /// The presenter-facing route anchor: bearing/range are derived fresh
    /// from the cached target and the live player pose (cheap trig), so the
    /// needle tracks head movement between reselections.
    pub(super) fn route_anchor(&self) -> Option<RouteAnchor> {
        let target = self.route_target?;
        let player = [self.player.position[0], self.player.position[2]];
        Some(RouteAnchor {
            target_level: target.target_level,
            bearing_deg: relative_bearing_deg(self.player.yaw, player, target.position),
            range_m: range_m(player, target.position),
        })
    }
}
