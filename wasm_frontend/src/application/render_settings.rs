//! Renderer optimization toggles — one switch per labeled optimization.
//!
//! Every renderer in the engine (GPU surfaces, GPU face splats, the SVO
//! raymarcher, and the CPU splatter) is written as a *correct, simple
//! reference path* plus a set of clearly labeled optimizations layered on
//! top. Each optimization can be disabled at runtime, which serves three
//! purposes:
//!
//! 1. **Debugging** — when a frame looks wrong, binary-search the toggles to
//!    find which optimization broke it.
//! 2. **Benchmarking** — measure exactly what each optimization buys on a
//!    given machine (the HUD shows frame times).
//! 3. **Documentation** — the toggle list *is* the catalog of every shortcut
//!    the pipeline takes.
//!
//! The struct is plain data with no browser dependencies, so it is natively
//! unit-tested. It crosses the wasm boundary as a bitfield (one `AtomicU32`
//! in `lib.rs`). Each browser driver snapshots it once at the frame boundary;
//! platform-free render code receives the snapshot by value.
//!
//! URL controls: `?rt_hiz=0&rt_shadows=0` … (`rt_<name>=0|1`). The settings
//! menu calls the `set_render_toggle(name, enabled)` wasm export with the
//! same names.

/// One on/off switch per optimization, across all render paths.
///
/// Correctness-preserving switches default on. The approximate diffuse-fill
/// bake defaults off: first boot uses the strict analytic-light image, while
/// `rt_bake=1` opts into the quantized field explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderToggles {
    // ------------------------------------------------------------------
    // CPU splatter (SoftwareRasterizer)
    // ------------------------------------------------------------------
    /// OPTIMIZATION (CPU): 8x8-tile coarse hierarchical z-buffer. Rejects a
    /// whole octree subtree with a handful of tile reads when something
    /// nearer already covers its screen bounds. Off = every node is tested
    /// per-pixel by the fine z-buffer only.
    pub hierarchical_z: bool,
    /// OPTIMIZATION (CPU + splat + raymarch): near-to-far traversal,
    /// submission, or resident-chunk selection. Maximizes early exits and
    /// z-rejection. Off retains each strategy's deterministic reference order.
    pub front_to_back: bool,
    /// OPTIMIZATION (raymarch): jump across the complete empty SVO leaf
    /// containing the current sample. Off = diagnostic finest-voxel DDA;
    /// both paths use the same exact point lookup and must report the same
    /// nearest hit/material/normal.
    pub empty_space_skip: bool,
    /// OPTIMIZATION (CPU): coarse-to-fine LOD — a subtree whose projection
    /// fits in ~a pixel is drawn as one MIP-filtered splat instead of being
    /// descended. Off = full descent to leaves regardless of distance, except
    /// for the renderer's non-optional malformed/workload safety cap.
    pub mip_lod: bool,
    /// OPTIMIZATION (CPU): the flashlight's occlusion ray (one SVO trace per
    /// shaded splat while the flashlight is on). This is a *quality*
    /// optimization: off means the beam shines through walls but shading
    /// gets cheaper.
    pub flashlight_occlusion: bool,
    /// OPTIMIZATION (CPU): query the fine depth buffer before evaluating
    /// fixture, hero, and flashlight radiance. Off = shade every projected
    /// splat and let the ordinary write-time depth test reject it. The image
    /// is identical; only hidden-splat shading work changes.
    pub deferred_shading: bool,
    /// APPROXIMATION (CPU): sibling-count ambient occlusion. Off is the
    /// material/lighting reference with unit ambient visibility.
    pub ambient_occlusion: bool,
    // ------------------------------------------------------------------
    // GPU paths (surface / splat / raymarch)
    // ------------------------------------------------------------------
    /// Direct-light visibility. Raster strategies use a hero-fixture depth
    /// map; raymarch traces finite SVO segments to analytic emitter samples.
    /// Off keeps otherwise-identical unoccluded direct lighting.
    pub shadow_pass: bool,
    /// OPTIMIZATION (splat): per-cell instance-range culling. Face instances
    /// are grouped into spatial cells; invisible cells are skipped and
    /// visible neighbors coalesce into one instanced draw. Off = every
    /// resident face is drawn every frame.
    pub cell_culling: bool,
    /// OPTIMIZATION (splat): stop submitting whole far chunks after the
    /// profile's face budget is exhausted. Off = draw every visible face;
    /// useful as the correctness/reference path when diagnosing pop-out.
    pub face_budget: bool,
    /// OPTIMIZATION (CPU + GPU): range rejection for every strategy, plus
    /// conservative camera visibility for raster paths. Raymarch filters
    /// complete chunk bounds before its resident-set cap. Off keeps all
    /// resident chunks eligible.
    pub distance_cull: bool,
    /// Dither/grain/banding-noise post effects (GPU). A visual feature more
    /// than a speedup; toggle to isolate its contribution to the image.
    pub dither: bool,
    /// Optional quantized diffuse fill in the CPU, indexed-surface, and splat
    /// strategies. Analytic fixtures remain authoritative; raymarch stays on
    /// analytic lighting because its stored bake is face-independent.
    pub baked_lighting: bool,
}

impl Default for RenderToggles {
    fn default() -> Self {
        Self {
            hierarchical_z: true,
            front_to_back: true,
            empty_space_skip: true,
            mip_lod: true,
            flashlight_occlusion: true,
            deferred_shading: true,
            ambient_occlusion: true,
            shadow_pass: true,
            cell_culling: true,
            face_budget: true,
            distance_cull: true,
            dither: true,
            baked_lighting: false,
        }
    }
}

impl RenderToggles {
    /// Baseline used when a renderer profile admits only a named subset of
    /// the global switchboard.
    pub const fn disabled() -> Self {
        Self {
            hierarchical_z: false,
            front_to_back: false,
            empty_space_skip: false,
            mip_lod: false,
            flashlight_occlusion: false,
            deferred_shading: false,
            ambient_occlusion: false,
            shadow_pass: false,
            cell_culling: false,
            face_budget: false,
            distance_cull: false,
            dither: false,
            baked_lighting: false,
        }
    }
}

/// Canonical query-name/bit ordering used by parsing and bitfield round trips.
const TOGGLE_BITS: [(&str, u32); 13] = [
    ("hiz", 1 << 0),
    ("f2b", 1 << 1),
    ("mips", 1 << 2),
    ("beam_occlusion", 1 << 3),
    ("shadows", 1 << 4),
    ("cells", 1 << 5),
    ("budget", 1 << 6),
    ("cull", 1 << 7),
    ("dither", 1 << 8),
    ("skip", 1 << 9),
    ("bake", 1 << 10),
    ("deferred", 1 << 11),
    ("ao", 1 << 12),
];

impl RenderToggles {
    fn field_mut(&mut self, name: &str) -> Option<&mut bool> {
        Some(match name {
            "hiz" => &mut self.hierarchical_z,
            "f2b" => &mut self.front_to_back,
            "skip" => &mut self.empty_space_skip,
            "mips" => &mut self.mip_lod,
            "beam_occlusion" => &mut self.flashlight_occlusion,
            "deferred" => &mut self.deferred_shading,
            "ao" => &mut self.ambient_occlusion,
            "shadows" => &mut self.shadow_pass,
            "cells" => &mut self.cell_culling,
            "budget" => &mut self.face_budget,
            "cull" => &mut self.distance_cull,
            "dither" => &mut self.dither,
            "bake" => &mut self.baked_lighting,
            _ => return None,
        })
    }

    fn field(&self, name: &str) -> Option<bool> {
        let mut copy = *self;
        copy.field_mut(name).map(|b| *b)
    }

    /// Sets one toggle by its short name. Unknown names are ignored (the
    /// settings menu and old URLs must never crash the engine).
    pub fn set(&mut self, name: &str, enabled: bool) {
        if let Some(field) = self.field_mut(name) {
            *field = enabled;
        }
    }

    /// Reads one toggle by its short name. Returning `None` for unknown
    /// names keeps diagnostics and browser tests on the same catalog as the
    /// parser and setter.
    pub fn enabled(&self, name: &str) -> Option<bool> {
        self.field(name)
    }

    /// Packs the toggles into a bitfield for the atomic global in `lib.rs`.
    pub fn to_bits(self) -> u32 {
        let mut bits = 0;
        for (name, bit) in TOGGLE_BITS {
            if self.field(name).expect("table names match fields") {
                bits |= bit;
            }
        }
        bits
    }

    /// Inverse of [`Self::to_bits`].
    pub fn from_bits(bits: u32) -> Self {
        let mut toggles = Self::default();
        for (name, bit) in TOGGLE_BITS {
            toggles.set(name, bits & bit != 0);
        }
        toggles
    }
}

/// Applies `rt_<name>=0|1` pairs from a URL query (with or without the
/// leading `?`) on top of the defaults. Unknown keys and malformed values
/// are ignored, matching the generation-knob parser's tolerance.
pub fn parse_render_toggles(query: &str) -> RenderToggles {
    let mut toggles = RenderToggles::default();
    for pair in query.trim_start_matches('?').split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        let Some(name) = key.strip_prefix("rt_") else {
            continue;
        };
        match value {
            "0" | "false" | "off" => toggles.set(name, false),
            "1" | "true" | "on" => toggles.set(name, true),
            _ => {}
        }
    }
    toggles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_use_analytic_static_lighting() {
        let t = RenderToggles::default();
        assert!(t.hierarchical_z && t.front_to_back && t.empty_space_skip && t.mip_lod);
        assert!(t.flashlight_occlusion && t.deferred_shading && t.ambient_occlusion);
        assert!(t.shadow_pass && t.cell_culling && t.face_budget);
        assert!(t.distance_cull && t.dither);
        assert!(!t.baked_lighting);
        assert_eq!(RenderToggles::disabled().to_bits(), 0);
    }

    #[test]
    fn bits_round_trip_every_combination_of_low_bits() {
        for bits in 0..(1u32 << TOGGLE_BITS.len()) {
            assert_eq!(RenderToggles::from_bits(bits).to_bits(), bits);
        }
    }

    #[test]
    fn query_disables_named_toggles_only() {
        let t = parse_render_toggles("?seed=7&rt_hiz=0&rt_shadows=off&rt_bogus=0");
        assert!(!t.hierarchical_z);
        assert!(!t.shadow_pass);
        assert!(t.front_to_back, "unrelated toggles stay on");
    }

    #[test]
    fn set_ignores_unknown_names() {
        let mut t = RenderToggles::default();
        t.set("not_a_toggle", false);
        assert_eq!(t, RenderToggles::default());
    }

    #[test]
    fn enabled_uses_the_same_name_catalog_as_set() {
        let mut t = RenderToggles::default();
        t.set("budget", false);
        assert_eq!(t.enabled("budget"), Some(false));
        assert_eq!(t.enabled("not_a_toggle"), None);
    }
}
