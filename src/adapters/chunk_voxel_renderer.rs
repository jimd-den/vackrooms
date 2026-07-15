/// Per-chunk voxel blueprint renderer.
///
/// # Why a separate module?
///
/// The existing `render_large_voxel_blueprint_svg` stitches N×N chunks into
/// one giant SVG and is designed for batch export.  This module renders a
/// *single* chunk — typically a 50×50 voxel grid at default spec — so the
/// WASM call completes in < 5 ms and the browser can display the result
/// instantly without a reload delay.
///
/// # Architecture (Clean Architecture – Interface Adapter layer)
///
/// This module occupies the Interface Adapter layer: it takes domain entities
/// (`VoxelGrid`, `RegionPlan`, `Position`) and converts them into a string
/// artifact (SVG) for the browser presentation layer.  It contains no
/// business logic; all meaning lives in the entities it receives.

use crate::domain::entities::architecture::{
    RegionPlan, SpaceProgram,
};
use crate::domain::entities::voxel_grid::{
    VoxelGrid,
    VOXEL_WALL, VOXEL_RED_WALL, VOXEL_FLOOR, VOXEL_CEILING,
    VOXEL_LIGHT, VOXEL_RED_LIGHT, VOXEL_GRASS, VOXEL_WATER, VOXEL_TREE,
};
use crate::entities::models::Position;

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

/// Display toggles for the chunk voxel blueprint.
///
/// Every field is a plain boolean so the JavaScript caller can wire each to a
/// checkbox.  The `pixels_per_voxel` float controls visual density.
#[derive(Debug, Clone, Copy)]
pub struct VoxelBlueprintOptions {
    /// SVG pixels per voxel cell.  8–12 works well for a 50×50 grid.
    pub pixels_per_voxel: f32,
    /// Draw hairline grid lines between every voxel cell.
    pub show_grid: bool,
    /// Fill each voxel cell with its material colour.
    pub show_voxels: bool,
    /// Draw coloured semantic bounding boxes (corridors, assemblies, doors…).
    pub show_semantics: bool,
    /// Print text labels on each semantic box.
    pub show_labels: bool,
    /// Include ceiling / light voxels in the fill pass.
    pub show_ceiling: bool,
}

impl Default for VoxelBlueprintOptions {
    fn default() -> Self {
        Self {
            pixels_per_voxel: 10.0,
            show_grid: true,
            show_voxels: true,
            show_semantics: true,
            show_labels: true,
            show_ceiling: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions (all deterministic, no side-effects)
// ---------------------------------------------------------------------------

/// Convert a single world-space axis value into the chunk-local voxel float
/// coordinate.  The VoxelGrid interior starts at index 1 (there is a 1-cell
/// padding border), so we add +1.0 to align semantic world objects with the
/// rendered cells.
#[inline]
fn world_to_voxel(world: f32, chunk_origin: f32, voxel_scale: f32) -> f32 {
    (world - chunk_origin) / voxel_scale + 1.0
}

/// Clip an axis-aligned rectangle given in voxel coordinates to the grid's
/// interior region [1, grid_dim-1].  Returns `None` if the result is empty.
///
/// Why +1 / -1 bounds? The VoxelGrid has a 1-cell padding on every face; the
/// "real" chunk interior occupies [1, width-1] × [1, depth-1].
#[inline]
fn clip_rect(
    x0: f32, z0: f32, x1: f32, z1: f32,
    grid_w: f32, grid_d: f32,
) -> Option<(f32, f32, f32, f32)> {
    let cx0 = x0.max(1.0);
    let cz0 = z0.max(1.0);
    let cx1 = x1.min(grid_w - 1.0);
    let cz1 = z1.min(grid_d - 1.0);
    if cx1 > cx0 && cz1 > cz0 { Some((cx0, cz0, cx1, cz1)) } else { None }
}

/// Return the canonical debug label for a `SpaceProgram` variant.
const fn program_label(p: SpaceProgram) -> &'static str {
    match p {
        SpaceProgram::Arrival            => "ARRIVAL",
        SpaceProgram::Reception          => "RECEPTION",
        SpaceProgram::MainCorridor       => "MAIN CORRIDOR",
        SpaceProgram::SecondaryHall      => "SECONDARY HALL",
        SpaceProgram::OpenOffice         => "OPEN OFFICE",
        SpaceProgram::PrivateOffice      => "PRIVATE OFFICE",
        SpaceProgram::ConferenceRoom     => "CONFERENCE RM",
        SpaceProgram::WaitingArea        => "WAITING AREA",
        SpaceProgram::BreakRoom          => "BREAK ROOM",
        SpaceProgram::Storage            => "STORAGE",
        SpaceProgram::ServerRoom         => "SERVER ROOM",
        SpaceProgram::RestroomCore       => "RESTROOM",
        SpaceProgram::Stair              => "STAIR",
        SpaceProgram::Mechanical         => "MECHANICAL",
        SpaceProgram::Atrium             => "ATRIUM",
        SpaceProgram::AbandonedExpansion => "ABANDONED",
    }
}

/// Priority value for material layering: higher priority wins when a voxel
/// column contains multiple material types.
#[inline]
const fn material_priority(voxel: u8) -> u8 {
    match voxel {
        VOXEL_WALL | VOXEL_RED_WALL | VOXEL_TREE => 4,
        VOXEL_LIGHT | VOXEL_RED_LIGHT             => 3,
        VOXEL_CEILING                             => 2,
        VOXEL_FLOOR | VOXEL_GRASS | VOXEL_WATER   => 1,
        _                                         => 0,
    }
}

/// Map a winning material byte to its CSS class name.
#[inline]
const fn material_class(voxel: u8) -> &'static str {
    match voxel {
        VOXEL_WALL      => "vw",
        VOXEL_RED_WALL  => "vrw",
        VOXEL_TREE      => "vwt",
        VOXEL_LIGHT     => "vli",
        VOXEL_RED_LIGHT => "vrli",
        VOXEL_CEILING   => "vce",
        VOXEL_FLOOR     => "vfl",
        VOXEL_GRASS     => "vgr",
        VOXEL_WATER     => "vwa",
        _               => "vair",
    }
}

// ---------------------------------------------------------------------------
// Main public function
// ---------------------------------------------------------------------------

/// Render a single chunk as a voxel-level blueprint SVG.
///
/// # Render order (matches the spec)
/// 1. Dark background (set on the `<svg>` element itself).
/// 2. Per-voxel cell fills (column-collapsed material colour).
/// 3. Hairline voxel grid lines.
/// 4. Semantic bounding-box fills at ~13% opacity.
/// 5. Semantic outlines with contrasting stroke colour.
/// 6. Door / portal markers (green boxes).
/// 7. Text labels on each semantic box.
/// 8. Header bar and legend.
///
/// # Arguments
/// * `grid`         – The generated `VoxelGrid` for this chunk.
/// * `plan`         – The `RegionPlan` that covers this chunk's world region.
/// * `chunk_origin` – World-space (x, z) of the chunk's min corner.
/// * `voxel_scale`  – World units per voxel (0.2 for low-spec, 0.1 for high-spec).
/// * `options`      – Display toggles.
pub fn render_chunk_voxel_blueprint_svg(
    grid: &VoxelGrid,
    plan: &RegionPlan,
    chunk_origin: Position,
    voxel_scale: f32,
    options: VoxelBlueprintOptions,
) -> String {
    let ppv = options.pixels_per_voxel;

    let gw = grid.width();   // includes 1-cell padding on each side
    let gd = grid.depth();
    let gh = grid.height();

    let canvas_w = gw as f32 * ppv;
    let canvas_h = gd as f32 * ppv;

    // Derive chunk grid coordinates for the header label.
    // Interior cell count = width - 2 (strip the padding border).
    let interior = (gw as i32 - 2).max(1) as f32;
    let chunk_size_world = interior * voxel_scale;
    let cx_idx = if chunk_size_world > 0.0 {
        (chunk_origin.x / chunk_size_world).floor() as i32
    } else { 0 };
    let cz_idx = if chunk_size_world > 0.0 {
        (chunk_origin.z / chunk_size_world).floor() as i32
    } else { 0 };

    // ── Coordinate transform closures ─────────────────────────────────
    //
    // SVG Y axis is flipped: Y=0 is the top of the canvas, Y=canvas_h is the
    // bottom.  We want world-Z increasing upward, so:
    //   svg_y  = canvas_h - (vz + 1) * ppv   for integer voxel index vz
    //   svg_y  = canvas_h - vz_float * ppv    for float voxel coordinate
    //
    // The +1 in the integer version accounts for the padding border offset
    // already baked into the float version via world_to_voxel().

    let vox_px_x = |vx: usize| -> f32 { vx as f32 * ppv };
    let vox_px_y = |vz: usize| -> f32 { canvas_h - (vz + 1) as f32 * ppv };

    // World coord → float voxel index (accounts for padding offset).
    let w2vx = |wx: f32| -> f32 { world_to_voxel(wx, chunk_origin.x, voxel_scale) };
    let w2vz = |wz: f32| -> f32 { world_to_voxel(wz, chunk_origin.z, voxel_scale) };

    // Float voxel → SVG pixel.
    let vpx = |vx_f: f32| -> f32 { vx_f * ppv };
    let vpy_top = |vz_f: f32| -> f32 { canvas_h - vz_f * ppv };   // top edge of a box whose Z min = vz_f

    let label_fs  = (ppv * 1.1).max(7.0_f32).min(13.0_f32);
    let label_sm  = (ppv * 0.85).max(5.5_f32).min(10.0_f32);

    // ── Build SVG string ───────────────────────────────────────────────
    let mut svg = String::with_capacity(256 * 1024);

    // Header.
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg"
  viewBox="0 0 {cw} {ch}" width="{cw}" height="{ch}"
  style="display:block;background:#0b0f19;font-family:monospace">
<defs>
  <style>
    .vw   {{ fill:#1c2333 }}
    .vrw  {{ fill:#3b0b0b }}
    .vwt  {{ fill:#251510 }}
    .vli  {{ fill:#ff2222 }}
    .vrli {{ fill:#ff6666 }}
    .vce  {{ fill:#1a1f2e }}
    .vfl  {{ fill:#1a120880 }}
    .vgr  {{ fill:#0d1f0d80 }}
    .vwa  {{ fill:#0a102080 }}
    .vair {{ fill:#0b1422 }}
    .vgrid{{ stroke:#1e2d40; stroke-width:0.4; fill:none }}
    .lbl  {{ font-size:{lfs}px; dominant-baseline:hanging; pointer-events:none }}
    .lsm  {{ font-size:{lsm}px; dominant-baseline:hanging; pointer-events:none }}
  </style>
</defs>
"#,
        cw = canvas_w, ch = canvas_h,
        lfs = label_fs, lsm = label_sm,
    ));

    // ── Pass 1: voxel fills ────────────────────────────────────────────
    if options.show_voxels {
        svg.push_str("<g id=\"voxels\">\n");
        for vz in 0..gd {
            for vx in 0..gw {
                // Column collapse: scan all Y layers, keep highest-priority material.
                let mut winning: u8 = 0;
                let mut winning_pri: u8 = 0;
                for vy in 0..gh {
                    let v = grid.get(vx, vy, vz);
                    // Skip ceiling/fixture voxels when show_ceiling is off.
                    // Glimmers are ceiling fixtures too: leaving them in made
                    // blackout debug captures look like floor-level lights.
                    if !options.show_ceiling
                        && (v == VOXEL_CEILING
                            || v == VOXEL_LIGHT
                            || v == VOXEL_RED_LIGHT
                            || v == crate::domain::entities::voxel_grid::VOXEL_GLIMMER)
                    {
                        continue;
                    }
                    let pri = material_priority(v);
                    if pri > winning_pri {
                        winning_pri = pri;
                        winning = v;
                    }
                }
                let px = vox_px_x(vx);
                let py = vox_px_y(vz);
                let cls = material_class(winning);
                svg.push_str(&format!(
                    "<rect class=\"{cls}\" x=\"{px}\" y=\"{py}\" width=\"{ppv}\" height=\"{ppv}\"/>\n"
                ));
            }
        }
        svg.push_str("</g>\n");
    }

    // ── Pass 2: voxel grid lines ───────────────────────────────────────
    if options.show_grid {
        svg.push_str("<g id=\"grid\" class=\"vgrid\">\n");
        for vx in 0..=gw {
            let px = vx as f32 * ppv;
            svg.push_str(&format!(
                "<line x1=\"{px}\" y1=\"0\" x2=\"{px}\" y2=\"{canvas_h}\"/>\n"
            ));
        }
        for vz in 0..=gd {
            let py = vz as f32 * ppv;
            svg.push_str(&format!(
                "<line x1=\"0\" y1=\"{py}\" x2=\"{canvas_w}\" y2=\"{py}\"/>\n"
            ));
        }
        svg.push_str("</g>\n");
    }

    // ── Pass 3+4+5: semantic overlays ─────────────────────────────────
    if options.show_semantics {
        svg.push_str("<g id=\"semantics\" pointer-events=\"none\">\n");

        // Inner macro: emit one semantic bounding box clipped to the chunk.
        //
        // Why a macro? The borrow checker forbids closures that capture `svg`
        // mutably while also borrowing coordinate-transform closures.  A macro
        // expands inline and avoids the borrow conflict entirely.
        macro_rules! emit_box {
            ($wx0:expr, $wz0:expr, $wx1:expr, $wz1:expr,
             $fill:literal, $stroke:literal, $dash:expr, $label:expr) => {{
                let vx0 = w2vx($wx0);
                let vz0 = w2vz($wz0);
                let vx1 = w2vx($wx1);
                let vz1 = w2vz($wz1);
                let (rx0, rz0, rx1, rz1) = (vx0.min(vx1), vz0.min(vz1),
                                             vx0.max(vx1), vz0.max(vz1));
                if let Some((cx0, cz0, cx1, cz1)) =
                    clip_rect(rx0, rz0, rx1, rz1, gw as f32, gd as f32)
                {
                    // SVG box: top-left corner is the largest Z (because Y is flipped).
                    let sx  = vpx(cx0);
                    let sy  = vpy_top(cz1);  // SVG top = world Z max
                    let bw  = vpx(cx1) - sx;
                    let bh  = vpy_top(cz0) - sy; // world Z min → SVG bottom
                    let da: &str = if $dash { "stroke-dasharray=\"5,3\"" } else { "" };
                    svg.push_str(&format!(
                        "<rect fill=\"{fill}\" fill-opacity=\"0.13\" stroke=\"{stroke}\" stroke-width=\"1.5\" {da} x=\"{sx}\" y=\"{sy}\" width=\"{bw}\" height=\"{bh}\"/>\n",
                        fill = $fill, stroke = $stroke, da = da,
                        sx = sx, sy = sy, bw = bw, bh = bh,
                    ));
                    if options.show_labels {
                        let label_str: &str = $label;
                        if !label_str.is_empty() {
                            svg.push_str(&format!(
                                "<text class=\"lbl\" x=\"{tx}\" y=\"{ty}\" fill=\"{stroke}\">{label}</text>\n",
                                tx = sx + 3.0, ty = sy + 3.0,
                                stroke = $stroke, label = label_str,
                            ));
                        }
                    }
                }
            }};
        }

        // Corridors — draw each path segment's bounding rect ± half-width.
        // We draw per-segment rather than the whole path AABB so that
        // L-shaped corridors clip correctly at the chunk boundary.
        for spine in &plan.corridors {
            if spine.path.len() < 2 { continue; }
            let hw = spine.width * 0.5;
            let (fill, stroke, label_text) = match spine.spine_kind {
                SpaceProgram::MainCorridor =>
                    ("#00ffff", "#00e5ff",
                     format!("{} · {:.1}m", program_label(spine.spine_kind), spine.width)),
                _ =>
                    ("#a855f7", "#c084fc",
                     format!("{} · {:.1}m", program_label(spine.spine_kind), spine.width)),
            };
            for seg in spine.path.windows(2) {
                let (a, b) = (seg[0], seg[1]);
                let x0 = a.x.min(b.x) - hw;
                let z0 = a.z.min(b.z) - hw;
                let x1 = a.x.max(b.x) + hw;
                let z1 = a.z.max(b.z) + hw;
                if fill == "#00ffff" {
                    emit_box!(x0, z0, x1, z1, "#00ffff", "#00e5ff", false, label_text.as_str());
                } else {
                    emit_box!(x0, z0, x1, z1, "#a855f7", "#c084fc", false, label_text.as_str());
                }
            }
        }

        // Assemblies (rooms).
        for asm in &plan.assemblies {
            let (mx, mz, xx, xz) = asm.footprint.bounds();
            let label_text = format!("#{} {}", asm.id, program_label(asm.program));
            if asm.corruption.abandoned {
                emit_box!(mx, mz, xx, xz, "#ff00ff", "#e879f9", false, label_text.as_str());
            } else {
                emit_box!(mx, mz, xx, xz, "#ffffff", "#cbd5e1", false, label_text.as_str());
            }

            // Sub-spaces within the assembly.
            for space in &asm.spaces {
                let (sx0, sz0, sx1, sz1) = space.footprint.bounds();
                emit_box!(sx0, sz0, sx1, sz1, "#94a3b8", "#64748b", false,
                          program_label(space.program));
            }

            // Ceiling zones (dashed gold).
            for cz in &asm.ceiling_zones {
                let (zx0, zz0, zx1, zz1) = cz.area.bounds();
                let clabel = format!("VAULT · {:.1}m", cz.height_units);
                emit_box!(zx0, zz0, zx1, zz1, "#ca8a04", "#fbbf24", true, clabel.as_str());
            }

            // Openings / portals (green).
            for opening in &asm.entrances {
                let hw = opening.width * 0.5;
                let (ox0, oz0, ox1, oz1) = if opening.through_x_wall {
                    (opening.center.x - hw, opening.center.z - voxel_scale,
                     opening.center.x + hw, opening.center.z + voxel_scale)
                } else {
                    (opening.center.x - voxel_scale, opening.center.z - hw,
                     opening.center.x + voxel_scale, opening.center.z + hw)
                };
                let plabel = format!("PORTAL · {:.1}m", opening.width);
                emit_box!(ox0, oz0, ox1, oz1, "#10b981", "#34d399", false, plabel.as_str());
            }
        }

        svg.push_str("</g>\n");
    }

    // ── Chunk boundary dashed border ──────────────────────────────────
    let bi = 0.5_f32;
    svg.push_str(&format!(
        "<rect id=\"chunk-border\" x=\"{bi}\" y=\"{bi}\" width=\"{bw}\" height=\"{bh}\" \
         fill=\"none\" stroke=\"#6b7280\" stroke-width=\"1.5\" stroke-dasharray=\"6,4\"/>\n",
        bi = bi,
        bw = canvas_w - bi * 2.0,
        bh = canvas_h - bi * 2.0,
    ));

    // ── Header strip ──────────────────────────────────────────────────
    let header = format!(
        "CHUNK ({cx}, {cz})  ·  {ic}×{ic} voxels  ·  {vs}m/voxel",
        cx = cx_idx, cz = cz_idx,
        ic = interior as usize, vs = voxel_scale,
    );
    let header_h = ppv * 1.8;
    svg.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{cw}\" height=\"{hh}\" fill=\"#0f172a\" fill-opacity=\"0.9\"/>\n\
         <text class=\"lbl\" x=\"6\" y=\"5\" fill=\"#38bdf8\" font-weight=\"bold\">{header}</text>\n",
        cw = canvas_w, hh = header_h, header = header,
    ));

    // ── Legend ─────────────────────────────────────────────────────────
    if options.show_labels {
        let legend_items: &[(&str, &str)] = &[
            ("vw",   "WALL"),
            ("vrw",  "RED WALL"),
            ("vli",  "LIGHT"),
            ("vrli", "RED LIGHT"),
            ("vfl",  "FLOOR"),
            ("vce",  "CEILING"),
        ];
        let legend_col_w = ppv * 8.5;
        let legend_row_h = ppv * 1.1;
        let legend_x = canvas_w - legend_col_w - 4.0;
        let legend_total_h = legend_row_h * (legend_items.len() as f32) + 6.0;
        let legend_y = canvas_h - legend_total_h - 4.0;
        svg.push_str(&format!(
            "<rect x=\"{lx}\" y=\"{ly}\" width=\"{lw}\" height=\"{lh}\" \
             fill=\"#0f172a\" fill-opacity=\"0.88\" rx=\"3\"/>\n",
            lx = legend_x, ly = legend_y,
            lw = legend_col_w, lh = legend_total_h,
        ));
        for (i, (cls, label)) in legend_items.iter().enumerate() {
            let ry = legend_y + 3.0 + i as f32 * legend_row_h;
            svg.push_str(&format!(
                "<rect class=\"{cls}\" x=\"{sx}\" y=\"{ry}\" width=\"{ppv}\" height=\"{ppv}\"/>\n\
                 <text class=\"lsm\" x=\"{tx}\" y=\"{ry}\" fill=\"#94a3b8\">{label}</text>\n",
                cls = cls,
                sx = legend_x + 3.0,
                ry = ry, ppv = ppv,
                tx = legend_x + ppv + 6.0,
                label = label,
            ));
        }
    }

    svg.push_str("</svg>\n");
    svg
}
