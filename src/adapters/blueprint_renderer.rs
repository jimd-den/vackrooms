use crate::domain::entities::architecture::{
    RegionPlan, SpaceProgram, StructuralSystem,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlueprintOptions {
    pub pixels_per_unit: f32,
    pub show_chunk_grid: bool,
    pub show_fixtures: bool,
    pub show_structure: bool,
    pub seed: u32,
    pub level: u32,
    pub voxel_scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkBounds {
    pub min_x: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_z: f32,
    pub label: String,
}

pub fn render_region_blueprint_svg(
    plan: &RegionPlan,
    chunk_bounds: &[ChunkBounds],
    options: BlueprintOptions,
) -> String {
    let scale = options.pixels_per_unit;
    let size_px = plan.size_world * scale;

    let mut svg = String::new();
    svg.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  <style>
    .bg {{ fill: #0b0f19; }}
    .grid {{ stroke: #1e293b; stroke-width: 0.5; }}
    .chunk-grid {{ stroke: #f43f5e; stroke-width: 1.0; stroke-dasharray: 4,4; fill: none; }}
    .chunk-label {{ fill: #f43f5e; font-family: monospace; font-size: 10px; font-weight: bold; }}
    .assembly {{ fill: #1e293b; fill-opacity: 0.2; stroke: #475569; stroke-width: 1.0; }}
    .assembly-corrupt {{ fill: #a21caf; fill-opacity: 0.15; stroke: #d946ef; stroke-width: 1.0; }}
    .assembly-label {{ fill: #94a3b8; font-family: monospace; font-size: 9px; font-weight: bold; }}
    .corridor-main {{ stroke: #38bdf8; stroke-linecap: round; stroke-linejoin: round; fill: none; }}
    .corridor-secondary {{ stroke: #818cf8; stroke-linecap: round; stroke-linejoin: round; fill: none; }}
    .entrance {{ stroke: #10b981; stroke-width: 2; stroke-linecap: square; }}
    .entrance-node {{ fill: #10b981; }}
    .fixture-lit {{ fill: #fbbf24; stroke: #d97706; stroke-width: 0.5; }}
    .fixture-unlit {{ fill: #475569; stroke: #334155; stroke-width: 0.5; }}
    .column {{ fill: #64748b; stroke: #475569; stroke-width: 0.5; }}
    .title-bg {{ fill: #0f172a; fill-opacity: 0.9; stroke: #475569; stroke-width: 1.5; }}
    .title-text {{ fill: #f8fafc; font-family: monospace; font-size: 11px; font-weight: bold; }}
    .title-sub {{ fill: #94a3b8; font-family: monospace; font-size: 9px; }}
  </style>
  <rect class="bg" width="{}" height="{}" />
"##,
        size_px, size_px, size_px, size_px, size_px, size_px
    ));

    // Grid lines every 5 world units
    let grid_step = 5.0 * scale;
    let mut grid_y = grid_step;
    while grid_y < size_px {
        svg.push_str(&format!(
            r##"  <line class="grid" x1="0" y1="{}" x2="{}" y2="{}" />
  <line class="grid" x1="{}" y1="0" x2="{}" y2="{}" />
"##,
            grid_y, size_px, grid_y, grid_y, grid_y, size_px
        ));
        grid_y += grid_step;
    }

    let to_svg_x = |wx: f32| -> f32 {
        (wx - plan.origin_world.x) * scale
    };
    let to_svg_y = |wz: f32| -> f32 {
        (plan.size_world - (wz - plan.origin_world.z)) * scale
    };

    // Layer 1: Corridors
    for s in &plan.corridors {
        if s.path.len() < 2 {
            continue;
        }
        let class_name = if s.spine_kind == SpaceProgram::MainCorridor {
            "corridor-main"
        } else {
            "corridor-secondary"
        };
        let mut points_str = String::new();
        for pos in &s.path {
            points_str.push_str(&format!("{},{} ", to_svg_x(pos.x), to_svg_y(pos.z)));
        }
        let stroke_w = s.width * scale;
        svg.push_str(&format!(
            r##"  <polyline class="{}" points="{}" stroke-width="{}" />
"##,
            class_name, points_str, stroke_w
        ));
    }

    // Layer 2: Assembly Footprints
    for a in &plan.assemblies {
        let mut points_str = String::new();
        for &(vx, vz) in &a.footprint.vertices {
            points_str.push_str(&format!("{},{} ", to_svg_x(vx), to_svg_y(vz)));
        }
        let class_name = if a.corruption.misalignment.0 != 0.0 || a.corruption.misalignment.1 != 0.0 || a.corruption.abandoned {
            "assembly-corrupt"
        } else {
            "assembly"
        };
        svg.push_str(&format!(
            r##"  <polygon class="{}" points="{}" />
"##,
            class_name, points_str
        ));
    }

    // Layer 3: Entrances / Portals
    for a in &plan.assemblies {
        for e in a.entrances() {
            let door_half_w = e.width * 0.5;
            let (x1, z1, x2, z2) = if e.through_x_wall {
                (e.center.x - door_half_w, e.center.z, e.center.x + door_half_w, e.center.z)
            } else {
                (e.center.x, e.center.z - door_half_w, e.center.x, e.center.z + door_half_w)
            };

            let svg_x1 = to_svg_x(x1);
            let svg_y1 = to_svg_y(z1);
            let svg_x2 = to_svg_x(x2);
            let svg_y2 = to_svg_y(z2);

            svg.push_str(&format!(
                r##"  <line class="entrance" x1="{}" y1="{}" x2="{}" y2="{}" />
  <circle class="entrance-node" cx="{}" cy="{}" r="3.5" />
"##,
                svg_x1, svg_y1, svg_x2, svg_y2,
                to_svg_x(e.center.x), to_svg_y(e.center.z)
            ));

            let txt_x = to_svg_x(e.center.x);
            let txt_y = to_svg_y(e.center.z) - 6.0;
            svg.push_str(&format!(
                r##"  <text x="{}" y="{}" fill="#10b981" font-family="monospace" font-size="8px" font-weight="bold" text-anchor="middle">{:.1}m</text>
"##,
                txt_x, txt_y, e.width
            ));
        }
    }

    // Layer 4: Fixtures
    if options.show_fixtures {
        for a in &plan.assemblies {
            for f in &a.fixtures {
                let w_px = f.half_x * 2.0 * scale;
                let h_px = f.half_z * 2.0 * scale;
                let svg_f_x = to_svg_x(f.at.x) - w_px * 0.5;
                let svg_f_y = to_svg_y(f.at.z) - h_px * 0.5;

                let class_name = if f.lit { "fixture-lit" } else { "fixture-unlit" };
                svg.push_str(&format!(
                    r##"  <rect class="{}" x="{}" y="{}" width="{}" height="{}" rx="1" />
"##,
                    class_name, svg_f_x, svg_f_y, w_px, h_px
                ));
            }
        }
    }

    // Layer 5: Columns
    if options.show_structure {
        for a in &plan.assemblies {
            let st = &a.structure;
            if st.system == StructuralSystem::CoreAndShell {
                continue;
            }
            let (min_x, min_z, max_x, max_z) = a.footprint.bounds();

            let start_x = ((min_x - st.phase.0) / st.bay_x).floor() as i32 - 1;
            let end_x = ((max_x - st.phase.0) / st.bay_x).ceil() as i32 + 1;
            let start_z = ((min_z - st.phase.1) / st.bay_z).floor() as i32 - 1;
            let end_z = ((max_z - st.phase.1) / st.bay_z).ceil() as i32 + 1;

            for iz in start_z..=end_z {
                for ix in start_x..=end_x {
                    let mut wx = st.phase.0 + ix as f32 * st.bay_x;
                    let wz = st.phase.1 + iz as f32 * st.bay_z;
                    if st.system == StructuralSystem::OffsetGrid {
                        if iz.rem_euclid(2) == 1 {
                            wx += st.bay_x * 0.5;
                        }
                    }

                    let col_center_x = wx + st.column_side * 0.5;
                    let col_center_z = wz + st.column_side * 0.5;
                    if a.footprint.contains(col_center_x, col_center_z) {
                        let svg_col_x = to_svg_x(wx);
                        let svg_col_y = to_svg_y(wz + st.column_side);
                        let w_px = st.column_side * scale;
                        svg.push_str(&format!(
                            r##"  <rect class="column" x="{}" y="{}" width="{}" height="{}" />
"##,
                            svg_col_x, svg_col_y, w_px, w_px
                        ));
                    }
                }
            }
        }
    }

    // Layer 6: Labels
    for a in &plan.assemblies {
        let mut sum_x = 0.0;
        let mut sum_z = 0.0;
        for &(vx, vz) in &a.footprint.vertices {
            sum_x += vx;
            sum_z += vz;
        }
        let cx = sum_x / a.footprint.vertices.len() as f32;
        let cz = sum_z / a.footprint.vertices.len() as f32;

        let label = format!("#{} · {:?}", a.id, a.program);
        let svg_cx = to_svg_x(cx);
        let svg_cz = to_svg_y(cz);

        svg.push_str(&format!(
            r##"  <text class="assembly-label" x="{}" y="{}" text-anchor="middle" dominant-baseline="middle">{}</text>
"##,
            svg_cx, svg_cz, label
        ));
    }

    // Layer 7: Chunk Grid Overlays
    if options.show_chunk_grid {
        for cb in chunk_bounds {
            let svg_min_x = to_svg_x(cb.min_x);
            let svg_max_x = to_svg_x(cb.max_x);
            let svg_min_y = to_svg_y(cb.max_z);
            let svg_max_y = to_svg_y(cb.min_z);

            let w_px = svg_max_x - svg_min_x;
            let h_px = svg_max_y - svg_min_y;

            svg.push_str(&format!(
                r##"  <rect class="chunk-grid" x="{}" y="{}" width="{}" height="{}" />
  <text class="chunk-label" x="{}" y="{}" text-anchor="start" dominant-baseline="hanging">{}</text>
"##,
                svg_min_x, svg_min_y, w_px, h_px,
                svg_min_x + 6.0, svg_min_y + 6.0, cb.label
            ));
        }
    }

    // Layer 8: Title Block
    let title_x = size_px - 230.0;
    let title_y = size_px - 140.0;
    let rx = (plan.origin_world.x / plan.size_world).round() as i32;
    let rz = (plan.origin_world.z / plan.size_world).round() as i32;
    svg.push_str(&format!(
        r##"  <g transform="translate({}, {})">
    <rect class="title-bg" width="215" height="125" rx="6" />
    <text class="title-text" x="15" y="25">VACKROOMS ARCHITECTURE</text>
    <text class="title-sub" x="15" y="45">Seed: {} | Level: {}</text>
    <text class="title-sub" x="15" y="60">Region: ({}, {})</text>
    <text class="title-sub" x="15" y="75">Region Size: {:.1}m x {:.1}m</text>
    <text class="title-sub" x="15" y="90">Voxel Scale: {}m</text>
    <text class="title-sub" x="15" y="105">Scale: 1 unit = {}px</text>
  </g>
"##,
        title_x, title_y,
        options.seed, options.level,
        rx, rz,
        plan.size_world, plan.size_world,
        options.voxel_scale, scale
    ));

    svg.push_str("</svg>\n");
    svg
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlueprintSlice {
    FloorPlan,       // y = 0: floor, walls projected upward
    WallPlan,        // any WALL / REDWALL in y = 1..ceiling
    CeilingPlan,     // ceiling and LIGHT voxels
    Composite,       // wall + floor + light + semantic overlay
}

pub fn render_voxel_chunk_svg(
    grid: &crate::domain::entities::voxel_grid::VoxelGrid,
    origin: crate::domain::entities::position::Position,
    config: &crate::use_cases::generate_chunk::GeneratorConfig,
    slice: BlueprintSlice,
    semantics: Option<&RegionPlan>,
) -> String {
    use crate::domain::entities::voxel_grid::{
        VOXEL_AIR, VOXEL_CEILING, VOXEL_FLOOR, VOXEL_GRASS, VOXEL_LIGHT,
        VOXEL_RED_LIGHT, VOXEL_RED_WALL, VOXEL_TREE, VOXEL_WALL, VOXEL_WATER,
    };

    let w_voxels = grid.width().saturating_sub(2);
    let d_voxels = grid.depth().saturating_sub(2);

    let cell_size = 12.0;
    let width_px = w_voxels as f32 * cell_size;
    let height_px = d_voxels as f32 * cell_size;

    let mut svg = String::new();
    svg.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  <style>
    .bg {{ fill: #070a13; }}
    .cell-air {{ fill: #0b0f19; }}
    .cell-main-corridor {{ fill: #083344; fill-opacity: 0.8; }}
    .cell-secondary-corridor {{ fill: #1e1b4b; fill-opacity: 0.8; }}
    .cell-assembly {{ fill: #111827; fill-opacity: 0.8; }}
    
    .voxel-wall {{ fill: #1e293b; stroke: #475569; stroke-width: 0.5; }}
    .voxel-red-wall {{ fill: #7f1d1d; stroke: #b91c1c; stroke-width: 0.5; }}
    .voxel-floor {{ fill: #78350f; stroke: #92400e; stroke-dasharray: 1,1; stroke-width: 0.5; }}
    .voxel-grass {{ fill: #14532d; stroke: #166534; stroke-width: 0.5; }}
    .voxel-water {{ fill: #1e3a8a; stroke: #1d4ed8; stroke-width: 0.5; }}
    .voxel-tree {{ fill: #064e3b; stroke: #047857; stroke-width: 0.5; }}
    .voxel-ceiling {{ fill: #334155; stroke: #475569; stroke-width: 0.5; }}
    .voxel-light {{ fill: #fef08a; stroke: #ca8a04; stroke-width: 0.5; filter: drop-shadow(0px 0px 2px #fef08a); }}
    .voxel-red-light {{ fill: #fecaca; stroke: #dc2626; stroke-width: 0.5; filter: drop-shadow(0px 0px 2px #fecaca); }}

    .grid-line {{ stroke: #1e293b; stroke-width: 0.25; }}
    .label-text {{ fill: #94a3b8; font-family: monospace; font-size: 10px; }}
  </style>
  <rect class="bg" width="{}" height="{}" />
"##,
        width_px, height_px, width_px, height_px, width_px, height_px
    ));

    // Render cells
    for z in 0..d_voxels {
        for x in 0..w_voxels {
            // Voxel coordinates are 1-indexed due to padding
            let vx = x + 1;
            let vz = z + 1;

            // World coordinates
            let wx = origin.x + x as f32 * config.voxel_scale;
            let wz = origin.z + z as f32 * config.voxel_scale;

            // 1. Semantic background classification
            let mut semantic_class = "cell-air";
            if let Some(plan) = semantics {
                let mut inside_main = false;
                let mut inside_secondary = false;
                for c in &plan.corridors {
                    if c.path.len() >= 2 {
                        let dist = point_to_polyline_distance(
                            wx + config.voxel_scale * 0.5,
                            wz + config.voxel_scale * 0.5,
                            &c.path,
                        );
                        if dist <= c.width * 0.5 {
                            if c.spine_kind == SpaceProgram::MainCorridor {
                                inside_main = true;
                            } else {
                                inside_secondary = true;
                            }
                        }
                    }
                }
                if inside_main {
                    semantic_class = "cell-main-corridor";
                } else if inside_secondary {
                    semantic_class = "cell-secondary-corridor";
                } else {
                    for a in &plan.assemblies {
                        if a.footprint.contains(
                            wx + config.voxel_scale * 0.5,
                            wz + config.voxel_scale * 0.5,
                        ) {
                            semantic_class = "cell-assembly";
                            break;
                        }
                    }
                }
            }

            // 2. Voxel classification based on BlueprintSlice
            let mut has_wall = false;
            let mut has_red_wall = false;
            let mut has_light = false;
            let mut has_red_light = false;
            let mut has_floor = false;
            let mut has_ceiling = false;
            let mut has_tree = false;
            let mut has_grass = false;
            let mut has_water = false;

            for y in 0..grid.height() {
                match grid.get(vx, y, vz) {
                    VOXEL_WALL => has_wall = true,
                    VOXEL_RED_WALL => has_red_wall = true,
                    VOXEL_LIGHT => has_light = true,
                    VOXEL_RED_LIGHT => has_red_light = true,
                    VOXEL_FLOOR => has_floor = true,
                    VOXEL_CEILING => has_ceiling = true,
                    VOXEL_TREE => has_tree = true,
                    VOXEL_GRASS => has_grass = true,
                    VOXEL_WATER => has_water = true,
                    _ => {}
                }
            }

            let voxel_class = match slice {
                BlueprintSlice::FloorPlan => {
                    if has_wall || has_red_wall || has_tree {
                        Some(if has_red_wall { "voxel-red-wall" } else { "voxel-wall" })
                    } else if has_floor || has_grass || has_water {
                        Some(if has_water { "voxel-water" } else if has_grass { "voxel-grass" } else { "voxel-floor" })
                    } else {
                        None
                    }
                }
                BlueprintSlice::WallPlan => {
                    if has_wall || has_red_wall || has_tree {
                        Some(if has_red_wall { "voxel-red-wall" } else { "voxel-wall" })
                    } else {
                        None
                    }
                }
                BlueprintSlice::CeilingPlan => {
                    if has_light || has_red_light {
                        Some(if has_red_light { "voxel-red-light" } else { "voxel-light" })
                    } else if has_ceiling {
                        Some("voxel-ceiling")
                    } else {
                        None
                    }
                }
                BlueprintSlice::Composite => {
                    if has_wall || has_red_wall || has_tree {
                        Some(if has_red_wall { "voxel-red-wall" } else { "voxel-wall" })
                    } else if has_light || has_red_light {
                        Some(if has_red_light { "voxel-red-light" } else { "voxel-light" })
                    } else if has_floor || has_grass || has_water {
                        Some(if has_water { "voxel-water" } else if has_grass { "voxel-grass" } else { "voxel-floor" })
                    } else {
                        None
                    }
                }
            };

            // Render rect in SVG coordinates (origin top-left)
            let svg_x = x as f32 * cell_size;
            let svg_y = (d_voxels - 1 - z) as f32 * cell_size;

            if let Some(vc) = voxel_class {
                svg.push_str(&format!(
                    r##"  <rect class="{}" x="{}" y="{}" width="{}" height="{}" />
"##,
                    vc, svg_x, svg_y, cell_size, cell_size
                ));
            } else {
                svg.push_str(&format!(
                    r##"  <rect class="{}" x="{}" y="{}" width="{}" height="{}" />
"##,
                    semantic_class, svg_x, svg_y, cell_size, cell_size
                ));
            }
        }
    }

    // Draw coordinate label on top
    svg.push_str(&format!(
        r##"  <text class="label-text" x="10" y="20">Chunk: ({:.1}, {:.1}) | Slice: {:?}</text>
"##,
        origin.x, origin.z, slice
    ));

    svg.push_str("</svg>\n");
    svg
}

fn point_to_polyline_distance(
    x: f32,
    z: f32,
    path: &[crate::domain::entities::position::Position],
) -> f32 {
    let mut min_dist = f32::MAX;
    for seg in path.windows(2) {
        let p1 = seg[0];
        let p2 = seg[1];
        let dist = point_to_segment_distance(x, z, p1.x, p1.z, p2.x, p2.z);
        if dist < min_dist {
            min_dist = dist;
        }
    }
    min_dist
}

fn point_to_segment_distance(x: f32, z: f32, x1: f32, z1: f32, x2: f32, z2: f32) -> f32 {
    let dx = x2 - x1;
    let dz = z2 - z1;
    let len2 = dx * dx + dz * dz;
    if len2 == 0.0 {
        let t_dx = x - x1;
        let t_dz = z - z1;
        return (t_dx * t_dx + t_dz * t_dz).sqrt();
    }
    let t = ((x - x1) * dx + (z - z1) * dz) / len2;
    let t_clamped = t.clamp(0.0, 1.0);
    let proj_x = x1 + t_clamped * dx;
    let proj_z = z1 + t_clamped * dz;
    let diff_x = x - proj_x;
    let diff_z = z - proj_z;
    (diff_x * diff_x + diff_z * diff_z).sqrt()
}

pub fn render_large_voxel_blueprint_svg(
    combined_cells: &[u8],
    total_w: usize,
    total_d: usize,
    rx_origin: f32,
    rz_origin: f32,
    voxel_scale: f32,
    size_world: f32,
    slice: BlueprintSlice,
    semantics: Option<&RegionPlan>,
    is_bw: bool,
) -> String {
    let scale = 10.0; // 10 pixels per world unit
    let size_px = size_world * scale;

    let mut svg = String::new();
    if is_bw {
        svg.push_str(&format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  <style>
    .bg {{ fill: #ffffff; }}
    .voxel-wall {{ fill: #000000; }}
    .voxel-red-wall {{ fill: #000000; }}
    .voxel-light {{ fill: #ff0000; }}
    .voxel-red-light {{ fill: #ff0000; }}
  </style>
  <rect class="bg" width="{}" height="{}" />
"##,
            size_px, size_px, size_px, size_px, size_px, size_px
        ));
    } else {
        svg.push_str(&format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  <style>
    .bg {{ fill: #070a13; }}
    .grid {{ stroke: #1e293b; stroke-width: 0.5; }}
    .chunk-grid {{ stroke: #f43f5e; stroke-width: 1.0; stroke-dasharray: 4,4; fill: none; }}
    .chunk-label {{ fill: #f43f5e; font-family: monospace; font-size: 10px; font-weight: bold; }}
    
    .assembly {{ fill: #1e293b; fill-opacity: 0.25; stroke: #475569; stroke-width: 1.0; }}
    .assembly-corrupt {{ fill: #a21caf; fill-opacity: 0.2; stroke: #d946ef; stroke-width: 1.0; }}
    .assembly-label {{ fill: #94a3b8; font-family: monospace; font-size: 9px; font-weight: bold; }}
    .corridor-main {{ stroke: #083344; stroke-width: 3.0; stroke-linecap: round; stroke-linejoin: round; fill: none; }}
    .corridor-secondary {{ stroke: #1e1b4b; stroke-width: 2.0; stroke-linecap: round; stroke-linejoin: round; fill: none; }}

    .voxel-wall {{ fill: #1e293b; stroke: #475569; stroke-width: 0.25; }}
    .voxel-red-wall {{ fill: #7f1d1d; stroke: #b91c1c; stroke-width: 0.25; }}
    .voxel-floor {{ fill: #78350f; fill-opacity: 0.4; stroke: #92400e; stroke-width: 0.1; }}
    .voxel-grass {{ fill: #14532d; fill-opacity: 0.4; stroke: #166534; stroke-width: 0.1; }}
    .voxel-water {{ fill: #1e3a8a; fill-opacity: 0.4; stroke: #1d4ed8; stroke-width: 0.1; }}
    .voxel-ceiling {{ fill: #334155; fill-opacity: 0.4; stroke: #475569; stroke-width: 0.1; }}
    .voxel-light {{ fill: #fef08a; stroke: #ca8a04; stroke-width: 0.25; }}
    .voxel-red-light {{ fill: #fecaca; stroke: #dc2626; stroke-width: 0.25; }}

    .title-bg {{ fill: #0f172a; fill-opacity: 0.95; stroke: #475569; stroke-width: 1.5; }}
    .title-text {{ fill: #f8fafc; font-family: monospace; font-size: 11px; font-weight: bold; }}
    .title-sub {{ fill: #94a3b8; font-family: monospace; font-size: 9px; }}
  </style>
  <rect class="bg" width="{}" height="{}" />
"##,
            size_px, size_px, size_px, size_px, size_px, size_px
        ));
    }

    // Helpers
    let to_svg_x = |wx: f32| -> f32 { (wx - rx_origin) * scale };
    let to_svg_y = |wz: f32| -> f32 { size_px - (wz - rz_origin) * scale };

    // 1. Draw semantics first (corridors and assemblies)
    if !is_bw {
        if let Some(plan) = semantics {
            // Assemblies
            for a in &plan.assemblies {
                let mut points_str = String::new();
                for &(vx, vz) in &a.footprint.vertices {
                    points_str.push_str(&format!("{},{} ", to_svg_x(vx), to_svg_y(vz)));
                }
                let class_name = if a.corruption.misalignment.0 != 0.0 || a.corruption.misalignment.1 != 0.0 || a.corruption.abandoned {
                    "assembly-corrupt"
                } else {
                    "assembly"
                };
                svg.push_str(&format!(
                    r##"  <polygon class="{}" points="{}" />
"##,
                    class_name, points_str
                ));
            }

            // Corridors
            for c in &plan.corridors {
                if c.path.len() >= 2 {
                    let mut d_str = String::new();
                    for (idx, p) in c.path.iter().enumerate() {
                        let prefix = if idx == 0 { "M" } else { "L" };
                        d_str.push_str(&format!("{} {},{} ", prefix, to_svg_x(p.x), to_svg_y(p.z)));
                    }
                    let class_name = if c.spine_kind == SpaceProgram::MainCorridor {
                        "corridor-main"
                    } else {
                        "corridor-secondary"
                    };
                    svg.push_str(&format!(
                        r##"  <path class="{}" d="{}" stroke-width="{}" />
"##,
                        class_name, d_str, c.width * scale
                    ));
                }
            }
        }
    }

    // 2. Draw 5-unit background grid lines
    if !is_bw {
        let grid_step = 5.0 * scale;
        let mut grid_val = 0.0;
        while grid_val < size_px {
            svg.push_str(&format!(
                r##"  <line class="grid" x1="{}" y1="0" x2="{}" y2="{}" />
  <line class="grid" x1="0" y1="{}" x2="{}" y2="{}" />
"##,
                grid_val, grid_val, size_px, grid_val, size_px, grid_val
            ));
            grid_val += grid_step;
        }
    }

    // 3. Draw voxels
    let voxel_px = voxel_scale * scale;
    for z in 0..total_d {
        for x in 0..total_w {
            let idx = z * total_w + x;
            let val = combined_cells[idx];
            if val == 0 {
                continue; // Air is transparent to show background plan
            }

            let class_name = match val {
                1 => "voxel-wall",
                2 => "voxel-red-wall",
                3 => "voxel-light",
                4 => "voxel-red-light",
                5 if !is_bw => "voxel-floor",
                6 if !is_bw => "voxel-grass",
                7 if !is_bw => "voxel-water",
                9 if !is_bw => "voxel-ceiling",
                _ => continue,
            };

            let svg_x = x as f32 * voxel_px;
            let svg_y = size_px - (z + 1) as f32 * voxel_px;

            // In BW mode, we don't want strokes on elements
            let stroke_attr = if is_bw { "" } else { "" };

            svg.push_str(&format!(
                r##"  <rect class="{}" x="{}" y="{}" width="{}" height="{}"{} />
"##,
                class_name, svg_x, svg_y, voxel_px, voxel_px, stroke_attr
            ));
        }
    }

    // 4. Draw chunk boundaries on top
    if !is_bw {
        let chunk_size = if voxel_scale <= 0.1 { 20.0 } else { 10.0 };
        let steps = (size_world / chunk_size).round() as i32;
        for i in 0..steps {
            for j in 0..steps {
                let min_x = rx_origin + i as f32 * chunk_size;
                let min_z = rz_origin + j as f32 * chunk_size;
                let svg_min_x = to_svg_x(min_x);
                let svg_max_x = to_svg_x(min_x + chunk_size);
                let svg_min_y = to_svg_y(min_z + chunk_size);
                let svg_max_y = to_svg_y(min_z);

                let w_px = svg_max_x - svg_min_x;
                let h_px = svg_max_y - svg_min_y;

                let chunk_coord_x = (min_x / chunk_size).round() as i64;
                let chunk_coord_z = (min_z / chunk_size).round() as i64;

                svg.push_str(&format!(
                    r##"  <rect class="chunk-grid" x="{}" y="{}" width="{}" height="{}" />
  <text class="chunk-label" x="{}" y="{}" text-anchor="start" dominant-baseline="hanging">[{}, {}]</text>
"##,
                    svg_min_x, svg_min_y, w_px, h_px,
                    svg_min_x + 6.0, svg_min_y + 6.0, chunk_coord_x, chunk_coord_z
                ));
            }
        }
    }

    // 5. Title Block
    if !is_bw {
        let title_x = size_px - 230.0;
        let title_y = size_px - 140.0;
        svg.push_str(&format!(
            r##"  <g transform="translate({}, {})">
    <rect class="title-bg" width="215" height="125" rx="6" />
    <text class="title-text" x="15" y="25">VOXEL MAP BLUEPRINT</text>
    <text class="title-sub" x="15" y="45">Slice Mode: {:?}</text>
    <text class="title-sub" x="15" y="60">Region: ({:.1}, {:.1})</text>
    <text class="title-sub" x="15" y="75">Region Size: {:.1}m x {:.1}m</text>
    <text class="title-sub" x="15" y="90">Voxel Scale: {}m</text>
    <text class="title-sub" x="15" y="105">Resolution: {}x{} cells</text>
  </g>
"##,
            title_x, title_y,
            slice,
            rx_origin, rz_origin,
            size_world, size_world,
            voxel_scale,
            total_w, total_d
        ));
    }

    svg.push_str("</svg>\n");
    svg
}
