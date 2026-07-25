use std::fs;
use std::path::Path;
use vackrooms::domain::entities::position::Position;
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::{REGION_SIZE, generate_region_plan};
use vackrooms::adapters::blueprint_renderer::{
    render_region_blueprint_svg, BlueprintOptions, ChunkBounds,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seed = 42;
    for i in 0..args.len() {
        if args[i] == "--seed" && i + 1 < args.len() {
            if let Ok(s) = args[i + 1].parse::<u32>() {
                seed = s;
            }
        }
    }

    let noise = SimpleNoiseProvider::new();
    let rx = 0i64;
    let rz = 0i64;
    let plan = generate_region_plan(
        seed,
        Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
        REGION_SIZE,
        &GeneratorConfig::low_spec(),
        &noise,
    );

    let options = BlueprintOptions {
        pixels_per_unit: 10.0,
        show_chunk_grid: true,
        show_fixtures: true,
        show_structure: true,
        seed,
        level: 0,
        voxel_scale: 0.2,
    };

    let main_crossing_cb = ChunkBounds {
        min_x: 40.0,
        min_z: 10.0,
        max_x: 50.0,
        max_z: 20.0,
        label: "main-spine-crossing".to_string(),
    };

    let secondary_branch_cb = ChunkBounds {
        min_x: 10.0,
        min_z: 0.0,
        max_x: 20.0,
        max_z: 10.0,
        label: "secondary-branch".to_string(),
    };

    let portal_boundary_cb = ChunkBounds {
        min_x: 20.0,
        min_z: 0.0,
        max_x: 30.0,
        max_z: 10.0,
        label: "portal-at-boundary".to_string(),
    };

    let fixtures = [
        ("main-spine-crossing.svg", main_crossing_cb),
        ("secondary-branch.svg", secondary_branch_cb),
        ("portal-at-boundary.svg", portal_boundary_cb),
    ];

    for (filename, cb) in &fixtures {
        let svg = render_region_blueprint_svg(&plan, &[cb.clone()], options);
        fs::write(filename, &svg).unwrap();
        println!("Wrote blueprint SVG: {}", filename);
        
        let art_dir = "/home/dbslim/.gemini/antigravity-cli/brain/631f1036-e351-406b-91f2-5e89e6a6e89f";
        if Path::new(art_dir).exists() {
            let art_path = format!("{}/{}", art_dir, filename);
            fs::write(&art_path, &svg).unwrap();
            println!("Copied blueprint SVG to artifacts: {}", art_path);
        }
    }

    // Generate a square of 100 chunks (10x10 grid of 10.0m chunks = 100.0m region)
    let plan_100 = generate_region_plan(
        seed,
        Position::new(0.0, 0.0),
        100.0,
        &GeneratorConfig::low_spec(),
        &noise,
    );

    let mut bounds_100 = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            bounds_100.push(ChunkBounds {
                min_x: i as f32 * 10.0,
                min_z: j as f32 * 10.0,
                max_x: (i + 1) as f32 * 10.0,
                max_z: (j + 1) as f32 * 10.0,
                label: format!("[{}, {}]", i, j),
            });
        }
    }

    let svg_100 = render_region_blueprint_svg(&plan_100, &bounds_100, options);
    let filename_100 = "grid-100-chunks.svg";
    fs::write(filename_100, &svg_100).unwrap();
    println!("Wrote 100-chunk blueprint SVG: {}", filename_100);

    let art_dir = "/home/dbslim/.gemini/antigravity-cli/brain/631f1036-e351-406b-91f2-5e89e6a6e89f";
    if Path::new(art_dir).exists() {
        let art_path = format!("{}/{}", art_dir, filename_100);
        fs::write(&art_path, &svg_100).unwrap();
        println!("Copied 100-chunk blueprint SVG to artifacts: {}", art_path);
    }
}
