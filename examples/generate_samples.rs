use std::fs;
use vackrooms::entities::models::Position;
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::generate_region_plan;
use vackrooms::adapters::blueprint_renderer::{
    render_large_voxel_blueprint_svg, BlueprintSlice,
};

fn main() {
    let out_dir = "/home/dbslim/vackrooms/static/samples";
    fs::create_dir_all(out_dir).unwrap();

    let noise = SimpleNoiseProvider::new();
    let config = GeneratorConfig::low_spec();
    let chunk_size = config.chunk_size;
    let size_world = 100.0;
    let steps = (size_world / chunk_size).round() as i32;

    let chunk_w = (chunk_size / config.voxel_scale).round() as usize;
    let chunk_d = (chunk_size / config.voxel_scale).round() as usize;
    let total_w = steps as usize * chunk_w;
    let total_d = steps as usize * chunk_d;

    println!("Generating 100 B&W/Red SVGs (seeds 1 to 100) inside static/samples...");

    for seed in 1..=100 {
        // Generate plan
        let _plan = generate_region_plan(
            seed,
            Position::new(0.0, 0.0),
            size_world,
            &config,
            &noise,
        );

        let generator = vackrooms::use_cases::generate_chunk::GenerateChunkArchitectureUseCase::new(&noise);

        // Generate combined cells (1 for wall, 3 for light)
        let mut combined_cells = vec![0u8; total_w * total_d];

        for i in 0..steps {
            for j in 0..steps {
                let chunk_pos = Position::new(i as f32 * chunk_size, j as f32 * chunk_size);
                let grid = generator.execute(chunk_pos, seed, config.clone());

                for cz in 0..chunk_d {
                    for cx in 0..chunk_w {
                        let vx = cx + 1;
                        let vz = cz + 1;

                        let mut has_wall = false;
                        let mut has_light = false;

                        for y in 0..grid.height() {
                            let v = grid.get(vx, y, vz);
                            if v == 1 || v == 5 || v == 8 {
                                has_wall = true;
                            }
                            if v == 4 || v == 9 {
                                has_light = true;
                            }
                        }

                        let target_x = i as usize * chunk_w + cx;
                        let target_z = j as usize * chunk_d + cz;
                        let idx = target_z * total_w + target_x;

                        if has_wall {
                            combined_cells[idx] = 1; // Wall
                        } else if has_light {
                            combined_cells[idx] = 3; // Light
                        }
                    }
                }
            }
        }

        // Render composite view in B&W style (white bg, black walls, red lights)
        let svg = render_large_voxel_blueprint_svg(
            &combined_cells,
            total_w,
            total_d,
            0.0,
            0.0,
            config.voxel_scale,
            size_world,
            BlueprintSlice::Composite,
            None,
            true, // is_bw = true
        );

        // Write file
        fs::write(format!("{}/seed_{}.svg", out_dir, seed), &svg).unwrap();
    }

    println!("All 100 SVGs written successfully!");
}
