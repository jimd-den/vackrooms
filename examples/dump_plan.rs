use vackrooms::entities::models::Position;
use vackrooms::frameworks_drivers::simple_noise::SimpleNoiseProvider;
use vackrooms::use_cases::generate_chunk::GeneratorConfig;
use vackrooms::use_cases::region_plan::{REGION_SIZE, debug_region_ascii, generate_region_plan};

fn main() {
    let noise = SimpleNoiseProvider::new();
    for (rx, rz) in [(0i64, 0i64), (1, 0)] {
        let plan = generate_region_plan(
            42,
            Position::new(rx as f32 * REGION_SIZE, rz as f32 * REGION_SIZE),
            REGION_SIZE,
            &GeneratorConfig::low_spec(),
            &noise,
        );
        println!(
            "region ({rx},{rz}) circulation={:?} assemblies={}",
            plan.architects[0].circulation,
            plan.assemblies.len()
        );
        println!("{}", debug_region_ascii(&plan, 1.0));
    }
}
