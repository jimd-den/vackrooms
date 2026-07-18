//! Production WebGPU strategies. Each module owns one resource cache and one
//! render pass; browser and native Vulkan tests call the same code.

mod cpu_present;
mod raster_shadow;
mod raymarch;
mod splat;
mod supply_labels;
mod surface;
mod visibility;

pub use cpu_present::CpuPresentPipeline;
pub use raster_shadow::HeroShadowOptions;
pub use raymarch::{RaymarchPipeline, RaymarchRuntimeOptions};
pub use splat::{SplatFrameStats, SplatPipeline};
pub use supply_labels::SupplyLabelPipeline;
pub use surface::SurfacePipeline;
