//! Canvas-2D blit driver for the CPU splatting renderer.
//!
//! All rendering logic lives in the platform-free
//! `adapters::cpu_splatter::SoftwareRasterizer`; this wrapper's only job is
//! to copy the finished RGBA framebuffer into the canvas via `ImageData` —
//! no WebGL context is created at all.

use wasm_bindgen::{Clamped, JsCast, JsValue};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

use crate::adapters::cpu_splatter::SoftwareRasterizer;
use crate::application::ports::{ChunkDraw, FrameParams, RendererPort};

pub struct CpuCanvasRenderer {
    rasterizer: SoftwareRasterizer,
    ctx: CanvasRenderingContext2d,
    settings: crate::adapters::cpu_splatter::CpuRenderSettings,
}

impl CpuCanvasRenderer {
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, JsValue> {
        let ctx = canvas
            .get_context("2d")?
            .ok_or_else(|| JsValue::from_str("no 2d context available"))?
            .dyn_into::<CanvasRenderingContext2d>()?;
        let settings = crate::get_cpu_settings();
        Ok(Self {
            rasterizer: SoftwareRasterizer::new(canvas.width() as usize, canvas.height() as usize),
            ctx,
            settings,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        // Impose a low CPU backing resolution cap (480x270 max), keeping aspect ratio
        let max_w = 480.0;
        let max_h = 270.0;
        let scale = (max_w / width as f32).min(max_h / height as f32).min(1.0) * self.settings.internal_scale;
        let w = ((width as f32 * scale).round() as usize).max(1);
        let h = ((height as f32 * scale).round() as usize).max(1);

        self.rasterizer.resize(w, h);
    }

    pub fn telemetry_string(&self) -> String {
        let stats = self.rasterizer.telemetry();
        format!(
            "V:{} (D:{})/S:{}/P:{}K{}",
            stats.visited_nodes,
            stats.max_virtual_depth,
            stats.splat_count,
            stats.pixel_writes / 1000,
            if stats.budget_exhausted { "!" } else { "" }
        )
    }
}

impl RendererPort for CpuCanvasRenderer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        self.rasterizer.upload_atlas(texels);
    }

    fn cpu_telemetry_string(&self) -> Option<String> {
        Some(self.telemetry_string())
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        let old_scale = self.settings.internal_scale;
        self.settings = crate::get_cpu_settings();
        self.settings.fov_tan = crate::drivers::webgl::fov_tan();
        if self.settings.internal_scale != old_scale {
            if let Some(canvas) = self.ctx.canvas() {
                self.resize(canvas.width(), canvas.height());
            }
        }
        self.rasterizer.settings = self.settings;
        self.rasterizer.draw(frame, chunks);
        let width = self.rasterizer.width() as u32;
        let height = self.rasterizer.height() as u32;
        let fb = self.rasterizer.framebuffer();
        
        if let Ok(image) = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(fb),
            width,
            height,
        ) {
            let _ = self.ctx.put_image_data(&image, 0.0, 0.0);
        }
    }
}
