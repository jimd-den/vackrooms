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
}

impl CpuCanvasRenderer {
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, JsValue> {
        let ctx = canvas
            .get_context("2d")?
            .ok_or_else(|| JsValue::from_str("no 2d context available"))?
            .dyn_into::<CanvasRenderingContext2d>()?;
        Ok(Self {
            rasterizer: SoftwareRasterizer::new(canvas.width() as usize, canvas.height() as usize),
            ctx,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.rasterizer.resize(width as usize, height as usize);
    }
}

impl RendererPort for CpuCanvasRenderer {
    fn upload_atlas(&mut self, texels: &[u32]) {
        self.rasterizer.upload_atlas(texels);
    }

    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]) {
        self.rasterizer.draw(frame, chunks);
        let width = self.rasterizer.width() as u32;
        if let Ok(image) = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(self.rasterizer.framebuffer()),
            width,
            self.rasterizer.height() as u32,
        ) {
            let _ = self.ctx.put_image_data(&image, 0.0, 0.0);
        }
    }
}
