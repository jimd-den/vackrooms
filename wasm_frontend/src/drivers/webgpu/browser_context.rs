//! Browser surface/device ownership. Pipeline modules receive only ordinary
//! `wgpu::Device`, `Queue`, and target views, so native Vulkan tests reuse the
//! production rendering code without a DOM.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;

pub struct BrowserGpuContext {
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    pub adapter_info: wgpu::AdapterInfo,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    device_lost: Arc<AtomicBool>,
}

impl BrowserGpuContext {
    pub async fn new(canvas: &HtmlCanvasElement) -> Result<Self, JsValue> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(js_error)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: true,
            })
            .await
            .map_err(js_error)?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("vackrooms-webgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(js_error)?;
        let device_lost = Arc::new(AtomicBool::new(false));
        device.set_device_lost_callback({
            let device_lost = Arc::clone(&device_lost);
            move |reason, message| {
                device_lost.store(true, Ordering::Release);
                web_sys::console::error_1(
                    &format!("WebGPU device lost ({reason:?}): {message}").into(),
                );
            }
        });
        device.on_uncaptured_error(Arc::new(|error| {
            web_sys::console::error_1(&format!("Uncaptured WebGPU error: {error}").into());
        }));

        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .ok_or_else(|| {
                JsValue::from_str(
                    "WebGPU surface exposes no linear color format; Vackrooms encodes display color in WGSL",
                )
            })?;
        let present_mode = capabilities
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::AutoVsync)
            .or_else(|| capabilities.present_modes.first().copied())
            .ok_or_else(|| JsValue::from_str("WebGPU surface exposes no present modes"))?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: canvas.width().max(1),
            height: canvas.height().max(1),
            desired_maximum_frame_latency: 2,
            present_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Ok(Self {
            _instance: instance,
            surface,
            adapter_info,
            device,
            queue,
            config,
            device_lost,
        })
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn width(&self) -> u32 {
        self.config.width
    }

    pub fn height(&self) -> u32 {
        self.config.height
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.config.width && height == self.config.height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn current_texture(&self) -> wgpu::CurrentSurfaceTexture {
        self.surface.get_current_texture()
    }

    pub fn reconfigure(&self) {
        self.surface.configure(&self.device, &self.config);
    }

    pub fn is_device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Acquire)
    }
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
