//! Bind group zero: one immutable layout and two streaming buffers shared by
//! every GPU strategy.

use std::num::NonZeroU64;

use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

use super::gpu_types::{GpuFrameUniforms, GpuLight};

pub struct FrameResources {
    layout: wgpu::BindGroupLayout,
    frame_buffer: wgpu::Buffer,
    light_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    light_capacity: usize,
}

impl FrameResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("webgpu.frame-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<GpuFrameUniforms>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<GpuLight>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("webgpu.frame-uniforms"),
            size: size_of::<GpuFrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_capacity = 1;
        let light_buffer = create_light_buffer(device, light_capacity);
        let bind_group = create_bind_group(device, &layout, &frame_buffer, &light_buffer);
        Self {
            layout,
            frame_buffer,
            light_buffer,
            bind_group,
            light_capacity,
        }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &GpuFrameUniforms,
        lights: &[GpuLight],
    ) {
        if lights.len().max(1) > self.light_capacity {
            self.light_capacity = lights.len().max(1).next_power_of_two();
            self.light_buffer = create_light_buffer(device, self.light_capacity);
            self.bind_group =
                create_bind_group(device, &self.layout, &self.frame_buffer, &self.light_buffer);
        }
        queue.write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(frame));
        if lights.is_empty() {
            queue.write_buffer(
                &self.light_buffer,
                0,
                bytemuck::bytes_of(&GpuLight::zeroed()),
            );
        } else {
            queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(lights));
        }
    }
}

fn create_light_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("webgpu.frame-lights"),
        contents: &vec![0; capacity * size_of::<GpuLight>()],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    frame: &wgpu::Buffer,
    lights: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webgpu.frame-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lights.as_entire_binding(),
            },
        ],
    })
}
