//! GPU context for compute shader dispatch.

use super::error::GpuError;

/// Holds the wgpu device, queue, and compute pipeline.
///
/// Created once and reused across multiple [`GpuGraph`](super::graph::GpuGraph) evaluations.
/// The compute pipeline uses an interpreter-style WGSL shader that evaluates
/// any compiled computation graph.
///
/// Derives [`bevy_ecs::resource::Resource`] for use as a Bevy singleton resource.
#[derive(bevy_ecs::resource::Resource)]
pub struct GpuContext {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) pipeline: wgpu::ComputePipeline,
}

impl GpuContext {
    /// Create a new GPU context, auto-selecting the best available GPU.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;

        let pipeline = Self::create_pipeline(&device);
        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    /// Create a GPU context from an existing wgpu device and queue.
    ///
    /// Use this when integrating with a framework (e.g., Bevy) that already
    /// owns the GPU device.
    pub fn from_wgpu(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let pipeline = Self::create_pipeline(&device);
        Self {
            device,
            queue,
            pipeline,
        }
    }

    fn create_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bevy_autodiff eval"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bevy_autodiff compute"),
            layout: None, // auto layout from shader
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }
}
