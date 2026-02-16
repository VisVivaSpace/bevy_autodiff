//! GPU graph preparation and batch evaluation.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::error::GpuError;
use super::types::convert_nodes;
use crate::compiled::CompiledGraph;

/// Workgroup size — must match the value in shader.wgsl.
const WORKGROUP_SIZE: u32 = 64;

/// Params uniform (must match shader.wgsl Params struct layout).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    num_nodes: u32,
    num_samples: u32,
    num_inputs: u32,
    num_outputs: u32,
}

/// A computation graph prepared for GPU evaluation.
///
/// Created by [`GpuContext::prepare`]. Holds the translated node array
/// and output index mapping. Reusable across multiple [`eval_batch`](Self::eval_batch) calls.
///
/// Derives [`bevy_ecs::component::Component`] and [`bevy_ecs::resource::Resource`]
/// for use in Bevy ECS.
#[derive(bevy_ecs::component::Component, bevy_ecs::resource::Resource)]
pub struct GpuGraph {
    nodes_buffer: wgpu::Buffer,
    output_indices_buffer: wgpu::Buffer,
    num_nodes: u32,
    num_inputs: u32,
    num_outputs: u32,
    /// Maps multi_index → position in output_indices (0 is always the primal value).
    partial_lookup: HashMap<Vec<usize>, usize>,
}

/// Results from a GPU batch evaluation.
///
/// Contains the computed values and partial derivatives for all samples.
pub struct GpuResults {
    data: Vec<f32>,
    num_samples: usize,
    num_outputs: usize,
    partial_lookup: HashMap<Vec<usize>, usize>,
}

impl GpuContext {
    /// Prepare a [`CompiledGraph`] for GPU evaluation.
    ///
    /// Translates the node array to GPU format and creates persistent buffers.
    /// The returned [`GpuGraph`] can be reused across multiple [`eval_batch`](GpuGraph::eval_batch) calls.
    pub fn prepare(&self, graph: &CompiledGraph) -> Result<GpuGraph, GpuError> {
        let gpu_nodes = convert_nodes(graph.nodes());

        // Build output indices: primal first, then each partial
        let mut output_indices: Vec<u32> = vec![graph.output_index() as u32];
        let mut partial_lookup = HashMap::new();
        for (multi_index, node_idx) in graph.partial_outputs() {
            let pos = output_indices.len();
            partial_lookup.insert(multi_index.clone(), pos);
            output_indices.push(*node_idx as u32);
        }

        let nodes_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("autodiff nodes"),
            contents: bytemuck::cast_slice(&gpu_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_indices_buffer =
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("autodiff output_indices"),
                contents: bytemuck::cast_slice(&output_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        Ok(GpuGraph {
            nodes_buffer,
            output_indices_buffer,
            num_nodes: gpu_nodes.len() as u32,
            num_inputs: graph.num_inputs() as u32,
            num_outputs: output_indices.len() as u32,
            partial_lookup,
        })
    }
}

impl GpuGraph {
    /// Evaluate the graph at many input points in parallel on the GPU.
    ///
    /// `input_samples` contains one `&[f32]` slice per input variable.
    /// All slices must have the same length (the sample count).
    ///
    /// Returns a [`GpuResults`] with values and partial derivatives for each sample.
    pub fn eval_batch(
        &self,
        ctx: &GpuContext,
        input_samples: &[&[f32]],
    ) -> Result<GpuResults, GpuError> {
        // --- Validation ---
        if input_samples.len() != self.num_inputs as usize {
            return Err(GpuError::InputMismatch {
                expected: self.num_inputs as usize,
                got: input_samples.len(),
            });
        }
        if input_samples.is_empty() || input_samples[0].is_empty() {
            return Err(GpuError::EmptySamples);
        }
        let num_samples = input_samples[0].len();
        for (i, samples) in input_samples.iter().enumerate().skip(1) {
            if samples.len() != num_samples {
                return Err(GpuError::SampleCountMismatch {
                    input_idx: i,
                    expected: num_samples,
                    got: samples.len(),
                });
            }
        }

        let num_samples_u32 = num_samples as u32;

        // --- Params uniform ---
        let params = Params {
            num_nodes: self.num_nodes,
            num_samples: num_samples_u32,
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
        };
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("autodiff params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // --- Inputs buffer (SoA layout) ---
        let mut input_data = Vec::with_capacity(self.num_inputs as usize * num_samples);
        for samples in input_samples {
            input_data.extend_from_slice(samples);
        }
        let inputs_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("autodiff inputs"),
                contents: bytemuck::cast_slice(&input_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // --- Values buffer (read-write storage) ---
        let values_size = (self.num_nodes as u64) * (num_samples as u64) * 4;
        let values_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("autodiff values"),
            size: values_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // --- Outputs buffer ---
        let outputs_size = (self.num_outputs as u64) * (num_samples as u64) * 4;
        let outputs_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("autodiff outputs"),
            size: outputs_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // --- Staging buffer (for CPU readback) ---
        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("autodiff staging"),
            size: outputs_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bind group ---
        let bind_group_layout = ctx.pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("autodiff bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.output_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: outputs_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Dispatch ---
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("autodiff encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("autodiff pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = num_samples_u32.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&outputs_buffer, 0, &staging_buffer, 0, outputs_size);
        ctx.queue.submit(Some(encoder.finish()));

        // --- Readback ---
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("internal: channel receiver dropped");
        });
        ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv().expect("internal: channel sender dropped")?;

        let data_raw = slice.get_mapped_range();
        let data: Vec<f32> = bytemuck::cast_slice(&data_raw).to_vec();
        drop(data_raw);
        staging_buffer.unmap();

        Ok(GpuResults {
            data,
            num_samples,
            num_outputs: self.num_outputs as usize,
            partial_lookup: self.partial_lookup.clone(),
        })
    }
}

impl GpuResults {
    /// Returns the primal function values for all samples.
    pub fn values(&self) -> &[f32] {
        &self.data[0..self.num_samples]
    }

    /// Returns partial derivative values for all samples.
    ///
    /// The `multi_index` must match one of the partials compiled into the graph.
    /// For example, `&[1, 0]` returns df/dx for a 2-input function compiled with order >= 1.
    ///
    /// Returns `None` if the requested partial was not compiled.
    pub fn partials(&self, multi_index: &[usize]) -> Option<&[f32]> {
        let &pos = self.partial_lookup.get(multi_index)?;
        let start = pos * self.num_samples;
        Some(&self.data[start..start + self.num_samples])
    }

    /// Returns the number of samples in this result set.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Returns the number of output channels (primal + partials).
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }
}

#[cfg(test)]
mod tests {
    use crate::AutoDiff;

    use super::*;

    fn gpu() -> Option<GpuContext> {
        GpuContext::new().ok()
    }

    #[test]
    fn gpu_context_creates() {
        let ctx = gpu();
        assert!(ctx.is_some(), "GPU context should be available");
    }

    #[test]
    fn gpu_eval_linear() {
        // f(x) = 2*x + 1
        let Some(ctx) = gpu() else { return };

        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let c2 = ad.constant(2.0);
        let two_x = ad.mul(c2, x);
        let c1 = ad.constant(1.0);
        let f = ad.add(two_x, c1);
        let graph = ad.compile_primal(f, &[x]).unwrap();

        let gpu_graph = ctx.prepare(&graph).unwrap();
        let x_vals: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let results = gpu_graph.eval_batch(&ctx, &[&x_vals]).unwrap();

        let values = results.values();
        assert_eq!(values.len(), 100);
        for (i, &v) in values.iter().enumerate() {
            let x = i as f32 * 0.1;
            let expected = 2.0 * x + 1.0;
            assert!(
                (v - expected).abs() < 1e-5,
                "sample {i}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn gpu_eval_sin() {
        let Some(ctx) = gpu() else { return };

        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let f = ad.sin(x);
        let graph = ad.compile_primal(f, &[x]).unwrap();

        let gpu_graph = ctx.prepare(&graph).unwrap();
        let x_vals: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let results = gpu_graph.eval_batch(&ctx, &[&x_vals]).unwrap();

        for (i, &v) in results.values().iter().enumerate() {
            let expected = (i as f32 * 0.1).sin();
            assert!(
                (v - expected).abs() < 1e-5,
                "sin sample {i}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn gpu_eval_two_inputs() {
        // f(x, y) = x * y
        let Some(ctx) = gpu() else { return };

        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let y = ad.var(0.0).unwrap();
        let f = ad.mul(x, y);
        let graph = ad.compile_primal(f, &[x, y]).unwrap();

        let gpu_graph = ctx.prepare(&graph).unwrap();
        let x_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y_vals: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let results = gpu_graph.eval_batch(&ctx, &[&x_vals, &y_vals]).unwrap();

        let values = results.values();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 5.0).abs() < 1e-5);
        assert!((values[1] - 12.0).abs() < 1e-5);
        assert!((values[2] - 21.0).abs() < 1e-5);
        assert!((values[3] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_eval_with_partials() {
        // f(x) = x^2, df/dx = 2x
        let Some(ctx) = gpu() else { return };

        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let f = ad.square(x);
        let graph = ad.compile_order(f, &[x], 1).unwrap();

        let gpu_graph = ctx.prepare(&graph).unwrap();
        let x_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let results = gpu_graph.eval_batch(&ctx, &[&x_vals]).unwrap();

        // Primal: x^2
        let values = results.values();
        for (i, &v) in values.iter().enumerate() {
            let x = x_vals[i];
            assert!(
                (v - x * x).abs() < 1e-4,
                "x^2 at x={x}: got {v}, expected {}",
                x * x
            );
        }

        // Partial: df/dx = 2x
        let dfdx = results.partials(&[1]).expect("partial [1] should exist");
        for (i, &d) in dfdx.iter().enumerate() {
            let x = x_vals[i];
            let expected = 2.0 * x;
            assert!(
                (d - expected).abs() < 1e-4,
                "df/dx at x={x}: got {d}, expected {expected}"
            );
        }
    }

    #[test]
    fn gpu_eval_composition() {
        // f(x) = sin(exp(x))
        let Some(ctx) = gpu() else { return };

        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let exp_x = ad.exp(x);
        let f = ad.sin(exp_x);
        let graph = ad.compile_primal(f, &[x]).unwrap();

        let gpu_graph = ctx.prepare(&graph).unwrap();
        let x_vals: Vec<f32> = vec![0.0, 0.1, 0.2, 0.5, 1.0];
        let results = gpu_graph.eval_batch(&ctx, &[&x_vals]).unwrap();

        for (i, &v) in results.values().iter().enumerate() {
            let expected = x_vals[i].exp().sin();
            assert!(
                (v - expected).abs() < 1e-4,
                "sin(exp({})) = {v}, expected {expected}",
                x_vals[i]
            );
        }
    }

    #[test]
    fn gpu_error_input_mismatch() {
        let Some(ctx) = gpu() else { return };

        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let y = ad.var(0.0).unwrap();
        let f = ad.add(x, y);
        let graph = ad.compile_primal(f, &[x, y]).unwrap();

        let gpu_graph = ctx.prepare(&graph).unwrap();
        // Pass 1 input array when graph expects 2
        let result = gpu_graph.eval_batch(&ctx, &[&[1.0, 2.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn gpu_error_empty_samples() {
        let Some(ctx) = gpu() else { return };

        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let f = ad.sin(x);
        let graph = ad.compile_primal(f, &[x]).unwrap();

        let gpu_graph = ctx.prepare(&graph).unwrap();
        let empty: &[f32] = &[];
        let result = gpu_graph.eval_batch(&ctx, &[empty]);
        assert!(result.is_err());
    }

    #[test]
    fn gpu_reuse_graph() {
        // Verify GpuGraph can be reused across multiple eval_batch calls
        let Some(ctx) = gpu() else { return };

        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let f = ad.sin(x);
        let graph = ad.compile_primal(f, &[x]).unwrap();

        let gpu_graph = ctx.prepare(&graph).unwrap();

        let r1 = gpu_graph
            .eval_batch(&ctx, &[&[0.0, 1.0]])
            .unwrap();
        let r2 = gpu_graph
            .eval_batch(&ctx, &[&[2.0, 3.0]])
            .unwrap();

        assert!((r1.values()[0] - 0.0_f32.sin()).abs() < 1e-5);
        assert!((r1.values()[1] - 1.0_f32.sin()).abs() < 1e-5);
        assert!((r2.values()[0] - 2.0_f32.sin()).abs() < 1e-5);
        assert!((r2.values()[1] - 3.0_f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_types_bevy_trait_bounds() {
        fn assert_resource<T: bevy_ecs::resource::Resource>() {}
        fn assert_component<T: bevy_ecs::component::Component>() {}
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_resource::<GpuContext>();
        assert_send::<GpuContext>();
        assert_sync::<GpuContext>();

        assert_component::<GpuGraph>();
        assert_resource::<GpuGraph>();
        assert_send::<GpuGraph>();
        assert_sync::<GpuGraph>();
    }
}
