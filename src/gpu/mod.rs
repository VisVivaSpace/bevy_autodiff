//! GPU batch evaluation via wgpu.
//!
//! Evaluates a [`CompiledGraph`](crate::CompiledGraph) at many input points
//! in parallel on the GPU. All threads execute the same computation graph
//! with different input values — ideal for Monte Carlo simulation and
//! batch optimization.
//!
//! # Usage
//!
//! ```ignore
//! use bevy_autodiff::AutoDiff;
//! use bevy_autodiff::gpu::GpuContext;
//!
//! let gpu = GpuContext::new()?;
//!
//! let mut ad = AutoDiff::new();
//! let x = ad.var(0.0);
//! let f = ad.sin(x);
//! let graph = ad.compile_order(f, &[x], 1);
//!
//! let gpu_graph = gpu.prepare(&graph)?;
//! let results = gpu_graph.eval_batch(&gpu, &[&x_samples])?;
//! let values = results.values();
//! ```

mod context;
mod error;
mod graph;
mod types;

pub use context::GpuContext;
pub use error::GpuError;
pub use graph::{GpuGraph, GpuResults};
