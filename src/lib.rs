//! Automatic differentiation using Bevy ECS.
//!
//! `bevy_autodiff` implements automatic differentiation using symbolic graph
//! differentiation, with Bevy ECS as the computational graph backend.
//! Variables are ECS entities, operations are components, and derivatives
//! are computed by applying the chain rule symbolically with constant folding.
//!
//! # Core Concepts
//!
//! - **ECS as computation graph**: Variables are entities, operations are components
//! - **Symbolic differentiation**: [`AutoDiff::differentiate`] creates new entities
//!   representing the derivative graph via the chain rule
//! - **Successive differentiation**: For d²f/dxdy, differentiate f w.r.t. x then w.r.t. y
//! - **[`CompiledGraph`]**: Flattens the ECS graph into a `Vec<NodeOp>` for fast repeated evaluation
//! - **Reverse-mode gradient**: [`CompiledGraph::gradient`] computes all partial derivatives
//!   in a single backward pass, independent of input count
//! - **Forward-mode partials**: [`AutoDiff::compile_order`] pre-compiles symbolic derivative
//!   subgraphs for higher-order or mixed partial derivatives
//!
//! # Example
//!
//! ```
//! use bevy_autodiff::AutoDiff;
//!
//! let mut ad = AutoDiff::new();
//!
//! // Create input variables
//! let x = ad.var(2.0).unwrap();
//! let y = ad.var(3.0).unwrap();
//!
//! // Build computation graph: f = x * y
//! let f = ad.mul(x, y);
//! assert_eq!(ad.eval(f).unwrap(), 6.0);
//!
//! // Symbolic differentiation creates new graph entities
//! let dfdx = ad.differentiate(f, x).unwrap();
//! assert_eq!(ad.eval(dfdx).unwrap(), 3.0); // df/dx = y = 3
//!
//! // Higher-order: d²f/dxdy = 1
//! let d2fdxdy = ad.differentiate(dfdx, y).unwrap();
//! assert_eq!(ad.eval(d2fdxdy).unwrap(), 1.0);
//! ```
//!
//! # Reverse-mode gradient
//!
//! ```
//! use bevy_autodiff::AutoDiff;
//!
//! let mut ad = AutoDiff::new();
//! let x = ad.var(1.0).unwrap();
//! let y = ad.var(2.0).unwrap();
//!
//! let x2 = ad.square(x);
//! let y2 = ad.square(y);
//! let f = ad.add(x2, y2); // x² + y²
//!
//! // Compile primal only, then use reverse-mode for gradient
//! let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
//! cg.eval(&[1.0, 2.0]).unwrap();
//! assert_eq!(cg.value(), 5.0);
//!
//! let grad = cg.gradient();
//! assert_eq!(grad, &[2.0, 4.0]); // [2x, 2y]
//! ```

pub mod codegen;
pub mod compiled;
pub mod components;
pub mod context;
pub mod debug;
pub mod error;
pub mod graph;
#[macro_use]
pub mod macros;
pub mod ops;
pub mod var;

#[cfg(test)]
mod tests;

// Re-exports: essential public API
pub use compiled::{CompiledGraph, NodeOp};
pub use context::AutoDiff;
pub use error::AutoDiffError;
pub use var::Var;

// Re-export key component types
pub use components::{BinaryOp, Dependencies, IsConstant, IsInput, UnaryOp, Value, Variable};

// Re-export debug utilities
pub use debug::{count_operations, to_dot, validate_graph};

// Re-export proc-macro when feature is enabled
#[cfg(feature = "proc-macros")]
pub use bevy_autodiff_macros::autodiff;

// GPU batch evaluation via wgpu
#[cfg(feature = "wgpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "wgpu")))]
pub mod gpu;

// Compile-time assertions that all components are Send + Sync
// This is required for Bevy's parallel system execution
#[allow(dead_code)]
fn _assert_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    // Core data components
    assert_send::<Value>();
    assert_sync::<Value>();
    assert_send::<Dependencies>();
    assert_sync::<Dependencies>();

    // Marker components
    assert_send::<Variable>();
    assert_sync::<Variable>();
    assert_send::<IsInput>();
    assert_sync::<IsInput>();
    assert_send::<IsConstant>();
    assert_sync::<IsConstant>();

    // Operation components
    assert_send::<UnaryOp>();
    assert_sync::<UnaryOp>();
    assert_send::<BinaryOp>();
    assert_sync::<BinaryOp>();
    assert_send::<components::UnaryOpMarker>();
    assert_sync::<components::UnaryOpMarker>();
    assert_send::<components::BinaryOpMarker>();
    assert_sync::<components::BinaryOpMarker>();
    assert_send::<components::UnaryInput>();
    assert_sync::<components::UnaryInput>();
    assert_send::<components::BinaryInputs>();
    assert_sync::<components::BinaryInputs>();

    // Handle type
    assert_send::<Var>();
    assert_sync::<Var>();

    // Compiled graph (derives Component + Resource)
    assert_send::<CompiledGraph>();
    assert_sync::<CompiledGraph>();
}

// GPU types are Send + Sync (required for Bevy Resource/Component)
#[cfg(feature = "wgpu")]
#[allow(dead_code)]
fn _assert_gpu_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<gpu::GpuContext>();
    assert_sync::<gpu::GpuContext>();
    assert_send::<gpu::GpuGraph>();
    assert_sync::<gpu::GpuGraph>();
}
