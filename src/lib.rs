//! Automatic differentiation using Bevy ECS.
//!
//! `bevy_autodiff` implements automatic differentiation using symbolic graph
//! differentiation, with Bevy ECS as the computational graph backend.
//!
//! # Core Concepts
//!
//! - **ECS as computation graph**: Variables are entities, operations are components
//! - **Symbolic differentiation**: `differentiate(output, wrt)` creates new entities representing the derivative graph
//! - **Successive differentiation**: For d²f/dxdy, differentiate f w.r.t. x then w.r.t. y
//! - **Functional style**: Immutable graph nodes, pure differentiation
//!
//! # Example
//!
//! ```
//! use bevy_autodiff::AutoDiff;
//!
//! let mut ad = AutoDiff::new();
//!
//! // Create input variables
//! let x = ad.var(2.0);
//! let y = ad.var(3.0);
//!
//! // Build computation graph: f = x * y
//! let f = ad.mul(x, y);
//! assert_eq!(ad.eval(f), 6.0);
//!
//! // Symbolic differentiation creates new graph entities
//! let dfdx = ad.differentiate(f, x);
//! assert_eq!(ad.eval(dfdx), 3.0); // df/dx = y = 3
//!
//! // Higher-order: d²f/dxdy = 1
//! let d2fdxdy = ad.differentiate(dfdx, y);
//! assert_eq!(ad.eval(d2fdxdy), 1.0);
//! ```

pub mod compiled;
pub mod components;
pub mod context;
pub mod debug;
pub mod error;
pub mod graph;
#[macro_use]
pub mod macros;
pub mod optimize;
pub mod util;
pub mod var;

// Feature-gated operator overloading
pub mod ops;

#[cfg(test)]
mod tests;

// Re-exports for convenience
pub use compiled::{CompiledGraph, NodeOp};
pub use context::AutoDiff;
pub use error::{AutoDiffError, AutoDiffResult};
pub use var::Var;

// Re-export key component types
pub use components::{
    BinaryInputs, BinaryOp, Dependencies, IsConstant, IsInput, UnaryInput, UnaryOp, Value, Variable,
};

// Re-export debug utilities
pub use debug::{count_operations, to_dot, validate_graph};

// Re-export proc-macro when feature is enabled
#[cfg(feature = "proc-macros")]
pub use bevy_autodiff_macros::autodiff;

// Re-export graph traversal helpers
pub use graph::{
    collect_all_entities, find_all_inputs, get_binary_inputs, get_inputs, get_operation_name,
    get_unary_input, get_value, is_leaf, max_depth, visit_topological,
};

// Re-export optimization utilities
pub use optimize::{
    build_cse_table, count_cse_opportunities, simplify_binary, simplify_unary, CseTable,
    OpSignature, SimplifyResult,
};

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
    assert_send::<UnaryInput>();
    assert_sync::<UnaryInput>();
    assert_send::<BinaryInputs>();
    assert_sync::<BinaryInputs>();

    // Handle type
    assert_send::<Var>();
    assert_sync::<Var>();
}
