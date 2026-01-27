//! Taylor-mode automatic differentiation using Bevy ECS.
//!
//! `vvad` implements higher-order automatic differentiation using Taylor series
//! propagation, with Bevy ECS as the computational graph backend.
//!
//! # Core Concepts
//!
//! - **ECS as computation graph**: Variables are entities, operations are components
//! - **Taylor-mode AD**: O(n²) complexity for n-th derivative (vs O(exp(n)) for naive nesting)
//! - **Univariate decomposition**: Directional derivatives avoid multivariate Bell polynomial complexity
//! - **Functional style**: Immutable Taylor data, pure propagation functions
//!
//! # Example
//!
//! ```
//! use vvad::AutoDiff;
//!
//! let mut ad = AutoDiff::new();
//!
//! // Create input variables
//! let x = ad.var(2.0);
//! let y = ad.var(3.0);
//!
//! // Build computation graph
//! let f = ad.mul(x, y);  // f = x * y
//!
//! // Evaluate
//! assert_eq!(ad.eval(f), 6.0);
//! ```
//!
//! # Taylor Coefficient Storage
//!
//! Coefficients are stored **normalized** (divided by k!):
//! - `f_k = f^(k)(a) / k!` for numerical stability
//! - Recurrence formulas naturally produce normalized coefficients
//! - Multiply by k! only when extracting actual derivatives
//!
//! # Architecture
//!
//! The crate is organized into:
//! - `components`: ECS components for variables, operations, Taylor data
//! - `var`: Lightweight variable handle type
//! - `context`: Main `AutoDiff` API for graph construction
//! - `util`: Numerical utilities (factorial, binomial, Horner evaluation)

pub mod components;
pub mod context;
pub mod debug;
pub mod error;
pub mod graph;
#[macro_use]
pub mod macros;
pub mod optimize;
pub mod partials;
pub mod reverse;
pub mod taylor;
pub mod util;
pub mod var;

// Feature-gated operator overloading
pub mod ops;

#[cfg(test)]
mod tests;

// Re-exports for convenience
pub use context::AutoDiff;
pub use error::{TaylorError, TaylorResult};
pub use var::Var;

// Re-export key component types
pub use components::{
    AdjointTaylor, BinaryInputs, BinaryOp, Dependencies, Direction, IsConstant, IsInput,
    MultiIndex, TaylorData, UnaryInput, UnaryOp, Value, Variable,
};

// Re-export partial derivative functions
pub use partials::{
    compute_partial, extract_directional_derivative, get_mixed_partial, get_mixed_partial_2,
    get_pure_partial,
};

// Re-export reverse mode functions
pub use reverse::{compute_gradient_reverse, reverse_accumulate};

// Re-export debug utilities
pub use debug::{count_operations, debug_taylor_data, to_dot, validate_graph};

// Re-export proc-macro when feature is enabled
#[cfg(feature = "proc-macros")]
pub use vvad_macros::autodiff;

// Re-export graph traversal helpers
pub use graph::{
    collect_all_entities, get_binary_inputs, get_inputs, get_operation_name, get_unary_input,
    get_value, is_leaf, max_depth, visit_topological, GraphTraverser,
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
    assert_send::<TaylorData>();
    assert_sync::<TaylorData>();
    assert_send::<AdjointTaylor>();
    assert_sync::<AdjointTaylor>();
    assert_send::<Value>();
    assert_sync::<Value>();
    assert_send::<Dependencies>();
    assert_sync::<Dependencies>();

    // Type wrappers
    assert_send::<Direction>();
    assert_sync::<Direction>();
    assert_send::<MultiIndex>();
    assert_sync::<MultiIndex>();

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
    assert_send::<UnaryInput>();
    assert_sync::<UnaryInput>();
    assert_send::<BinaryInputs>();
    assert_sync::<BinaryInputs>();

    // Handle type
    assert_send::<Var>();
    assert_sync::<Var>();
}
