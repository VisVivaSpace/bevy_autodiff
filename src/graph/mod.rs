//! Graph utilities for the computation graph.
//!
//! This module provides:
//! - Topological sorting of variables for forward propagation
//! - Cache management for Taylor data
//! - Graph traversal helpers

pub mod cache;
pub mod topology;
pub mod traverse;

pub use cache::{invalidate_cache, invalidate_dependents, is_cache_valid};
pub use topology::topological_order;
pub use traverse::{
    collect_all_entities, get_binary_inputs, get_inputs, get_operation_name, get_unary_input,
    get_value, is_leaf, max_depth, visit_topological, GraphTraverser,
};
