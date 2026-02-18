//! Graph utilities for the computation graph.
//!
//! This module provides:
//! - Topological sorting of variables for forward propagation
//! - Graph traversal helpers

pub(crate) mod topology;
pub(crate) mod traverse;

pub(crate) use topology::topological_order;
